import copy
import re
import requests
import sseclient
import json
import time
from dotenv import load_dotenv
from constants import *
from modules.injection import Injection
from safety import SafetyFilter


class AbstractLLMWrapper:

    def __init__(self, signals, tts, llmState, modules=None):
        self.signals = signals
        self.llmState = llmState
        self.tts = tts
        self.API = self.API(self)
        if modules is None:
            self.modules = {}
        else:
            self.modules = modules

        self.headers = {"Content-Type": "application/json"}

        load_dotenv()

        #Below constants must be set by child classes
        self.SYSTEM_PROMPT = None
        self.LLM_ENDPOINT = None
        self.CONTEXT_SIZE = None
        self.tokenizer = None

    def is_filtered(self, text):
        """True when this text must not be spoken. See safety.py for what is checked."""
        safe, reason = SafetyFilter(self.llmState.blacklist).check(text)
        if not safe:
            print(f"SAFETY: blocked ({reason})")
        return not safe

    # Assembles all the injections from all modules into a single prompt by increasing priority
    def assemble_injections(self, injections=None):
        if injections is None:
            injections = []

        # Gather all injections from all modules
        for module in self.modules.values():
            injections.append(module.get_prompt_injection())

        # Let all modules clean up once the prompt injection has been fetched from all modules
        for module in self.modules.values():
            module.cleanup()

        # Sort injections by priority
        injections = sorted(injections, key=lambda x: x.priority)

        # Assemble injections
        prompt = ""
        for injection in injections:
            prompt += injection.text
        return prompt

    def generate_prompt(self):
        """Build the message list sent to the model.

        Ordering is chosen so that llama.cpp can reuse its KV cache across turns. Everything
        that never changes comes first and stays byte-identical:

            [system] persona          <- constant
            [few-shot turns]          <- constant
            [conversation history]    <- append-only, so the prefix still matches
            [volatile context]        <- memory recall, what's on screen, /no_think

        Anything that changes every turn therefore lives in the LAST message. Put it any
        earlier and the whole prefix after it is invalidated, which on this card meant
        reprocessing ~950 tokens (about 14 seconds) before generation could even start.

        This used to be one flat string in a single "user" message. That made the model treat
        its own character sheet as user-supplied text and follow it only loosely; as a real
        system message plus real assistant turns, the persona actually holds.
        """
        history = copy.deepcopy(self.signals.history)

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages += [{"role": role, "content": content} for role, content in FEW_SHOT]
        fixed_count = len(messages)

        for message in history:
            if not message["content"]:
                continue
            messages.append({"role": message["role"], "content": message["content"]})

        # Collected once per prompt: assemble_injections() also runs each module's cleanup().
        context = self.assemble_injections()
        # Kept for the turn log: this is the part of the prompt that differs turn to turn, and
        # therefore the part most often behind "why did she say that".
        self._last_context = context.strip()
        self._last_trimmed = 0

        # /no_think disables Qwen3's chain-of-thought mode, which otherwise spends max_tokens on
        # internal reasoning before the one-line reply -- fatal for conversational tempo.
        # Harmless on models without a thinking mode.
        trailing = (context.strip() + "\n" if context.strip() else "") + "/no_think"
        if messages[-1]["role"] == "user":
            messages[-1] = {"role": "user",
                            "content": messages[-1]["content"] + "\n" + trailing}
        else:
            messages.append({"role": "user", "content": trailing})

        while True:
            prompt_tokens = len(self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0])

            # Maximum 90% context size usage before prompting LLM
            if prompt_tokens < 0.9 * self.CONTEXT_SIZE:
                break

            # Drop the oldest real exchange. The system prompt and the few-shot examples are
            # what the persona is made of, so they are never what gets trimmed.
            if len(messages) <= fixed_count + 1:
                raise RuntimeError("Prompt too long even with no conversation history")
            messages.pop(fixed_count)
            self._last_trimmed += 1
            print("Prompt too long, removing earliest message")

        self._last_prompt_tokens = prompt_tokens
        self.signals.sio_queue.put(
            ("full_prompt", "\n".join(f'{m["role"]}: {m["content"]}' for m in messages)))
        return messages

    def warm_up(self):
        """Make the model process the constant part of the prompt once, before anyone is waiting.

        The persona and the few-shot turns come to ~1000 tokens, and llama.cpp has to run them
        through the model once before it can cache them -- about 16 seconds on this card. After
        that every turn reuses the cache and costs only the new tokens.

        Paying it here means the user waits during startup, which they expect, instead of on
        the first thing they say, which they do not. Failure is not worth reporting: the only
        consequence is that the first real turn is slow, exactly as it was before.
        """
        try:
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
            messages += [{"role": role, "content": content} for role, content in FEW_SHOT]
            messages.append({"role": "user", "content": "/no_think"})
            # Generate a few real tokens rather than one. Tool calling is enforced by a grammar
            # the backend compiles on first use, and that compilation is part of what the first
            # turn was paying for; a single token does not reliably trigger it.
            payload = {"mode": "instruct", "stream": False, "max_tokens": 16,
                       "messages": messages}

            # The tool definitions must be here too. The chat template renders them into the
            # system block, so warming up without them produces a different prefix from the one
            # every real turn sends -- the cache misses and the first reply pays the full cost
            # anyway, which is the whole thing this is meant to avoid.
            action_router = self.modules.get("action")
            if action_router is not None:
                tools = action_router.get_tool_specs()
                if tools:
                    payload["tools"] = tools

            # Two passes. The first makes the backend process the constant prefix; the second
            # is shaped like a real turn (a short user line on the end, streamed) because that
            # is measurably not the same work -- warming only with the prefix still left the
            # first real reply taking ~9s instead of the ~2.7s every later reply takes.
            for extra in ("", "こんにちは"):
                if extra:
                    payload["messages"] = messages[:-1] + [
                        {"role": "user", "content": extra + "\n/no_think"}]
                    payload["stream"] = True
                response = requests.post(self.LLM_ENDPOINT + "/v1/chat/completions",
                                         headers=self.headers, json=payload,
                                         verify=False, timeout=180, stream=payload["stream"])
                # Drain it, otherwise the connection closes early and the work may be abandoned.
                for _ in response.iter_lines():
                    pass
            print("LLM: prompt cache warmed")
        except Exception as e:
            print(f"LLM: warm-up skipped ({type(e).__name__}) -- first reply will be slow")

    def generate_line(self, instruction, max_tokens=60, timeout=60):
        """Ask for one in-character line outside the conversation loop.

        Used for things the AI should say that nobody prompted -- the startup greeting, for
        one. The persona and few-shot turns are sent exactly as a normal turn sends them,
        tool definitions included, so this hits the same warmed prompt cache instead of
        forcing a fresh ~950-token prefix through the model.

        Returns the raw reply (emotion tag still attached), or None if anything goes wrong.
        The caller is expected to have a written fallback.
        """
        try:
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
            messages += [{"role": role, "content": content} for role, content in FEW_SHOT]
            messages.append({"role": "user", "content": instruction + "\n/no_think"})

            payload = {
                "mode": "instruct",
                "stream": False,
                "max_tokens": max_tokens,
                # Only slightly looser than a normal reply. Pushed to 0.9/1.15 to get more
                # variety, this model started producing malformed Japanese ("わたしった") and
                # sentences that meant nothing -- the same 4-bit fragility noted in constants.
                # Variety has to come from the instruction, not from loosening sampling.
                "temperature": 0.8,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "messages": messages,
            }
            action_router = self.modules.get("action")
            if action_router is not None:
                tools = action_router.get_tool_specs()
                if tools:
                    payload["tools"] = tools

            response = requests.post(self.LLM_ENDPOINT + "/v1/chat/completions",
                                     headers=self.headers, json=payload,
                                     verify=False, timeout=timeout)
            content = response.json()["choices"][0]["message"].get("content", "")
            return self.fix_chinese_chars(content.strip()) or None
        except Exception as e:
            print(f"LLM: generate_line failed ({type(e).__name__}: {e})")
            return None

    def prepare_payload(self):
        raise NotImplementedError("Must implement prepare_payload in child classes")

    @staticmethod
    def fix_chinese_chars(text):
        """Replace simplified-Chinese characters the model leaks into its Japanese."""
        for cn, jp in CN_TO_JP_CHARS.items():
            if cn in text:
                text = text.replace(cn, jp)
        return text

    @staticmethod
    def strip_leading_interjection(text):
        """Drop an opening interjection like "へぇ〜！" or "ええっ！".

        Used only when the reflex model already voiced one this turn. Asking the main model not
        to open with one is unreliable, so the duplicate is removed after the fact instead. The
        length guard keeps real content (e.g. a reply that is itself just "そうなんだ！") intact.
        """
        # The tilde class needs every variant a model might emit: 〜 U+301C (wave dash),
        # ～ U+FF5E (fullwidth tilde) and plain ASCII ~ are visually identical but distinct
        # codepoints, and missing one silently defeats the whole strip.
        stripped = re.sub(r'^[ぁ-んァ-ンーｰ〜～~]{1,4}[！!？?、,。.…]+\s*', '', text)
        if stripped and len(stripped) >= 5:
            return stripped
        return text

    def clean_sentence(self, sentence, is_first, reflex_spoken):
        """Apply every text fix-up to one sentence, in the order they must happen.

        Runs per sentence rather than per reply so speech can start before generation finishes.
        """
        if is_first:
            # The emotion tag only ever appears at the very start of the reply.
            tag_match = re.match(r'^\s*\[(\w+)\]\s*', sentence)
            if tag_match and tag_match.group(1).lower() in EMOTION_TAGS:
                self.signals.current_emotion = tag_match.group(1).lower()
                sentence = sentence[tag_match.end():]
            else:
                self.signals.current_emotion = "neutral"

            if reflex_spoken:
                sentence = self.strip_leading_interjection(sentence)

        return self.fix_chinese_chars(sentence)

    def speak_sentence(self, sentence, is_first, reflex_spoken):
        """Clean one sentence and send it to the TTS queue. Returns what was actually said."""
        raw = sentence
        log = self.signals.log
        turn_id = self.signals.current_turn_id

        sentence = self.clean_sentence(sentence, is_first, reflex_spoken)
        if not sentence.strip():
            log.record_sentence(turn_id, raw, "", filtered="empty after cleaning")
            return ""

        # Nothing but punctuation survived cleaning. Streaming can still produce this when
        # a terminator arrives in a later chunk than the clause it belongs to, and handing
        # it to the voice makes her emit a stray noise between two real sentences.
        if not re.search(r'[^\s。、！？!?.\,…・ー~〜～]', sentence):
            log.record_sentence(turn_id, raw, "", filtered="punctuation only")
            return ""

        # Filtering happens per sentence, before that sentence is voiced. Checking the whole
        # reply at the end would be too late once earlier sentences are already playing.
        if self.is_filtered(sentence):
            print("FILTERED: " + sentence)
            self.signals.sio_queue.put(("next_chunk", "[filtered]"))
            log.record_sentence(turn_id, raw, "", filtered="safety filter")
            return ""

        log.record_sentence(turn_id, raw, sentence)
        self.tts.play(sentence)
        return sentence

    def stream_reply(self, data, reflex_spoken, is_first):
        """Stream one response, speaking each sentence as it completes.

        Returns (spoken_text, tool_calls). Sentences go to the TTS the moment they are finished
        rather than after the whole reply, which is what keeps time-to-first-word around a
        second instead of the length of the entire generation (design doc sec.04.1).
        """
        stream_response = requests.post(self.LLM_ENDPOINT + "/v1/chat/completions", headers=self.headers,
                                        json=data, verify=False, stream=True)
        response_stream = sseclient.SSEClient(stream_response)

        spoken_message = ''
        buffer = ''
        tool_calls = []
        # The reply as generated, before the emotion tag is stripped and before any sentence
        # is filtered. The log keeps this next to what was actually spoken.
        self._last_raw_reply = ''
        for event in response_stream.events():
            if self.llmState.next_cancelled:
                continue

            # Some streams end with a literal "[DONE]" sentinel instead of a JSON payload
            if event.data.strip() == "[DONE]":
                continue

            delta = json.loads(event.data)['choices'][0]['delta']

            # Tool calls arrive complete in a single chunk rather than piecewise.
            if delta.get('tool_calls'):
                tool_calls.extend(delta['tool_calls'])
                continue

            # The final chunk of a stream (and any reasoning-only chunk from a thinking model)
            # may have no "content" key -- only "role", or nothing at all.
            chunk = delta.get('content')
            if not chunk:
                continue
            buffer += chunk
            self._last_raw_reply += chunk
            self.signals.sio_queue.put(("next_chunk", chunk))

            while True:
                # Match a RUN of terminators, so "？！" ends one sentence instead of two
                # and the second one is not a lone "！" sent to the voice on its own.
                boundary = re.search(r'[。！？!?\n]+', buffer)
                if not boundary:
                    break
                sentence, buffer = buffer[:boundary.end()], buffer[boundary.end():]
                if not self.signals.AI_speaking:
                    self.signals.AI_speaking = True
                spoken_message += self.speak_sentence(sentence, is_first, reflex_spoken)
                is_first = False

        # Whatever is left never reached a sentence ending (the model stopped mid-thought, or the
        # whole reply was one clause with no final punctuation).
        if buffer.strip() and not self.llmState.next_cancelled:
            if not self.signals.AI_speaking:
                self.signals.AI_speaking = True
            spoken_message += self.speak_sentence(buffer, is_first, reflex_spoken)

        return spoken_message, tool_calls

    def run_tool_calls(self, tool_calls):
        """Execute the actions the model asked for. Unregistered names are dropped."""
        action_router = self.modules.get("action")
        if action_router is None:
            return

        for call in tool_calls:
            function = call.get("function", {})
            action_router.execute(function.get("name", ""), function.get("arguments", ""))

    def prompt(self):
        if not self.llmState.enabled:
            return

        self.signals.AI_thinking = True
        self.signals.new_message = False
        self.signals.sio_queue.put(("reset_next_message", None))

        data = self.prepare_payload()

        # Everything the model was given, filed under this turn. When a reply is wrong, this
        # is what tells apart "the instruction was not in the prompt" from "it was there and
        # the model ignored it" -- two different fixes.
        log = self.signals.log
        turn_id = self.signals.current_turn_id
        log.record_prompt(turn_id, data.get("messages", []), data,
                          getattr(self, "_last_prompt_tokens", None),
                          getattr(self, "_last_context", ""),
                          len(self.signals.history), getattr(self, "_last_trimmed", 0))

        # The reflex hint is baked into the payload now; it must not leak into the next turn.
        # Kept locally because the reply still needs de-duplicating against it below.
        reflex_spoken = self.signals.pending_reflex
        self.signals.pending_reflex = None

        spoken_message, tool_calls = self.stream_reply(data, reflex_spoken, is_first=True)
        raw_reply = self._last_raw_reply

        if self.llmState.next_cancelled:
            self.llmState.next_cancelled = False
            self.signals.sio_queue.put(("reset_next_message", None))
            self.signals.AI_thinking = False
            log.end_turn(turn_id, raw_reply, spoken_message, self.signals.current_emotion,
                         cancelled=True)
            return

        # When the model calls a tool it returns no spoken line at all, so run the action and
        # then ask again for something to actually say about it.
        if tool_calls:
            log.record_tool_calls(turn_id, tool_calls)
            self.run_tool_calls(tool_calls)
            follow_up = self.prepare_payload()
            # Drop the tools on the second pass: the action already ran, and leaving them in
            # invites the model to call it again instead of speaking.
            follow_up.pop("tools", None)
            more, _ = self.stream_reply(follow_up, reflex_spoken, is_first=not spoken_message)
            spoken_message += more
            raw_reply += self._last_raw_reply

        print("AI OUTPUT: " + spoken_message)
        self.signals.last_message_time = time.time()
        self.signals.AI_thinking = False

        # Store what was actually said, not the raw generation -- otherwise the emotion tag and
        # any filtered sentence would come back as context on the next turn.
        self.signals.history.append({"role": "assistant", "content": spoken_message})
        log.end_turn(turn_id, raw_reply, spoken_message, self.signals.current_emotion)

    class API:
        def __init__(self, outer):
            self.outer = outer

        def get_blacklist(self):
            return self.outer.llmState.blacklist

        def set_blacklist(self, new_blacklist):
            self.outer.llmState.blacklist = new_blacklist
            with open('blacklist.txt', 'w', encoding='utf-8') as file:
                for word in new_blacklist:
                    file.write(word + "\n")

            # Notify clients
            self.outer.signals.sio_queue.put(('get_blacklist', new_blacklist))

        def set_LLM_status(self, status):
            self.outer.llmState.enabled = status
            if status:
                self.outer.signals.AI_thinking = False
            self.outer.signals.sio_queue.put(('LLM_status', status))

        def get_LLM_status(self):
            return self.outer.llmState.enabled

        def cancel_next(self):
            self.outer.llmState.next_cancelled = True
            # For text-generation-webui: Immediately stop generation
            requests.post(self.outer.LLM_ENDPOINT + "/v1/internal/stop-generation", headers={"Content-Type": "application/json"})
