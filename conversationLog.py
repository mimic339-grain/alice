"""One record per conversational turn, with everything needed to explain it afterwards.

This exists to answer "why did she say that?" after the fact. A transcript alone cannot: the
same input produces different replies depending on what memory was recalled, what was on
screen, how long the history had grown, and which sampling settings were in force. So each
turn records the exact message list the model was sent, alongside what came back, what was
actually spoken, and any feedback the user later attached to it.

The file is append-only JSONL under logs/. One line per record, each with a "type":

    turn      a complete exchange: input -> prompt -> reply -> spoken -> timing
    feedback  the user's verdict on a turn, keyed by turn_id
    event     things that are not replies to anyone: the startup greeting, a cancelled turn

Append-only means a crash never damages what was already written, and feedback can arrive
long after its turn without rewriting anything. showlog.py joins the pieces for reading.
"""

import json
import os
import threading
import time
import uuid


class ConversationLog:
    def __init__(self, path=None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.path = path or os.path.join(base, 'logs', 'conversation.jsonl')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self._open = {}   # turn_id -> partial record, until end_turn writes it

    # ---- low level -------------------------------------------------------------------------

    def _write(self, record):
        record.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S"))
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            try:
                with open(self.path, 'a', encoding='utf-8') as f:
                    f.write(line + "\n")
            except Exception as e:
                # Logging must never be the thing that breaks the conversation.
                print(f"LOG: write failed ({type(e).__name__}: {e})")

    # ---- a turn ----------------------------------------------------------------------------

    def begin_turn(self, source, text):
        """Input arrived. Returns the turn_id that everything else about it hangs off."""
        # Millisecond stamp first so ids sort into conversation order; the random tail only
        # separates two inputs that land in the same millisecond.
        now = time.time()
        turn_id = (time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
                   + f"{int((now % 1) * 1000):03d}-" + uuid.uuid4().hex[:3])
        with self._lock:
            self._open[turn_id] = {
                "type": "turn",
                "turn_id": turn_id,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "input": {"source": source, "text": text},
                "prompt": None,
                "sentences": [],
                "tool_calls": [],
                "reply": None,
                "timing": {"t0": time.time()},
            }
        # The input line is written straight away as well, so a crash mid-generation still
        # leaves a trace of what was asked. showlog.py hides it once the full turn exists.
        self._write({"type": "input", "turn_id": turn_id, "source": source, "text": text})
        return turn_id

    def record_prompt(self, turn_id, messages, params, prompt_tokens, context,
                      history_len, trimmed):
        """What the model was actually given. This is the part that explains the reply."""
        rec = self._open.get(turn_id)
        if rec is None:
            return
        rec["prompt"] = {
            "messages": messages,
            "params": {k: v for k, v in params.items() if k != "messages"},
            "prompt_tokens": prompt_tokens,
            # The volatile part on its own, so it can be read without digging through the
            # full message list: memory that was recalled, what was on screen, and so on.
            "context": context,
            "history_len": history_len,
            "history_trimmed": trimmed,
        }

    def record_sentence(self, turn_id, raw, spoken, filtered=None):
        """One sentence as generated and as it went to the voice (or why it did not)."""
        rec = self._open.get(turn_id)
        if rec is None:
            return
        entry = {"raw": raw, "spoken": spoken}
        if filtered:
            entry["filtered"] = filtered
        rec["sentences"].append(entry)
        if spoken and "first_word" not in rec["timing"]:
            rec["timing"]["first_word"] = time.time()

    def record_tool_calls(self, turn_id, calls):
        rec = self._open.get(turn_id)
        if rec is None:
            return
        for call in calls:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            rec["tool_calls"].append({"name": fn.get("name"), "arguments": fn.get("arguments")})

    def end_turn(self, turn_id, raw_reply, spoken_reply, emotion, cancelled=False):
        """Reply finished (or was cut off). Writes the complete record."""
        with self._lock:
            rec = self._open.pop(turn_id, None)
        if rec is None:
            return
        now = time.time()
        t0 = rec["timing"].pop("t0")
        first = rec["timing"].pop("first_word", None)
        rec["timing"] = {
            "first_word_ms": round((first - t0) * 1000) if first else None,
            "total_ms": round((now - t0) * 1000),
        }
        rec["reply"] = {
            "raw": raw_reply,
            "spoken": spoken_reply,
            "emotion": emotion,
            "cancelled": cancelled,
        }
        self._write(rec)

    # ---- things that are not turns ---------------------------------------------------------

    def record_feedback(self, turn_id, verdict, user, alice, better, emotion):
        self._write({
            "type": "feedback",
            "turn_id": turn_id,
            "verdict": verdict,
            "user": user,
            "alice": alice,
            "better": better,
            "emotion": emotion,
        })

    def record_event(self, kind, **data):
        self._write({"type": "event", "kind": kind, **data})
