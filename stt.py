import collections
import logging
import re
import time
from RealtimeSTT import AudioToTextRecorder
from audioDevices import resolve_input_device, check_input_health
from constants import *


class STT:
    def __init__(self, signals, tts=None, llm_api=None):
        self.recorder = None
        self.signals = signals
        self.API = self.API(self)
        self.enabled = True
        # Needed for barge-in: interrupt TTS playback and in-flight LLM generation the moment
        # the human starts talking again, instead of letting the AI talk over them (design doc sec.04.2).
        self.tts = tts
        self.llm_api = llm_api
        # Echo guard state: one snapshot per recording, taken when the recording OPENS, because
        # by the time Whisper has finished she has long since gone quiet.
        #
        # A QUEUE and not a single value. Whisper 'small' runs on the CPU here and falls behind,
        # so several recordings can be open or waiting while an earlier one is still being
        # transcribed. With one shared slot, the later recording's snapshot overwrote the
        # earlier one's and the transcription was judged against the wrong recording -- which is
        # exactly how 「なんでもあるのかな？」 came back as 「何ではあるのか.」 and got through.
        # RealtimeSTT transcribes recordings in the order they finished, so the oldest snapshot
        # belongs to the next transcription. Bounded in case the two ever get out of step: a
        # wrong-but-recent snapshot beats unbounded growth.
        self._recording_snapshots = collections.deque(maxlen=8)

    # Whisper always returns something, so noise and half-syllables come back as "あ." or "まず."
    # or an empty string. Answering those is worse than ignoring them: there is nothing to react
    # to, so the reply is invented, and the log showed her turning that into random irritation
    # ("[angry] うざったい。話しないで。" to an input of just "アリス.").
    MIN_MEANINGFUL_CHARS = 3

    def is_meaningful(self, text):
        """False for transcriptions too short or too empty to be something a person said."""
        stripped = re.sub(r'[\s。、．，.,!?！？…]', '', text or '')
        if len(stripped) < self.MIN_MEANINGFUL_CHARS:
            return False
        # A run of one repeated character ("ああああ") is noise, not speech.
        return len(set(stripped)) > 1

    def is_echo(self):
        """True when this transcription is most likely the AI's own voice coming back in.

        Measured on this machine: she said 「アリス、起動。おひさま、こんばんわ。」, the
        microphone heard it, and Whisper returned 「お前の前は.」 and 「お前はアジで.」. Both
        became user turns, and she answered them with 「え？そんなこと言ってないよ。」 -- a reply
        that reads like a persona failure and is nothing of the kind.

        The guard that was here before checked signals.AI_speaking inside process_text, which
        never fires: post_speech_silence_duration plus transcription time means Whisper returns
        well after AI_speaking has gone back to False. The log showed the sequence plainly --
        "AI Talking Stop", then "STT OUTPUT", then "New Message".

        So both halves of the question are answered at recording_start instead. A recording
        that opened while she was talking is her. A recording that opened in the moment right
        after she stopped still holds the tail of her last word, because the audio is buffered
        ahead of the flag. Without acoustic echo cancellation, timing is the only signal there
        is.
        """
        if self._recording_snapshots:
            overlapped, quiet_for = self._recording_snapshots.popleft()
        else:
            # No snapshot to pair with. Fall back to reading the state now, which is weak but
            # better than treating an unknown recording as definitely a person.
            overlapped = self.signals.AI_speaking
            quiet_for = time.time() - self.signals.ai_quiet_since
        return overlapped or quiet_for < STT_ECHO_GUARD_TAIL

    def process_text(self, text):
        # Consumed first, unconditionally: is_echo() pops the snapshot queue, so returning
        # early without calling it would leave every later transcription paired with the wrong
        # recording. One transcription in, one snapshot out, whatever happens next.
        echo = self.is_echo()

        if not self.enabled:
            return

        if not self.is_meaningful(text):
            print(f"STT IGNORED (too short to be speech): {text!r}")
            return

        print("STT OUTPUT: " + text)

        if STT_ECHO_GUARD and echo:
            # Dropped completely -- not logged as a turn, not appended to history, no prompt.
            # Appending it "but not answering it" is not a safe middle ground: it stays in the
            # context window as something the user supposedly said, and she answers it on the
            # next real turn instead.
            print(f"STT DROPPED (her own voice coming back): {text!r}")
            self.signals.sio_queue.put(("heard_dropped", text))
            return

        # Barge-in, second half: abandon a reply she is still generating, now that there is
        # actual transcribed speech to replace it with.
        #
        # This used to happen the instant the VAD detected *sound*, before anything had been
        # transcribed. With a sensitive condenser mic in a room, that meant every small noise
        # cancelled the reply in progress -- the log showed "PROMPTING AI" followed immediately
        # by "AI Thinking Stop", over and over, and she never finished a single sentence.
        # Waiting for real words costs the transcription delay and makes the cancel deliberate.
        if self.signals.AI_thinking and self.llm_api is not None:
            self.llm_api.cancel_next()

        # Show it. The chat window had no way to display a spoken turn at all, so the only
        # visible sign of a misheard sentence was her answering something nobody said -- which
        # reads as the persona being broken when it is the transcription. 「ゲーム作りたい」
        # came back as 「キノクロからもう一度」 and she answered that.
        self.signals.sio_queue.put(("heard", text))

        self.signals.current_turn_id = self.signals.log.begin_turn("voice", text)
        self.signals.history.append({"role": "user", "content": text})

        self.signals.last_message_time = time.time()
        # Kept as a second line of defence. It cannot catch an echo on its own -- that is what
        # is_echo() above is for -- but it still stops a genuine barge-in from queueing a
        # prompt on top of the reply she is in the middle of speaking.
        if not self.signals.AI_speaking:
            self.signals.new_message = True

    def recording_start(self):
        # Snapshot for the echo guard, taken before anything else: was she talking when this
        # recording opened, and if not, how long had she been quiet? Read again in
        # process_text, where neither question can still be answered.
        self._recording_snapshots.append(
            (self.signals.AI_speaking, time.time() - self.signals.ai_quiet_since))

        # Barge-in, first half: stop the voice the moment sound arrives, so talking over her
        # feels immediate. Worst case a stray noise clips a word.
        #
        # With her own voice audible to the microphone this also fires on herself, cutting her
        # own sentences short. Nothing here can tell her voice from a person's, so STT_BARGE_IN
        # exists to turn the feature off on a machine where that keeps happening.
        if STT_BARGE_IN and self.signals.AI_speaking and self.tts is not None:
            self.tts.API.abort_current()

        self.signals.human_speaking = True

    def recording_stop(self):
        self.signals.human_speaking = False

    def feed_audio(self, data):
        self.recorder.feed_audio(data)

    def listen_loop(self):
        print("STT Starting")
        device_index = resolve_input_device(INPUT_DEVICE_NAME, INPUT_DEVICE_INDEX)
        check_input_health(device_index)
        recorder_config = {
            'spinner': False,
            'language': 'ja',
            'use_microphone': True,
            'input_device_index': device_index,
            # These were all pushed well below their defaults to shave latency, back when the
            # microphone was the broken built-in array that produced no speech at all, so the
            # cost never showed up. With a real condenser mic they made the VAD fire on every
            # small room noise: Whisper then hallucinated a sentence out of it ("そうですね、
            # 他のパディーションとして"), which counted as the user speaking and cancelled
            # whatever she was saying. Back to something close to the library defaults.
            'silero_sensitivity': 0.4,       # 0 = least sensitive, 1 = most
            'silero_use_onnx': True,
            'post_speech_silence_duration': 0.6,
            # The important one: at 0, a single click or breath was a complete "recording".
            'min_length_of_recording': 0.5,
            'min_gap_between_recordings': 0.5,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0.2,
            # 'tiny.en'/'small.en' etc. are English-only; Japanese needs the plain multilingual sizes.
            'model': 'small',
            'realtime_model_type': 'tiny',
            'compute_type': 'auto',
            # Keep STT off the GPU so the full 6GB stays available for the LLM (see tech comparison doc, sec.04).
            'device': 'cpu',
            'on_recording_start': self.recording_start,
            'on_recording_stop': self.recording_stop,
            'level': logging.ERROR
        }

        with AudioToTextRecorder(**recorder_config) as recorder:
            self.recorder = recorder
            print("STT Ready")
            self.signals.stt_ready = True
            while not self.signals.terminate:
                if not self.enabled:
                    time.sleep(0.2)
                    continue
                recorder.text(self.process_text)

    class API:
        def __init__(self, outer):
            self.outer = outer

        def set_STT_status(self, status):
            self.outer.enabled = status
            self.outer.signals.sio_queue.put(('STT_status', status))

        def get_STT_status(self):
            return self.outer.enabled

        def shutdown(self):
            self.outer.recorder.stop()
            self.outer.recorder.interrupt_stop_event.set()
