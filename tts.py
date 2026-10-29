import queue
import threading
import time

import numpy as np
from RealtimeTTS import TextToAudioStream

from voicevoxEngine import VoicevoxEngine, VoicevoxVoice
from constants import *


class TTS:
    def __init__(self, signals):
        self.stream = None
        self.signals = signals
        self.API = self.API(self)
        self.enabled = True

        # Requires VOICEVOX Engine to be running locally (default port 50021)
        engine = VoicevoxEngine(
            voice=VoicevoxVoice(speaker_id=VOICEVOX_SPEAKER_ID, speed_scale=VOICEVOX_SPEED),
        )
        tts_config = {
            'on_audio_stream_start': self.audio_started,
            'on_audio_stream_stop': self.audio_ended,
            'output_device_index': OUTPUT_DEVICE_INDEX,
        }
        self.stream = TextToAudioStream(engine, **tts_config)

        # Sentences wait here and a single worker speaks them one at a time.
        #
        # RealtimeTTS cannot be handed sentences as they are generated. Its player consumes the
        # text it has and then shuts the session down, and text fed during that shutdown lands
        # in a buffer nobody reads again -- is_playing() still says True, so there is no moment
        # at which a caller can tell that feeding is no longer safe. Generation here produces a
        # sentence every second or two, which puts almost every sentence after the first into
        # that window: the AI spoke her opening line, went silent, and the rest was simply lost.
        #
        # Feeding one sentence per playback session removes the window entirely. Each session is
        # started only after the previous one has fully finished, so nothing can arrive late.
        self.pending = queue.Queue()
        self.play_thread = threading.Thread(target=self.play_loop, daemon=True)
        self.play_thread.start()

        self.signals.tts_ready = True

    def on_audio_chunk(self, chunk):
        """Publish how loud the audio playing right now is, for VTube Studio lip sync.

        Reading the loudness straight off the audio we are already generating avoids the usual
        setup, where TTS output is routed through a virtual audio cable into VTube Studio's
        microphone input. One less piece of software to install, and no risk of the model
        lip-syncing to room noise.
        """
        try:
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size == 0:
                return
            # RMS tracks perceived loudness better than peak, which spikes on consonants and
            # makes the mouth flap open on every syllable edge.
            rms = float(np.sqrt(np.mean(samples ** 2)))
            self.signals.mouth_openness = min(1.0, rms * MOUTH_GAIN)
        except Exception:
            # Lip sync is cosmetic; never let it interfere with speech.
            pass

    def play(self, message):
        """Queue one sentence to be spoken. Returns immediately."""
        if not self.enabled:
            return

        # If the message is only whitespace, don't attempt to play it
        if not message.strip():
            return

        self.signals.sio_queue.put(("current_message", message))
        # Held here rather than fed to the stream now: see the note in __init__ for why the
        # stream is only ever given one sentence at a time.
        self.signals.AI_speaking = True
        self.pending.put(message)

    def play_loop(self):
        """Speak queued sentences in order, one playback session each."""
        while not self.signals.terminate:
            try:
                message = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.enabled:
                continue

            try:
                self.stream.feed(message)
                # Synchronous: returns once this sentence has actually been spoken, which is
                # what makes the next feed safe. on_audio_chunk belongs to the play call rather
                # than the constructor, so it is passed every time.
                self.stream.play(on_audio_chunk=self.on_audio_chunk)
                # Logged after the fact, so this line means "was said out loud", not "was
                # queued". That distinction is what made the dropped-sentence bug hard to see.
                print(f"SPOKE: {message}")
            except Exception as e:
                # One sentence failing to synthesize must not take the whole voice down.
                print(f"TTS: failed to speak {message[:30]!r}: {type(e).__name__}: {e}")

    def stop(self):
        # Drop anything still queued as well as what is playing. Without this, a reply the user
        # interrupted would carry on out of the queue seconds later, on top of the next one.
        while True:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                break
        self.stream.stop()
        self.signals.AI_speaking = False

    def audio_started(self):
        self.signals.AI_speaking = True

    def audio_ended(self):
        self.signals.last_message_time = time.time()
        self.signals.mouth_openness = 0.0
        # Fires after every sentence now, so she is only really finished once nothing is left
        # waiting. Reporting her as silent between sentences would let the rest of the app treat
        # a mid-reply pause as her turn being over.
        if self.pending.empty():
            self.signals.AI_speaking = False

    class API:
        def __init__(self, outer):
            self.outer = outer

        def set_TTS_status(self, status):
            self.outer.enabled = status
            if not status:
                self.outer.stop()
            self.outer.signals.sio_queue.put(('TTS_status', status))

        def get_TTS_status(self):
            return self.outer.enabled

        def abort_current(self):
            self.outer.stop()
