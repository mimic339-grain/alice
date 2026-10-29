import logging
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

    def process_text(self, text):
        if not self.enabled:
            return

        print("STT OUTPUT: " + text)

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

        self.signals.current_turn_id = self.signals.log.begin_turn("voice", text)
        self.signals.history.append({"role": "user", "content": text})

        self.signals.last_message_time = time.time()
        if not self.signals.AI_speaking:
            self.signals.new_message = True

    def recording_start(self):
        # Barge-in, first half: stop the voice the moment sound arrives, so talking over her
        # feels immediate. Worst case a stray noise clips a word.
        if self.signals.AI_speaking and self.tts is not None:
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
