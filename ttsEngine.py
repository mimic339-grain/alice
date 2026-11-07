"""RealtimeTTS engine for VOICEVOX-compatible HTTP engines.

Two engines speak the same HTTP API, so one class covers both:

    VOICEVOX ENGINE      http://127.0.0.1:50021   24 kHz, many characters and styles
    AivisSpeech Engine   http://127.0.0.1:10101   44.1 kHz, Style-Bert-VITS2 (JP-Extra)

AivisSpeech is the one in use. It matters for two reasons beyond sounding better: its models
are Style-Bert-VITS2, the same family Shizuku's V2.0 voice is built on, and it loads them from
a plain folder of .aivmx files -- so training a voice of our own later is "drop one more file
into Models", not a rewrite. Nothing in this file is tied to a particular voice.

Three things here are not obvious:

1. THE SAMPLE RATE IS PROBED, NOT ASSUMED. AivisSpeech returns 44.1 kHz where VOICEVOX
   returns 24 kHz. RealtimeTTS opens its output stream from get_stream_info() before the
   first synthesis, so a hardcoded rate plays 44.1 kHz audio through a 24 kHz stream: the
   voice comes out slow and deep, which reads as "this model sounds wrong" rather than as
   a bug in our code.

2. ONLY KEYS THE ENGINE ITSELF RETURNED ARE OVERRIDDEN. The query dict comes back from
   /audio_query, so it already contains exactly the fields that engine understands.
   tempoDynamicsScale exists only on AivisSpeech; pauseLength is accepted there but ignored.
   Filtering against the returned dict means neither engine is ever sent a field it would
   reject, without this file having to track which is which.

3. pitchScale IS LEFT ALONE. AivisSpeech documents that moving it off 0.0 degrades audio
   quality, so emotion is expressed through speed, intonation and tempo dynamics instead.
"""

import io
import wave
from typing import Optional, Union

import pyaudio
import requests
from RealtimeTTS.engines.base_engine import BaseEngine


class SynthVoice:
    """Base voice settings. Emotion is layered on top of these, not substituted for them."""

    def __init__(self, speaker_id: int, speed: float = 1.0, intonation: float = 1.0,
                 tempo_dynamics: float = 1.0, volume: float = 1.0,
                 pre_phoneme: float = 0.1, post_phoneme: float = 0.1):
        self.speaker_id = speaker_id
        self.speed = speed
        # On AivisSpeech this is the strength of the style's emotional expression (0.0-2.0),
        # NOT overall pitch movement as on VOICEVOX. Past about 1.5 the voice starts to sound
        # strained, so the emotion table stays below that.
        self.intonation = intonation
        # AivisSpeech only: how much the speaking rate varies within a line. This is the single
        # setting that most separates "reading a sentence out" from "talking".
        self.tempo_dynamics = tempo_dynamics
        self.volume = volume
        self.pre_phoneme = pre_phoneme
        self.post_phoneme = post_phoneme

    def __repr__(self):
        return (f"voice {self.speaker_id} (speed {self.speed}, intonation {self.intonation}, "
                f"tempo {self.tempo_dynamics})")


class SynthEngine(BaseEngine):
    def __init__(self, base_url: str, voice: Union[int, SynthVoice],
                 emotion_voice: Optional[dict] = None,
                 emotion_styles: Optional[dict] = None,
                 timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # tag -> {"speed": .., "intonation": .., "tempo_dynamics": ..}. Missing keys fall back
        # to the base voice, and an absent tag means "say it plainly".
        self.emotion_voice = emotion_voice or {}
        # tag -> style id, for voice models that ship separate emotional styles. The AI Maid
        # models have only one style each, so this is normally empty; filling it in is all it
        # takes to make a multi-style model switch voice per emotion.
        self.emotion_styles = emotion_styles or {}
        self._emotion = None
        self._sample_rate = 44100
        self.set_voice(voice)
        self._probe_sample_rate()

    def post_init(self):
        self.engine_name = "vv-compatible"

    # ---- setup -----------------------------------------------------------------------------

    def _probe_sample_rate(self):
        """Ask the engine what it will actually return, before any audio is played."""
        try:
            resp = requests.post(f"{self.base_url}/audio_query",
                                 params={"text": "あ", "speaker": self.voice.speaker_id},
                                 timeout=self.timeout)
            resp.raise_for_status()
            rate = resp.json().get("outputSamplingRate")
            if rate:
                self._sample_rate = int(rate)
        except Exception as e:
            print(f"TTS: could not probe sample rate, assuming {self._sample_rate} Hz "
                  f"({type(e).__name__}: {e})")

    def warm_up(self, text: str = "ん。"):
        """Force the voice model into memory.

        The first synthesis with a given speaker takes about 7 seconds while the model is
        loaded; every one after it takes well under a second. Paying that once at startup is
        the difference between her first reply of the session arriving late and on time.
        """
        try:
            for speaker_id in {self.voice.speaker_id} | set(self.emotion_styles.values()):
                self._synthesis(self._query(text, speaker_id), speaker_id)
            return True
        except Exception as e:
            print(f"TTS: warm-up failed ({type(e).__name__}: {e})")
            return False

    # ---- synthesis -------------------------------------------------------------------------

    def _resolved(self):
        """The base voice with the current emotion's adjustments applied."""
        over = self.emotion_voice.get(self._emotion) or {}
        return (
            over.get("speed", self.voice.speed),
            over.get("intonation", self.voice.intonation),
            over.get("tempo_dynamics", self.voice.tempo_dynamics),
            self.emotion_styles.get(self._emotion, self.voice.speaker_id),
        )

    def _query(self, text, speaker_id):
        resp = requests.post(f"{self.base_url}/audio_query",
                             params={"text": text, "speaker": speaker_id},
                             timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _synthesis(self, query, speaker_id):
        resp = requests.post(f"{self.base_url}/synthesis",
                             params={"speaker": speaker_id}, json=query,
                             timeout=self.timeout * 4)
        resp.raise_for_status()
        return resp.content

    def set_emotion(self, tag: Optional[str]):
        """Set the emotion the next synthesize() call speaks in."""
        self._emotion = (tag or "").lower() or None

    def get_stream_info(self):
        return pyaudio.paInt16, 1, self._sample_rate

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        speed, intonation, tempo, speaker_id = self._resolved()

        try:
            query = self._query(text, speaker_id)
            # Only touch fields this engine actually returned -- see note 2 at the top.
            wanted = {
                "speedScale": speed,
                "intonationScale": intonation,
                "tempoDynamicsScale": tempo,
                "volumeScale": self.voice.volume,
                "prePhonemeLength": self.voice.pre_phoneme,
                "postPhonemeLength": self.voice.post_phoneme,
            }
            for key, value in wanted.items():
                if key in query:
                    query[key] = value

            audio = self._synthesis(query, speaker_id)
            with wave.open(io.BytesIO(audio), "rb") as wf:
                # This should match the probe, but a mismatch would be silent corruption,
                # so trust the file we actually received.
                self._sample_rate = wf.getframerate()
                self.queue.put(wf.readframes(wf.getnframes()))
            return True
        except Exception as e:
            print(f"TTS synthesis error ({type(e).__name__}: {e})")
            return False

    # ---- voice bookkeeping -----------------------------------------------------------------

    def get_voices(self):
        try:
            resp = requests.get(f"{self.base_url}/speakers", timeout=self.timeout)
            resp.raise_for_status()
            return [SynthVoice(speaker_id=style["id"])
                    for speaker in resp.json() for style in speaker["styles"]]
        except Exception as e:
            print(f"TTS get_voices error ({type(e).__name__}: {e})")
            return []

    def set_voice(self, voice: Union[int, SynthVoice]):
        self.voice = voice if isinstance(voice, SynthVoice) else SynthVoice(int(voice))

    def set_voice_parameters(self, **params):
        for key in ("speaker_id", "speed", "intonation", "tempo_dynamics", "volume"):
            if key in params:
                setattr(self.voice, key, params[key])
