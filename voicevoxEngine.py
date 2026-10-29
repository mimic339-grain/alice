"""
RealtimeTTS engine that talks to a locally running VOICEVOX Engine
(https://voicevox.hiroshiba.jp/) over its local HTTP API.

Requires VOICEVOX Engine to already be running (default: http://127.0.0.1:50021).
Written by hand because RealtimeTTS has no built-in VOICEVOX engine.
"""

import io
import wave
from typing import Union

import pyaudio
import requests
from RealtimeTTS.engines.base_engine import BaseEngine


class VoicevoxVoice:
    def __init__(self, speaker_id: int = 3, speed_scale: float = 1.0):
        self.speaker_id = speaker_id
        self.speed_scale = speed_scale

    def __repr__(self):
        return f"VOICEVOX speaker {self.speaker_id} (speed {self.speed_scale})"


class VoicevoxEngine(BaseEngine):
    def __init__(self, voice: Union[int, VoicevoxVoice] = VoicevoxVoice(), base_url: str = "http://127.0.0.1:50021"):
        self.base_url = base_url
        self._sample_rate = 24000
        self.set_voice(voice)

    def post_init(self):
        self.engine_name = "voicevox"

    def get_stream_info(self):
        return pyaudio.paInt16, 1, self._sample_rate

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)

        try:
            query_resp = requests.post(
                f"{self.base_url}/audio_query",
                params={"text": text, "speaker": self.voice.speaker_id},
                timeout=30,
            )
            query_resp.raise_for_status()
            query = query_resp.json()
            query["speedScale"] = self.voice.speed_scale

            synth_resp = requests.post(
                f"{self.base_url}/synthesis",
                params={"speaker": self.voice.speaker_id},
                json=query,
                timeout=30,
            )
            synth_resp.raise_for_status()

            with wave.open(io.BytesIO(synth_resp.content), "rb") as wf:
                self._sample_rate = wf.getframerate()
                audio_data = wf.readframes(wf.getnframes())
                self.queue.put(audio_data)
            return True
        except Exception as e:
            print(f"VOICEVOX synthesis error: {e}")
            return False

    def get_voices(self):
        try:
            resp = requests.get(f"{self.base_url}/speakers", timeout=10)
            resp.raise_for_status()
            voices = []
            for speaker in resp.json():
                for style in speaker["styles"]:
                    voices.append(VoicevoxVoice(speaker_id=style["id"]))
            return voices
        except Exception as e:
            print(f"VOICEVOX get_voices error: {e}")
            return []

    def set_voice(self, voice: Union[int, VoicevoxVoice]):
        if isinstance(voice, VoicevoxVoice):
            self.voice = voice
        else:
            self.voice = VoicevoxVoice(speaker_id=int(voice))

    def set_voice_parameters(self, **voice_parameters):
        if "speaker_id" in voice_parameters:
            self.voice.speaker_id = voice_parameters["speaker_id"]
        if "speed_scale" in voice_parameters:
            self.voice.speed_scale = voice_parameters["speed_scale"]
