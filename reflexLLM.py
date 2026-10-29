"""
Reflex LLM: a tiny model that emits an immediate one-breath interjection ("へぇ〜", "なるほどね")
while the main LLM is still generating its real reply.

Purpose is tempo, not content (design doc sec.04.3): the main 8B model takes long enough that the
gap reads as a hang, so a small CPU-resident model fills it in a few hundred milliseconds. Runs on
its own endpoint so it never competes with the main model for VRAM.
"""

import random
import re
import requests

from constants import (REFLEX_ENDPOINT, REFLEX_FALLBACKS, REFLEX_MAX_CHARS, REFLEX_MAX_TOKENS,
                       REFLEX_PROMPT, REFLEX_TIMEOUT, REFLEX_USE_LLM)


class ReflexLLM:
    def __init__(self, signals, tts):
        self.signals = signals
        self.tts = tts
        self.enabled = True
        self.headers = {"Content-Type": "application/json"}
        self.last_phrase = None

    def _clean(self, text):
        # The reflex model sometimes wraps its interjection in quotes or adds a trailing
        # explanation; keep only the first short line.
        text = text.strip().split("\n")[0]
        text = re.sub(r'^["\'「『\[]|["\'」』\]]$', "", text).strip()
        return text

    @staticmethod
    def _is_valid_interjection(phrase):
        """Accept only a short kana interjection; anything else is model noise."""
        if not phrase or len(phrase) > REFLEX_MAX_CHARS:
            return False
        # Kana, prolongation marks and light punctuation only. Kanji or latin means the model
        # started forming a real sentence instead of an interjection.
        return bool(re.fullmatch(r'[ぁ-んァ-ンーｰ〜～~！!？?、,。.…]+', phrase))

    def interject(self, user_message):
        """Speak a short filler immediately. Returns the phrase spoken, or None."""
        if not self.enabled:
            return None

        phrase = ""
        if REFLEX_USE_LLM:
            data = {
                "mode": "instruct",
                "max_tokens": REFLEX_MAX_TOKENS,
                "temperature": 1.0,
                "messages": [
                    {"role": "system", "content": REFLEX_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            }

            try:
                response = requests.post(
                    REFLEX_ENDPOINT + "/v1/chat/completions",
                    headers=self.headers,
                    json=data,
                    timeout=REFLEX_TIMEOUT,
                    verify=False,
                )
                phrase = self._clean(response.json()["choices"][0]["message"]["content"])
            except Exception as e:
                # A reflex filler is never worth delaying or breaking the real reply for.
                print(f"REFLEX: request failed ({e}), using fallback")
                phrase = ""

        if not self._is_valid_interjection(phrase):
            # Model produced a sentence rather than an interjection. The filler exists purely for
            # tempo, so a canned phrase serves the same purpose without the model's mistakes.
            phrase = random.choice([p for p in REFLEX_FALLBACKS if p != self.last_phrase])

        # Hearing the identical filler twice running is more jarring than a plain pause.
        if phrase == self.last_phrase:
            print("REFLEX: skipped (repeat of last phrase)")
            return None
        self.last_phrase = phrase

        print("REFLEX OUTPUT: " + phrase)
        self.signals.pending_reflex = phrase
        self.tts.play(phrase)
        return phrase

    class API:
        def __init__(self, outer):
            self.outer = outer

        def set_reflex_status(self, status):
            self.outer.enabled = status
            self.outer.signals.sio_queue.put(("reflex_status", status))

        def get_reflex_status(self):
            return self.outer.enabled
