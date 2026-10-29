"""
Screen vision: lets the AI look at what is on screen and talk about it.

Runs against a second textgen instance holding a small vision-language model, so the main chat
model never has to be unloaded. The look is *pull-based*: the AI asks to look by calling the
look_screen action, and the description it gets back is injected into its next prompt. That is
closer to how a person streams than describing every frame, and it keeps the cost off the turns
where nobody cares what is on screen.
"""

import base64
import time

import cv2
import mss
import numpy as np
import requests

from constants import (PRIMARY_MONITOR, VISION_ENDPOINT, VISION_MAX_TOKENS, VISION_PROMPT,
                       VISION_TIMEOUT)


class Vision:
    def __init__(self, signals):
        self.signals = signals
        self.headers = {"Content-Type": "application/json"}
        self._mss = None
        self.last_description = None
        self.last_look_time = 0.0

    def _screenshot_base64(self, max_width=1280):
        # mss is not thread safe, so the instance is created lazily on whichever thread first
        # takes a screenshot and reused from there.
        if self._mss is None:
            self._mss = mss.mss()

        frame = np.array(self._mss.grab(self._mss.monitors[PRIMARY_MONITOR]))

        # Downscale before sending. A full 4K screenshot costs a lot of image tokens and the
        # model does not need that resolution to say what is happening.
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / width
            frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)

        # mss gives BGRA; JPEG wants BGR.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("failed to encode screenshot")
        return base64.b64encode(encoded).decode("utf-8")

    def look(self):
        """Describe what is on screen right now, or None if the look failed.

        Failure is always survivable: the AI just carries on without having seen anything, which
        is much better than dropping the turn.
        """
        try:
            image = self._screenshot_base64()
        except Exception as e:
            print(f"VISION: screenshot failed ({e})")
            return None

        data = {
            "mode": "instruct",
            "max_tokens": VISION_MAX_TOKENS,
            "temperature": 0.3,  # description, not creative writing
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                ],
            }],
        }

        try:
            started = time.time()
            response = requests.post(VISION_ENDPOINT + "/v1/chat/completions", headers=self.headers,
                                     json=data, timeout=VISION_TIMEOUT, verify=False)
            response.raise_for_status()
            description = (response.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            print(f"VISION: request failed ({e})")
            return None

        if not description:
            return None

        self.last_description = description
        self.last_look_time = time.time()
        print(f"VISION ({time.time() - started:.1f}s): {description}")
        self.signals.sio_queue.put(("vision_description", description))
        return description
