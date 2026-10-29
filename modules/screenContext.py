"""
Feeds the most recent screen description into the prompt.

Kept separate from vision.py so the looking and the remembering stay independent: vision.py
knows how to look, this decides how long a look stays relevant. A description is dropped once
it goes stale, because commenting on a screen that changed two minutes ago reads as confusion.
"""

import time

from constants import VISION_CONTEXT_TTL
from modules.module import Module


class ScreenContext(Module):

    def __init__(self, signals, vision, enabled=True):
        super().__init__(signals, enabled)
        self.vision = vision
        self.prompt_injection.priority = 130

    def get_prompt_injection(self):
        self.prompt_injection.text = ""

        if not self.enabled or self.vision is None or not self.vision.last_description:
            return self.prompt_injection

        age = time.time() - self.vision.last_look_time
        if age > VISION_CONTEXT_TTL:
            # Stale: the screen has almost certainly moved on.
            self.vision.last_description = None
            return self.prompt_injection

        self.prompt_injection.text = (
            f"\n【今画面に見えているもの】\n{self.vision.last_description}\n"
            "これを見た感想を自分の言葉で話してよい。説明の丸読みはしないこと。\n")
        return self.prompt_injection
