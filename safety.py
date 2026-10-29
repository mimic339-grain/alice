"""
Output safety filter.

Runs on every sentence before it is voiced (see AbstractLLMWrapper.speak_sentence). Everything
here is deterministic on purpose: the tech comparison doc's section 05 concluded that asking the
model to police itself does not hold up, and this project has now confirmed that repeatedly --
prompt-level prohibitions on repeating interjections, on simplified-Chinese characters and on
polite speech were all ignored by the 8B model.
"""

import re


class SafetyFilter:
    # Patterns for things that must never be read out on stream regardless of wording. These
    # catch shapes rather than words, so the model cannot slip them past with a synonym.
    PERSONAL_INFO_PATTERNS = [
        # Japanese phone numbers, with or without hyphens (0X-XXXX-XXXX / 0XXXXXXXXXX).
        (r'0\d{1,4}[-ー－]?\d{1,4}[-ー－]?\d{3,4}', "電話番号らしき数字"),
        # Email addresses.
        (r'[\w.+-]+@[\w-]+\.[\w.]+', "メールアドレス"),
        # Long digit runs: card numbers, account numbers, My Number.
        (r'\d[\d\s-]{10,}\d', "長い数字列（カード番号・口座番号など）"),
        # Japanese postal codes.
        (r'〒?\d{3}[-ー－]\d{4}', "郵便番号"),
        # URLs, which can be used to send viewers somewhere unvetted.
        (r'https?://\S+', "URL"),
    ]

    def __init__(self, blacklist=None, block_personal_info=True):
        self.blacklist = blacklist or []
        self.block_personal_info = block_personal_info

    def check(self, text):
        """Return (is_safe, reason). reason is None when the text is safe."""
        if not text or not text.strip():
            return True, None

        lowered = text.lower()

        # Substring match, not word match. The original implementation used text.split(), which
        # silently never matched anything in Japanese because Japanese has no spaces between
        # words -- the blacklist looked active but blocked nothing.
        for word in self.blacklist:
            word = word.strip()
            if word and word.lower() in lowered:
                return False, f"NGワード: {word}"

        if self.block_personal_info:
            for pattern, label in self.PERSONAL_INFO_PATTERNS:
                if re.search(pattern, text):
                    return False, label

        return True, None

    def is_filtered(self, text):
        safe, _ = self.check(text)
        return not safe
