import os
import mss, cv2, base64
import numpy as np
from transformers import AutoTokenizer
from constants import *
from llmWrappers.abstractLLMWrapper import AbstractLLMWrapper


class ImageLLMWrapper(AbstractLLMWrapper):

    def __init__(self, signals, tts, llmState, modules=None):
        super().__init__(signals, tts, llmState, modules)
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.LLM_ENDPOINT = MULTIMODAL_ENDPOINT
        self.CONTEXT_SIZE = MULTIMODAL_CONTEXT_SIZE
        self.tokenizer = AutoTokenizer.from_pretrained(MULTIMODAL_MODEL, token=os.getenv("HF_TOKEN"), trust_remote_code=True)

        self.MSS = None

    def screen_shot(self):
        if self.MSS is None:
            self.MSS = mss.mss()

        # Take a screenshot of the main screen
        frame_bytes = self.MSS.grab(self.MSS.monitors[PRIMARY_MONITOR])

        frame_array = np.array(frame_bytes)
        # resize
        frame_resized = cv2.resize(frame_array, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        result, frame_encoded = cv2.imencode('.jpg', frame_resized, encode_param)
        # base64
        frame_base64 = base64.b64encode(frame_encoded).decode("utf-8")
        return frame_base64

    def prepare_payload(self):
        return {
            "mode": "instruct",
            "stream": True,
            "max_tokens": 200,
            "skip_special_tokens": False,  # Necessary for Llama 3
            "custom_token_bans": BANNED_TOKENS,
            "stop": STOP_STRINGS,
            # generate_prompt() returns real chat turns. The image has to ride along with the
            # last user turn, so that one turn's content becomes the text+image list the
            # vision API expects while every earlier turn stays an ordinary message.
            "messages": self._attach_image(self.generate_prompt())
        }

    def _attach_image(self, messages):
        """Turn the final user turn into the text+image content list the vision API expects."""
        messages = list(messages)
        image_part = {"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{self.screen_shot()}"}}

        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                text_part = {"type": "text", "text": messages[i]["content"]}
                messages[i] = {"role": "user", "content": [text_part, image_part]}
                return messages

        messages.append({"role": "user", "content": [{"type": "text", "text": ""}, image_part]})
        return messages
