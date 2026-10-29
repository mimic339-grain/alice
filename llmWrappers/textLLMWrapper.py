import os

from transformers import AutoTokenizer
from constants import *
from llmWrappers.abstractLLMWrapper import AbstractLLMWrapper


class TextLLMWrapper(AbstractLLMWrapper):

    def __init__(self, signals, tts, llmState, modules=None):
        super().__init__(signals, tts, llmState, modules)
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.LLM_ENDPOINT = LLM_ENDPOINT
        self.CONTEXT_SIZE = CONTEXT_SIZE
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, token=os.getenv("HF_TOKEN"))

    def prepare_payload(self):
        payload = self._base_payload()
        # Advertise registered actions as callable tools. Constrained decoding makes the model
        # actually use them, unlike the prompt-level tag scheme this replaced.
        action_router = self.modules.get("action")
        if action_router is not None:
            tools = action_router.get_tool_specs()
            if tools:
                payload["tools"] = tools
        return payload

    def _base_payload(self):
        return {
            "mode": "instruct",
            "stream": True,
            # She is told to answer in 1-2 sentences, and at ~10 tok/s every allowed token is
            # latency the user waits through. 200 bought nothing but the occasional ramble.
            "max_tokens": 90,
            "skip_special_tokens": False,  # Necessary for Llama 3
            "custom_token_bans": BANNED_TOKENS,
            "stop": STOP_STRINGS,
            # Without these the model returned the identical sentence for different inputs --
            # deadly for a VTuber that has to keep talking for a whole stream. Temperature is
            # held down because the 4-bit Qwen3 starts leaking simplified-Chinese characters
            # and invented words into its Japanese when sampling gets loose.
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            # generate_prompt() now returns real system/user/assistant turns rather than one
            # flat user message -- see AbstractLLMWrapper.generate_prompt for why.
            "messages": self.generate_prompt()
        }