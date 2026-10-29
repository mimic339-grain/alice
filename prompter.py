import time
from constants import PATIENCE


class Prompter:
    def __init__(self, signals, llms, modules=None, reflex=None):
        self.signals = signals
        self.llms = llms
        if modules is None:
            self.modules = {}
        else:
            self.modules = modules
        self.reflex = reflex

        self.system_ready = False
        self.timeSinceLastMessage = 0.0

    def prompt_now(self):
        # Don't prompt AI if system isn't ready yet
        if not self.signals.stt_ready or not self.signals.tts_ready:
            return False
        # Don't prompt AI when anyone is currently talking
        if self.signals.human_speaking or self.signals.AI_thinking or self.signals.AI_speaking:
            return False
        # Prompt AI if human said something
        if self.signals.new_message:
            return True
        # Prompt AI if there are unprocessed chat messages
        if len(self.signals.recentTwitchMessages) > 0:
            return True
        # Prompt if some amount of seconds has passed without anyone talking.
        # PATIENCE = 0 disables this: without a human turn in between, the AI would be replying
        # to her own last line, and each reply becomes the next prompt.
        if PATIENCE > 0 and self.timeSinceLastMessage > PATIENCE:
            return True

    def last_user_message(self):
        for message in reversed(self.signals.history):
            if message["role"] == "user" and message["content"].strip():
                return message["content"]
        return ""

    def chooseLLM(self):
        if "multimodal" in self.modules and self.modules["multimodal"].API.multimodal_now():
            return self.llms["image"]
        else:
            return self.llms["text"]

    def prompt_loop(self):
        print("Prompter loop started")

        while not self.signals.terminate:
            # Set lastMessageTime to now if program is still starting
            if self.signals.last_message_time == 0.0 or (not self.signals.stt_ready or not self.signals.tts_ready):
                self.signals.last_message_time = time.time()
                self.timeSinceLastMessage = 0.0
            else:
                if not self.system_ready:
                    print("SYSTEM READY")
                    self.system_ready = True

            # Calculate and set time since last message
            self.timeSinceLastMessage = time.time() - self.signals.last_message_time
            self.signals.sio_queue.put(("patience_update", {"crr_time": self.timeSinceLastMessage, "total_time": PATIENCE}))

            # Decide and prompt LLM
            if self.prompt_now():
                print("PROMPTING AI")
                # Fire the short filler first so it reaches the speaker while the main model is
                # still generating. TTS queues it ahead of the real reply (design doc sec.04.3).
                if self.reflex is not None and self.signals.new_message:
                    self.reflex.interject(self.last_user_message())
                llmWrapper = self.chooseLLM()
                llmWrapper.prompt()
                self.signals.last_message_time = time.time()

            # Sleep for 0.1 seconds before checking again.
            time.sleep(0.1)
