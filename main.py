# Python Module Imports
import signal
import sys
import time
import threading
import asyncio

# Class Imports
from greeting import greet
from signals import Signals
from prompter import Prompter
from llmWrappers.llmState import LLMState
from llmWrappers.textLLMWrapper import TextLLMWrapper
from llmWrappers.imageLLMWrapper import ImageLLMWrapper
from stt import STT
from tts import TTS
from reflexLLM import ReflexLLM
from vision import Vision
from constants import REFLEX_ENABLED, VISION_ENABLED, VTUBE_STUDIO_ENABLED
from modules.twitchClient import TwitchClient
from modules.audioPlayer import AudioPlayer
from modules.vtubeStudio import VtubeStudio
from modules.multimodal import MultiModal
from modules.customPrompt import CustomPrompt
from modules.memory import Memory
from modules.action import ActionRouter
from modules.screenContext import ScreenContext
from socketioServer import SocketIOServer


def register_default_actions(action_router, modules, vision):
    """Register the actions the AI is allowed to take.

    Deliberately minimal. Anything registered here becomes something the model can trigger on
    its own, so add only what is safe to fire at any moment -- never a purchase, an account
    change, or anything else that is awkward to undo.
    """
    vtube_studio = modules.get('vtube_studio')

    if vision is not None:
        def look_at_screen(_arg):
            # The description lands in the vision module, and the ScreenContext module feeds it
            # into the next prompt -- so the AI comments on what it saw on the following breath.
            vision.look()

        action_router.register(
            "look_screen", look_at_screen,
            "画面を見たいとき（ゲームの状況や、相手が見せてきたものを確認したいとき）に使う")

    if vtube_studio is not None and getattr(vtube_studio, "enabled", False):
        def wave(_arg):
            # VTube Studio hotkeys are set up by the user inside VTube Studio itself; this only
            # asks for one by name, so a missing hotkey is a no-op rather than an error.
            vtube_studio.queue.put(("hotkey", "wave"))

        action_router.register("wave", wave, "視聴者に手を振るとき")


async def main():
    print("Starting Project...")

    # Register signal handler so that all threads can be exited.
    def signal_handler(sig, frame):
        print('Received CTRL + C, attempting to gracefully exit. Close all dashboard windows to speed up shutdown.')
        signals.terminate = True
        stt.API.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # CORE FILES

    # Singleton object that every module will be able to read/write to
    signals = Signals()

    # MODULES
    # Modules that start disabled CANNOT be enabled while the program is running.
    modules = {}
    module_threads = {}

    # Create TTS
    tts = TTS(signals)
    # Create LLMWrappers
    llmState = LLMState()
    llms = {
        "text": TextLLMWrapper(signals, tts, llmState, modules),
    }
    # ImageLLMWrapper (Vision/multimodal, Phase 4) is optional: the multimodal module starts
    # disabled by default, and prompter.chooseLLM() never touches llms["image"] while it's
    # disabled. Guarded so a broken vision model doesn't block text+voice from starting.
    try:
        llms["image"] = ImageLLMWrapper(signals, tts, llmState, modules)
    except Exception as e:
        print(f"WARNING: Failed to load ImageLLMWrapper (Vision disabled): {e}")
    # Create STT (after tts/llms so it can interrupt them on barge-in, see stt.py)
    stt = STT(signals, tts=tts, llm_api=llms["text"].API)
    # Create Reflex LLM (small model that fills the gap while the main model thinks)
    reflex = ReflexLLM(signals, tts) if REFLEX_ENABLED else None
    # Create Prompter
    prompter = Prompter(signals, llms, modules, reflex=reflex)

    # Create Discord bot
    # modules['discord'] = DiscordClient(signals, stt, enabled=False)
    # Create Twitch bot
    modules['twitch'] = TwitchClient(signals, enabled=False)
    # Create audio player
    modules['audio_player'] = AudioPlayer(signals, enabled=True)
    # Create Vtube Studio plugin
    modules['vtube_studio'] = VtubeStudio(signals, enabled=VTUBE_STUDIO_ENABLED)
    # Create Multimodal module
    modules['multimodal'] = MultiModal(signals, enabled=False)
    # Create Custom Prompt module
    modules['custom_prompt'] = CustomPrompt(signals, enabled=True)
    # Create Memory module
    modules['memory'] = Memory(signals, enabled=True)
    # Screen vision, and the module that carries what it saw into the next prompt.
    vision = Vision(signals) if VISION_ENABLED else None
    modules['screen_context'] = ScreenContext(signals, vision, enabled=VISION_ENABLED)
    # Create Action router. Nothing is registered by default -- an action the AI cannot name
    # cannot be run, which is the point (see modules/action.py).
    action_router = ActionRouter(signals, enabled=True)
    modules['action'] = action_router
    register_default_actions(action_router, modules, vision)

    # Create Socket.io server
    # The specific llmWrapper it gets doesn't matter since state is shared between all llmWrappers
    sio = SocketIOServer(signals, stt, tts, llms["text"], prompter, modules=modules)

    # Warm the model's prompt cache, then let her introduce herself, on one background thread.
    #
    # Ordered, not parallel. The greeting's second half is generated, and a generation issued
    # before the cache is warm queues behind the warm-up and then pays full prompt processing --
    # she would say "起動。" and finish the sentence fifteen seconds later. Warming first costs
    # a few seconds of silence at startup, which reads as an application loading, and then the
    # whole greeting lands together.
    def warm_then_greet():
        llms["text"].warm_up()
        greet(signals, tts, llm=llms["text"])

    threading.Thread(target=warm_then_greet, daemon=True).start()

    # Create threads (As daemons, so they exit when the main thread exits)
    prompter_thread = threading.Thread(target=prompter.prompt_loop, daemon=True)
    stt_thread = threading.Thread(target=stt.listen_loop, daemon=True)
    sio_thread = threading.Thread(target=sio.start_server, daemon=True)
    # Start Threads
    sio_thread.start()
    prompter_thread.start()
    stt_thread.start()

    # Create and start threads for modules
    for name, module in modules.items():
        module_thread = threading.Thread(target=module.init_event_loop, daemon=True)
        module_threads[name] = module_thread
        module_thread.start()

    while not signals.terminate:
        time.sleep(0.1)
    print("TERMINATING ======================")

    # Wait for child threads to exit before exiting main thread

    # Wait for all modules to finish
    for module_thread in module_threads.values():
        module_thread.join()

    sio_thread.join()
    print("SIO EXITED ======================")
    prompter_thread.join()
    print("PROMPTER EXITED ======================")
    # stt_thread.join()
    # print("STT EXITED ======================")

    print("All threads exited, shutdown complete")
    sys.exit(0)

if __name__ == '__main__':
    asyncio.run(main())
