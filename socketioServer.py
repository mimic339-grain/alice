import json
import os
import time
from aiohttp import web
import socketio
from aiohttp.web_runner import GracefulExit

from constants import PATIENCE


class SocketIOServer:
    def __init__(self, signals, stt, tts, llmWrapper, prompter, modules=None):
        if modules is None:
            modules = {}
        self.signals = signals
        self.stt = stt
        self.tts = tts
        self.llmWrapper = llmWrapper
        self.prompter = prompter
        self.modules = modules

    def start_server(self):
        print("Starting Socket.io server")
        sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
        app = web.Application()
        sio.attach(app)

        @sio.event
        async def get_blacklist(sid):
            await sio.emit('get_blacklist', self.llmWrapper.API.get_blacklist())

        @sio.event
        async def set_blacklist(sid, message):
            self.llmWrapper.API.set_blacklist(message)

        @sio.event
        async def disable_LLM(sid):
            self.llmWrapper.API.set_LLM_status(False)

        @sio.event
        async def enable_LLM(sid):
            self.llmWrapper.API.set_LLM_status(True)

        @sio.event
        async def disable_TTS(sid):
            self.tts.API.set_TTS_status(False)

        @sio.event
        async def enable_TTS(sid):
            self.tts.API.set_TTS_status(True)

        @sio.event
        async def disable_STT(sid):
            self.stt.API.set_STT_status(False)

        @sio.event
        async def enable_STT(sid):
            self.stt.API.set_STT_status(True)

        @sio.event
        async def disable_movement(sid):
            if "vtube_studio" in self.modules:
                self.modules["vtube_studio"].API.set_movement_status(False)

        @sio.event
        async def enable_movement(sid):
            if "vtube_studio" in self.modules:
                self.modules["vtube_studio"].API.set_movement_status(True)

        @sio.event
        async def disable_multimodal(sid):
            if "multimodal" in self.modules:
                self.modules["multimodal"].API.set_multimodal_status(False)

        @sio.event
        async def enable_multimodal(sid):
            if "multimodal" in self.modules:
                self.modules["multimodal"].API.set_multimodal_status(True)

        @sio.event
        async def get_hotkeys(sid):
            if "vtube_studio" in self.modules:
                self.modules["vtube_studio"].API.get_hotkeys()

        @sio.event
        async def send_hotkey(sid, hotkey):
            if "vtube_studio" in self.modules:
                self.modules["vtube_studio"].API.send_hotkey(hotkey)

        @sio.event
        async def trigger_prop(sid, prop_action):
            if "vtube_studio" in self.modules:
                self.modules["vtube_studio"].API.trigger_prop(prop_action)

        @sio.event
        async def move_model(sid, mode):
            if "vtube_studio" in self.modules:
                self.modules["vtube_studio"].API.move_model(mode)

        @sio.event
        async def disable_twitch(sid):
            if "twitch" in self.modules:
                self.modules["twitch"].API.set_twitch_status(False)

        @sio.event
        async def enable_twitch(sid):
            if "twitch" in self.modules:
                self.modules["twitch"].API.set_twitch_status(True)

        @sio.event
        async def cancel_next_message(sid):
            self.llmWrapper.API.cancel_next()

        @sio.event
        async def abort_current_message(sid):
            self.tts.API.abort_current()

        @sio.event
        async def fun_fact(sid):
            self.signals.history.append({"role": "user", "content": "Let's move on. Can we get a fun fact?"})
            self.signals.new_message = True

        @sio.event
        async def new_topic(sid, message):
            # Logged with its own prefix so the transcript shows which route a turn came in on.
            # Without this, a typed message and a spoken one are indistinguishable in the log --
            # the AI just starts talking, and it looks like the mic fired when it did not.
            print("CHAT INPUT: " + message)
            self.signals.current_turn_id = self.signals.log.begin_turn("chat", message)
            # The page needs the id so feedback on this reply can be filed under it.
            await sio.emit('turn_id', self.signals.current_turn_id)
            self.signals.history.append({"role": "user", "content": message})
            self.signals.new_message = True

        # Feedback lands in a JSONL file next to the code, one record per line. This is the
        # training set: fine-tuning needs hundreds of (what was said -> what she should have
        # said) pairs, and the only way to get them is to collect them while actually talking
        # to her. Appending rather than rewriting means a crash can never lose what came before.
        feedback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'feedback.jsonl')

        @sio.event
        async def feedback(sid, entry):
            if not isinstance(entry, dict):
                return
            record = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "verdict": entry.get("verdict"),          # "good" or "rewrite"
                "user": entry.get("user", ""),            # what was said to her
                "alice": entry.get("alice", ""),          # what she actually replied
                "better": entry.get("better", ""),        # what she should have said
                "emotion": entry.get("emotion", ""),
            }
            with open(feedback_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Also filed against the turn it judges, so showlog.py can put the verdict next to
            # the prompt that produced the reply being judged.
            self.signals.log.record_feedback(
                entry.get("turn_id"), record["verdict"], record["user"], record["alice"],
                record["better"], record["emotion"])

            total = sum(1 for _ in open(feedback_path, encoding='utf-8'))
            label = "OK" if record["verdict"] == "good" else "REWRITE"
            print(f"FEEDBACK [{label}] {record['alice'][:30]!r}"
                  + (f" -> {record['better'][:30]!r}" if record['better'] else "")
                  + f"  (total {total})")
            await sio.emit('feedback_count', total)

        @sio.event
        async def get_feedback_count(sid):
            if not os.path.exists(feedback_path):
                await sio.emit('feedback_count', 0)
                return
            await sio.emit('feedback_count',
                           sum(1 for _ in open(feedback_path, encoding='utf-8')))

        @sio.event
        async def nuke_history(sid):
            self.signals.history = []

        @sio.event
        async def play_audio(sid, file_name):
            if "audio_player" in self.modules:
                self.modules["audio_player"].API.play_audio(file_name)

        @sio.event
        async def pause_audio(sid):
            if "audio_player" in self.modules:
                self.modules["audio_player"].API.pause_audio()

        @sio.event
        async def resume_audio(sid):
            if "audio_player" in self.modules:
                self.modules["audio_player"].API.resume_audio()

        @sio.event
        async def abort_audio(sid):
            if "audio_player" in self.modules:
                self.modules["audio_player"].API.stop_playing()

        @sio.event
        async def set_custom_prompt(sid, data):
            if "custom_prompt" in self.modules:
                self.modules["custom_prompt"].API.set_prompt(data["prompt"], priority=int(data["priority"]))
                await sio.emit("get_custom_prompt", self.modules["custom_prompt"].API.get_prompt())

        @sio.event
        async def clear_short_term(sid):
            if "memory" in self.modules:
                self.modules["memory"].API.clear_short_term()
                await sio.emit("get_memories", self.modules["memory"].API.get_memories())

        @sio.event
        async def import_json(sid):
            if "memory" in self.modules:
                self.modules["memory"].API.import_json()

        @sio.event
        async def export_json(sid):
            if "memory" in self.modules:
                self.modules["memory"].API.export_json()

        @sio.event
        async def delete_memory(sid, data):
            if "memory" in self.modules:
                self.modules["memory"].API.delete_memory(data)
                await sio.emit("get_memories", self.modules["memory"].API.get_memories())

        @sio.event
        async def get_memories(sid, data):
            if "memory" in self.modules:
                await sio.emit("get_memories", self.modules["memory"].API.get_memories(data))

        @sio.event
        async def create_memory(sid, data):
            if "memory" in self.modules:
                self.modules["memory"].API.create_memory(data)
                await sio.emit("get_memories", self.modules["memory"].API.get_memories())

        # When a new client connects, send them the status of everything
        @sio.event
        async def connect(sid, environ):
            # Set signals to themselves to trigger setter function and the sio.emit
            self.signals.AI_thinking = self.signals.AI_thinking
            self.signals.AI_speaking = self.signals.AI_speaking
            self.signals.human_speaking = self.signals.human_speaking
            self.signals.recentTwitchMessages = self.signals.recentTwitchMessages
            await sio.emit("patience_update", {"crr_time": time.time() - self.signals.last_message_time, "total_time": PATIENCE})
            await sio.emit('get_blacklist', self.llmWrapper.API.get_blacklist())

            if "twitch" in self.modules:
                await sio.emit('twitch_status', self.modules["twitch"].API.get_twitch_status())
            if "audio_player" in self.modules:
                await sio.emit('audio_list', self.modules["audio_player"].API.get_audio_list())
            if "vtube_studio" in self.modules:
                await sio.emit('movement_status', self.modules["vtube_studio"].API.get_movement_status())
                self.modules["vtube_studio"].API.get_hotkeys()
            if "custom_prompt" in self.modules:
                await sio.emit('get_custom_prompt', self.modules["custom_prompt"].API.get_prompt())
            if "multimodal" in self.modules:
                await sio.emit('multimodal_status', self.modules["multimodal"].API.get_multimodal_status())

            # Collect the enabled status of the llm, tts, stt, and movement and send it to the client
            await sio.emit('LLM_status', self.llmWrapper.API.get_LLM_status())
            await sio.emit('TTS_status', self.tts.API.get_TTS_status())
            await sio.emit('STT_status', self.stt.API.get_STT_status())

        @sio.event
        def disconnect(sid):
            print('Client disconnected')

        async def send_messages():
            while True:
                if self.signals.terminate:
                    raise GracefulExit

                while not self.signals.sio_queue.empty():
                    event, data = self.signals.sio_queue.get()
                    # print(f"Sending {event} with {data}")
                    await sio.emit(event, data)
                await sio.sleep(0.1)

        # Serve the chat page from the server itself so it has a real http:// origin.
        # Opened straight off disk, the page's origin is "null", and the browser blocks its
        # socket.io connection -- the window renders but sits on "接続中…" forever.
        # run_app binds every interface, so this URL also works from a phone on the same Wi-Fi.
        chat_html = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat.html')

        async def chat_page(_request):
            return web.FileResponse(chat_html)

        app.router.add_get('/', chat_page)
        app.router.add_get('/chat', chat_page)

        async def init_app():
            sio.start_background_task(send_messages)
            return app

        web.run_app(init_app())
