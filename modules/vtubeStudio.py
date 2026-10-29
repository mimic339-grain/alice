import asyncio
import logging
import queue
import pyvts

# The websockets library logs every frame at DEBUG, which buries the app's own output once the
# lip-sync loop starts sending parameter updates several times a second.
logging.getLogger("websockets").setLevel(logging.WARNING)
from modules.module import Module
from constants import VTUBE_MODEL_POSITIONS, VTUBE_MIC_POSITION, EMOTION_HOTKEYS


class VtubeStudio(Module):
    def __init__(self, signals, enabled=True):
        super().__init__(signals, enabled)
        self.queue = queue.SimpleQueue()
        self.API = self.API(self)

        self.item_list = []
        self.prop_instance_ids = {}
        self.available_hotkeys = []

        plugin_info = {
            "plugin_name": "Neuro VTS Plugin",
            "developer": "KimJammer",
            "authentication_token_path": "./vtubeStudio_token.txt",
        }
        self.vts = pyvts.vts(plugin_info=plugin_info)

    async def get_hotkeys(self):
        response = await self.send_request(self.vts.vts_request.requestHotKeyList())
        hotkey_list = []
        for hotkey in response['data']['availableHotkeys']:
            hotkey_list.append(hotkey['name'])
        return hotkey_list

    async def send_hotkey(self, hotkey):
        request = self.vts.vts_request.requestTriggerHotKey(hotkey)
        await self.send_request(request)

    async def get_item_list(self):
        request = self.vts.vts_request.BaseRequest(
            "ItemListRequest",
            {
                "includeAvailableSpots": False,
                "includeItemInstancesInScene": False,
                "includeAvailableItemFiles": True
            }
        )
        response = (await self.send_request(request))["data"]
        items = []
        for item in response["availableItemFiles"]:
            items.append(item["fileName"])
        return items

    async def spawn_microphone(self):
        # Ensure microphone item exists
        if "microphone_red (@7MDigital).png" not in self.item_list:
            self.signals.sio_queue.put(("error", "Microphone item not found in Vtube Studio"))
            return
        request = self.vts.vts_request.BaseRequest(
            "ItemLoadRequest",
            {
                "fileName": "microphone_red (@7MDigital).png",
                "positionX": VTUBE_MIC_POSITION["x"],
                "positionY": VTUBE_MIC_POSITION["y"] - 1,
                "size": VTUBE_MIC_POSITION["size"],
                "rotation": VTUBE_MIC_POSITION["rotation"],
                "fadeTime": 0,
                "order": 4,
                "failIfOrderTaken": False,
            }
        )
        self.prop_instance_ids["microphone"] = (await self.send_request(request))["data"]["instanceID"]
        request = self.vts.vts_request.BaseRequest(
            "ItemMoveRequest",
            {
                "itemsToMove":
                    [
                        {
                            "itemInstanceID": self.prop_instance_ids["microphone"],
                            "timeInSeconds": 0.6,
                            "fadeMode": "zip",
                            "positionX": VTUBE_MIC_POSITION["x"],
                            "positionY": VTUBE_MIC_POSITION["y"],
                        }
                    ]
            }
        )
        await self.send_request(request)

    async def despawn_microphone(self):
        if "microphone" in self.prop_instance_ids:
            request = self.vts.vts_request.BaseRequest(
                "ItemMoveRequest",
                {
                    "itemsToMove":
                        [
                            {
                                "itemInstanceID": self.prop_instance_ids["microphone"],
                                "timeInSeconds": 0.6,
                                "fadeMode": "easeBoth",
                                "positionX": VTUBE_MIC_POSITION["x"],
                                "positionY": VTUBE_MIC_POSITION["y"] - 1,
                            }
                        ]
                }
            )
            await self.send_request(request)
        await asyncio.sleep(0.6)
        request = self.vts.vts_request.BaseRequest(
            "ItemUnloadRequest",
            {
                "fileNames":
                    [
                        "microphone_red (@7MDigital).png"
                    ]
            }
        )
        await self.send_request(request)

    async def move_model(self, mode):
        if mode not in VTUBE_MODEL_POSITIONS:
            self.signals.sio_queue.put(("error", "Invalid model location: " + mode))
            return

        request = self.vts.vts_request.BaseRequest(
            "MoveModelRequest",
            {
                "timeInSeconds": 0.8,
                "valuesAreRelativeToModel": False,
                "positionX": VTUBE_MODEL_POSITIONS[mode]["x"],
                "positionY": VTUBE_MODEL_POSITIONS[mode]["y"],
                "rotation": VTUBE_MODEL_POSITIONS[mode]["rotation"],
                "size": VTUBE_MODEL_POSITIONS[mode]["size"]
            }
        )
        await self.send_request(request)

    async def set_mouth(self, openness):
        """Push one mouth-open value to the model."""
        request = self.vts.vts_request.requestSetParameterValue("MouthOpen", openness)
        await self.vts.request(request)

    async def apply_emotion(self, emotion):
        """Trigger the hotkey mapped to an emotion, if the loaded model has one.

        Expressions live in the model, so the mapping is by name: whatever hotkey the user has
        called "happy" gets fired for [happy]. A model with no matching hotkey simply keeps its
        current face rather than erroring.
        """
        hotkey = EMOTION_HOTKEYS.get(emotion)
        if not hotkey or hotkey not in self.available_hotkeys:
            return
        await self.send_hotkey(hotkey)

    async def send_request(self, request):
        response = await self.vts.request(request)
        if response["messageType"] == "APIError":
            self.signals.sio_queue.put(("error", "Vtube Studio API Error: " + response["data"]["message"]))
            return
        return response

    async def run(self):
        if not self.enabled:
            return

        # Connect
        try:
            await self.vts.connect()
        except:
            print("Failed to connect to Vtube Studio. Disabling Vtube Studio module.")
            self.enabled = False
            return

        # Authenticate
        await self.vts.request_authenticate_token()  # get token
        await self.vts.request_authenticate()  # use token

        self.item_list = await self.get_item_list()
        try:
            self.available_hotkeys = await self.get_hotkeys()
            print(f"VTS: hotkeys available: {self.available_hotkeys}")
        except Exception as e:
            print(f"VTS: could not read hotkeys ({e})")
            self.available_hotkeys = []

        closing_frames = 0
        last_emotion = None

        # Run the request processor loop - Read requests from queue and process them
        # This is done because the API functions are called directly by other threads but actually interacting with the
        # VTS API must be done in this thread.

        while True:
            if self.signals.terminate:
                await self.vts.close()
                return

            if not self.enabled:
                await asyncio.sleep(0.1)
                continue

            # Lip sync. A value injected through the API decays back to the model's default
            # unless it keeps being refreshed, so this resends every tick while talking rather
            # than only when the number changes. Once speech ends it sends a few closing frames
            # and then goes quiet, so a silent model costs no traffic.
            mouth = self.signals.mouth_openness
            if self.signals.AI_speaking or mouth > 0.01 or closing_frames > 0:
                closing_frames = 5 if self.signals.AI_speaking else max(0, closing_frames - 1)
                try:
                    await self.set_mouth(mouth)
                except Exception:
                    pass

            # Expression follows the emotion tag the LLM produced.
            if self.signals.current_emotion != last_emotion:
                last_emotion = self.signals.current_emotion
                try:
                    await self.apply_emotion(last_emotion)
                except Exception as e:
                    print(f"VTS: expression change failed ({e})")

            if self.queue.qsize() > 0:
                crr_action = self.queue.get()

                # If-Elif chains are ugly but like whatever I'll refactor it later :tm:

                if crr_action.action == "get_hotkeys":
                    self.signals.sio_queue.put(("get_hotkeys", await self.get_hotkeys()))
                elif crr_action.action == "send_hotkey":
                    await self.send_hotkey(crr_action.data)
                elif crr_action.action == "move_model":
                    await self.move_model(crr_action.data)
                elif crr_action.action == "spawn_microphone":
                    await self.spawn_microphone()
                elif crr_action.action == "despawn_microphone":
                    await self.despawn_microphone()
                else:
                    print(f"Unknown Vtube Studio action: {crr_action.action}")

            # Pace the loop. It used to spin with sleep(0), which was harmless when the loop only
            # drained a queue, but now that every tick can send a websocket request it saturates
            # the connection and floods the log. 20fps is smooth for a mouth and costs nothing.
            await asyncio.sleep(0.05)

    class Action:
        def __init__(self, action, data):
            self.action = action
            self.data = data
    class API:
        def __init__(self, outer):
            self.outer = outer

        def set_movement_status(self, status):
            self.outer.enabled = status
            self.outer.signals.sio_queue.put(('movement_status', status))
            if status:
                #Clear queue
                while not self.outer.queue.empty():
                    self.outer.queue.get()

        def get_movement_status(self):
            return self.outer.enabled

        def get_hotkeys(self):
            if not self.outer.enabled:
                self.outer.signals.sio_queue.put(("error", "Vtube Studio Module is disabled"))
                return
            self.outer.queue.put(self.outer.Action("get_hotkeys", None))

        def send_hotkey(self, hotkey):
            if not self.outer.enabled:
                self.outer.signals.sio_queue.put(("error", "Vtube Studio Module is disabled"))
                return
            self.outer.queue.put(self.outer.Action("send_hotkey", hotkey))

        def trigger_prop(self, prop_action):
            if not self.outer.enabled:
                self.outer.signals.sio_queue.put(("error", "Vtube Studio Module is disabled"))
                return
            self.outer.queue.put(self.outer.Action(prop_action, None))

        def move_model(self, mode):
            if not self.outer.enabled:
                self.outer.signals.sio_queue.put(("error", "Vtube Studio Module is disabled"))
                return
            self.outer.queue.put(self.outer.Action("move_model", mode))
