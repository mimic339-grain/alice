"""
Action layer: lets the AI act on the world, not just talk about it.

Design (see the design doc, sec.02.2 and the tech comparison doc, sec.05.2):

  LLM -> tool call -> whitelist lookup -> handler

The model never executes anything. It can only *name* an action, and only actions registered
here can run; anything else is dropped. That is the whole safety story: a capability that is not
registered cannot be invoked, no matter what the model decides or what a viewer talks it into.
Never register a handler that spends money, changes account settings, or is otherwise hard to
undo.

Actions are advertised through the API's `tools` field, not described in the prompt. An earlier
version asked the model to type "[ACTION:name]" into its reply and it simply never did, even
with a worked example -- the same failure mode as every other prompt-level rule in this project.
Tool calling works because the decoder enforces it instead of the model choosing to comply.
"""

from modules.module import Module


class ActionRouter(Module):

    def __init__(self, signals, enabled=True):
        super().__init__(signals, enabled)
        self.API = self.API(self)
        # name -> {"handler": callable, "description": str}
        self.actions = {}
        self.prompt_injection.priority = 120

    def register(self, name, handler, description):
        """Make one action callable by the model. Registration is the only way in."""
        self.actions[name] = {"handler": handler, "description": description}
        print(f"ACTION: registered '{name}'")

    def unregister(self, name):
        self.actions.pop(name, None)

    def get_prompt_injection(self):
        # Nothing to inject: actions are advertised through the API's `tools` field instead of
        # through prompt text. Describing them in prose as well only invited the model to type
        # the tag out loud as part of its line.
        self.prompt_injection.text = ""
        return self.prompt_injection

    def get_tool_specs(self):
        """Registered actions in OpenAI `tools` format, or None when there are none.

        Sent with every request so the model can call an action through constrained generation.
        This replaced an earlier "[ACTION:name]" text-tag scheme: the 8B model ignored the tag
        instruction entirely, while the same model calls these tools correctly, because tool
        calling is enforced by the decoder rather than requested in the prompt.
        """
        if not self.enabled or not self.actions:
            return None

        return [{
            "type": "function",
            "function": {
                "name": name,
                "description": action["description"],
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        } for name, action in self.actions.items()]

    def execute(self, name, arg):
        """Run one named action if, and only if, it was registered."""
        action = self.actions.get(name)
        if action is None:
            # Expected and harmless: the model invents action names it was never given.
            print(f"ACTION: ignored unregistered '{name}'")
            return False

        try:
            action["handler"](arg)
            print(f"ACTION: executed '{name}' arg={arg!r}")
            self.signals.sio_queue.put(("action_executed", {"name": name, "arg": arg}))
            return True
        except Exception as e:
            # A failing action must never take the stream down with it.
            print(f"ACTION: '{name}' failed: {e}")
            return False

    class API:
        def __init__(self, outer):
            self.outer = outer

        def list_actions(self):
            return {name: a["description"] for name, a in self.outer.actions.items()}

        def set_action_status(self, status):
            self.outer.enabled = status
            self.outer.signals.sio_queue.put(("action_status", status))
