class LLMState:
    def __init__(self):
        self.enabled = True
        self.next_cancelled = False

        # Read in blacklist from file
        with open('blacklist.txt', 'r', encoding='utf-8') as file:
            self.blacklist = file.read().splitlines()