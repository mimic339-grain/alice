"""What the AI says on startup, before anyone has said anything to her.

Two halves, deliberately. "アリス、起動。" is fixed -- it is the character's boot phrase and
should land the instant she comes up, with no wait. The line after it is generated, so the
greeting is different every session instead of cycling a short written list.

Generation is affordable here in a way it is not mid-conversation: at startup nobody is waiting
on a specific moment, and the pause between the boot phrase and the line reads as a machine
coming up rather than as lag. The written lines below are the fallback for when generation
fails or the model is not up yet, so she always says something.
"""

import json
import os
import random
import re
import time

from constants import (EMOTION_TAGS, STARTUP_LINES, STARTUP_LINES_LONG_ABSENCE,
                       STARTUP_LONG_ABSENCE_HOURS, STARTUP_PREFIX, STARTUP_TIME_SLOTS)

_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.last_launch.json')

# The emotion is chosen here rather than left to the model. Asked to pick one, it answered
# "neutral" every single time across every slot tested -- in conversation it uses all eight,
# but with no dialogue to react to it defaults and stays there. Choosing first and telling it
# which tag to use gets the variety, and lets the mood match the hour deliberately.
_SLOT_EMOTIONS = {
    "early":     ["sleepy", "surprised", "neutral"],
    "morning":   ["happy", "excited", "embarrassed"],
    "noon":      ["happy", "excited", "neutral"],
    "evening":   ["neutral", "happy", "sleepy"],
    "night":     ["happy", "neutral", "sleepy"],
    "latenight": ["sleepy", "surprised", "embarrassed"],
}
_ABSENCE_EMOTIONS = ["excited", "happy", "sad", "surprised"]

_SLOT_LABELS = {
    "early": "早朝",
    "morning": "朝",
    "noon": "昼",
    "evening": "夕方",
    "night": "夜",
    "latenight": "深夜",
}


def _time_slot(hour):
    for name, start, end in STARTUP_TIME_SLOTS:
        if start <= hour < end:
            return name
    return "latenight"


def _read_state():
    try:
        with open(_STATE_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        # Missing or unreadable is the same as a first run.
        return {}


def _write_state(line, emotion=None):
    try:
        with open(_STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump({"last_launch": time.time(), "last_line": line,
                       "last_emotion": emotion}, f, ensure_ascii=False)
    except Exception:
        # Losing this only costs a slightly less apt greeting next time.
        pass


def _hours_since_last_launch(state):
    last = state.get("last_launch")
    if last is None:
        return None
    return (time.time() - float(last)) / 3600.0


def _fallback_line(slot, long_absence, avoid=None):
    pool = STARTUP_LINES_LONG_ABSENCE if long_absence else STARTUP_LINES[slot]
    choices = [entry for entry in pool if entry[1] != avoid] or list(pool)
    return random.choice(choices)


def _pick_emotion(slot, long_absence, avoid=None):
    pool = _ABSENCE_EMOTIONS if long_absence else _SLOT_EMOTIONS[slot]
    choices = [e for e in pool if e != avoid] or list(pool)
    return random.choice(choices)


def _build_instruction(slot, long_absence, gap, emotion):
    """Describe the situation to the model, and let it write the line itself."""
    when = _SLOT_LABELS[slot]
    lines = [
        f"いま起動したところ。時刻は{time.strftime('%H時%M分')}、{when}。",
    ]
    if long_absence and gap is not None:
        lines.append(f"前に起動されてから{int(gap)}時間ぶり。しばらく会っていなかった。")
    elif gap is not None and gap < 1:
        lines.append("さっきも起動されたばかり。")

    lines.append(
        "起動して最初のひとことを1文だけ言って。"
        f"{when}であることが伝わる挨拶でもいいし、相手に話しかけるひとことでもいい。"
        "毎回ちがう言い回しにすること。相手はまだ何も言っていないので、質問に答えようとしないこと。"
        "「わたし」などの一人称は使わないこと。敬語（です・ます・すみません）は使わないこと。"
        f"感情タグは必ず [{emotion}] にすること。その気分に合う言い方にすること。"
        "時刻を勝手に別の時間に言い換えないこと。"
    )
    return "\n".join(lines)


# Rejected rather than spoken. The 4-bit model occasionally emits a mangled particle run
# ("わたしった") or a sentence carrying no meaning, and a startup line is worth one retry
# because it is the first thing anyone hears.
_BAD_LINE = re.compile(r'わたしった|私った|ってきたよ')

# She does not use polite forms -- it is in the persona, and the instruction repeats it, and
# the model still produces "あ、すみません。" often enough to matter. Cheaper to catch here
# than to keep rewording the prompt: a rejected line just costs one more generation, and the
# written fallback is in character by construction.
_POLITE = re.compile(r'すみません|ごめんなさい|ください|でしょうか|ですか|します|ました|ですね|でした|まして')


def _looks_broken(line):
    if len(line) < 4:
        return True
    if _BAD_LINE.search(line) or _POLITE.search(line):
        return True
    # No kana at all is not this character speaking.
    return not re.search(r'[ぁ-んァ-ン]', line)


def _parse(reply):
    """Split "[emotion] text" into its parts, tolerating a missing or unknown tag."""
    match = re.match(r'^\s*\[(\w+)\]\s*(.+)$', reply, re.S)
    if match and match.group(1).lower() in EMOTION_TAGS:
        return match.group(1).lower(), match.group(2).strip()
    return "neutral", reply.strip()


def _clean(text):
    """Keep the first sentence only, and drop anything that is not her speaking.

    Asked for one line, the model sometimes returns two, or wraps it in quotes, or prefixes it
    with her own name the way the sample dialogue is laid out.
    """
    text = text.strip().strip('「」"\'')
    text = re.sub(r'^アリス\s*[:：]\s*', '', text)
    parts = re.split(r'(?<=[。！？!?])', text)
    first = next((p.strip() for p in parts if p.strip()), "")
    return first or text


def greet(signals, tts, llm=None):
    """Say the boot phrase immediately, then a generated line once it is ready.

    Runs once at startup. The greeting goes into history as an assistant turn so the model
    knows she has already said hello -- without it she opens the first real reply by greeting
    all over again.
    """
    try:
        state = _read_state()
        gap = _hours_since_last_launch(state)
        long_absence = gap is not None and gap >= STARTUP_LONG_ABSENCE_HOURS
        slot = _time_slot(time.localtime().tm_hour)

        # The boot phrase first, with nothing to wait on.
        signals.sio_queue.put(("reset_next_message", None))
        signals.sio_queue.put(("next_chunk", STARTUP_PREFIX))
        tts.play(STARTUP_PREFIX)

        emotion = line = None
        generated = False
        if llm is not None:
            emotion = _pick_emotion(slot, long_absence, state.get("last_emotion"))
            instruction = _build_instruction(slot, long_absence, gap, emotion)
            for _ in range(2):
                reply = llm.generate_line(instruction)
                if not reply:
                    break
                _, candidate = _parse(reply)   # our chosen tag wins over whatever it emitted
                candidate = _clean(candidate)
                if candidate and not _looks_broken(candidate):
                    line = candidate
                    generated = True
                    break
                print(f"GREETING: discarded a malformed line ({candidate!r})")
                line = None

        if not line:
            emotion, line = _fallback_line(slot, long_absence, state.get("last_line"))
            print("GREETING: using a written line (model unavailable)")

        signals.current_emotion = emotion
        signals.history.append({"role": "assistant",
                                "content": f"[{emotion}] {STARTUP_PREFIX}{line}"})
        signals.sio_queue.put(("next_chunk", f"[{emotion}] {line}"))
        signals.sio_queue.put(("current_emotion", emotion))
        print(f"GREETING: {STARTUP_PREFIX}{line}")
        signals.log.record_event("startup_greeting", emotion=emotion,
                                 line=f"{STARTUP_PREFIX}{line}",
                                 generated=generated,
                                 slot=slot, long_absence=long_absence)

        tts.play(line)
        _write_state(line, emotion)
    except Exception as e:
        # She can run perfectly well without a greeting; never let this stop startup.
        print(f"GREETING: skipped ({type(e).__name__}: {e})")
