# -*- coding: utf-8 -*-
"""Read logs/conversation.jsonl as a conversation, and dig into any single turn.

    python showlog.py                 last 20 turns, one block each
    python showlog.py -n 50           last 50
    python showlog.py --feedback      only turns the user gave a verdict on
    python showlog.py --grep すみません  turns whose reply contains the text
    python showlog.py --slow 4000     turns that took longer than 4000 ms to first word
    python showlog.py --turn ID       everything about one turn, including the full prompt
    python showlog.py --why ID        just the prompt: what the model was actually given

The last two are the ones for "why did she say that". --why prints the message list the model
received, with the volatile part (recalled memory, screen context) pulled out separately, so
it is possible to check whether an instruction was in the prompt at all before deciding the
model ignored it.
"""

import argparse
import json
import os
import sys

if sys.platform == "win32":
    # The console codepage cannot show most of what is in this log.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'conversation.jsonl')


def load():
    turns, feedback, events, inputs = {}, {}, [], {}
    if not os.path.exists(LOG):
        return turns, feedback, events, inputs
    with open(LOG, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("type")
            if kind == "turn":
                turns[rec["turn_id"]] = rec
            elif kind == "feedback":
                feedback.setdefault(rec.get("turn_id"), []).append(rec)
            elif kind == "event":
                events.append(rec)
            elif kind == "input":
                # Only interesting if the turn never completed (crash mid-generation).
                inputs[rec["turn_id"]] = rec
    return turns, feedback, events, inputs


def ms(v):
    return "   -" if v is None else f"{v:>5}"


def brief(turn, fb):
    """One block per turn: what was said, what she said, how fast, and any verdict."""
    inp = turn["input"]
    rep = turn["reply"] or {}
    tm = turn.get("timing") or {}
    src = {"chat": "chat ", "voice": "voice"}.get(inp["source"], inp["source"][:5])
    print(f"── {turn['turn_id']}   {turn['time'][11:]}   [{src}]   "
          f"first {ms(tm.get('first_word_ms'))}ms  total {ms(tm.get('total_ms'))}ms")
    print(f"   あなた : {inp['text']}")
    tag = f"[{rep.get('emotion')}] " if rep.get("emotion") else ""
    print(f"   アリス : {tag}{rep.get('spoken', '')}")

    raw = (rep.get("raw") or "").strip()
    spoken = (rep.get("spoken") or "").strip()
    # Show what was generated but never spoken, which is where filtering and cleaning act.
    dropped = [s for s in turn.get("sentences", []) if s.get("filtered")]
    if dropped:
        for s in dropped:
            print(f"   （未発話: {s['raw'].strip()!r}  ← {s['filtered']}）")
    if turn.get("tool_calls"):
        print(f"   道具   : {', '.join(t.get('name') or '?' for t in turn['tool_calls'])}")
    if rep.get("cancelled"):
        print("   （途中で中断）")
    p = turn.get("prompt") or {}
    if p.get("history_trimmed"):
        print(f"   ※ 履歴が長すぎて古い{p['history_trimmed']}件を削って送った")
    if p.get("context"):
        print(f"   注入   : {p['context'][:100]}{'…' if len(p['context']) > 100 else ''}")
    for v in fb:
        if v.get("verdict") == "good":
            print("   ✓ これでいい")
        else:
            print(f"   ✏ こう言ってほしかった: {v.get('better')}")
    print()


def full(turn, fb):
    brief(turn, fb)
    p = turn.get("prompt") or {}
    print("   --- 生成の設定 ---")
    for k in ("temperature", "top_p", "repetition_penalty", "max_tokens"):
        if k in p.get("params", {}):
            print(f"   {k:<20} {p['params'][k]}")
    print(f"   {'prompt_tokens':<20} {p.get('prompt_tokens')}")
    print(f"   {'history_len':<20} {p.get('history_len')}")
    print(f"   {'tools offered':<20} {'yes' if p.get('params', {}).get('tools') else 'no'}")
    print("   --- 生成された原文 ---")
    print(f"   {(turn.get('reply') or {}).get('raw', '')}")
    print("   --- 文ごと ---")
    for s in turn.get("sentences", []):
        mark = f"  ✗ {s['filtered']}" if s.get("filtered") else "  ✓ 発話"
        print(f"   {s['raw'].strip()!r}{mark}")
    print()
    why(turn)


def why(turn):
    """The prompt itself. Long, by design: this is the evidence."""
    p = turn.get("prompt") or {}
    msgs = p.get("messages", [])
    print(f"   === モデルに送ったメッセージ（{len(msgs)}件, {p.get('prompt_tokens')} tokens） ===")
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):   # vision turns carry text+image parts
            content = " ".join(part.get("text", "[image]") for part in content)
        marker = ""
        if role == "system":
            marker = "  ← 人格設定"
        elif i == len(msgs) - 1:
            marker = "  ← 直前の入力 + 注入分（毎回変わる部分）"
        print(f"   [{i:>2}] {role}{marker}")
        for line in (content or "").split("\n"):
            print(f"        {line}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--feedback", action="store_true")
    ap.add_argument("--grep")
    ap.add_argument("--slow", type=int, metavar="MS")
    ap.add_argument("--turn", metavar="ID")
    ap.add_argument("--why", metavar="ID")
    ap.add_argument("--events", action="store_true", help="起動時の挨拶など、会話以外の記録")
    args = ap.parse_args()

    turns, feedback, events, inputs = load()
    if not turns and not events:
        print(f"まだ記録がありません: {LOG}")
        return

    if args.turn or args.why:
        tid = args.turn or args.why
        match = [t for t in turns.values() if t["turn_id"].startswith(tid)]
        if not match:
            print(f"turn {tid!r} は見つかりません")
            return
        (full if args.turn else why)(match[0], feedback.get(match[0]["turn_id"], []))
        return

    if args.events:
        for e in events:
            print(f"{e['time'][11:]}  {e['kind']:<18} {json.dumps({k: v for k, v in e.items() if k not in ('type','kind','time')}, ensure_ascii=False)}")
        return

    ordered = sorted(turns.values(), key=lambda t: t["turn_id"])
    if args.feedback:
        ordered = [t for t in ordered if t["turn_id"] in feedback]
    if args.grep:
        ordered = [t for t in ordered
                   if args.grep in ((t.get("reply") or {}).get("raw") or "")
                   or args.grep in ((t.get("reply") or {}).get("spoken") or "")]
    if args.slow:
        ordered = [t for t in ordered
                   if ((t.get("timing") or {}).get("first_word_ms") or 0) > args.slow]

    unfinished = [i for tid, i in inputs.items() if tid not in turns]
    if unfinished:
        print(f"（返答が完了しなかった入力が {len(unfinished)} 件: "
              + ", ".join(repr(i['text'][:20]) for i in unfinished[-3:]) + "）\n")

    shown = ordered[-args.n:]
    print(f"{len(shown)} / {len(ordered)} 件\n")
    for t in shown:
        brief(t, feedback.get(t["turn_id"], []))


if __name__ == "__main__":
    main()
