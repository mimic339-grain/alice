"""Read back what she actually said, and propose persona changes from it.

The problem this exists to solve: the persona in constants.py was written by guessing, then
corrected by guessing again, and the guesses were wrong twice. What was missing was not more
thinking about the character but a way to look at the character as it already behaves. The
conversation log is a record of that. This turns it into a list of proposals.

The method is lifted from bluesura/AI-VTuber-Shizuku-log, whose persona/ folder analyses stream
logs to build up a character record rather than writing one in advance. Four of its design
choices are kept because each one prevents a mistake already made in this project:

  * NOTHING IS APPLIED AUTOMATICALLY. Every finding is a proposal with its evidence attached.
    An automatic edit is a guess wearing a number.
  * EVERY CLAIM CARRIES A COUNT AND A VERBATIM EXAMPLE. "She sounds too polite" is an opinion;
    "17 of 103 replies contain ですます, e.g. ..." is something that can be checked.
  * CONTRADICTIONS ARE FLAGGED, NOT RESOLVED. Marked 【要確認】 and left for a human.
  * VOLATILE AND STABLE ARE SEPARATED. A phrase she used twice this week is not a trait.

Everything here is deterministic -- counting, not interpreting. safety.py already records the
reason: asking the 8B model to judge its own output did not hold up, and asking it to judge its
own personality would be the same mistake with more words. Numbers from this tool and judgement
from the person reading them is the division that works.

Usage:
    python personaExtract.py                     # full report
    python personaExtract.py --since 2026-11-01  # only recent turns
    python personaExtract.py --top 15            # longer frequency lists
    python personaExtract.py --json report.json  # machine-readable, for diffing runs
"""

import argparse
import collections
import difflib
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE, "logs", "conversation.jsonl")
FEEDBACK_PATH = os.path.join(BASE, "feedback.jsonl")

# Sentence boundaries, matching how tts.py actually splits a reply into spoken sentences, so
# "first sentence" here means the same thing it means to the listener.
SENT_SPLIT = re.compile(r"[。！？!?\n]+")
EMOTION_TAG = re.compile(r"^\s*\[([a-zA-Z_]+)\]\s*")
PUNCT = re.compile(r"[\s。、．，.,!?！？…「」『』ー~〜]")

# --- the persona rules that are checkable from text alone ------------------------------------
#
# Each entry is (label, compiled pattern, the rule it belongs to). Only rules that can be
# verified by looking at the words are here; "面倒見がよい" cannot be counted and is left to the
# person reading the samples at the end of the report.
POLITE = re.compile(r"(です|ます|ました|ません|でした|ございま|いたしま|ましょう)")
BRACKETED = re.compile(r"[（(][^）)]{1,30}[）)]")
# Simplified-Chinese characters that are not used in Japanese. Not exhaustive on purpose: these
# are the ones Qwen has actually produced here, and a false positive would be worse than a miss.
SIMPLIFIED = set("这说么对们还过开问经种时间样从为现书东车长门马鸟龙风飞气发爱听买卖实关应单双丽乐习乡冲纪级练红")

# Phrases the character is not supposed to say. This list was in the persona prompt and was
# dropped when it was rewritten, after which the phrases came back -- so it lives here too, as
# something measured rather than only something asked for.
BANNED_PHRASES = [
    "一緒に頑張",
    "頑張ろう",
    "頑張ってね",
    "応援してる",
    "なぁ",
    "だね！",
    # The original complaint about this project, in the user's words: "ずっとちょっとまってて
    # ねって言ってる". It was a symptom of a self-prompting loop that has since been fixed, but
    # the phrase itself came back in generated data ("お仕事中だよ！ちょっとまってて！"), so it
    # is banned outright. She has nothing to make anyone wait for.
    "ちょっとまって",
    "ちょっと待って",
    "まっててね",
    "待っててね",
    "お仕事中",
    # Added 2026-11-07 from the user's own notes on six consecutive replies. 「……うん。」 had
    # become a verbal tic: it closed 4 of 7 replies in one sitting, and the note on it reads
    # 「この反応が多すぎる」. 「だもの」 came with 「半のおがおかしい」 on a reply that also
    # invented a League of Legends champion out of nothing.
    "……うん。",
    "だもの",
    "なんだもの",
]


def read_jsonl(path):
    """Yield parsed records, skipping damaged lines rather than dying on them."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def strip_tag(text):
    return EMOTION_TAG.sub("", text or "").strip()


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text or "") if s.strip()]


def visible_len(text):
    """Length in characters a listener would notice, ignoring punctuation and spacing."""
    return len(PUNCT.sub("", text or ""))


def ending(sentence, n=2):
    """The last n non-punctuation characters: a crude but honest stand-in for 語尾."""
    core = PUNCT.sub("", sentence or "")
    return core[-n:] if len(core) >= n else core


def pct(part, whole):
    return (100.0 * part / whole) if whole else 0.0


class Corpus:
    """Everything the log says about how she has actually spoken."""

    def __init__(self, since=None):
        self.since = since
        self.turns = []
        self.events = []
        self.feedback = []
        self.inputs = 0
        self._load()

    def _load(self):
        for rec in read_jsonl(LOG_PATH):
            when = rec.get("time", "")
            if self.since and when < self.since:
                continue
            kind = rec.get("type")
            if kind == "turn":
                self.turns.append(rec)
            elif kind == "event":
                self.events.append(rec)
            elif kind == "feedback":
                self.feedback.append(rec)
            elif kind == "input":
                self.inputs += 1

        # feedback.jsonl is the older, standalone file; conversation.jsonl only started carrying
        # feedback later. Merge both and drop duplicates so counts are not inflated.
        seen = {(f.get("user"), f.get("alice"), f.get("better")) for f in self.feedback}
        for rec in read_jsonl(FEEDBACK_PATH):
            if self.since and rec.get("time", "") < self.since:
                continue
            key = (rec.get("user"), rec.get("alice"), rec.get("better"))
            if key not in seen:
                seen.add(key)
                self.feedback.append(rec)

    # ---- what was said ---------------------------------------------------------------------

    def spoken_turns(self):
        """Turns that produced actual speech. A cancelled turn says nothing about character."""
        out = []
        for t in self.turns:
            reply = t.get("reply") or {}
            text = strip_tag(reply.get("spoken") or reply.get("raw") or "")
            if reply.get("cancelled") or not text:
                continue
            out.append((t, text))
        return out

    def self_initiated(self):
        """Lines nobody prompted: the startup greeting.

        Kept apart from replies on purpose. The Shizuku method weights self-initiated
        utterances more heavily for tone, because a reply is partly shaped by whatever it is
        answering. Worth being honest that there are very few of these here -- she is almost
        entirely reactive, which is itself a finding.
        """
        return [e for e in self.events if e.get("kind") == "startup_greeting"]


def analyse(corpus, top):
    spoken = corpus.spoken_turns()
    n = len(spoken)

    stats = {
        "turns_logged": len(corpus.turns),
        "turns_spoken": n,
        "turns_cancelled": len(corpus.turns) - n,
        "inputs_logged": corpus.inputs,
        "events": len(corpus.events),
        "feedback": len(corpus.feedback),
        "first_time": min((t.get("time", "") for t in corpus.turns), default=""),
        "last_time": max((t.get("time", "") for t in corpus.turns), default=""),
    }

    sources = collections.Counter(
        (t.get("input") or {}).get("source", "?") for t, _ in spoken)
    emotions = collections.Counter(
        ((t.get("reply") or {}).get("emotion") or "?") for t, _ in spoken)

    first_lens, sent_counts, endings = [], [], collections.Counter()
    violations = collections.defaultdict(list)   # rule -> [(text, detail)]
    all_sentences = collections.Counter()

    for t, text in spoken:
        sents = sentences(text)
        if not sents:
            continue
        sent_counts.append(len(sents))
        first_lens.append(visible_len(sents[0]))
        for s in sents:
            endings[ending(s)] += 1
            all_sentences[s] += 1

        # The persona asks for a reaction of 10 characters or fewer as the opening sentence;
        # this is the single most concrete rule in the prompt, so it is worth measuring alone.
        if visible_len(sents[0]) > 10:
            violations["opening_too_long"].append((text, f"1文目 {visible_len(sents[0])}字"))

        m = POLITE.search(text)
        if m:
            violations["polite"].append((text, m.group(0)))

        m = BRACKETED.search(text)
        if m:
            violations["bracketed"].append((text, m.group(0)))

        bad = sorted(set(text) & SIMPLIFIED)
        if bad:
            violations["simplified"].append((text, "".join(bad)))

        for phrase in BANNED_PHRASES:
            if phrase in text:
                violations["banned"].append((text, phrase))
                break

    repeats = {s: c for s, c in all_sentences.items() if c > 1 and visible_len(s) >= 4}

    timings = [(t.get("timing") or {}) for t, _ in spoken]
    firsts = sorted(x["first_word_ms"] for x in timings if x.get("first_word_ms"))
    totals = sorted(x["total_ms"] for x in timings if x.get("total_ms"))

    return {
        "stats": stats,
        "sources": sources,
        "emotions": emotions,
        "first_lens": first_lens,
        "sent_counts": sent_counts,
        "endings": endings.most_common(top),
        "violations": violations,
        "repeats": sorted(repeats.items(), key=lambda kv: -kv[1])[:top],
        "firsts": firsts,
        "totals": totals,
        "spoken": spoken,
    }


def wanted_phrases(pairs, min_len=2, max_len=8, min_pairs=2):
    """Phrases the person wrote into two or more separate rewrites.

    Not done with difflib. Character-level diffs chop Japanese into fragments that line up
    differently in each edit, so the same requested word comes out as 「マスター」 in one pair
    and 「だ？マ」 in another and neither reaches a count of two. That is not a cosmetic flaw:
    the first version of this file reported six rewrites as merely "shorter" and missed that
    two of them had asked for 「マスター」 by name.

    Instead: every substring of the wanted reply that does not occur in what she actually said
    is a candidate, counted once per rewrite. Substrings that always appear inside a longer
    candidate with the same count are dropped, so 「マスタ」 does not clutter 「マスター」.
    """
    counts = collections.Counter()
    for p in pairs:
        alice, better = p["alice"], p["better"]
        seen = set()
        for size in range(min_len, max_len + 1):
            for i in range(len(better) - size + 1):
                frag = better[i:i + size]
                if frag.strip() != frag or not frag.strip():
                    continue
                # Punctuation-only or near-punctuation fragments say nothing about wording.
                if visible_len(frag) < min_len:
                    continue
                if frag in alice:
                    continue
                seen.add(frag)
        for frag in seen:
            counts[frag] += 1

    kept = {f: c for f, c in counts.items() if c >= min_pairs}
    # Keep only maximal fragments: drop any that is contained in a longer kept fragment that
    # was wanted just as often.
    result = []
    for frag, c in sorted(kept.items(), key=lambda kv: (-kv[1], -len(kv[0]))):
        if any(frag != other and frag in other and kept[other] >= c for other in kept):
            continue
        result.append((frag, c))
    return result


def analyse_feedback(corpus):
    """What the rewrites say, by comparing what she said against what was wanted.

    This is the highest-value signal in the whole file and the scarcest: a rewrite is the one
    place where the target is stated rather than inferred.
    """
    out = {"good": 0, "rewrite": 0, "with_why": 0, "test_looking": 0, "pairs": [],
           # Records that say WHY a reply was wrong without supplying a replacement line.
           # Kept separate from pairs on purpose: they carry no target text, so none of the
           # length/politeness arithmetic below applies -- but they are the only place a RULE
           # is stated rather than inferred from a single example, which makes them the more
           # directly actionable of the two. Before 'ここがおかしい（理由だけ）' became its own
           # button in the chat window there were zero of these, and they were dropped here
           # without a word: counted in with_why, then skipped because `better` was empty.
           "notes": []}
    for f in corpus.feedback:
        verdict = f.get("verdict")
        out["good" if verdict == "good" else "rewrite"] += 1
        if (f.get("why") or "").strip():
            out["with_why"] += 1

        alice = strip_tag(f.get("alice") or "")
        better = strip_tag(f.get("better") or "")
        user = f.get("user") or ""
        # Early smoke-test rows would otherwise be read as real character evidence.
        if "テスト" in user or "テスト" in alice:
            out["test_looking"] += 1
            continue
        if verdict != "rewrite":
            continue
        why = (f.get("why") or "").strip()
        if not better:
            if why:
                out["notes"].append({"user": user, "alice": alice, "why": why})
            continue

        out["pairs"].append({
            "user": user,
            "alice": alice,
            "better": better,
            "why": (f.get("why") or "").strip(),
            "len_from": visible_len(alice),
            "len_to": visible_len(better),
            "sents_from": len(sentences(alice)),
            "sents_to": len(sentences(better)),
            "polite_removed": bool(POLITE.search(alice)) and not POLITE.search(better),
            "banned_removed": [p for p in BANNED_PHRASES if p in alice and p not in better],
            "removed": [],
            "added": [],
        })

    for p in out["pairs"]:
        # Both sides of the edit. difflib on characters is crude for Japanese but good enough to
        # surface "the whole second sentence went away" and "this word was put in".
        #
        # The added side is the more valuable one and was missing from the first version of this
        # file. What the person writes IN is a direct statement of what they want, where a
        # deletion only says what they did not want. The first run without it reported six
        # rewrites as "shorter" and missed that two of them had written 「マスター」 in -- a form
        # of address the persona had been changed to remove.
        sm = difflib.SequenceMatcher(None, p["alice"], p["better"])
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("delete", "replace") and i2 - i1 >= 2:
                p["removed"].append(p["alice"][i1:i2])
            if tag in ("insert", "replace") and j2 - j1 >= 2:
                p["added"].append(p["better"][j1:j2])
        p["removed"] = p["removed"][:4]
        p["added"] = p["added"][:4]

    out["wanted"] = wanted_phrases(out["pairs"])

    # Where a rewrite asks for something the current prompt forbids. These are the findings
    # worth a human's attention: both sides are deliberate, so one of them has to give.
    out["conflicts"] = []
    for p in out["pairs"]:
        if BRACKETED.search(p["better"]):
            out["conflicts"].append(
                ("括弧書き", p["better"], "プロンプトは括弧を禁止している（音声でそのまま読まれるため）"))
        if POLITE.search(p["better"]):
            out["conflicts"].append(
                ("敬語", p["better"], "プロンプトは敬語を禁止している"))
    return out


# ---- reporting ------------------------------------------------------------------------------

def bar(count, total, width=28):
    filled = int(round(width * count / total)) if total else 0
    return "#" * filled + "." * (width - filled)


def dist(label, values, buckets):
    """One line per bucket, so a distribution is readable without a plotting library."""
    total = len(values)
    print(f"\n  {label}  (n={total})")
    if not total:
        print("    データなし")
        return
    for lo, hi, name in buckets:
        c = sum(1 for v in values if lo <= v <= hi)
        print(f"    {name:<12} {c:>4}  {pct(c, total):>5.1f}%  {bar(c, total)}")


def show_samples(items, limit, indent="      "):
    for text, detail in items[:limit]:
        snippet = text if len(text) <= 60 else text[:57] + "…"
        print(f"{indent}・{snippet}    ← {detail}")


def report(corpus, a, fb, top, samples):
    s = a["stats"]
    line = "=" * 78

    print(line)
    print(" アリス人格レポート  —  ログから観測した事実と、そこからの提案")
    print(line)
    print(f"  期間      : {s['first_time'][:16] or '?'}  〜  {s['last_time'][:16] or '?'}")
    print(f"  記録ターン: {s['turns_logged']}  "
          f"(発話あり {s['turns_spoken']} / 中断 {s['turns_cancelled']})")
    print(f"  入力行    : {s['inputs_logged']}    自発発言(起動挨拶): {len(corpus.self_initiated())}")
    print(f"  フィードバック: {s['feedback']}  "
          f"(good {fb['good']} / rewrite {fb['rewrite']} / "
          f"理由記入 {fb['with_why']} / テスト行 {fb['test_looking']})")

    n = s["turns_spoken"]
    if not n:
        print("\n  発話のあるターンが0件です。アリスと話してからもう一度実行してください。")
        return

    # ---- A. 観測 ---------------------------------------------------------------------------
    print("\n" + line)
    print(" A. 観測（事実のみ）")
    print(line)

    print("\n  入力経路")
    for src, c in a["sources"].most_common():
        print(f"    {src:<12} {c:>4}  {pct(c, n):>5.1f}%")

    print("\n  感情タグの実際の分布")
    for tag, c in a["emotions"].most_common():
        print(f"    {tag:<12} {c:>4}  {pct(c, n):>5.1f}%  {bar(c, n)}")

    dist("1文目の長さ（句読点を除いた文字数）", a["first_lens"], [
        (0, 4, "1-4字"), (5, 10, "5-10字 ◎"), (11, 20, "11-20字"),
        (21, 40, "21-40字"), (41, 9999, "41字以上"),
    ])
    dist("1返答あたりの文数", a["sent_counts"], [
        (1, 1, "1文"), (2, 2, "2文 ◎"), (3, 3, "3文"), (4, 9999, "4文以上"),
    ])

    print(f"\n  文末表現 上位{top}（語尾の癖）")
    for end, c in a["endings"]:
        print(f"    …{end:<6} {c:>4}")

    if a["firsts"] or a["totals"]:
        def p50p90(xs):
            if not xs:
                return "—", "—"
            return f"{xs[len(xs)//2]}ms", f"{xs[int(len(xs)*0.9)]}ms"
        f50, f90 = p50p90(a["firsts"])
        t50, t90 = p50p90(a["totals"])
        print("\n  応答速度       中央値 / 90%ile")
        print(f"    最初の声まで  {f50:>8} / {f90:>8}")
        print(f"    返答おわりまで{t50:>8} / {t90:>8}")

    # ---- B. ルール違反 ---------------------------------------------------------------------
    print("\n" + line)
    print(" B. プロンプトのルールが守られているか")
    print(line)

    checks = [
        ("opening_too_long", "1文目を10文字以内にする", "破られた回数"),
        ("polite", "敬語を使わない（ですます）", "敬語が出た回数"),
        ("bracketed", "括弧書きを使わない", "括弧が出た回数"),
        ("simplified", "簡体字を使わない", "簡体字が出た回数"),
        ("banned", "禁止フレーズを言わない", "禁止語が出た回数"),
    ]
    v = a["violations"]
    for key, rule, label in checks:
        hits = v.get(key, [])
        rate = pct(len(hits), n)
        mark = "OK  " if not hits else ("警告" if rate < 15 else "問題")
        print(f"\n  [{mark}] {rule}")
        print(f"         {label}: {len(hits)} / {n}  ({rate:.1f}%)")
        if hits:
            show_samples(hits, samples)

    if a["repeats"]:
        print(f"\n  [問題] 毎回ちがう言い回しをする")
        print(f"         同じ文が2回以上: {len(a['repeats'])}種")
        for text, c in a["repeats"][:samples]:
            snippet = text if len(text) <= 56 else text[:53] + "…"
            print(f"      ・{snippet}    ← {c}回")
    else:
        print(f"\n  [OK  ] 毎回ちがう言い回しをする")
        print("         同じ文の繰り返しなし")

    # ---- C. フィードバック -----------------------------------------------------------------
    print("\n" + line)
    print(" C. フィードバックが示していること")
    print(line)

    pairs = fb["pairs"]
    if not pairs:
        print("\n  使える書き換え例が0件です。")
        print("  chat.html の「本当は何と言ってほしかった？」を埋めると、ここに")
        print("  『あなたが実際に何を削って何を足したか』が出ます。")
    else:
        shorter = sum(1 for p in pairs if p["len_to"] < p["len_from"])
        fewer = sum(1 for p in pairs if p["sents_to"] < p["sents_from"])
        polite = sum(1 for p in pairs if p["polite_removed"])
        banned = sum(1 for p in pairs if p["banned_removed"])
        avg_from = sum(p["len_from"] for p in pairs) / len(pairs)
        avg_to = sum(p["len_to"] for p in pairs) / len(pairs)

        print(f"\n  書き換え {len(pairs)}件の傾向")
        print(f"    短くされた        {shorter}/{len(pairs)}")
        print(f"    文数を減らされた  {fewer}/{len(pairs)}")
        print(f"    敬語を外された    {polite}/{len(pairs)}")
        print(f"    禁止語を消された  {banned}/{len(pairs)}")
        print(f"    平均文字数        {avg_from:.1f}字 → {avg_to:.1f}字")

        print("\n  個別（アリスの発言 → 望まれた発言）")
        for p in pairs:
            print(f"\n    入力: {p['user'][:50]}")
            print(f"    言った  : {p['alice'][:60]}")
            print(f"    望まれた: {p['better'][:60]}")
            if p["removed"]:
                print(f"    消された: {' / '.join(r.strip() for r in p['removed'] if r.strip())}")
            if p["added"]:
                print(f"    足された: {' / '.join(r.strip() for r in p['added'] if r.strip())}")
            if p["why"]:
                print(f"    理由    : {p['why'][:70]}")

        if fb["wanted"]:
            print("\n  2件以上の書き換えで『足された』表現 — くり返し求められているもの")
            for frag, c in fb["wanted"]:
                print(f"    「{frag}」  {c}件")

        if fb["conflicts"]:
            print("\n  【要確認】 書き換えが、今のプロンプトと矛盾している箇所")
            for kind, text, why in fb["conflicts"]:
                print(f"    {kind}: 「{text[:50]}」")
                print(f"        → {why}")

    if fb["notes"]:
        print(f"\n  指摘 {len(fb['notes'])}件（書き直し文なし、理由だけ）")
        print("    ここに書かれているのは規則です。1件の例ではなく、プロンプトに書ける形に近い。")
        for n in fb["notes"]:
            print(f"\n    入力  : {n['user'][:50]}")
            print(f"    言った: {n['alice'][:60]}")
            print(f"    指摘  : {n['why'][:120]}")

    # ---- D. 提案 ---------------------------------------------------------------------------
    print("\n" + line)
    print(" D. 更新案（提案のみ。自動適用はしない）")
    print(line)

    proposals = build_proposals(a, fb, n)
    if not proposals:
        print("\n  提案なし。観測されたルール違反はありません。")
    for i, p in enumerate(proposals, 1):
        print(f"\n  {i}. [{p['kind']}] {p['title']}")
        print(f"     根拠: {p['evidence']}")
        print(f"     提案: {p['action']}")
        if p.get("note"):
            print(f"     {p['note']}")

    # ---- E. 口調サンプル -------------------------------------------------------------------
    print("\n" + line)
    print(" E. 口調サンプル（原文のまま。few-shot の材料）")
    print(line)

    selfs = corpus.self_initiated()
    if selfs:
        print("\n  自発発言（起動挨拶）— 相手に合わせていないので人格が出やすい")
        for e in selfs[-samples:]:
            print(f"    [{e.get('emotion')}] {e.get('line')}")

    print("\n  返答（短い順に、重複を除いて）")
    seen, picked = set(), []
    for t, text in sorted(a["spoken"], key=lambda x: visible_len(x[1])):
        if text in seen or visible_len(text) < 4:
            continue
        seen.add(text)
        emo = (t.get("reply") or {}).get("emotion")
        picked.append(f"    [{emo}] {text}")
        if len(picked) >= samples * 2:
            break
    print("\n".join(picked) or "    データなし")

    print("\n" + line)
    print(" 注意: 上の提案はどれも自動適用されていません。採否はあなたが決めてください。")
    print(line)
    return proposals


def build_proposals(a, fb, n):
    """Turn the counts into concrete, evidence-backed suggestions.

    Thresholds are deliberately crude. A proposal is a prompt to look, not a verdict, and a
    tunable threshold would just move the guessing somewhere less visible.
    """
    out = []
    v = a["violations"]

    def rate(key):
        return pct(len(v.get(key, [])), n)

    if rate("banned") > 0:
        hits = collections.Counter(d for _, d in v["banned"])
        out.append({
            "kind": "復活",
            "title": "禁止フレーズのリストをプロンプトに戻す",
            "evidence": f"{len(v['banned'])}/{n} 回 ({rate('banned'):.1f}%)  "
                        f"内訳: {dict(hits)}",
            "action": "constants.py の【話し方】に「次の言い方はしない」節を作り、"
                      "観測された語をそのまま列挙する",
            "note": "このリストは persona_v4 で消えた。消えた後に再出現しているので、"
                    "プロンプトが効いていた証拠でもある",
        })

    if rate("polite") > 10:
        out.append({
            "kind": "強化",
            "title": "敬語の禁止がプロンプトで効いていない",
            "evidence": f"{len(v['polite'])}/{n} 回 ({rate('polite'):.1f}%) で ですます が出ている",
            "action": "プロンプトを強める案と、safety.py 側で文を弾く案の二択。"
                      "safety.py のコメントどおり、プロンプトは8Bに無視された実績がある",
            "note": "【要確認】視聴者には少し丁寧に話す設定があるため、"
                    "全面禁止にすると設定と矛盾する",
        })
    elif v.get("polite"):
        out.append({
            "kind": "確認",
            "title": "敬語はおおむね抑えられている",
            "evidence": f"{len(v['polite'])}/{n} 回 ({rate('polite'):.1f}%) のみ",
            "action": "現状維持。ルールの確信度を上げてよい",
        })

    if rate("bracketed") > 0:
        out.append({
            "kind": "緊急",
            "title": "括弧書きが音声に流れている",
            "evidence": f"{len(v['bracketed'])}/{n} 回。例: {v['bracketed'][0][1]}",
            "action": "プロンプト任せをやめ、tts に渡す前に括弧を除去する。"
                      "括弧の中身はそのまま読み上げられる（「（怯え）」→「オビエ」）",
        })

    if rate("simplified") > 0:
        out.append({
            "kind": "緊急",
            "title": "簡体字が出力に混ざっている",
            "evidence": f"{len(v['simplified'])}/{n} 回。例: {v['simplified'][0][1]}",
            "action": "safety.py に決定的なチェックを追加する。"
                      "プロンプトでの禁止は無視された実績がある",
        })

    long_rate = rate("opening_too_long")
    if long_rate > 40:
        out.append({
            "kind": "再検討",
            "title": "「1文目は10文字以内」が守られていない",
            "evidence": f"{len(v['opening_too_long'])}/{n} 回 ({long_rate:.1f}%) で超過",
            "action": "ルールを守らせる（few-shot を短い反応だけに差し替える）か、"
                      "ルール自体をやめるかを決める",
            "note": "【要確認】半分以上破られているルールは、守れていないのではなく"
                    "キャラに合っていない可能性がある",
        })

    if a["repeats"]:
        worst = a["repeats"][0]
        out.append({
            "kind": "強化",
            "title": "同じ文を繰り返している",
            "evidence": f"{len(a['repeats'])}種が2回以上。最多: 「{worst[0][:30]}」{worst[1]}回",
            "action": "直近に話した文を揮発コンテキストに入れて避けさせる。"
                      "プロンプトの「繰り返さない」指示だけでは足りていない",
        })

    # Emotion: the anger bug came from a prompt line, and a flat distribution is how it showed.
    emo = a["emotions"]
    if emo:
        top_tag, top_count = emo.most_common(1)[0]
        if pct(top_count, n) > 60:
            out.append({
                "kind": "観察",
                "title": f"感情タグが {top_tag} に偏っている",
                "evidence": f"{top_tag} が {top_count}/{n} ({pct(top_count, n):.1f}%)",
                "action": "偏り自体は不具合ではない。無理に散らす指示を入れると"
                          "「怒る理由がないのに angry」の不具合が再発する",
                "note": "【要確認】偏りを直すべきかは、聞いていて不自然かどうかで決める",
            })
        unused = [t for t in ("neutral", "happy", "excited", "sad", "angry",
                              "surprised", "embarrassed", "sleepy") if t not in emo]
        if unused:
            out.append({
                "kind": "観察",
                "title": "一度も選ばれていない感情タグがある",
                "evidence": f"未出現: {', '.join(unused)}",
                "action": "few-shot に該当タグの例が無いものを足す。"
                          "sleepy は persona_v4 で例が消えている",
            })

    pairs = fb["pairs"]
    if pairs:
        shorter = sum(1 for p in pairs if p["len_to"] < p["len_from"])
        if shorter >= max(2, len(pairs) * 0.6):
            avg_to = sum(p["len_to"] for p in pairs) / len(pairs)
            out.append({
                "kind": "提案",
                "title": "望まれている返答は、今より短い",
                "evidence": f"書き換え {len(pairs)}件のうち {shorter}件が短縮。"
                            f"望まれた平均 {avg_to:.1f}字",
                "action": f"few-shot の assistant 側を平均 {avg_to:.0f}字程度に差し替える。"
                          "指示文より例文のほうが効く",
            })

    for frag, c in fb.get("wanted", []):
        out.append({
            "kind": "要確認",
            "title": f"書き換えで「{frag}」が繰り返し足されている",
            "evidence": f"別々の書き換え {c}件で挿入されている",
            "action": f"「{frag}」を話し方のルールとして採用するか決める。"
                      "書き換えは『こう言ってほしい』の直接的な表明なので、"
                      "推測より強い根拠になる",
            "note": "【要確認】20問の回答から外した要素がここに再出現している場合、"
                    "20問の回答とログのどちらを採るかの判断になる",
        })

    for kind, text, why in fb.get("conflicts", []):
        out.append({
            "kind": "矛盾",
            "title": f"望まれた返答に{kind}が含まれている",
            "evidence": f"「{text[:40]}」",
            "action": f"{why}。どちらかを変える必要がある",
            "note": "【要確認】括弧の場合、表示には残して音声に渡す直前で取り除く、"
                    "という折衷ができる",
        })

    if fb["with_why"] == 0 and fb["rewrite"] > 0:
        out.append({
            "kind": "要対応",
            "title": "「どこがおかしいか」が1件も記入されていない",
            "evidence": f"rewrite {fb['rewrite']}件すべて why が空",
            "action": "書き換えだけだと『何を直すか』は分かるが『どのルールが壊れたか』"
                      "が分からない。why が埋まると提案を規則として書ける",
        })

    total_usable = len(pairs)
    if total_usable < 100:
        out.append({
            "kind": "前提",
            "title": "ファインチューニングには、まだデータが足りない",
            "evidence": f"使える (発言 → 望まれた発言) の組: {total_usable}組",
            "action": "スタイル適応の実務的な下限は 100〜200組。"
                      "まず組数を増やすこと（self-chat で種から増やせる）",
            "note": "この数で学習すると、学習ではなく暗記になる。"
                    "few-shot の入力でテストして9/10一致した件と同じ失敗",
        })

    return out


def main():
    ap = argparse.ArgumentParser(description="アリスのログから人格の観測と提案を出す")
    ap.add_argument("--since", help="この日時以降のみ (例: 2026-11-01)")
    ap.add_argument("--top", type=int, default=10, help="頻度リストの表示件数")
    ap.add_argument("--samples", type=int, default=4, help="各項目の例示件数")
    ap.add_argument("--json", help="結果をJSONでも書き出す")
    args = ap.parse_args()

    corpus = Corpus(since=args.since)
    if not corpus.turns and not corpus.feedback:
        print(f"ログが見つかりません: {LOG_PATH}")
        return 1

    a = analyse(corpus, args.top)
    fb = analyse_feedback(corpus)
    proposals = report(corpus, a, fb, args.top, args.samples)

    if args.json:
        payload = {
            "stats": a["stats"],
            "sources": dict(a["sources"]),
            "emotions": dict(a["emotions"]),
            "endings": a["endings"],
            "violation_counts": {k: len(v) for k, v in a["violations"].items()},
            "repeats": a["repeats"],
            "feedback": {k: v for k, v in fb.items() if k != "pairs"},
            "feedback_pairs": fb["pairs"],
            "proposals": proposals,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nJSON を書き出しました: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
