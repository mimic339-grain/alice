"""Grow the fine-tuning set from the handful of real examples into something trainable.

The blocker measured by personaExtract.py: six usable (what she said -> what was wanted) pairs.
Practitioner guidance for style adaptation with LoRA puts the floor at 100-200 high-quality,
diverse examples, and LIMA's result -- 1,000 examples beating a 52,000-example model -- is the
reason the floor is that low rather than tens of thousands. Six is not a small dataset, it is
memorisation waiting to happen: fine-tuning on it would reproduce the earlier failure where
testing on the few-shot's own inputs gave 9/10 verbatim matches.

The method is Self-Instruct's, adapted. Self-Instruct bootstrapped 52k instructions from 175
hand-written seeds by sampling a few seeds per prompt; Baize and UltraChat do the dialogue
version by having a model play both sides. Two things make it work here rather than just
producing more of the same:

  1. THE SEEDS ARE REAL. 118 user inputs are already in the log -- things actually said to her,
     with the actual distribution of half-heard mumbles, one-word pokes and memes. Those cost
     nothing and no generator invents them. New inputs are generated only to extend that set.

  2. REPLIES ARE FILTERED, NOT TRUSTED. Every generated reply is checked against the rules
     personaExtract.py measures: no polite forms, no brackets, no banned phrases, opening
     sentence within 10 characters, at most three sentences, no simplified characters, not a
     duplicate, and close to the length the rewrites actually asked for. Most generations are
     thrown away. Rejection sampling is the part that turns "more output" into "data".

What this does NOT do, and it matters: an 8B model generating its own training data cannot
invent character it does not already have. This produces CONSISTENCY -- the rules obeyed every
time instead of 83% of the time -- not depth. The six real rewrites remain the only statements
of what is actually wanted, so they are weighted in as few-shot for every generation and are
written into the output verbatim. Depth has to come from more rewrites, which only you can
write.

Usage:
    python dataBootstrap.py --dry-run            # show seeds and filters, call nothing
    python dataBootstrap.py --target 120         # generate until 120 accepted pairs
    python dataBootstrap.py --new-inputs 40      # also invent 40 fresh user inputs
    python dataBootstrap.py --out data/sft.jsonl
"""

import argparse
import collections
import json
import os
import random
import re
import sys
import time

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from constants import (AI_NAME, EMOTION_TAGS, FEW_SHOT, LLM_ENDPOINT, PERSONA_REMINDER,
                       SYSTEM_PROMPT)
from personaExtract import (BANNED_PHRASES, BRACKETED, Corpus, POLITE, SIMPLIFIED,
                            analyse_feedback, sentences, strip_tag, visible_len)

DEFAULT_OUT = os.path.join(BASE, "data", "persona_sft.jsonl")

# Copying a few-shot example back verbatim is not a new data point, it is the example again.
FEW_SHOT_REPLIES = {re.sub(r"^\[\w+\]\s*", "", text)
                    for role, text in FEW_SHOT if role == "assistant"}

# Accept/reject thresholds. Every one of these is a rule personaExtract.py already counts, so a
# generated reply that passes is a reply that would have shown up clean in the report.
MAX_SENTENCES = 3
MAX_FIRST_SENTENCE = 10
# Was 2, which was the main reason the first full run averaged 4.4 characters against the 8.7
# the rewrites asked for. A filter that only rejects selects for whatever passes most cheaply,
# and the cheapest way to satisfy "short, no polite forms, no banned phrases" is to say almost
# nothing: 「ああ、死ぬ。」「またなき。」. Rejection sampling needs a floor as well as a ceiling.
MIN_CHARS = 4
MAX_CHARS = 36

# Share of rows allowed to carry the same emotion tag. greeting.py already established that
# this model answers "neutral" to almost anything when left to choose; the first full run came
# back 93.3% neutral even with the examples carrying real tags. So the emotion is requested per
# generation, the same fix greeting.py uses, and capped here as a backstop.
EMOTION_SHARE = 0.45

# Seeds longer than this are not someone talking to her.
#
# The microphone is an open condenser next to the speakers, and the log contains several minutes
# of a webinar it picked up and transcribed as user input -- job-hunting talk about 3D
# portfolios, hundreds of characters long, which she then answered. Left in, those would be the
# longest and most distinctive seeds in the set and would steer every generated input towards
# polite business Japanese. Real things said to her are short.
MAX_SEED_CHARS = 24

# Business-polite Japanese is the signature of the same webinar bleed, and some of it is short
# enough to pass the length cap ("ありがとうございます。スタラミアスターさんですかね、").
SEED_NOT_ADDRESSED_TO_HER = re.compile(
    r"(ございます|よろしくお願い|いただき|恐縮|御社|ポートフォリオ|説明会|質問の方法"
    r"|ですけど|ますけど|でしょうか|ということで|となりますが)")

# Replies that are nothing but an acknowledgement. Each one is legitimate -- she should be able
# to just say "うん" -- but the accept filter rewards brevity, so the model collapses onto them
# and the first trial run produced "うん。" / "うん" / "うん？" over and over. Capped as a share
# of the set rather than banned.
GENERIC_REPLIES = {
    "うん", "ううん", "うんうん", "はい", "へぇ", "ふーん", "そっか", "なに", "ん", "ああ",
    "うーん", "そう", "まあ", "はいはい", "おお",
}
GENERIC_SHARE = 0.12

# Share of replies allowed to open with the same couple of syllables. Without this the set
# measured 20.8% starting with 「うん」 and 17.5% with 「へえ/へへ/へえ～」 -- 38% of all replies
# beginning one of two ways. Every individual row passed every other rule; the tic only exists
# at the level of the whole set, which is also the level the model would learn it at.
OPENER_SHARE = 0.12

# Replies that claim a body or an errand. The persona says not to invent these ("身体がない
# ことについて聞かれたら嘘はつかない"), and the first clean run still produced
# 「ラーメン作ってたの」 and 「お仕事中だよ」. A rule in the prompt did not stop it, so it is
# checked here instead.
# Inputs that give her a reason to be angry.
#
# This exists because of a mistake made here. The emotion tag is requested per generation to
# stop the set coming back 93% neutral, and requesting them round-robin meant asking for
# [angry] on inputs like 「お腹空いた。」 and 「今週のスケジュールは？」. The model obliged:
# 「うるさい。食べる時間じゃない。」. That is exactly the bug reported against the live system
# -- "何も怒ってないのに怒り出す" -- which the prompt rule 怒る理由がないのに angry を選ばない
# こと was written to fix. Training on those rows would have put the bug into the weights, where
# a prompt rule can no longer reach it.
#
# So anger is only ever requested, and only ever accepted, when the input earns it.
HOSTILE_INPUT = re.compile(
    r"(うざ|ウザ|きもい|キモ|黙れ|だまれ|うるさい|ばか|バカ|馬鹿|あほ|アホ|死ね|消えろ"
    r"|くそ|クソ|ゴミ|役立たず|使えない|つまらない|きらい|嫌い|嘘|うそつき|だまされ"
    r"|しらない|興味ない|どうでもいい|やめろ)")

BODY_CLAIM = re.compile(
    r"(食べた|食べてた|作ってた|飲んだ|飲んでた|寝てた|歩いて|触っ|行ってきた|買ってきた"
    r"|お仕事中|仕事してた)")


class Rejected(Exception):
    """Raised with the reason a generated reply was thrown away."""


def opener(text):
    """The first sound of a reply, used to stop one opening taking over the set.

    Keyed on a single character, not two or three. Keying on more looked right and did almost
    nothing: 「へえ」「へ」「へえー」 counted as three separate openings, each comfortably under
    the cap, while 「へ」-initial replies were 16.7% of the set. A Japanese verbal tic lives at
    the mora, so that is where it has to be counted. Coarse on purpose -- it lumps 「うん」 with
    「うわ」 -- which costs some good rows and is the right trade for not training in a tic.
    """
    m = re.match(r"^([ぁ-んァ-ヶ])", text or "")
    return m.group(1) if m else ""


def normalise(text):
    """Strip punctuation and elongation so near-identical replies collide in the dedupe set.

    Without this, "うん" / "うん。" / "うん？" / "うーん" are four distinct rows that teach the
    same thing, and the model will happily supply all four.
    """
    return re.sub(r"[\s。、．，.,!?！？…ー〜~]", "", text or "")


def check_reply(text, seen, user_text="", generic_count=0, generic_cap=10 ** 9,
                emotion_counts=None, emotion_cap=10 ** 9,
                opener_counts=None, opener_cap=10 ** 9):
    """Raise Rejected with a reason, or return the cleaned reply and its emotion tag."""
    raw = (text or "").strip()
    m = re.match(r"^\s*\[([a-zA-Z_]+)\]\s*(.+)$", raw, re.S)
    if not m:
        raise Rejected("感情タグなし")
    emotion, body = m.group(1).lower(), m.group(2).strip()
    body = body.split("\n")[0].strip()

    if not body:
        raise Rejected("本文が空")
    if visible_len(body) < MIN_CHARS:
        raise Rejected("短すぎる")
    if visible_len(body) > MAX_CHARS:
        raise Rejected(f"長すぎる ({visible_len(body)}字)")

    sents = sentences(body)
    if not sents:
        raise Rejected("文に分割できない")
    if len(sents) > MAX_SENTENCES:
        raise Rejected(f"{len(sents)}文（{MAX_SENTENCES}文まで）")
    if visible_len(sents[0]) > MAX_FIRST_SENTENCE:
        raise Rejected(f"1文目 {visible_len(sents[0])}字")

    if POLITE.search(body):
        raise Rejected("敬語")
    if BRACKETED.search(body):
        raise Rejected("括弧書き")
    if set(body) & SIMPLIFIED:
        raise Rejected("簡体字")
    for phrase in BANNED_PHRASES:
        if phrase in body:
            raise Rejected(f"禁止語: {phrase}")
    # An LLM asked to be in character will sometimes narrate instead of speaking.
    if re.search(r"(アリスは|彼女は|と言った|と答え)", body):
        raise Rejected("地の文")
    if BODY_CLAIM.search(body):
        raise Rejected("身体や用事があるかのような返答")
    # 「へえ～？ tasty？」. Qwen reaches for English when it runs out of Japanese, and the
    # character does not speak it.
    if re.search(r"[A-Za-z]{2,}", body):
        raise Rejected("ラテン文字が混ざっている")
    if emotion == "angry" and not HOSTILE_INPUT.search(user_text or ""):
        raise Rejected("怒る理由のない入力に angry")
    if body in FEW_SHOT_REPLIES:
        raise Rejected("few-shot の文の丸写し")

    key = normalise(body)
    # Echoing the input back is the other way a reply can pass every rule and still be useless.
    # The first run produced 「6枚？」 to 「今、6枚とかはちょっと…」 and 「何の曲？」 to
    # 「あ、あれって何の曲？」 -- grammatical, in character, and teaching her to parrot.
    user_key = normalise(user_text)
    if len(key) >= 2 and user_key and key in user_key:
        raise Rejected("入力の反復（オウム返し）")

    if key in seen:
        raise Rejected("重複（句読点違いを含む）")
    if key in GENERIC_REPLIES and generic_count >= generic_cap:
        raise Rejected("相づちだけの返答が上限に達している")
    if emotion_counts is not None and emotion_counts.get(emotion, 0) >= emotion_cap:
        raise Rejected(f"{emotion} が上限に達している")
    head = opener(body)
    if head and opener_counts is not None and opener_counts.get(head, 0) >= opener_cap:
        raise Rejected(f"「{head}」で始まる返答が上限に達している")
    return body, emotion


def gather_seeds(corpus):
    """Real user inputs from the log, plus the rewrite pairs, which are the only ground truth."""
    inputs, seen = [], set()
    dropped = collections.Counter()
    for rec in corpus.turns:
        text = ((rec.get("input") or {}).get("text") or "").strip()
        if not text or text in seen:
            dropped["空または重複"] += 1
            continue
        if len(text) > MAX_SEED_CHARS:
            dropped[f"長すぎる（{MAX_SEED_CHARS}字超・スピーカーからの混入）"] += 1
            continue
        if SEED_NOT_ADDRESSED_TO_HER.search(text):
            dropped["アリス宛てでない（ビジネス敬語＝混入）"] += 1
            continue
        # Mis-transcriptions are part of real life and worth keeping some of, but a line that is
        # mostly repeated syllables teaches nothing.
        stripped = re.sub(r"[\s。、．，.,!?！？…]", "", text)
        if len(set(stripped)) <= 2:
            dropped["同じ文字の繰り返し（ノイズ）"] += 1
            continue
        seen.add(text)
        inputs.append(text)

    fb = analyse_feedback(corpus)
    return inputs, fb["pairs"], dropped


def chat(messages, max_tokens=80, temperature=0.9, timeout=90):
    payload = {
        "mode": "instruct",
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "messages": messages,
    }
    resp = requests.post(LLM_ENDPOINT + "/v1/chat/completions",
                         headers={"Content-Type": "application/json"},
                         json=payload, timeout=timeout, verify=False)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def screen_seeds(seeds, batch=12):
    """Drop logged inputs that are transcription damage rather than something a person said.

    Whisper always returns something, so the log contains lines like 「どっちに広く死ぬだろ」 and
    「なんでそんな変身来た」 -- half-heard speech rendered as grammatical nonsense. These passed
    every mechanical filter and then produced replies like 「ああ、死ぬ。」 in the first full run.
    No length or character rule separates them from real speech; judging whether a sentence
    means anything is exactly the kind of thing to hand to the model, unlike judging the
    character, which it cannot do.
    """
    keep = []
    for i in range(0, len(seeds), batch):
        chunk = seeds[i:i + batch]
        numbered = "\n".join(f"{n+1}. {s}" for n, s in enumerate(chunk))
        instruction = (
            f"次の各行は音声認識の出力です。配信者が AI VTuber「{AI_NAME}」に向かって"
            "カジュアルに話しかけた言葉だと言えるものだけを選び、"
            "その番号だけをカンマ区切りで答えてください。\n"
            "選ばないもの: 聞き取り失敗で意味が崩れている行。"
            "会議・説明会・商談のような、その場の他人に向けた丁寧な話し方の行。"
            "文として成立していない行。\n"
            "番号以外は何も書かないこと。\n\n" + numbered + "\n/no_think")
        try:
            answer = chat([{"role": "user", "content": instruction}],
                          max_tokens=60, temperature=0.1)
        except Exception as e:
            print(f"  種の選別に失敗、全件採用します ({type(e).__name__}: {e})")
            return seeds
        picked = {int(n) for n in re.findall(r"\d+", answer) if 1 <= int(n) <= len(chunk)}
        keep += [chunk[n - 1] for n in sorted(picked)]
    return keep


def invent_inputs(seeds, count, batch=8):
    """Extend the real inputs with generated ones, Self-Instruct style.

    Sampling a handful of real seeds per prompt is what keeps the generated inputs in the same
    register as the real ones. Asked cold, the model produces tidy full-sentence questions,
    which is not how anyone talks to her.
    """
    out, seen = [], set(seeds)
    attempts = 0
    while len(out) < count and attempts < count * 4:
        attempts += 1
        sample = random.sample(seeds, min(batch, len(seeds)))
        instruction = (
            f"以下は、配信者が AI VTuber「{AI_NAME}」に実際に話しかけた言葉です。\n\n"
            + "\n".join(f"- {s}" for s in sample)
            + "\n\nこれと同じような調子で、別の話しかけを5個だけ作ってください。"
              "長い質問ではなく、雑談・呼びかけ・短い報告・ふざけた一言にすること。"
              "各行を「- 」で始めること。説明は書かないこと。/no_think"
        )
        try:
            text = chat([{"role": "user", "content": instruction}],
                        max_tokens=180, temperature=1.0)
        except Exception as e:
            print(f"  入力生成に失敗 ({type(e).__name__}: {e})")
            break
        for line in text.splitlines():
            line = line.strip().lstrip("-・*0123456789. 　").strip()
            if not line or len(line) > 40 or line in seen:
                continue
            seen.add(line)
            out.append(line)
            if len(out) >= count:
                break
    return out


def build_messages(user_text, rewrite_pairs, want_emotion=None, k=4):
    """The prompt used to generate one reply.

    The real rewrites go in as assistant turns, not as instructions. The rewrites are the only
    place anyone has stated what the right answer looks like, and an example of the right answer
    outweighs a description of it -- which is the same reason the persona itself lives in a real
    system message rather than in a user turn.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": role, "content": content} for role, content in FEW_SHOT]
    for p in random.sample(rewrite_pairs, min(k, len(rewrite_pairs))):
        messages.append({"role": "user", "content": p["user"]})
        # The emotion recorded with the rewrite, not a fixed tag. Hardcoding [neutral] here
        # taught the model that every reply is neutral, and the first trial run came back 100%
        # neutral as a result -- the examples won, as they always do.
        messages.append({"role": "assistant",
                         "content": f"[{p.get('emotion') or 'neutral'}] {p['better']}"})
    ask = PERSONA_REMINDER.strip()
    if want_emotion:
        # Named, not left open. greeting.py found the same thing: asked to pick an emotion this
        # model picks neutral, every time. Telling it which one also diversifies the wording,
        # because the tag changes what kind of sentence it writes.
        ask += f"\n- 今回の感情タグは必ず [{want_emotion}] にすること"
    ask += f"\n- {MIN_CHARS}〜{MAX_CHARS}字で答える。一言で済ませず、ひと呼吸ぶん話す"
    ask += "\n- 相手の言葉をそのまま繰り返さない"
    messages.append({"role": "user", "content": user_text + "\n" + ask + "\n/no_think"})
    return messages


def main():
    ap = argparse.ArgumentParser(description="アリスの学習データを種から増やす")
    ap.add_argument("--target", type=int, default=120, help="採用したい組数")
    ap.add_argument("--new-inputs", type=int, default=0, help="追加で生成する入力の数")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="LLMを呼ばず、種と設定だけ表示")
    ap.add_argument("--no-screen", action="store_true",
                    help="種のLLM選別をしない（文字起こしの失敗も使う）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    corpus = Corpus()
    seeds, pairs, dropped = gather_seeds(corpus)

    print("=" * 74)
    print(" データ増強  —  Self-Instruct 方式、生成は棄却サンプリングで絞る")
    print("=" * 74)
    print(f"  実在する入力（ログから）  : {len(seeds)}")
    for reason, c in dropped.most_common():
        print(f"      除外 {c:>3}  {reason}")
    print(f"  実在する書き換え（正解）  : {len(pairs)}")
    print(f"  目標                      : {args.target} 組")
    print(f"  出力                      : {args.out}")
    print("\n  採用条件")
    print(f"    感情タグあり / {MAX_SENTENCES}文以内 / 1文目{MAX_FIRST_SENTENCE}字以内 "
          f"/ 全体{MIN_CHARS}〜{MAX_CHARS}字")
    print("    敬語なし / 括弧なし / 簡体字なし / 地の文なし / 重複なし")
    print(f"    禁止語なし: {', '.join(BANNED_PHRASES)}")

    if not pairs:
        print("\n  書き換えが0件です。正解の例がないまま生成すると、"
              "今のアリスの癖をそのまま増やすだけになります。")
        print("  chat.html で数件書き換えてから実行してください。")
        return 1

    print("\n  種のサンプル（実際に言われた言葉）")
    for s in random.sample(seeds, min(6, len(seeds))):
        print(f"    - {s}")
    print("\n  正解のサンプル（あなたが書いた望まれる返答）")
    for p in pairs[:4]:
        print(f"    {p['user'][:28]:<30} -> {p['better'][:34]}")

    if args.dry_run:
        print("\n  --dry-run のため、ここで終了します。")
        return 0

    if not args.no_screen:
        print(f"\n  種を選別します（{len(seeds)}件）…")
        before = len(seeds)
        seeds = screen_seeds(seeds)
        print(f"    {before} → {len(seeds)} 件  "
              f"（文字起こしの失敗 {before - len(seeds)} 件を除外）")
        if len(seeds) < 10:
            print("  種が少なすぎます。--no-screen で全件使うか、もっと会話してください。")
            return 1

    if args.new_inputs:
        print(f"\n  入力を {args.new_inputs} 個生成します…")
        extra = invent_inputs(seeds, args.new_inputs)
        print(f"    {len(extra)} 個できました。例: {extra[:3]}")
        seeds = seeds + extra

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    accepted, rejected = [], collections.Counter()
    seen_replies = set()
    generic_cap = max(2, int(args.target * GENERIC_SHARE))
    emotion_cap = max(3, int(args.target * EMOTION_SHARE))
    emotion_counts = collections.Counter()
    opener_cap = max(3, int(args.target * OPENER_SHARE))
    opener_counts = collections.Counter()
    # Every real rewrite goes in verbatim and first. These are the only rows nobody generated.
    for p in pairs:
        emo = p.get("emotion") or "neutral"
        accepted.append({"user": p["user"], "assistant": p["better"],
                         "emotion": emo, "origin": "rewrite"})
        seen_replies.add(normalise(p["better"]))
        emotion_counts[emo] += 1
        opener_counts[opener(p["better"])] += 1

    queue = seeds[:]
    random.shuffle(queue)
    started = time.time()
    attempts = 0
    print(f"\n  生成開始（実在の書き換え {len(accepted)} 組からスタート）")

    while len(accepted) < args.target and attempts < args.target * 8:
        if not queue:
            queue = seeds[:]
            random.shuffle(queue)
        user_text = queue.pop()
        attempts += 1
        # Round-robin through the tags, skipping any already at its cap, so the set comes out
        # spread across emotions instead of 93% neutral.
        choices = [t for t in EMOTION_TAGS if emotion_counts.get(t, 0) < emotion_cap]
        # Never ask for anger unless the input provoked it. See HOSTILE_INPUT above.
        if not HOSTILE_INPUT.search(user_text):
            choices = [t for t in choices if t != "angry"]
        want = random.choice(choices or ["neutral"])
        try:
            reply = chat(build_messages(user_text, pairs, want_emotion=want))
        except Exception as e:
            rejected[f"生成エラー: {type(e).__name__}"] += 1
            continue
        generic_so_far = sum(1 for r in accepted if normalise(r["assistant"]) in GENERIC_REPLIES)
        try:
            body, emotion = check_reply(reply, seen_replies, user_text, generic_so_far,
                                        generic_cap, emotion_counts, emotion_cap,
                                        opener_counts, opener_cap)
        except Rejected as r:
            rejected[str(r)] += 1
            continue
        emotion_counts[emotion] += 1
        opener_counts[opener(body)] += 1
        seen_replies.add(normalise(body))
        accepted.append({"user": user_text, "assistant": body,
                         "emotion": emotion, "origin": "generated"})
        if len(accepted) % 10 == 0:
            rate = 100.0 * len(accepted) / max(1, attempts)
            print(f"    {len(accepted):>4} 組  (試行 {attempts}, 採用率 {rate:.0f}%, "
                  f"{time.time()-started:.0f}s)")

    with open(args.out, "w", encoding="utf-8") as f:
        for row in accepted:
            # The shape SFT tooling expects, with the persona as the system message so the
            # trained model learns the reply given that persona rather than in a vacuum.
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["user"]},
                    {"role": "assistant", "content": f"[{row['emotion']}] {row['assistant']}"},
                ],
                "origin": row["origin"],
            }, ensure_ascii=False) + "\n")

    print("\n" + "=" * 74)
    print(f"  採用 {len(accepted)} 組  (うち実在の書き換え "
          f"{sum(1 for r in accepted if r['origin'] == 'rewrite')} 組)")
    print(f"  試行 {attempts}  採用率 {100.0*len(accepted)/max(1,attempts):.1f}%  "
          f"所要 {time.time()-started:.0f}s")
    print(f"  書き出し: {args.out}")
    print("\n  棄却の理由（多い順）")
    for reason, c in rejected.most_common(12):
        print(f"    {c:>4}  {reason}")

    # Printed because the first trial run was 100% neutral and nothing in the output said so.
    emo = collections.Counter(r["emotion"] for r in accepted)
    print("\n  感情タグの分布")
    for tag, c in emo.most_common():
        print(f"    {tag:<12} {c:>4}  {100.0*c/len(accepted):>5.1f}%")
    lens = [visible_len(r["assistant"]) for r in accepted]
    print(f"\n  返答の長さ  平均 {sum(lens)/len(lens):.1f}字  "
          f"最短 {min(lens)}  最長 {max(lens)}  "
          f"(書き換えが求めた平均は 8.7字)")
    generic = sum(1 for r in accepted if normalise(r["assistant"]) in GENERIC_REPLIES)
    print(f"  相づちだけの行: {generic} / {len(accepted)} "
          f"({100.0*generic/len(accepted):.1f}%, 上限 {generic_cap})")
    print("\n  採用されたサンプル")
    for row in random.sample([r for r in accepted if r["origin"] == "generated"],
                             min(8, max(0, len(accepted) - len(pairs)))):
        print(f"    {row['user'][:26]:<28} -> [{row['emotion']}] {row['assistant']}")
    print("\n  注意: 生成分は『ルールを守った返答』であって『正解』ではありません。")
    print("  人格の深さは書き換えの件数でしか増えません。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
