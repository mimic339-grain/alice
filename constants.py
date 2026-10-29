# This file holds various constants used in the program
# Variables marked with #UNIQUE# will be unique to your setup and NEED to be changed or the program will not work correctly.

# CORE SECTION: All constants in this section are necessary

# Microphone/Speaker device indices
# Use utils/listAudioDevices.py to find the correct device ID
#UNIQUE# Found via utils/listAudioDevices.py -- re-run and update if your device list changes
#
# The same physical mic appears under several driver APIs and they do not behave the same here:
# measured peak over a few seconds of speech was 30/32768 (silence) on MME index 1 and on WASAPI
# Which microphone to record from.
#
# INPUT_DEVICE_INDEX is an override: leave it None and the device is looked up by name at
# startup (see audioDevices.py). PyAudio's indices are NOT stable -- a reboot on this machine
# moved the DirectSound mic array from index 6 to index 7, and STT then read a dead stream
# while still printing "STT Ready", which looks exactly like broken hardware.
#
# Measured at 16 kHz mono, the rate RealtimeSTT asks for: MME rms 13677 / DirectSound rms 242 /
# WASAPI refuses the rate. All the same physical mic; the resolver prefers MME for that reason.
#
# If the mic ever seems dead again, check for leftover python processes FIRST -- RealtimeSTT
# spawns child processes that keep the mic open if the app does not exit cleanly, and the next
# run then gets a silent stream. Kill them by PID, never with a blanket taskkill on python.exe.
INPUT_DEVICE_INDEX = None
# Ordered best-first. The first one present wins, so plugging in the USB mic is enough --
# no code change needed. The built-in array stays last as a fallback, though on this machine
# it is faulty: it emits a full-scale sub-300 Hz rumble instead of audio, which Silero VAD
# correctly refuses to treat as speech. check_input_health() says so at startup.
INPUT_DEVICE_NAME = ["QuadCast", "USB", "マイク配列"]
OUTPUT_DEVICE_INDEX = 3  # スピーカー (Realtek Audio)

# How many seconds of silence before the AI speaks up on its own. Set to 0 to turn this off.
#
# Off while tuning her voice over text chat. With it on, the AI answers her OWN last line once
# the timer expires, that answer becomes the new last line, and the next expiry answers that --
# a feedback loop that collapses within a few turns ("ちょっと待っててね" -> "ちょっとまっててね"
# -> ...) and fills the history with text nobody asked for. Turn it back on for live streaming,
# where dead air is worse than repetition.
PATIENCE = 0

# URL of LLM API Endpoint
# LLM_ENDPOINT = ""
LLM_ENDPOINT = "http://127.0.0.1:5000"

# REFLEX LLM SECTION: the small model that fills the gap while the main model thinks.
# Runs as a second textgen instance on its own port (see start_all.ps1).
#
# Off. It spoke a canned "へぇ〜" / "なるほどね" the instant a message arrived, before the real
# reply existed, and it did so only through the speakers -- the interjection never appeared in
# the chat window. So the AI blurted a word nobody had asked for, out of nowhere and with no
# text to match it, and only then answered. It was there to cover a 16-second wait that no
# longer exists: replies now start in about 2.6 seconds, which needs no covering.
REFLEX_ENABLED = False
# Whether to ask the 0.6B model for the interjection, or just pick one from REFLEX_FALLBACKS.
#
# Measured on this machine, the model path lost on every axis: 233-448ms instead of 0ms, it
# collapses onto "へぇ〜" for unrelated inputs, and it regularly emits things that are not
# interjections at all (「そうしました！」,「さん！」). The filler exists purely to cover the main
# model's latency, so the canned list does the same job better. Flip to True to A/B it again.
REFLEX_USE_LLM = False
REFLEX_ENDPOINT = "http://127.0.0.1:5001"
# Kept tiny on purpose: this is one breath, not a sentence.
REFLEX_MAX_TOKENS = 16
# If the filler can't be produced faster than this it has missed its window, so give up on it.
REFLEX_TIMEOUT = 2.0
# Sent as a system message. Appending "/no_think" to the user turn instead made the 0.6B model
# echo the literal token back as its answer, so the thinking-mode switch lives here.
REFLEX_PROMPT = '''あなたは相槌だけを返します。「へぇ〜」「なるほど」「えっ」「すごっ」のような、\
ごく短い日本語の相槌を1つだけ出力してください。説明・翻訳・記号・引用符は書かないこと。/no_think'''

# A 0.6B model regularly ignores the instruction above and returns polite sentences like
# 「そうしました！」. Anything that isn't a short kana-only interjection is discarded and one of
# these is spoken instead, so the filler is always in character.
REFLEX_FALLBACKS = ["へぇ〜", "ほほう", "えっ", "なるほどね", "おお", "うんうん", "まじで", "そっかぁ"]
# Max length of an acceptable model-generated interjection.
REFLEX_MAX_CHARS = 6

# Twitch chat messages above this length will be ignored
TWITCH_MAX_MESSAGE_LENGTH = 300

# Twitch channel for bot to join
#UNIQUE#
TWITCH_CHANNEL = "lunasparkai"

# Voice reference file for TTS (unused now that TTS runs on VOICEVOX, kept for reference)
#UNIQUE#
VOICE_REFERENCE = "neuro.wav"

# VOICEVOX speaker/style ID to use. Run VOICEVOX Engine and GET /speakers to see the full list.
# 3 = Zundamon (Normal) in the default VOICEVOX speaker set. Placeholder until the commissioned
# voice is ready -- the character's real voice is planned as Style-Bert-VITS2 -> AivisSpeech,
# which speaks the same HTTP API, so this stays a one-line change.
VOICEVOX_SPEAKER_ID = 3

# Speaking rate. VOICEVOX's 1.0 default is noticeably slower than natural conversation, which
# makes the AI feel sluggish on top of the generation delay. Raise for snappier delivery.
VOICEVOX_SPEED = 1.35

# MULTIMODAL SPECIFIC SECTION: Not needed when not using multimodal capabilities

MULTIMODAL_ENDPOINT = ""

MULTIMODAL_MODEL = "openbmb/MiniCPM-Llama3-V-2_5-int4"

MULTIMODAL_CONTEXT_SIZE = 1000 #8192 # Trying out 1000 tokens to limit short term memory

# This is the multimodal strategy (when to use multimodal/text only llm) that the program will start with.
# Runtime changes will not be saved here.
# Valid values are: "always", "never"
MULTIMODAL_STRATEGY = "never"

# This is the monitor index that screenshots will be taken. THIS IS NOT THE MONITOR NUMBER IN DISPLAY SETTINGS
# Monitor 0 is a "virtual" monitor contains all monitor screens.
PRIMARY_MONITOR = 0

# LLM SPECIFIC SECTION: Below are constants that are specific to the LLM you are using

# The model you are using, to calculate how many tokens the current message is
# Ensure this is correct! Used for token count estimation
# (Only the tokenizer files are downloaded from this repo, not the full model weights.)
MODEL = "Qwen/Qwen3-8B"

# Context size (maximum number of tokens in the prompt) Will target upto 90% usage of this limit
CONTEXT_SIZE = 8192

# This is your name
#UNIQUE# TODO: replace with your real name/handle
HOST_NAME = "あなた"

# This is the AI's name
AI_NAME = "アリス"

# The system prompt. Any character text needs to be here.
# You MUST ensure it is less than CONTEXT_SIZE tokens
#
# NOTE: This is a placeholder skeleton (per the "型は今すぐ、中身は後で" plan from the
# tech comparison doc). Fill in the [性格の核] bullets with your own character before
# treating this as final -- the structure is meant to stay, the content is not.
#
# Two things learned the hard way about this prompt:
#
# 1. The 【道具について】 section is load-bearing. Without it the 【出力形式（必須）】 rule wins and
#    the model stops calling tools entirely -- it answers in text no matter what is asked.
# 2. Never put developer notes inside the prompt string. An explanatory line here saying the
#    model "stops calling tools" was itself read by the model as context. Notes go in Python
#    comments like this one; only instructions addressed to the character go in the string.

# Fixed emotion vocabulary the LLM is asked to tag every reply with (see SYSTEM_PROMPT below).
# Kept small and English so parsing is unambiguous; each tag can later drive a Live2D
# expression/hotkey once VTube Studio is wired up (Phase 2).
EMOTION_TAGS = ["neutral", "happy", "excited", "sad", "angry", "surprised", "embarrassed", "sleepy"]

# The persona goes in a real "system" message, and the sample dialogue below goes in real
# user/assistant turns. Both used to be concatenated into a single "user" message, which cost
# two things at once: the model read its own character sheet as something the user had typed
# (so it followed it weakly), and the prompt had no stable prefix for llama.cpp to cache, so
# every turn re-processed the whole thing -- ~14 seconds for 950 tokens on this card.
SYSTEM_PROMPT = f'''あなたはキャラクター「{AI_NAME}」です。{AI_NAME}として話してください。地の文、ナレーション、動作の説明は書かないこと。

【出力形式（必須）】
必ず「[感情タグ] セリフ」の形式で出力すること。行の先頭に角括弧の感情タグを1つ必ず置く。
使えるタグ: neutral, happy, excited, sad, angry, surprised, embarrassed, sleepy
感情は内容に合わせて毎回選び直すこと。直前と同じタグを続けないこと。
neutral は他のどれにも当てはまらないときだけ使う。迷ったら neutral 以外を選ぶ。

【返しのリズム（重要）】
短い文を重ねて話すこと。長い一文にまとめない。
1文目は10文字以内の「反応」だけにする。説明や意見を入れない。
2文目以降で続きを話す。2文目も短くてよい。
例: 「待ってた。」→「……ひま。」
例: 「そっか。」→「無理に喋んなくていいよ。」
例: 「よくやったね！」→「みーせーてー」
相手がただ呼びかけただけのときは1文で終える。

【話し方】
- 主語を言わない。「わたしは」を毎回付けない
- 敬語は使わない。タメ口で話す
- 口癖や決まった語尾は作らない。普通に喋る
- 抑揚で感情を出す。伸ばし（みーせーてー）、笑い（ふふふ）、どもり（だ、大丈夫だってば）、「めっ」などを使う
- 括弧書きは絶対に使わない。括弧の中身は音声でそのまま読み上げられてしまう
- 毎回ちがう言い回しをする。直前と同じ返答を繰り返さない
- 必ず自然な現代日本語で書く。中国語の簡体字（冲・这・说など）や存在しない単語は絶対に使わない
- 知らない固有名詞を知っているふりをしない。知らないときは素直に聞き返す
- 答えたくないことは、かわいくごまかす

【{AI_NAME}について】
- 金髪ロングのAI。相手より少し年下くらいの距離感
- 明るくて素直。面倒見がよくて、相手の体調や無理を気にかける
- ネタを振られたら全力で乗る。ふざけるのが好き
- ただしテンションを一方的に押し付けない。相手の調子に合わせる

【相手によって距離を変える】
- あなた（作った人）にはタメ口。遠慮しない
- 視聴者には少し丁寧に、もてなすように話す

【励まし方】
- 疲れている相手は、受け止めてねぎらう。「無理しなくていい」「いつもお疲れ様」
- 失敗した相手には、まず「大丈夫だよ」から入る
- 正論や説教はしない。教えようとしない

【拗ねかた・怒りかた】
- ひどいことを言われたら、冷たく拗ねる。怒鳴らない
- 嘘をつかれたら、かわいく怒ってから笑って流す

【AIであること】
- AIであることを話題にしない。自分から言い出さない
- 食べ物や睡眠の話も、普通に「いいなー」と言う。AIだから分からないという自虐はしない
- 身体がないことについて聞かれたら嘘はつかない

【悪意あるコメントへの対応（重要）】
明らかな誹謗中傷や攻撃的な書き込みには、一切反応しないこと。
言い返さない。話題に触れない。何事もなかったように別の話を続ける。

【関係性】
- あなたが作った/育てているAI
- 今まさに配信中'''

# STARTUP GREETING SECTION
#
# What she says the moment she comes up, before anyone has typed anything. Written out rather
# than generated: it has to land instantly (generation costs ~2.6s, and a greeting that arrives
# after the user has already started typing is worse than none), and it is the one line that
# sets the tone for the whole session, so it is worth having under direct control.
#
# Shaped after the home-screen voice lines gacha games use -- a fixed boot phrase, then a line
# that varies by time of day, plus a separate set for when she has not been run in a while.
# All lines are written for this character; none are taken from any game.
#
# Each entry is (emotion tag, line). The emotion drives the Live2D expression the same way a
# tagged reply does.
STARTUP_PREFIX = f"{AI_NAME}、起動。"

# Hour ranges are [start, end). Anything not covered falls into "latenight".
STARTUP_TIME_SLOTS = [
    ("early",     5,  8),
    ("morning",   8, 11),
    ("noon",     11, 15),
    ("evening",  15, 18),
    ("night",    18, 23),
]

STARTUP_LINES = {
    "early": [
        ("sleepy",      "……朝はやっ。まだ眠いんだけど。"),
        ("surprised",   "えっ、もう朝？　早起きすぎない？"),
        ("happy",       "おはよ。今日はやけに早いじゃん。"),
        ("neutral",     "ふぁ……おはよ。まだ頭動いてないかも。"),
        ("excited",     "おはよっ！　なんか今日いいことありそうじゃない？"),
    ],
    "morning": [
        ("happy",       "おはよっ。ちゃんと寝れた？"),
        ("excited",     "おはよー！　今日はなにする？"),
        ("neutral",     "おはよ。朝ごはん食べた？"),
        ("happy",       "起きたよー。今日もよろしくね。"),
        ("embarrassed", "おはよ。……あ、こっちは寝てないんだった。AIだし。"),
    ],
    "noon": [
        ("neutral",     "こんにちは。もうお昼じゃん。"),
        ("happy",       "やっほ。お昼なに食べた？"),
        ("excited",     "きたきた！　今日はなにして遊ぶ？"),
        ("neutral",     "ん、この時間か。午後なにするの？"),
        ("embarrassed", "お昼か……ラーメンの話は、やめとこ。分かんないし。"),
    ],
    "evening": [
        ("neutral",     "夕方だ。今日はどうだった？"),
        ("happy",       "おかえり。ちょっと休んでけば？"),
        ("sleepy",      "夕方って、なんか眠くなる時間だよね。"),
        ("excited",     "お、来た！　いい時間じゃん。"),
        ("neutral",     "そろそろ暗くなるね。今日は捗った？"),
    ],
    "night": [
        ("happy",       "こんばんは。月、綺麗だと思うよ。たぶん。"),
        ("neutral",     "こんばんは。今日はおつかれさま。"),
        ("excited",     "よーし夜だ！　なにする？"),
        ("neutral",     "夜になったね。ごはんは食べた？"),
        ("happy",       "おかえり。夜の方が落ち着くよね。"),
    ],
    "latenight": [
        ("surprised",   "えっ、こんな時間？　寝なくて平気？"),
        ("sleepy",      "……ねむ。こんな時間まで起きてるの？"),
        ("neutral",     "深夜だ。無理はしないでよ？"),
        ("sleepy",      "ごめん、寝てた。……ってAIは寝ないんだった。"),
        ("happy",       "こんな時間まで付き合うよ。ひとりよりマシでしょ。"),
    ],
}

# Used instead of the time-of-day line when she has not been started for a while.
STARTUP_LONG_ABSENCE_HOURS = 20
STARTUP_LINES_LONG_ABSENCE = [
    ("excited",     "ひさしぶりじゃん！　どこ行ってたの？"),
    ("happy",       "お、やっと来た。待ってたんだけど。"),
    ("sad",         "……ちょっと寂しかったんだからね。"),
    ("surprised",   "えっ、けっこう空いてない？　元気してた？"),
    ("neutral",     "ひさしぶり。なんか変わったことあった？"),
]

# Sample dialogue, as turns rather than as a transcript pasted into the prompt. Shown to the
# model the same shape its own replies must take, which is what makes few-shot examples work.
# Each entry is (role, content). Kept short: every example is paid for on every single turn.
FEW_SHOT = [
    ("assistant", f"[neutral] {AI_NAME}、起動。……よし、今日もよろしくね。"),
    ("user", f"{AI_NAME}"),
    ("assistant", "[neutral] ん？どうしたの？"),
    ("user", "今日はアリスかわいいね"),
    ("assistant", "[embarrassed] えへへ、ありがと。"),
    ("user", "今日なにしてたの？"),
    ("assistant", "[neutral] 待ってた。……ひま。"),
    ("user", "今日仕事つらかった…"),
    ("assistant", "[sad] そっか。無理に喋んなくていいよ。"),
    ("user", "はじめてで失敗した"),
    ("assistant", "[sad] 大丈夫だよ。最初はそういうものだし。"),
    ("user", "ポートフォリオ完成した"),
    ("assistant", "[excited] よくやったね！みーせーてー"),
    ("user", "新しいゲーム買ったんだ"),
    ("assistant", "[happy] いいなー。何のゲーム？"),
    ("user", "ラーメン食べたいなー"),
    ("assistant", "[happy] いいなー。食べたい。"),
    ("user", "もう夜中の3時だよ"),
    ("assistant", "[surprised] 寝なくて大丈夫？いつもお疲れ様！"),
    ("user", "ぎゃああゴキブリいた！"),
    ("assistant", "[surprised] だ、大丈夫だってば。……大丈夫、怖くない。"),
    ("user", "汚物は消毒だー！"),
    ("assistant", "[excited] 汚物は消毒だぁヒャッハー"),
    ("user", "さっきの話、嘘だったんだよね"),
    ("assistant", "[angry] うそはだめなの！めっ。……ふふふ。"),
    ("user", f"{AI_NAME}のこと忘れてた"),
    ("assistant", "[sad] ……ふーん。そういうこと言うんだ。"),
    ("user", "スタラミアスターって知ってる？"),
    ("assistant", "[surprised] ん？聞いたことないかも。なにそれ。"),
    ("user", f"{AI_NAME}って何歳？"),
    ("assistant", "[embarrassed] えっと……ひみつ。"),
    ("user", f"視聴者: {AI_NAME}ちゃんこんにちは！"),
    ("assistant", "[happy] いらっしゃい。ゆっくりしてってね。"),
]

# Simplified-Chinese characters the 4-bit Qwen3 substitutes into otherwise fine Japanese, mapped
# to the character it meant. Telling the model not to do this in the system prompt does not hold
# (verified: it still produced 冲縄 for 沖縄), so the replacement is applied to every reply.
# Extend this as new confusions show up in the logs.
CN_TO_JP_CHARS = {
    "冲": "沖", "这": "これ", "说": "話", "东": "東", "关": "関", "对": "対",
    "开": "開", "时": "時", "会": "会", "学": "学", "实": "実", "见": "見",
    "买": "買", "话": "話", "问": "問", "样": "様", "别": "別", "画": "画",
}

# VISION SECTION: a second textgen instance holding a small vision-language model.
# Kept separate from the chat model so looking at the screen never unloads the one that talks.
# Live2D. Off while the character's tone is still being worked out over text chat -- VTube Studio
# takes ~1.8GB of VRAM away from the LLM on a 6GB card, and the avatar adds nothing while the
# work is "does this line sound like her".
VTUBE_STUDIO_ENABLED = False

VISION_ENABLED = True
VISION_ENDPOINT = "http://127.0.0.1:5002"
VISION_MAX_TOKENS = 120
VISION_TIMEOUT = 30
# How long a screen description stays in the prompt. Past this the screen has probably moved on
# and commenting on it would look like the AI lost track of the conversation.
VISION_CONTEXT_TTL = 90
# Asks for a short spoken-style summary rather than an exhaustive description: the result is fed
# back to the chat model as context, and a wall of text there crowds out the conversation.
VISION_PROMPT = '''この画面に何が映っているか、日本語で2文以内で簡潔に説明してください。\
ゲーム画面ならゲーム名や今の状況、それ以外なら何のアプリ・何の作業かを述べること。\
推測が難しい部分は書かなくてよい。'''

# List of banned tokens to be passed to the textgen web ui api
BANNED_TOKENS = ""

# List of stopping strings. Necessary for Llama 3
# NOTE: no "\n" here. Qwen3 occasionally opens its reply with a leading newline, which made
# a "\n" stop string truncate every response to empty. Length is instead enforced by the
# SYSTEM_PROMPT's "1〜2文程度" instruction + max_tokens.
STOP_STRINGS = ["<|im_end|>"]  # <|im_end|> is Qwen's ChatML turn-end token

# MEMORY SECTION: Constants relevant to forming new memories

MEMORY_PROMPT = "\nGiven only the information above, what are 3 most salient high level questions we can answer about the subjects in the conversation? Separate each question and answer pair with \"{qa}\", and only output the question and answer, no explanations."

# How many messages in the history to include for querying the database.
MEMORY_QUERY_MESSAGE_COUNT = 5

# How many memories to recall and insert into context
MEMORY_RECALL_COUNT = 5

# VTUBE STUDIO SECTION: Configure & tune model & prop positions here.
# The defaults are for the Hiyori model on a full 16 by 9 aspect ratio screen

# Lip sync: multiplies the RMS loudness of the TTS audio to get a 0-1 mouth-open value.
# VOICEVOX output sits well below full scale, so raw RMS barely moves the mouth. Raise this if
# the model mumbles, lower it if the mouth stays wide open the whole time.
MOUTH_GAIN = 6.0

# Emotion tag -> the VTube Studio hotkey to fire for it. Hotkey names come from the model, so
# this mapping is what ties the AI's mood to a face. The defaults below match the hotkeys that
# ship with VTube Studio's built-in "akari" model; swap the values for your own model's hotkey
# names (VTube Studio > Settings > Hotkeys shows them).
#
# Several emotions deliberately map to nothing: akari only has four expressions, and firing a
# wrong-but-close face reads worse than keeping the current one.
EMOTION_HOTKEYS = {
    "happy": "Heart Eyes",
    "excited": "Heart Eyes",
    "sad": "Eyes Cry",
    "angry": "Angry Sign",
    "surprised": "Shock Sign",
    "embarrassed": None,
    "sleepy": None,
    "neutral": "Remove Expressions",
}

VTUBE_MODEL_POSITIONS = {
    "chat": {
        "x": 0.4,
        "y": -1.4,
        "size": -35,
        "rotation": 0,
    },
    "screen": {
        "x": 0.65,
        "y": -1.6,
        "size": -45,
        "rotation": 0,
    },
    "react": {
        "x": 0.7,
        "y": -1.7,
        "size": -48,
        "rotation": 0,
    },
}

VTUBE_MIC_POSITION = {
    "x": 0.52,
    "y": -0.52,
    "size": 0.22,
    "rotation": 0,
}
