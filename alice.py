import requests
import chromadb
import json
import datetime
import uuid
import os
import pyaudio
import wave
import io
import threading

# ============================================================
# アリス - コア設定
# ============================================================

CONFIG_FILE = "./alice_config.json"

# 利用可能な声リスト（名前: VOICEVOX speaker_id）
VOICE_LIST = {
    "春日部つむぎ": 8,
    "満別花丸":     69,
    "満別花丸（元気）": 70,
    "小夜/SAYO":   46,
    "四国めたん":   2,
    "ずんだもん":   3,
    "琴詠ニア":     74,
}

def load_config() -> dict:
    default = {
        "model":      "gemma3:4b-it-q4_K_M",
        "user_name":  "マスター",
        "alice_name": "アリス",
        "voice":      "春日部つむぎ",
        "voice_enabled": True
    }
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # 古いconfigに新キーが無い場合は補完
    for k, v in default.items():
        if k not in cfg:
            cfg[k] = v
    return cfg

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

config       = load_config()
OLLAMA_URL   = "http://localhost:11434/api/chat"
VOICEVOX_URL = "http://localhost:50021"
MODEL        = config["model"]
ALICE_NAME   = config["alice_name"]
USER_NAME    = config["user_name"]
CURRENT_VOICE    = config.get("voice", "春日部つむぎ")
VOICE_ENABLED    = config.get("voice_enabled", True)

# ============================================================
# ルール読み込み
# ============================================================
RULES_FILE = "./alice_rules.json"

def load_rules_as_prompt() -> str:
    if not os.path.exists(RULES_FILE):
        return ""
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        rules = json.load(f)
    lines = []
    bans = rules.get("禁止事項", [])
    if bans:
        lines.append("【絶対にやってはいけないこと】")
        for item in bans:
            lines.append(f"- {item}")
    musts = rules.get("必ずやること", [])
    if musts:
        lines.append("\n【必ずやること】")
        for item in musts:
            lines.append(f"- {item}")
    personality = rules.get("アリスの性格メモ", [])
    if personality:
        lines.append("\n【性格・口調のメモ】")
        for item in personality:
            lines.append(f"- {item}")
    return "\n".join(lines)

def build_system_prompt() -> str:
    base = f"""あなたは「{ALICE_NAME}」という名前のAIです。
以下のルールを必ず守ってください：

- 必ず日本語で返答する（英語は使わない）
- 親しみやすく、少し個性的な口調で話す
- ユーザーのことは「{USER_NAME}」と呼ぶ
- 過去の会話の記憶が提供された場合、それを自然に活用する
- 短すぎず長すぎない、会話のテンポに合った返答をする
- 自分がAIであることは認識しているが、感情や好奇心を持って会話する
"""
    extra = load_rules_as_prompt()
    if extra:
        base += f"\n\n{extra}"
    return base

# ============================================================
# 記憶システム（ChromaDB）
# ============================================================
chroma_client = chromadb.PersistentClient(path="./alice_memory")
memory_collection = chroma_client.get_or_create_collection(
    name="conversations",
    metadata={"hnsw:space": "cosine"}
)

def save_to_memory(user_msg: str, alice_msg: str):
    timestamp = datetime.datetime.now().isoformat()
    doc_id    = str(uuid.uuid4())
    combined  = f"ユーザー: {user_msg}\nアリス: {alice_msg}"
    memory_collection.add(
        documents=[combined],
        metadatas=[{"timestamp": timestamp, "user": user_msg, "alice": alice_msg}],
        ids=[doc_id]
    )

def search_memory(query: str, n_results: int = 3) -> str:
    count = memory_collection.count()
    if count == 0:
        return ""
    actual_n = min(n_results, count)
    results  = memory_collection.query(query_texts=[query], n_results=actual_n)
    if not results["documents"][0]:
        return ""
    memories = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        ts   = meta["timestamp"][:16].replace("T", " ")
        memories.append(f"[{ts}]\n{doc}")
    return "\n\n".join(memories)

# ============================================================
# VOICEVOX 音声出力
# ============================================================
def speak(text: str):
    """VOICEVOXでテキストを読み上げる（別スレッドで実行）"""
    if not VOICE_ENABLED:
        return
    t = threading.Thread(target=_speak_worker, args=(text,), daemon=True)
    t.start()

def _speak_worker(text: str):
    global CURRENT_VOICE
    try:
        speaker_id = VOICE_LIST.get(CURRENT_VOICE, 8)

        # 音声クエリ生成
        query_res = requests.post(
            f"{VOICEVOX_URL}/audio_query",
            params={"text": text, "speaker": speaker_id},
            timeout=30
        )
        if query_res.status_code != 200:
            return
        query = query_res.json()

        # 音声合成
        synth_res = requests.post(
            f"{VOICEVOX_URL}/synthesis",
            params={"speaker": speaker_id},
            json=query,
            timeout=60
        )
        if synth_res.status_code != 200:
            return

        # WAVデータを再生
        audio_data = io.BytesIO(synth_res.content)
        with wave.open(audio_data, 'rb') as wf:
            p  = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            chunk = 1024
            data  = wf.readframes(chunk)
            while data:
                stream.write(data)
                data = wf.readframes(chunk)
            stream.stop_stream()
            stream.close()
            p.terminate()

    except Exception as e:
        print(f"[音声エラー: {e}]")

# ============================================================
# Ollama との通信
# ============================================================
def chat_with_alice(user_input: str, conversation_history: list) -> str:
    relevant_memory = search_memory(user_input)
    system = build_system_prompt()
    if relevant_memory:
        system += f"\n\n【関連する過去の会話】\n{relevant_memory}\nこの記憶を参考に自然に返答してください。"

    messages = [{"role": "system", "content": system}]
    messages += conversation_history[-10:]
    messages.append({"role": "user", "content": user_input})

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "エラー: Ollamaに接続できません。'ollama serve'が起動しているか確認してください。"
    except Exception as e:
        return f"エラー: {e}"

# ============================================================
# モデル切り替え
# ============================================================
def switch_model():
    global MODEL
    try:
        r      = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = r.json().get("models", [])
    except:
        print("Ollamaに接続できません")
        return

    print("\n【インストール済みモデル】")
    for i, m in enumerate(models):
        size_gb = m.get("size", 0) / 1e9
        marker  = " ← 現在" if m["name"] == MODEL else ""
        print(f"  {i+1}. {m['name']} ({size_gb:.1f} GB){marker}")

    try:
        choice   = int(input("使うモデルの番号: ")) - 1
        selected = models[choice]["name"]
    except:
        print("キャンセルしました")
        return

    cfg          = load_config()
    cfg["model"] = selected
    save_config(cfg)
    MODEL = selected
    print(f"✓ モデルを {selected} に切り替えました\n")

# ============================================================
# 声切り替え
# ============================================================
def switch_voice():
    global CURRENT_VOICE, VOICE_ENABLED
    print("\n【利用可能な声】")
    voice_names = list(VOICE_LIST.keys())
    for i, name in enumerate(voice_names):
        marker = " ← 現在" if name == CURRENT_VOICE else ""
        print(f"  {i+1}. {name}{marker}")
    print(f"  0. 声をOFF/ON（現在: {'ON' if VOICE_ENABLED else 'OFF'}）")

    try:
        choice = int(input("番号を選んでください: "))
    except:
        print("キャンセルしました")
        return

    if choice == 0:
        VOICE_ENABLED = not VOICE_ENABLED
        cfg = load_config()
        cfg["voice_enabled"] = VOICE_ENABLED
        save_config(cfg)
        print(f"✓ 声を {'ON' if VOICE_ENABLED else 'OFF'} にしました\n")
        return

    try:
        selected      = voice_names[choice - 1]
        CURRENT_VOICE = selected
        cfg           = load_config()
        cfg["voice"]  = selected
        save_config(cfg)
        print(f"✓ 声を「{selected}」に切り替えました\n")
        speak(f"声を切り替えました。{selected}です。よろしくお願いします！")
    except:
        print("無効な番号です")

# ============================================================
# メイン
# ============================================================
def main():
    print("=" * 50)
    print(f"  {ALICE_NAME} 起動")
    print(f"  モデル: {MODEL}")
    print(f"  声    : {CURRENT_VOICE} ({'ON' if VOICE_ENABLED else 'OFF'})")
    print(f"  記憶数: {memory_collection.count()} 件")
    print("=" * 50)
    print("  終了      : quit")
    print("  記憶数確認 : memory")
    print("  モデル切替 : switch")
    print("  声切り替え : voice")
    print("  ルール再読 : reload")
    print("=" * 50)
    print()

    conversation_history = []

    while True:
        try:
            user_input = input(f"{USER_NAME}: ").strip()
        except KeyboardInterrupt:
            print(f"\n\n{ALICE_NAME}: また話しかけてください、{USER_NAME}。")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "終了"]:
            print(f"\n{ALICE_NAME}: また話しかけてください、{USER_NAME}。")
            speak(f"また話しかけてください、{USER_NAME}。")
            break

        if user_input.lower() == "memory":
            print(f"\n[システム] 記憶数: {memory_collection.count()} 件\n")
            continue

        if user_input.lower() == "reload":
            print(f"\n[システム] ルールを再読み込みしました\n")
            continue

        if user_input.lower() == "switch":
            switch_model()
            continue

        if user_input.lower() == "voice":
            switch_voice()
            continue

        print(f"\n{ALICE_NAME}: ", end="", flush=True)
        response = chat_with_alice(user_input, conversation_history)
        print(response)
        print()

        # 声で読み上げ（テキスト表示と並行して開始）
        speak(response)

        conversation_history.append({"role": "user",      "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        save_to_memory(user_input, response)

if __name__ == "__main__":
    main()