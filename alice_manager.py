import chromadb
import json
import datetime
import os

# ============================================================
# アリス - メモリ管理ツール
# ============================================================
MEMORY_PATH = "./alice_memory"
EXPORT_PATH = "./alice_memory_export"
RULES_FILE = "./alice_rules.json"

chroma_client = chromadb.PersistentClient(path=MEMORY_PATH)
memory_collection = chroma_client.get_or_create_collection(
    name="conversations",
    metadata={"hnsw:space": "cosine"}
)

os.makedirs(EXPORT_PATH, exist_ok=True)

# ============================================================
# ルール管理
# ============================================================
def load_rules() -> dict:
    if not os.path.exists(RULES_FILE):
        default = {
            "禁止事項": [],
            "必ずやること": [],
            "アリスの性格メモ": [],
            "最終更新": ""
        }
        save_rules(default)
        return default
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_rules(rules: dict):
    rules["最終更新"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

def show_rules():
    rules = load_rules()
    print("\n" + "="*50)
    print("  アリスのルール一覧")
    print("="*50)
    for category, items in rules.items():
        if category == "最終更新":
            print(f"\n最終更新: {items}")
            continue
        print(f"\n【{category}】")
        if not items:
            print("  (なし)")
        for i, item in enumerate(items):
            print(f"  {i+1}. {item}")
    print()

def add_rule():
    rules = load_rules()
    print("\nカテゴリを選んでください:")
    categories = [k for k in rules.keys() if k != "最終更新"]
    for i, cat in enumerate(categories):
        print(f"  {i+1}. {cat}")
    
    try:
        choice = int(input("番号: ")) - 1
        category = categories[choice]
    except:
        print("無効な入力です")
        return
    
    new_rule = input(f"追加するルール: ").strip()
    if new_rule:
        rules[category].append(new_rule)
        save_rules(rules)
        print(f"✓ 追加しました: 【{category}】{new_rule}")

def delete_rule():
    rules = load_rules()
    print("\nカテゴリを選んでください:")
    categories = [k for k in rules.keys() if k != "最終更新"]
    for i, cat in enumerate(categories):
        print(f"  {i+1}. {cat}")
    
    try:
        choice = int(input("番号: ")) - 1
        category = categories[choice]
    except:
        print("無効な入力です")
        return
    
    items = rules[category]
    if not items:
        print("このカテゴリにルールはありません")
        return
    
    for i, item in enumerate(items):
        print(f"  {i+1}. {item}")
    
    try:
        del_choice = int(input("削除する番号: ")) - 1
        deleted = items.pop(del_choice)
        save_rules(rules)
        print(f"✓ 削除しました: {deleted}")
    except:
        print("無効な入力です")

# ============================================================
# メモリ閲覧・編集
# ============================================================
def show_all_memories(limit: int = 20):
    count = memory_collection.count()
    print(f"\n記憶の総数: {count} 件")
    
    if count == 0:
        print("まだ記憶がありません")
        return []
    
    actual_limit = min(limit, count)
    results = memory_collection.get(
        limit=actual_limit,
        include=["documents", "metadatas"]
    )
    
    memories = []
    for i in range(len(results["ids"])):
        mem = {
            "id": results["ids"][i],
            "index": i + 1,
            "timestamp": results["metadatas"][i].get("timestamp", "不明")[:16].replace("T", " "),
            "user": results["metadatas"][i].get("user", ""),
            "alice": results["metadatas"][i].get("alice", "")
        }
        memories.append(mem)
    
    # 時系列でソート
    memories.sort(key=lambda x: x["timestamp"])
    
    print("="*50)
    for mem in memories:
        print(f"\n[No.{mem['index']}] {mem['timestamp']}")
        print(f"  マスター: {mem['user']}")
        print(f"  アリス  : {mem['alice'][:80]}{'...' if len(mem['alice']) > 80 else ''}")
    print()
    return memories

def search_memories():
    query = input("検索キーワード: ").strip()
    if not query:
        return
    
    count = memory_collection.count()
    if count == 0:
        print("記憶がありません")
        return
    
    n = min(5, count)
    results = memory_collection.query(
        query_texts=[query],
        n_results=n
    )
    
    print(f"\n「{query}」の検索結果:")
    print("="*50)
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        ts = meta.get("timestamp", "不明")[:16].replace("T", " ")
        print(f"\n[{i+1}] {ts}")
        print(f"  マスター: {meta.get('user', '')}")
        print(f"  アリス  : {meta.get('alice', '')[:80]}")
    print()

def edit_memory():
    memories = show_all_memories()
    if not memories:
        return
    
    try:
        choice = int(input("編集する記憶の番号: ")) - 1
        if choice < 0 or choice >= len(memories):
            print("無効な番号です")
            return
    except:
        print("無効な入力です")
        return
    
    mem = memories[choice]
    print(f"\n現在の内容:")
    print(f"  マスター: {mem['user']}")
    print(f"  アリス  : {mem['alice']}")
    
    print("\n何を編集しますか?")
    print("  1. マスターの発言")
    print("  2. アリスの返答")
    print("  3. この記憶を削除")
    
    try:
        edit_choice = int(input("番号: "))
    except:
        print("無効な入力です")
        return
    
    if edit_choice == 3:
        memory_collection.delete(ids=[mem["id"]])
        print("✓ 記憶を削除しました")
        return
    
    if edit_choice == 1:
        new_text = input("新しい内容: ").strip()
        field = "user"
        other_field = "alice"
        other_text = mem["alice"]
    elif edit_choice == 2:
        new_text = input("新しい内容: ").strip()
        field = "alice"
        other_field = "user"
        other_text = mem["user"]
    else:
        print("無効な入力です")
        return
    
    if not new_text:
        return
    
    # 更新（削除して再追加）
    new_user = new_text if field == "user" else other_text
    new_alice = new_text if field == "alice" else other_text
    
    memory_collection.delete(ids=[mem["id"]])
    memory_collection.add(
        documents=[f"ユーザー: {new_user}\nアリス: {new_alice}"],
        metadatas=[{
            "timestamp": mem["timestamp"].replace(" ", "T"),
            "user": new_user,
            "alice": new_alice
        }],
        ids=[mem["id"]]
    )
    print("✓ 記憶を更新しました")

def delete_all_memories():
    count = memory_collection.count()
    if count == 0:
        print("記憶がありません")
        return
    
    confirm = input(f"⚠️  全{count}件の記憶を削除します。本当によいですか？ (yes/no): ")
    if confirm.lower() == "yes":
        all_ids = memory_collection.get()["ids"]
        memory_collection.delete(ids=all_ids)
        print("✓ 全記憶を削除しました")

# ============================================================
# エクスポート（ファイルに保存）
# ============================================================
def export_memories():
    count = memory_collection.count()
    if count == 0:
        print("記憶がありません")
        return
    
    results = memory_collection.get(include=["documents", "metadatas"])
    
    memories = []
    for i in range(len(results["ids"])):
        memories.append({
            "id": results["ids"][i],
            "timestamp": results["metadatas"][i].get("timestamp", ""),
            "user": results["metadatas"][i].get("user", ""),
            "alice": results["metadatas"][i].get("alice", "")
        })
    
    memories.sort(key=lambda x: x["timestamp"])
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON形式で保存
    json_path = os.path.join(EXPORT_PATH, f"alice_memory_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)
    
    # 読みやすいテキスト形式でも保存
    txt_path = os.path.join(EXPORT_PATH, f"alice_memory_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"アリス 記憶ログ - {timestamp}\n")
        f.write("="*50 + "\n\n")
        for mem in memories:
            ts = mem["timestamp"][:16].replace("T", " ")
            f.write(f"[{ts}]\n")
            f.write(f"マスター: {mem['user']}\n")
            f.write(f"アリス  : {mem['alice']}\n")
            f.write("-"*30 + "\n")
    
    print(f"✓ JSON: {json_path}")
    print(f"✓ テキスト: {txt_path}")
    print(f"  ({count}件をエクスポートしました)")

def import_memories():
    print(f"\nインポートするJSONファイルのパスを入力してください")
    print(f"(例: {EXPORT_PATH}\\alice_memory_20240101_120000.json)")
    path = input("パス: ").strip().strip('"')
    
    if not os.path.exists(path):
        print("ファイルが見つかりません")
        return
    
    with open(path, "r", encoding="utf-8") as f:
        memories = json.load(f)
    
    added = 0
    for mem in memories:
        try:
            memory_collection.add(
                documents=[f"ユーザー: {mem['user']}\nアリス: {mem['alice']}"],
                metadatas=[{
                    "timestamp": mem["timestamp"],
                    "user": mem["user"],
                    "alice": mem["alice"]
                }],
                ids=[mem["id"]]
            )
            added += 1
        except:
            pass  # 既存IDはスキップ
    
    print(f"✓ {added}件をインポートしました")

# ============================================================
# モデル情報表示
# ============================================================
def show_model_info():
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        data = r.json()
        print("\n【インストール済みモデル】")
        for model in data.get("models", []):
            size_gb = model.get("size", 0) / 1e9
            print(f"  - {model['name']} ({size_gb:.1f} GB)")
        
        print("\n【推奨: 日本語モデルに変更する場合】")
        print("  PowerShellで以下を実行:")
        print("  ollama pull 7shi/tanuki-dpo-v1.0-q4_k_m")
        print("\n  alice.py の MODEL = の行を以下に変更:")
        print('  MODEL = "7shi/tanuki-dpo-v1.0-q4_k_m"')
    except:
        print("Ollamaに接続できません。ollama serveを起動してください。")

# ============================================================
# メインメニュー
# ============================================================
def main():
    while True:
        print("\n" + "="*50)
        print("  アリス 管理ツール")
        print("="*50)
        count = memory_collection.count()
        print(f"  現在の記憶数: {count} 件")
        print()
        print("  【記憶の管理】")
        print("  1. 記憶を全件表示")
        print("  2. 記憶を検索")
        print("  3. 記憶を編集・削除")
        print("  4. 全記憶を削除")
        print("  5. 記憶をファイルにエクスポート")
        print("  6. ファイルから記憶をインポート")
        print()
        print("  【ルール管理】")
        print("  7. ルール一覧を表示")
        print("  8. ルールを追加")
        print("  9. ルールを削除")
        print()
        print("  【モデル情報】")
        print("  10. インストール済みモデルを表示")
        print()
        print("  0. 終了")
        print()
        
        choice = input("番号を選んでください: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            show_all_memories(50)
        elif choice == "2":
            search_memories()
        elif choice == "3":
            edit_memory()
        elif choice == "4":
            delete_all_memories()
        elif choice == "5":
            export_memories()
        elif choice == "6":
            import_memories()
        elif choice == "7":
            show_rules()
        elif choice == "8":
            add_rule()
        elif choice == "9":
            delete_rule()
        elif choice == "10":
            show_model_info()
        else:
            print("無効な入力です")

if __name__ == "__main__":
    main()