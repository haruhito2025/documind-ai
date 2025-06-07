import os
from dotenv import load_dotenv
from notion_client import Client as NotionClient
from datetime import datetime
from config.settings import settings
from modules.integrations.notion_client import NotionIntegration

# OpenAI APIのテスト用
try:
    import openai
except ImportError:
    openai = None

# .envファイルの読み込み
load_dotenv()

# 環境変数の読み込み
notion_token = os.getenv("NOTION_API_KEY")
notion_db_id = os.getenv("NOTION_DATABASE_ID")

def check_api_key():
    if not notion_token:
        print("❌ NOTION_API_KEYが設定されていません")
        return False
    print("✅ NOTION_API_KEYが設定されています")
    return True

def check_database_id():
    if not notion_db_id:
        print("❌ NOTION_DATABASE_IDが設定されていません")
        return False
    print("✅ NOTION_DATABASE_IDが設定されています")
    return True

def check_integration_permissions(notion):
    try:
        # インテグレーションの情報を取得
        print("\nインテグレーションの情報を取得中...")
        integration = notion.users.me()
        print(f"✅ インテグレーション名: {integration.get('name', '不明')}")
        
        # データベースの情報を取得
        print("\nデータベースの情報を取得中...")
        database = notion.databases.retrieve(database_id=notion_db_id)
        print(f"✅ データベース名: {database.get('title', [{'plain_text': '不明'}])[0]['plain_text']}")
        
        # データベースのプロパティを確認
        print("\nデータベースのプロパティを確認中...")
        required_properties = {
            "質問": "title",
            "回答": "rich_text",
            "評価": "multi_select",
            "日時": "date"
        }
        
        properties = database.get("properties", {})
        missing_properties = []
        
        for prop_name, prop_type in required_properties.items():
            if prop_name not in properties:
                missing_properties.append(f"{prop_name}（{prop_type}）")
            else:
                actual_type = properties[prop_name].get("type")
                if actual_type != prop_type:
                    print(f"⚠️ {prop_name}の型が異なります（期待: {prop_type}, 実際: {actual_type}）")
                else:
                    print(f"✅ {prop_name}（{prop_type}）")
        
        if missing_properties:
            print("\n❌ 以下のプロパティが不足しています：")
            for prop in missing_properties:
                print(f"  - {prop}")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {str(e)}")
        if "unauthorized" in str(e).lower():
            print("\n⚠️ インテグレーションにデータベースへのアクセス権限がありません。")
            print("以下の手順で権限を付与してください：")
            print("1. Notionでデータベースを開く")
            print("2. 右上の「...」をクリック")
            print("3. 「コネクションを追加」を選択")
            print("4. インテグレーションを選択して接続")
        return False

def test_notion_connection():
    try:
        # Notionクライアントの初期化
        notion = NotionClient(auth=notion_token)
        
        # インテグレーションの権限確認
        if not check_integration_permissions(notion):
            return False
        
        # テストページの作成
        print("\nテストページを作成中...")
        new_page = notion.pages.create(
            parent={"database_id": notion_db_id},
            properties={
                "質問": {"title": [{"text": {"content": "テスト接続"}}]},
                "回答": {"rich_text": [{"text": {"content": "これは接続テストです"}}]},
                "評価": {"multi_select": [{"name": "pending"}]},
                "日時": {"date": {"start": datetime.now().isoformat()}},
            },
        )
        print("✅ テストページの作成に成功しました！")
        print(f"作成したページのID: {new_page['id']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {str(e)}")
        return False

def test_api_connections():
    """
    OpenAI APIとNotion API/DBの接続確認テスト
    """
    openai_ok = False
    notion_ok = False

    # OpenAI API接続テスト
    print("\nOpenAI API接続テスト...")
    api_key = settings.openai_api_key
    if not api_key:
        print("❌ OPENAI_API_KEYが設定されていません")
    elif openai is None:
        print("❌ openaiパッケージがインストールされていません")
    else:
        try:
            openai.api_key = api_key
            # モデル一覧取得で接続確認
            models = openai.models.list()
            print(f"✅ OpenAI API接続成功！利用可能モデル数: {len(models.data)}")
            openai_ok = True
        except Exception as e:
            print(f"❌ OpenAI API接続エラー: {str(e)}")

    # Notion API/DB接続テスト
    print("\nNotion API/DB接続テスト...")
    notion = NotionIntegration()
    result = notion.test_connection()
    if result.get("success"):
        print(f"✅ Notion接続成功！DBタイトル: {result.get('database_title')}")
        notion_ok = True
    else:
        print(f"❌ Notion接続エラー: {result.get('error')}")

    return openai_ok and notion_ok

if __name__ == "__main__":
    print("API接続テストを開始します...")
    success = test_api_connections()
    if success:
        print("\n✅ すべてのAPI接続テストに成功しました！")
    else:
        print("\n❌ API接続に失敗しました。環境変数や認証情報を確認してください。") 