import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from notion_client import Client as NotionClient

from config.settings import settings

logger = logging.getLogger(__name__)

class NotionIntegration:
    """Notion統合クライアント"""
    
    def __init__(self):
        if not settings.notion_token:
            logger.warning("NOTION_TOKEN が設定されていません")
            self.client = None
            return
        
        if not settings.notion_database_id:
            logger.warning("NOTION_DATABASE_ID が設定されていません")
            self.client = None
            return
        
        try:
            self.client = NotionClient(auth=settings.notion_token)
            self.database_id = settings.notion_database_id
            logger.info("Notion統合を初期化しました")
        except Exception as e:
            logger.error(f"Notion初期化エラー: {e}")
            self.client = None
    
    def save_qa_feedback(self, question: str, answer: str, feedback: str = "pending", 
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """質問応答のフィードバックをNotionに保存"""
        if not self.client:
            logger.warning("Notionクライアントが利用できません")
            return False
        
        try:
            properties = {
                "質問": {
                    "title": [
                        {
                            "text": {
                                "content": question[:2000]
                            }
                        }
                    ]
                },
                "回答": {
                    "rich_text": [
                        {
                            "text": {
                                "content": answer[:2000]
                            }
                        }
                    ]
                },
                "評価": {
                    "multi_select": [
                        {
                            "name": feedback
                        }
                    ]
                },
                "日時": {
                    "date": {
                        "start": datetime.now().isoformat()
                    }
                }
            }
            
            if metadata:
                if "sources" in metadata:
                    source_text = self._format_sources(metadata["sources"])
                    properties["参照元"] = {
                        "rich_text": [
                            {
                                "text": {
                                    "content": source_text[:2000]
                                }
                            }
                        ]
                    }
                
                if "confidence" in metadata:
                    properties["信頼度"] = {
                        "number": float(metadata["confidence"])
                    }
                
                if "model_info" in metadata:
                    model_text = f"Model: {metadata['model_info'].get('model_name', 'unknown')}"
                    properties["モデル情報"] = {
                        "rich_text": [
                            {
                                "text": {
                                    "content": model_text
                                }
                            }
                        ]
                    }
            
            self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties
            )
            
            logger.info(f"Notionに保存しました: {feedback}")
            return True
            
        except Exception as e:
            logger.error(f"Notion保存エラー: {e}")
            return False
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """参照元情報をフォーマット"""
        formatted_sources = []
        for i, source in enumerate(sources[:3], 1):
            source_info = f"{i}. {source.get('source', '不明')} (ページ: {source.get('page', '不明')})"
            formatted_sources.append(source_info)
        return "\n".join(formatted_sources)
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """フィードバック統計を取得"""
        if not self.client:
            return {"error": "Notionクライアントが利用できません"}
        
        try:
            response = self.client.databases.query(
                database_id=self.database_id,
                page_size=100
            )
            
            stats = {
                "total_entries": len(response["results"]),
                "good_count": 0,
                "bad_count": 0,
                "pending_count": 0
            }
            
            for page in response["results"]:
                properties = page.get("properties", {})
                evaluation = properties.get("評価", {})
                multi_select = evaluation.get("multi_select", [])
                
                for item in multi_select:
                    feedback_type = item.get("name", "")
                    if feedback_type == "good":
                        stats["good_count"] += 1
                    elif feedback_type == "bad":
                        stats["bad_count"] += 1
                    elif feedback_type == "pending":
                        stats["pending_count"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"統計取得エラー: {e}")
            return {"error": str(e)}
    
    def is_available(self) -> bool:
        """Notion統合が利用可能かチェック"""
        return self.client is not None
    
    def test_connection(self) -> Dict[str, Any]:
        """接続テスト"""
        if not self.client:
            return {"success": False, "error": "クライアントが初期化されていません"}
        
        try:
            database = self.client.databases.retrieve(database_id=self.database_id)
            return {
                "success": True,
                "database_title": database.get("title", [{}])[0].get("text", {}).get("content", "不明"),
                "database_id": self.database_id
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class FeedbackManager:
    """フィードバック管理クラス"""
    
    def __init__(self):
        self.notion = NotionIntegration()
        self.local_feedback = []
    
    def save_feedback(self, question: str, answer: str, feedback: str, 
                     sources: Optional[List[Dict]] = None, 
                     confidence: Optional[float] = None,
                     model_info: Optional[Dict] = None) -> bool:
        """フィードバックを保存"""
        metadata = {}
        if sources:
            metadata["sources"] = sources
        if confidence is not None:
            metadata["confidence"] = confidence
        if model_info:
            metadata["model_info"] = model_info
        
        notion_success = self.notion.save_qa_feedback(question, answer, feedback, metadata)
        
        local_entry = {
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
            "notion_saved": notion_success
        }
        self.local_feedback.append(local_entry)
        
        if len(self.local_feedback) > 100:
            self.local_feedback = self.local_feedback[-100:]
        
        return notion_success
    
    def get_local_feedback(self) -> List[Dict[str, Any]]:
        """ローカルフィードバックを取得"""
        return self.local_feedback.copy()
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """フィードバック要約を取得"""
        local_stats = {
            "good": sum(1 for f in self.local_feedback if f["feedback"] == "good"),
            "bad": sum(1 for f in self.local_feedback if f["feedback"] == "bad"),
            "pending": sum(1 for f in self.local_feedback if f["feedback"] == "pending")
        }
        
        notion_stats = self.notion.get_feedback_stats()
        
        return {
            "local": local_stats,
            "notion": notion_stats,
            "notion_available": self.notion.is_available()
        }
