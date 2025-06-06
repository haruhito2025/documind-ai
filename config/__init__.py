
<create_file path="/home/ubuntu/repos/documind-ai/config/settings.py">
import os
from pathlib import Path
from typing import Dict, Any
import streamlit as st

class Settings:
    """アプリケーション設定管理"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.vector_store_dir = self.data_dir / "vector_store"
        self.uploads_dir = self.data_dir / "uploads"
        
        self.data_dir.mkdir(exist_ok=True)
        self.vector_store_dir.mkdir(exist_ok=True)
        self.uploads_dir.mkdir(exist_ok=True)
    
    @property
    def openai_api_key(self) -> str:
        """OpenAI APIキー取得"""
        return os.getenv("OPENAI_API_KEY", "")
    
    @property
    def notion_token(self) -> str:
        """Notion APIトークン取得"""
        return os.getenv("NOTION_TOKEN", "")
    
    @property
    def notion_database_id(self) -> str:
        """Notion データベースID取得"""
        return os.getenv("NOTION_DATABASE_ID", "")
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """OCR設定取得"""
        return {
            "default_engine": "easyocr",
            "fallback_engines": ["paddle", "multi"],
            "confidence_threshold": 0.5,
            "dpi": 200
        }
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """ベクトルストア設定取得"""
        return {
            "store_type": "chroma",
            "embedding_model": "text-embedding-ada-002",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "persist_directory": str(self.vector_store_dir)
        }
    
    def get_qa_config(self) -> Dict[str, Any]:
        """質問応答設定取得"""
        return {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 1000,
            "top_k": 5
        }

settings = Settings()
