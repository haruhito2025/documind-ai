from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

class BaseVectorStore(ABC):
    """ベクトルストアの基底クラス"""
    
    @abstractmethod
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """ドキュメントを追加"""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """類似検索を実行"""
        pass
    
    @abstractmethod
    def delete_collection(self) -> None:
        """コレクションを削除"""
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """コレクション情報を取得"""
        pass
    
    @abstractmethod
    def persist(self) -> None:
        """データを永続化"""
        pass

class DocumentChunk:
    """ドキュメントチャンクを表すクラス"""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata
        }
