import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RerankedDocument:
    """リランキングされたドキュメント"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    page: str
    engine: str

class Reranker:
    """リランキングエンジン"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """リランカーを初期化"""
        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"リランカーを初期化: {model_name}")
        except Exception as e:
            logger.error(f"リランカーの初期化に失敗: {e}")
            raise
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[RerankedDocument]:
        """ドキュメントをリランキング"""
        try:
            # クエリとドキュメントのペアを作成
            pairs = [(query, doc["content"]) for doc in documents]
            
            # スコアを計算
            scores = self.model.predict(pairs)
            
            # ドキュメントとスコアを組み合わせてソート
            reranked = []
            for doc, score in zip(documents, scores):
                reranked_doc = RerankedDocument(
                    content=doc["content"],
                    score=float(score),
                    metadata=doc.get("metadata", {}),
                    source=doc.get("source", "不明"),
                    page=doc.get("page", "不明"),
                    engine=doc.get("engine", "不明")
                )
                reranked.append(reranked_doc)
            
            # スコアでソート
            reranked.sort(key=lambda x: x.score, reverse=True)
            
            # top_kが指定されている場合は制限
            if top_k is not None:
                reranked = reranked[:top_k]
            
            return reranked
            
        except Exception as e:
            logger.error(f"リランキング中にエラーが発生: {e}")
            # エラーが発生した場合は元のドキュメントをそのまま返す
            return [
                RerankedDocument(
                    content=doc["content"],
                    score=0.0,
                    metadata=doc.get("metadata", {}),
                    source=doc.get("source", "不明"),
                    page=doc.get("page", "不明"),
                    engine=doc.get("engine", "不明")
                )
                for doc in documents
            ]
    
    def get_model_info(self) -> Dict[str, str]:
        """モデル情報を取得"""
        return {
            "model_name": self.model.get_config_dict().get("model_name", "unknown"),
            "max_length": str(self.model.get_config_dict().get("max_length", "unknown"))
        } 