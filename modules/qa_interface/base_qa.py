from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseQAInterface(ABC):
    """質問応答インターフェースの基底クラス"""
    
    @abstractmethod
    def ask_question(self, question: str) -> Dict[str, Any]:
        """質問に対する回答を生成"""
        pass
    
    @abstractmethod
    def set_temperature(self, temperature: float) -> None:
        """回答の創造性を設定"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """使用中のモデル情報を取得"""
        pass

class QAResult:
    """質問応答結果を格納するクラス"""
    
    def __init__(self, question: str, answer: str, sources: List[Dict], confidence: float = 0.0):
        self.question = question
        self.answer = answer
        self.sources = sources
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence
        }
