from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path

class BaseOCREngine(ABC):
    """OCRエンジンの基底クラス"""
    
    @abstractmethod
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """PDFからテキストを抽出"""
        pass
    
    @abstractmethod
    def extract_text_from_image(self, image_path: Path) -> str:
        """画像からテキストを抽出"""
        pass
    
    @abstractmethod
    def get_engine_name(self) -> str:
        """エンジン名を取得"""
        pass
    
    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """信頼度閾値を取得"""
        pass

class OCRResult:
    """OCR結果を格納するクラス"""
    
    def __init__(self, text: str, confidence: float, engine: str, metadata: Optional[Dict] = None):
        self.text = text
        self.confidence = confidence
        self.engine = engine
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "engine": self.engine,
            "metadata": self.metadata
        }
