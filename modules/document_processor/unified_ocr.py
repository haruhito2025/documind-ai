import os
import torch
import easyocr
import fitz
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from dataclasses import dataclass
import re

from modules.document_processor.base_ocr import BaseOCREngine, OCRResult
from config.engine_config import EngineConfig

logger = logging.getLogger(__name__)

try:
    import paddleocr
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR not available")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available")

@dataclass
class PageResult:
    page_num: int
    text: str
    confidence: float
    engine: str
    char_count: int

class EasyOCREngine(BaseOCREngine):
    """EasyOCRエンジン実装"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reader = easyocr.Reader(
            config.get("languages", ["ja", "en"]), 
            gpu=config.get("gpu", torch.cuda.is_available())
        )
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """PDFからテキストを抽出"""
        results = {"text_by_page": {}, "success": False, "total_chars": 0, "engine": "easyocr"}
        
        try:
            direct_text = self._extract_direct_text(pdf_path)
            if direct_text["total_chars"] > 100:
                return direct_text
            
            images = convert_from_path(pdf_path, dpi=200)
            
            for page_num, image in enumerate(images, 1):
                text = self.extract_text_from_image_pil(image)
                if text and len(text.strip()) > 10:
                    cleaned_text = self._clean_text(text)
                    results["text_by_page"][page_num] = cleaned_text
                    results["total_chars"] += len(cleaned_text)
            
            results["success"] = True
            return results
            
        except Exception as e:
            logger.error(f"EasyOCR処理エラー: {e}")
            results["error"] = str(e)
            return results
    
    def extract_text_from_image(self, image_path: Path) -> str:
        """画像からテキストを抽出"""
        try:
            result = self.reader.readtext(str(image_path))
            text_parts = []
            for (bbox, text, confidence) in result:
                if confidence > self.confidence_threshold:
                    text_parts.append(text)
            return " ".join(text_parts)
        except Exception as e:
            logger.error(f"EasyOCR画像処理エラー: {e}")
            return ""
    
    def extract_text_from_image_pil(self, image: Image.Image) -> str:
        """PIL画像からテキストを抽出"""
        try:
            image_array = np.array(image)
            result = self.reader.readtext(image_array)
            text_parts = []
            for (bbox, text, confidence) in result:
                if confidence > self.confidence_threshold:
                    text_parts.append(text)
            return " ".join(text_parts)
        except Exception as e:
            logger.error(f"EasyOCR PIL画像処理エラー: {e}")
            return ""
    
    def _extract_direct_text(self, pdf_path: Path) -> Dict[str, Any]:
        """PDF直接テキスト抽出"""
        results = {"text_by_page": {}, "success": False, "total_chars": 0, "engine": "direct"}
        
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip() and len(text.strip()) > 10:
                    cleaned_text = self._clean_text(text)
                    if len(cleaned_text) > 15:
                        results["text_by_page"][page_num + 1] = cleaned_text
                        results["total_chars"] += len(cleaned_text)
            
            results["success"] = True
            doc.close()
            return results
            
        except Exception as e:
            logger.error(f"直接テキスト抽出エラー: {e}")
            return results
    
    def _clean_text(self, text: str) -> str:
        """テキストクリーニング"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace('〇', '○').replace('0', '〇')
        text = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF。、！？]', '', text)
        return text
    
    def get_engine_name(self) -> str:
        return "easyocr"
    
    def get_confidence_threshold(self) -> float:
        return self.confidence_threshold

class PaddleOCREngine(BaseOCREngine):
    """PaddleOCRエンジン実装"""
    
    def __init__(self, config: Dict[str, Any]):
        if not PADDLE_AVAILABLE:
            raise ImportError("PaddleOCR is not available")
        
        self.config = config
        self.ocr = paddleocr.PaddleOCR(
            use_angle_cls=config.get("use_angle_cls", True),
            lang=config.get("lang", "japan")
        )
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """PDFからテキストを抽出"""
        results = {"text_by_page": {}, "success": False, "total_chars": 0, "engine": "paddle"}
        
        try:
            images = convert_from_path(pdf_path, dpi=200)
            
            for page_num, image in enumerate(images, 1):
                text = self.extract_text_from_image_pil(image)
                if text and len(text.strip()) > 10:
                    cleaned_text = self._clean_text(text)
                    results["text_by_page"][page_num] = cleaned_text
                    results["total_chars"] += len(cleaned_text)
            
            results["success"] = True
            return results
            
        except Exception as e:
            logger.error(f"PaddleOCR処理エラー: {e}")
            results["error"] = str(e)
            return results
    
    def extract_text_from_image(self, image_path: Path) -> str:
        """画像からテキストを抽出"""
        try:
            result = self.ocr.ocr(str(image_path), cls=True)
            text_parts = []
            for line in result:
                for word_info in line:
                    text, confidence = word_info[1]
                    if confidence > self.confidence_threshold:
                        text_parts.append(text)
            return " ".join(text_parts)
        except Exception as e:
            logger.error(f"PaddleOCR画像処理エラー: {e}")
            return ""
    
    def extract_text_from_image_pil(self, image: Image.Image) -> str:
        """PIL画像からテキストを抽出"""
        try:
            image_array = np.array(image)
            result = self.ocr.ocr(image_array, cls=True)
            text_parts = []
            for line in result:
                for word_info in line:
                    text, confidence = word_info[1]
                    if confidence > self.confidence_threshold:
                        text_parts.append(text)
            return " ".join(text_parts)
        except Exception as e:
            logger.error(f"PaddleOCR PIL画像処理エラー: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """テキストクリーニング"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace('〇', '○').replace('0', '〇')
        return text
    
    def get_engine_name(self) -> str:
        return "paddle"
    
    def get_confidence_threshold(self) -> float:
        return self.confidence_threshold

class TesseractOCREngine(BaseOCREngine):
    """Tesseract OCRエンジン実装"""
    
    def __init__(self, config: Dict[str, Any]):
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract is not available")
        
        self.config = config
        self.lang = config.get("lang", "jpn+eng")
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        
        # Tesseractの設定
        self.tesseract_config = {
            "lang": self.lang,
            "config": "--psm 3"  # 自動ページセグメンテーションモード
        }
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """PDFからテキストを抽出"""
        results = {"text_by_page": {}, "success": False, "total_chars": 0, "engine": "tesseract"}
        
        try:
            images = convert_from_path(pdf_path, dpi=300)  # より高解像度で変換
            
            for page_num, image in enumerate(images, 1):
                text = self.extract_text_from_image_pil(image)
                if text and len(text.strip()) > 10:
                    cleaned_text = self._clean_text(text)
                    results["text_by_page"][page_num] = cleaned_text
                    results["total_chars"] += len(cleaned_text)
            
            results["success"] = True
            return results
            
        except Exception as e:
            logger.error(f"Tesseract処理エラー: {e}")
            results["error"] = str(e)
            return results
    
    def extract_text_from_image(self, image_path: Path) -> str:
        """画像からテキストを抽出"""
        try:
            text = pytesseract.image_to_string(
                Image.open(image_path),
                **self.tesseract_config
            )
            return text
        except Exception as e:
            logger.error(f"Tesseract画像処理エラー: {e}")
            return ""
    
    def extract_text_from_image_pil(self, image: Image.Image) -> str:
        """PIL画像からテキストを抽出"""
        try:
            text = pytesseract.image_to_string(
                image,
                **self.tesseract_config
            )
            return text
        except Exception as e:
            logger.error(f"Tesseract PIL画像処理エラー: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """テキストクリーニング"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace('〇', '○').replace('0', '〇')
        text = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF。、！？]', '', text)
        return text
    
    def get_engine_name(self) -> str:
        return "tesseract"
    
    def get_confidence_threshold(self) -> float:
        return self.confidence_threshold

class MultiEngineOCR(BaseOCREngine):
    """複数エンジンを組み合わせたOCR"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engines = []
        self.confidence_threshold = config.get("confidence_threshold", 0.4)
        
        engine_names = config.get("engines", ["easyocr", "paddle"])
        for engine_name in engine_names:
            try:
                if engine_name == "easyocr":
                    engine_config = EngineConfig.get_ocr_engine_config("easyocr")
                    self.engines.append(EasyOCREngine(engine_config))
                elif engine_name == "paddle" and PADDLE_AVAILABLE:
                    engine_config = EngineConfig.get_ocr_engine_config("paddle")
                    self.engines.append(PaddleOCREngine(engine_config))
            except Exception as e:
                logger.warning(f"エンジン {engine_name} の初期化に失敗: {e}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """複数エンジンでPDFからテキストを抽出"""
        best_result = {"text_by_page": {}, "success": False, "total_chars": 0, "engine": "multi"}
        
        for engine in self.engines:
            try:
                result = engine.extract_text_from_pdf(pdf_path)
                if result["success"] and result["total_chars"] > best_result["total_chars"]:
                    best_result = result
                    best_result["engine"] = f"multi-{engine.get_engine_name()}"
            except Exception as e:
                logger.warning(f"エンジン {engine.get_engine_name()} でエラー: {e}")
        
        return best_result
    
    def extract_text_from_image(self, image_path: Path) -> str:
        """複数エンジンで画像からテキストを抽出"""
        best_text = ""
        max_length = 0
        
        for engine in self.engines:
            try:
                text = engine.extract_text_from_image(image_path)
                if len(text) > max_length:
                    best_text = text
                    max_length = len(text)
            except Exception as e:
                logger.warning(f"エンジン {engine.get_engine_name()} でエラー: {e}")
        
        return best_text
    
    def get_engine_name(self) -> str:
        return "multi"
    
    def get_confidence_threshold(self) -> float:
        return self.confidence_threshold

class OCRFactory:
    """OCRエンジンファクトリー"""
    
    @staticmethod
    def create_engine(engine_name: str) -> BaseOCREngine:
        """指定されたエンジン名のOCRエンジンを作成"""
        config = EngineConfig.get_ocr_config()
        
        if engine_name == "easyocr":
            return EasyOCREngine(config)
        elif engine_name == "paddle" and PADDLE_AVAILABLE:
            return PaddleOCREngine(config)
        elif engine_name == "tesseract" and TESSERACT_AVAILABLE:
            return TesseractOCREngine(config)
        elif engine_name == "multi":
            return MultiEngineOCR(config)
        else:
            raise ValueError(f"Unknown OCR engine: {engine_name}")
    
    @staticmethod
    def get_available_engines() -> List[str]:
        """利用可能なOCRエンジンのリストを取得"""
        engines = ["easyocr", "multi"]
        if PADDLE_AVAILABLE:
            engines.append("paddle")
        if TESSERACT_AVAILABLE:
            engines.append("tesseract")
        return engines
