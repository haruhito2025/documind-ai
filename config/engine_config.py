from typing import Dict, Any, List

class EngineConfig:
    """エンジン設定管理（アップグレード可能）"""
    
    @staticmethod
    def get_available_ocr_engines() -> List[str]:
        """利用可能なOCRエンジン一覧"""
        return ["easyocr", "paddle", "multi"]
    
    @staticmethod
    def get_ocr_engine_config(engine_name: str) -> Dict[str, Any]:
        """OCRエンジン別設定"""
        configs = {
            "easyocr": {
                "languages": ["ja", "en"],
                "gpu": True,
                "confidence_threshold": 0.5
            },
            "paddle": {
                "use_angle_cls": True,
                "lang": "japan",
                "confidence_threshold": 0.6
            },
            "multi": {
                "engines": ["easyocr", "paddle"],
                "selection_strategy": "max_chars",
                "confidence_threshold": 0.4
            }
        }
        return configs.get(engine_name, configs["easyocr"])
    
    @staticmethod
    def get_available_vector_stores() -> List[str]:
        """利用可能なベクトルストア一覧"""
        return ["chroma", "faiss"]
    
    @staticmethod
    def get_vector_store_config(store_name: str) -> Dict[str, Any]:
        """ベクトルストア別設定"""
        configs = {
            "chroma": {
                "collection_name": "documind_collection",
                "distance_metric": "cosine"
            },
            "faiss": {
                "index_type": "IndexFlatIP",
                "dimension": 1536
            }
        }
        return configs.get(store_name, configs["chroma"])
    
    @staticmethod
    def get_available_qa_models() -> List[str]:
        """利用可能なQAモデル一覧"""
        return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    
    @staticmethod
    def get_qa_model_config(model_name: str) -> Dict[str, Any]:
        """QAモデル別設定"""
        configs = {
            "gpt-3.5-turbo": {
                "max_tokens": 1000,
                "temperature": 0.1,
                "cost_per_token": 0.0015
            },
            "gpt-4": {
                "max_tokens": 2000,
                "temperature": 0.1,
                "cost_per_token": 0.03
            },
            "gpt-4-turbo": {
                "max_tokens": 4000,
                "temperature": 0.1,
                "cost_per_token": 0.01
            }
        }
        return configs.get(model_name, configs["gpt-3.5-turbo"])

    @staticmethod
    def get_ocr_config() -> Dict[str, Any]:
        """共通のOCR設定を取得"""
        return {
            "easyocr": {
                "languages": ["ja", "en"],
                "gpu": True,
                "confidence_threshold": 0.5
            },
            "paddle": {
                "use_angle_cls": True,
                "lang": "japan",
                "confidence_threshold": 0.6
            },
            "tesseract": {
                "lang": "jpn+eng",
                "confidence_threshold": 0.6
            },
            "multi": {
                "engines": ["easyocr", "paddle", "tesseract"],
                "selection_strategy": "max_chars",
                "confidence_threshold": 0.4
            }
        }
