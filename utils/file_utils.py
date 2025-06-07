import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import hashlib

logger = logging.getLogger(__name__)

class FileManager:
    """ファイル管理ユーティリティ"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.uploads_dir = self.base_dir / "uploads"
        self.processed_dir = self.base_dir / "processed"
        
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> Path:
        """アップロードされたファイルを保存"""
        try:
            safe_filename = self._sanitize_filename(filename)
            file_path = self.uploads_dir / safe_filename
            
            if file_path.exists():
                file_hash = hashlib.md5(file_content).hexdigest()[:8]
                name, ext = safe_filename.rsplit('.', 1) if '.' in safe_filename else (safe_filename, '')
                safe_filename = f"{name}_{file_hash}.{ext}" if ext else f"{name}_{file_hash}"
                file_path = self.uploads_dir / safe_filename
            
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"ファイル保存完了: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"ファイル保存エラー: {e}")
            raise
    
    def move_to_processed(self, file_path: Path) -> Path:
        """ファイルを処理済みディレクトリに移動"""
        try:
            processed_path = self.processed_dir / file_path.name
            shutil.move(str(file_path), str(processed_path))
            logger.info(f"ファイル移動完了: {processed_path}")
            return processed_path
        except Exception as e:
            logger.error(f"ファイル移動エラー: {e}")
            raise
    
    def get_uploaded_files(self) -> List[Dict[str, Any]]:
        """アップロード済みファイル一覧を取得"""
        files = []
        try:
            for file_path in self.uploads_dir.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "extension": file_path.suffix.lower()
                    })
            return sorted(files, key=lambda x: x["modified"], reverse=True)
        except Exception as e:
            logger.error(f"ファイル一覧取得エラー: {e}")
            return []
    
    def get_processed_files(self) -> List[Dict[str, Any]]:
        """処理済みファイル一覧を取得"""
        files = []
        try:
            for file_path in self.processed_dir.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "extension": file_path.suffix.lower()
                    })
            return sorted(files, key=lambda x: x["modified"], reverse=True)
        except Exception as e:
            logger.error(f"処理済みファイル一覧取得エラー: {e}")
            return []
    
    def delete_file(self, file_path: Path) -> bool:
        """ファイルを削除"""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"ファイル削除完了: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"ファイル削除エラー: {e}")
            return False
    
    def cleanup_old_files(self, days: int = 30) -> int:
        """古いファイルをクリーンアップ"""
        import time
        
        deleted_count = 0
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        try:
            for directory in [self.uploads_dir, self.processed_dir]:
                for file_path in directory.glob("*"):
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"古いファイルを削除: {file_path}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")
            return 0
    
    def get_file_info(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """ファイル情報を取得"""
        try:
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            return {
                "name": file_path.name,
                "path": str(file_path),
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime,
                "extension": file_path.suffix.lower(),
                "is_pdf": file_path.suffix.lower() == '.pdf'
            }
        except Exception as e:
            logger.error(f"ファイル情報取得エラー: {e}")
            return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """ファイル名をサニタイズ"""
        import re
        
        safe_chars = re.sub(r'[^\w\-_\.]', '_', filename)
        safe_chars = re.sub(r'_+', '_', safe_chars)
        
        if len(safe_chars) > 100:
            name, ext = safe_chars.rsplit('.', 1) if '.' in safe_chars else (safe_chars, '')
            safe_chars = f"{name[:90]}.{ext}" if ext else name[:100]
        
        return safe_chars

def validate_pdf_file(file_path: Path) -> Dict[str, Any]:
    """PDFファイルの妥当性をチェック"""
    result = {"valid": False, "error": None, "pages": 0, "size_mb": 0}
    
    try:
        import fitz
        
        if not file_path.exists():
            result["error"] = "ファイルが存在しません"
            return result
        
        if file_path.suffix.lower() != '.pdf':
            result["error"] = "PDFファイルではありません"
            return result
        
        stat = file_path.stat()
        result["size_mb"] = round(stat.st_size / (1024 * 1024), 2)
        
        if stat.st_size > 50 * 1024 * 1024:
            result["error"] = "ファイルサイズが大きすぎます (50MB以下)"
            return result
        
        doc = fitz.open(file_path)
        result["pages"] = len(doc)
        doc.close()
        
        if result["pages"] == 0:
            result["error"] = "ページが含まれていません"
            return result
        
        if result["pages"] > 300:
            result["error"] = "ページ数が多すぎます (300ページ以下)"
            return result
        
        result["valid"] = True
        return result
        
    except Exception as e:
        result["error"] = f"ファイル検証エラー: {str(e)}"
        return result
