import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from modules.vector_store.base_store import BaseVectorStore, DocumentChunk
from config.settings import settings

logger = logging.getLogger(__name__)

class ChromaVectorStore(BaseVectorStore):
    """ChromaDBを使用したベクトルストア実装"""
    
    def __init__(self, collection_name: str = "documind_collection"):
        self.collection_name = collection_name
        self.persist_directory = settings.vector_store_dir / "chroma_db"
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        config = settings.get_vector_store_config()
        
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY が設定されていません")
        
        self.embedding = OpenAIEmbeddings(
            model=config["embedding_model"],
            api_key=settings.openai_api_key
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """ベクトルストアを初期化"""
        try:
            if (self.persist_directory / "chroma.sqlite3").exists():
                logger.info("既存のベクトルストアを読み込み中...")
                self.vectorstore = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embedding,
                    collection_name=self.collection_name
                )
            else:
                logger.info("新しいベクトルストアを作成中...")
                self.vectorstore = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embedding,
                    collection_name=self.collection_name
                )
        except Exception as e:
            logger.error(f"ベクトルストア初期化エラー: {e}")
            raise
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """ドキュメントを追加"""
        try:
            all_chunks = []
            all_metadatas = []
            
            for i, text in enumerate(texts):
                chunks = self.text_splitter.split_text(text)
                all_chunks.extend(chunks)
                
                base_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                for j, chunk in enumerate(chunks):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": f"{i}_{j}",
                        "chunk_size": len(chunk),
                        "total_chunks": len(chunks)
                    })
                    all_metadatas.append(chunk_metadata)
            
            if all_chunks:
                self.vectorstore.add_texts(
                    texts=all_chunks,
                    metadatas=all_metadatas
                )
                logger.info(f"ドキュメント追加完了: {len(all_chunks)} チャンク")
            
        except Exception as e:
            logger.error(f"ドキュメント追加エラー: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """類似検索を実行"""
        try:
            if not self.vectorstore:
                return []
            
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            search_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"検索エラー: {e}")
            return []
    
    def delete_collection(self) -> None:
        """コレクションを削除"""
        try:
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
                logger.info("ベクトルストアを削除しました")
            self._initialize_vectorstore()
        except Exception as e:
            logger.error(f"コレクション削除エラー: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """コレクション情報を取得"""
        try:
            if not self.vectorstore:
                return {"document_count": 0, "collection_name": self.collection_name}
            
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.warning(f"コレクション情報取得エラー: {e}")
            return {"document_count": 0, "collection_name": self.collection_name}
    
    def persist(self) -> None:
        """データを永続化"""
        try:
            if self.vectorstore:
                self.vectorstore.persist()
                logger.info("ベクトルストアを永続化しました")
        except Exception as e:
            logger.error(f"永続化エラー: {e}")
            raise
    
    def get_all_documents(self) -> List[Any]:
        """すべてのドキュメントを取得"""
        try:
            results = self.vectorstore.get()
            documents = []
            
            for i in range(len(results["ids"])):
                doc = {
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                }
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"ドキュメント取得エラー: {e}")
            return []

class DocumentProcessor:
    """ドキュメント処理ユーティリティ"""
    
    @staticmethod
    def process_ocr_results(ocr_results: Dict[str, Any], source_file: str) -> List[DocumentChunk]:
        """OCR結果をドキュメントチャンクに変換"""
        chunks = []
        
        if not ocr_results.get("success", False):
            return chunks
        
        text_by_page = ocr_results.get("text_by_page", {})
        
        for page_num, text in text_by_page.items():
            if text and len(text.strip()) > 10:
                metadata = {
                    "source": source_file,
                    "page": page_num,
                    "engine": ocr_results.get("engine", "unknown"),
                    "char_count": len(text)
                }
                chunks.append(DocumentChunk(text, metadata))
        
        return chunks
    
    @staticmethod
    def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
        """ファイル名からメタデータを抽出"""
        path = Path(filename)
        return {
            "filename": path.name,
            "file_extension": path.suffix,
            "file_stem": path.stem
        }
