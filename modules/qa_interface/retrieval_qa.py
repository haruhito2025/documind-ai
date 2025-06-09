import os
from typing import Dict, List, Any, Optional
import logging

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from modules.qa_interface.base_qa import BaseQAInterface, QAResult
from modules.vector_store.chroma_store import ChromaVectorStore
from modules.reranking.reranker import Reranker
from config.settings import settings

logger = logging.getLogger(__name__)

class RetrievalQAInterface(BaseQAInterface):
    """検索拡張生成(RAG)を使用した質問応答インターフェース"""
    
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store
        self.config = settings.get_qa_config()
        
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY が設定されていません")
        
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            temperature=self.config["temperature"],
            model=self.config["model_name"],
            max_tokens=self.config["max_tokens"]
        )
        
        # リランカーを初期化
        self.reranker = Reranker()
        
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self) -> RetrievalQA:
        """QAチェーンを作成"""
        template = """
あなたは正確で詳細な回答を提供するAIアシスタントです。
与えられた情報源のみに基づいて回答してください。
情報源にない内容については「該当する情報がありません」と明確に伝えてください。
質問の文脈を理解し、最も関連性の高い情報を優先して回答に含めてください。

参考情報:
{context}

質問: {question}

回答は以下の形式で作成してください：
1. 直接的な回答（簡潔かつ具体的に）
2. 参考情報からの具体的な引用（該当箇所を「」で囲む）
3. 引用元の情報（ファイル名やページ番号など）
4. 補足説明（必要な場合のみ）
5. 情報が不足している場合は、その旨を明示

回答:"""
        
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        retriever = self.vector_store.vectorstore.as_retriever(
            search_kwargs={"k": self.config["top_k"] * 2}  # リランキング用に2倍のドキュメントを取得
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """質問に対する回答を生成"""
        try:
            # ベクトル検索でドキュメントを取得
            result = self.qa_chain(question)
            
            # ドキュメントをリランキング
            documents = []
            for doc in result.get("source_documents", []):
                doc_info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "不明"),
                    "page": doc.metadata.get("page", "不明"),
                    "engine": doc.metadata.get("engine", "不明")
                }
                documents.append(doc_info)
            
            # リランキングを実行
            reranked_docs = self.reranker.rerank(
                question,
                documents,
                top_k=self.config["top_k"]
            )
            
            # リランキングされたドキュメントをコンテキストとして使用
            context = "\n\n".join([doc.content for doc in reranked_docs])
            
            # 回答を生成
            answer = self.llm.predict(
                f"以下の情報に基づいて質問に答えてください：\n\n{context}\n\n質問：{question}"
            )
            
            # 結果を整形
            sources = []
            for doc in reranked_docs:
                source_info = {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "source": doc.source,
                    "page": doc.page,
                    "engine": doc.engine,
                    "score": doc.score
                }
                sources.append(source_info)
            
            qa_result = QAResult(
                question=question,
                answer=answer,
                sources=sources,
                confidence=self._calculate_confidence(reranked_docs)
            )
            
            return qa_result.to_dict()
            
        except Exception as e:
            logger.error(f"質問応答エラー: {e}")
            return {
                "question": question,
                "answer": f"エラーが発生しました: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _calculate_confidence(self, reranked_docs: List[Any]) -> float:
        """回答の信頼度を計算"""
        try:
            if not reranked_docs:
                return 0.0
            
            # リランキングスコアの平均を計算
            total_score = sum(doc.score for doc in reranked_docs)
            return min(total_score / len(reranked_docs), 1.0)
            
        except Exception:
            return 0.5
    
    def set_temperature(self, temperature: float) -> None:
        """回答の創造性を設定"""
        self.llm.temperature = max(0.0, min(1.0, temperature))
        logger.info(f"温度パラメータを {temperature} に設定")
    
    def get_model_info(self) -> Dict[str, str]:
        """使用中のモデル情報を取得"""
        return {
            "model_name": self.config["model_name"],
            "temperature": str(self.llm.temperature),
            "max_tokens": str(self.config["max_tokens"]),
            "top_k": str(self.config["top_k"]),
            "reranker": self.reranker.get_model_info()
        }

class EnhancedQAInterface(RetrievalQAInterface):
    """拡張された質問応答インターフェース"""
    
    def __init__(self, vector_store: ChromaVectorStore):
        super().__init__(vector_store)
        self.conversation_history = []
    
    def ask_question_with_context(self, question: str, use_history: bool = False) -> Dict[str, Any]:
        """文脈を考慮した質問応答"""
        if use_history and self.conversation_history:
            context_questions = [item["question"] for item in self.conversation_history[-3:]]
            enhanced_question = f"過去の質問: {' '.join(context_questions)}\n現在の質問: {question}"
        else:
            enhanced_question = question
        
        result = self.ask_question(enhanced_question)
        
        self.conversation_history.append({
            "question": question,
            "answer": result["answer"],
            "timestamp": os.times().elapsed
        })
        
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return result
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """会話履歴を取得"""
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """会話履歴をクリア"""
        self.conversation_history.clear()
        logger.info("会話履歴をクリアしました")
