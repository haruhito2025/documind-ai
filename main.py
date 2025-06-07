import streamlit as st
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
from dotenv import load_dotenv

# .envファイルの読み込み
load_dotenv()

from config.settings import settings
from modules.document_processor.unified_ocr import OCRFactory
from modules.vector_store.chroma_store import ChromaVectorStore, DocumentProcessor
from modules.qa_interface.retrieval_qa import RetrievalQAInterface, EnhancedQAInterface
from modules.integrations.notion_client import FeedbackManager
from utils.file_utils import FileManager, validate_pdf_file
from utils.text_processing import clean_ocr_text, format_text_for_display

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlitの設定
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_components():
    """アプリケーションコンポーネントを初期化"""
    try:
        file_manager = FileManager(settings.data_dir)
        vector_store = ChromaVectorStore()
        qa_interface = EnhancedQAInterface(vector_store)
        feedback_manager = FeedbackManager()
        
        return file_manager, vector_store, qa_interface, feedback_manager
    except Exception as e:
        st.error(f"初期化エラー: {str(e)}")
        st.stop()

def main():
    st.title("📄 DocuMind AI")
    st.markdown("**PDF処理・質問応答統合アプリケーション**")
    
    file_manager, vector_store, qa_interface, feedback_manager = initialize_components()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 ドキュメント入力", 
        "💬 質問応答", 
        "⚙️ 処理設定", 
        "📊 履歴・分析"
    ])
    
    with tab1:
        document_input_tab(file_manager, vector_store)
    
    with tab2:
        qa_tab(qa_interface, feedback_manager)
    
    with tab3:
        settings_tab()
    
    with tab4:
        history_tab(feedback_manager, vector_store)

def document_input_tab(file_manager: FileManager, vector_store: ChromaVectorStore):
    """ドキュメント入力タブ"""
    st.header("📄 ドキュメント処理")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("PDFファイルアップロード")
        uploaded_file = st.file_uploader(
            "PDFファイルを選択してください",
            type=['pdf'],
            help="最大50MB、300ページまでのPDFファイルをサポート"
        )
        
        if uploaded_file is not None:
            file_details = {
                "filename": uploaded_file.name,
                "size": uploaded_file.size,
                "size_mb": round(uploaded_file.size / (1024 * 1024), 2)
            }
            
            st.info(f"📁 {file_details['filename']} ({file_details['size_mb']} MB)")
            
            if st.button("📤 アップロード & 処理開始", type="primary"):
                process_uploaded_file(uploaded_file, file_manager, vector_store)
    
    with col2:
        st.subheader("OCRエンジン選択")
        available_engines = OCRFactory.get_available_engines()
        selected_engine = st.selectbox(
            "OCRエンジン",
            available_engines,
            index=0,
            help="EasyOCR: 高速、PaddleOCR: 高精度、Multi: 複数エンジン組み合わせ"
        )
        
        st.session_state.selected_ocr_engine = selected_engine
        
        st.subheader("処理済みファイル")
        processed_files = file_manager.get_processed_files()
        if processed_files:
            for file_info in processed_files[:5]:
                st.text(f"📄 {file_info['name']}")
        else:
            st.text("処理済みファイルはありません")

def process_uploaded_file(uploaded_file, file_manager: FileManager, vector_store: ChromaVectorStore):
    """アップロードされたファイルを処理"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📤 ファイルを保存中...")
        file_content = uploaded_file.read()
        saved_path = file_manager.save_uploaded_file(file_content, uploaded_file.name)
        progress_bar.progress(20)
        
        status_text.text("🔍 ファイルを検証中...")
        validation_result = validate_pdf_file(saved_path)
        if not validation_result["valid"]:
            st.error(f"❌ ファイル検証エラー: {validation_result['error']}")
            return
        progress_bar.progress(40)
        
        status_text.text("📖 OCR処理中...")
        engine_name = st.session_state.get('selected_ocr_engine', 'easyocr')
        ocr_engine = OCRFactory.create_engine(engine_name)
        ocr_results = ocr_engine.extract_text_from_pdf(saved_path)
        progress_bar.progress(70)
        
        if not ocr_results.get("success", False):
            st.error(f"❌ OCR処理エラー: {ocr_results.get('error', '不明なエラー')}")
            return
        
        status_text.text("🔄 ベクトル化中...")
        chunks = DocumentProcessor.process_ocr_results(ocr_results, uploaded_file.name)
        if chunks:
            texts = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            vector_store.add_documents(texts, metadatas)
        progress_bar.progress(90)
        
        status_text.text("✅ 処理完了")
        file_manager.move_to_processed(saved_path)
        progress_bar.progress(100)
        
        st.success(f"✅ 処理完了: {len(chunks)} チャンクを追加しました")
        
        with st.expander("📊 処理結果詳細"):
            st.json({
                "ファイル名": uploaded_file.name,
                "OCRエンジン": engine_name,
                "総文字数": ocr_results.get("total_chars", 0),
                "ページ数": len(ocr_results.get("text_by_page", {})),
                "チャンク数": len(chunks)
            })
        
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ 処理エラー: {str(e)}")
        logger.error(f"ファイル処理エラー: {e}")

def qa_tab(qa_interface: EnhancedQAInterface, feedback_manager: FeedbackManager):
    """質問応答タブ"""
    st.header("💬 質問応答")
    
    collection_info = qa_interface.vector_store.get_collection_info()
    if collection_info["document_count"] == 0:
        st.warning("📝 まずドキュメントを処理してください")
        return
    
    st.info(f"📚 {collection_info['document_count']} 件のドキュメントが利用可能です")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("⚙️ 設定")
        temperature = st.slider(
            "回答の創造性",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="値を大きくするとより創造的な回答になります"
        )
        qa_interface.set_temperature(temperature)
        
        use_history = st.checkbox(
            "会話履歴を使用",
            value=False,
            help="過去の質問を考慮して回答します"
        )
        
        if st.button("🗑️ 履歴クリア"):
            qa_interface.clear_history()
            st.success("履歴をクリアしました")
    
    with col1:
        st.subheader("質問入力")
        question = st.text_area(
            "質問を入力してください",
            height=100,
            placeholder="例: この文書の主要なポイントは何ですか？"
        )
        
        if st.button("🔍 質問する", type="primary", disabled=not question.strip()):
            with st.spinner("回答を生成中..."):
                if use_history:
                    result = qa_interface.ask_question_with_context(question, use_history=True)
                else:
                    result = qa_interface.ask_question(question)
                
                display_qa_result(result, feedback_manager)

def display_qa_result(result: Dict[str, Any], feedback_manager: FeedbackManager):
    """質問応答結果を表示"""
    st.markdown("### 📘 回答:")
    st.write(result["answer"])
    
    st.markdown("### 🔍 参照元:")
    sources = result.get("sources", [])
    
    if sources:
        for i, source in enumerate(sources, 1):
            with st.expander(f"参照元 {i}: {source.get('source', '不明')} (ページ: {source.get('page', '不明')})"):
                st.markdown(format_text_for_display(source.get("content", ""), 500))
                st.caption(f"OCRエンジン: {source.get('engine', '不明')}")
    else:
        st.info("参照元が見つかりませんでした")
    
    st.markdown("### 👍👎 フィードバック")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👍 Good", key=f"good_{hash(result['question'])}"):
            success = feedback_manager.save_feedback(
                result["question"], 
                result["answer"], 
                "good",
                sources,
                result.get("confidence", 0.0)
            )
            if success:
                st.success("✅ Good評価を保存しました")
            else:
                st.warning("⚠️ ローカルに保存しました（Notion接続なし）")
    
    with col2:
        if st.button("👎 Bad", key=f"bad_{hash(result['question'])}"):
            success = feedback_manager.save_feedback(
                result["question"], 
                result["answer"], 
                "bad",
                sources,
                result.get("confidence", 0.0)
            )
            if success:
                st.success("✅ Bad評価を保存しました")
            else:
                st.warning("⚠️ ローカルに保存しました（Notion接続なし）")
    
    with col3:
        if st.button("⏳ Pending", key=f"pending_{hash(result['question'])}"):
            success = feedback_manager.save_feedback(
                result["question"], 
                result["answer"], 
                "pending",
                sources,
                result.get("confidence", 0.0)
            )
            if success:
                st.success("✅ Pending評価を保存しました")
            else:
                st.warning("⚠️ ローカルに保存しました（Notion接続なし）")

def settings_tab():
    """設定タブ"""
    st.header("⚙️ 処理設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 OCRエンジン設定")
        
        available_engines = OCRFactory.get_available_engines()
        for engine in available_engines:
            with st.expander(f"{engine.upper()} エンジン"):
                if engine == "easyocr":
                    st.write("- 高速処理")
                    st.write("- GPU対応")
                    st.write("- 日本語・英語対応")
                elif engine == "paddle":
                    st.write("- 高精度OCR")
                    st.write("- 角度補正機能")
                    st.write("- 日本語特化")
                elif engine == "multi":
                    st.write("- 複数エンジン組み合わせ")
                    st.write("- 最適結果を自動選択")
                    st.write("- 高い信頼性")
        
        st.subheader("📊 ベクトルストア設定")
        vector_config = settings.get_vector_store_config()
        st.json(vector_config)
    
    with col2:
        st.subheader("🤖 QAモデル設定")
        qa_config = settings.get_qa_config()
        st.json(qa_config)
        
        st.subheader("🔗 API設定")
        api_status = {
            "OpenAI API": "✅ 設定済み" if settings.openai_api_key else "❌ 未設定",
            "Notion API": "✅ 設定済み" if settings.notion_token else "❌ 未設定"
        }
        
        for api, status in api_status.items():
            st.write(f"{api}: {status}")
        
        if not settings.openai_api_key:
            st.error("⚠️ OPENAI_API_KEY が設定されていません")
            st.code("export OPENAI_API_KEY=your_api_key")

def history_tab(feedback_manager: FeedbackManager, vector_store: ChromaVectorStore):
    """履歴・分析タブ"""
    st.header("📊 履歴・分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 フィードバック統計")
        feedback_summary = feedback_manager.get_feedback_summary()
        
        if feedback_summary["notion_available"]:
            notion_stats = feedback_summary["notion"]
            if "error" not in notion_stats:
                st.metric("総エントリ数", notion_stats["total_entries"])
                
                feedback_cols = st.columns(3)
                with feedback_cols[0]:
                    st.metric("👍 Good", notion_stats["good_count"])
                with feedback_cols[1]:
                    st.metric("👎 Bad", notion_stats["bad_count"])
                with feedback_cols[2]:
                    st.metric("⏳ Pending", notion_stats["pending_count"])
            else:
                st.error(f"Notion統計取得エラー: {notion_stats['error']}")
        else:
            st.warning("Notion統合が利用できません")
        
        local_stats = feedback_summary["local"]
        st.subheader("💾 ローカル統計")
        local_cols = st.columns(3)
        with local_cols[0]:
            st.metric("👍 Good", local_stats["good"])
        with local_cols[1]:
            st.metric("👎 Bad", local_stats["bad"])
        with local_cols[2]:
            st.metric("⏳ Pending", local_stats["pending"])
    
    with col2:
        st.subheader("📚 ドキュメント統計")
        collection_info = vector_store.get_collection_info()
        st.metric("ドキュメント数", collection_info["document_count"])
        
        st.subheader("🗂️ 最近のフィードバック")
        local_feedback = feedback_manager.get_local_feedback()
        
        if local_feedback:
            for feedback in local_feedback[-5:]:
                with st.expander(f"{feedback['feedback']} - {feedback['question'][:50]}..."):
                    st.write(f"**質問:** {feedback['question']}")
                    st.write(f"**回答:** {format_text_for_display(feedback['answer'], 200)}")
                    st.caption(f"時刻: {feedback['timestamp']}")
        else:
            st.info("フィードバック履歴がありません")
    
    if st.button("🗑️ ベクトルストアをリセット"):
        if st.checkbox("本当にリセットしますか？"):
            vector_store.delete_collection()
            st.success("ベクトルストアをリセットしました")
            st.rerun()

if __name__ == "__main__":
    main()
