import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all module imports"""
    print("Testing imports...")
    
    try:
        global settings, OCRFactory, ChromaVectorStore, EnhancedQAInterface, FeedbackManager, FileManager, validate_pdf_file, clean_ocr_text
        
        from config.settings import settings
        print("✅ Settings import successful")
        
        from modules.document_processor.unified_ocr import OCRFactory
        print("✅ OCR Factory import successful")
        
        from modules.vector_store.chroma_store import ChromaVectorStore
        print("✅ Vector Store import successful")
        
        from modules.qa_interface.retrieval_qa import EnhancedQAInterface
        print("✅ QA Interface import successful")
        
        from modules.integrations.notion_client import FeedbackManager
        print("✅ Notion Client import successful")
        
        from utils.file_utils import FileManager, validate_pdf_file
        print("✅ File Utils import successful")
        
        from utils.text_processing import clean_ocr_text
        print("✅ Text Processing import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_ocr_engines():
    """Test OCR engine availability"""
    print("\nTesting OCR engines...")
    
    try:
        available_engines = OCRFactory.get_available_engines()
        print(f"Available engines: {available_engines}")
        
        for engine_name in available_engines:
            try:
                engine = OCRFactory.create_engine(engine_name)
                print(f"✅ {engine_name} engine created successfully")
            except Exception as e:
                print(f"❌ {engine_name} engine error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR engine test error: {e}")
        return False

def test_vector_store():
    """Test vector store initialization"""
    print("\nTesting vector store...")
    
    try:
        vector_store = ChromaVectorStore()
        info = vector_store.get_collection_info()
        print(f"✅ Vector store initialized: {info}")
        return True
        
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return False

def test_settings():
    """Test settings configuration"""
    print("\nTesting settings...")
    
    try:
        print(f"Data directory: {settings.data_dir}")
        print(f"OpenAI API key configured: {'Yes' if settings.openai_api_key else 'No'}")
        print(f"Notion token configured: {'Yes' if settings.notion_token else 'No'}")
        
        ocr_config = settings.get_ocr_config()
        print(f"OCR config: {ocr_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 DocuMind AI - Component Testing")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_settings()
    success &= test_ocr_engines()
    success &= test_vector_store()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Ready to run Streamlit app.")
    else:
        print("❌ Some tests failed. Check the errors above.")
