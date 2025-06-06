import re
from typing import List, Dict, Any, Optional

def clean_ocr_text(text: str) -> str:
    """OCRで抽出されたテキストをクリーニング"""
    if not text:
        return ""
    
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('〇', '○').replace('0', '〇')
    text = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF。、！？]', '', text)
    
    return text

def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """テキストからメタデータを抽出"""
    metadata = {
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len(text.split('\n')),
        "has_japanese": bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text)),
        "has_english": bool(re.search(r'[a-zA-Z]', text)),
        "has_numbers": bool(re.search(r'\d', text))
    }
    
    return metadata

def split_text_by_sentences(text: str, max_length: int = 1000) -> List[str]:
    """文単位でテキストを分割"""
    if not text:
        return []
    
    sentences = re.split(r'[。！？\n]', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + "。"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + "。"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def calculate_text_similarity(text1: str, text2: str) -> float:
    """テキスト類似度を計算"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def format_text_for_display(text: str, max_length: int = 200) -> str:
    """表示用にテキストをフォーマット"""
    if not text:
        return ""
    
    text = text.strip()
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """テキストからキーワードを抽出"""
    if not text:
        return []
    
    words = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w]+', text.lower())
    
    stop_words = {'の', 'に', 'は', 'を', 'が', 'で', 'と', 'から', 'まで', 'より', 'も', 'て', 'だ', 'である', 'です', 'ます'}
    
    filtered_words = [word for word in words if len(word) > 1 and word not in stop_words]
    
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in sorted_words[:top_k]]
