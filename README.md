# DocuMind AI - PDF処理・質問応答統合アプリケーション

haruhito2025の既存リポジトリを統合し、PDF処理→ベクトル化→質問応答の完全なパイプラインを実現する統合アプリケーションです。

## 機能概要

- **PDF OCR処理**: 複数のOCRエンジン（EasyOCR、PaddleOCR、LayoutLMv3）を統合
- **ベクトルデータベース**: ChromaDBを使用した高速検索
- **質問応答システム**: OpenAI GPTを使用したRAGベースの回答生成
- **Notion連携**: フィードバック収集と履歴管理
- **モジュラー設計**: 各コンポーネントを独立してアップグレード可能

## クイックスタート

```bash
# 依存関係のインストール
pip install -r requirements.txt

# アプリケーション起動
streamlit run main.py
```

## アーキテクチャ

```
PDF入力 → OCR処理 → テキスト抽出 → ベクトル化 → 質問応答 → Notion保存
```

## 統合元リポジトリ

- Enhanced-PDF-OCR-Text-Extraction-Tool
- easyocr_layoutlmv3
- RAG_GPT
- chat-docs-qa-app
- chatgpt-notion-memo

## ライセンス

MIT License
