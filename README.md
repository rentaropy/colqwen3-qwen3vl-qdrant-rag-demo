# ColQwen3 + Qwen3-VL + Qdrant RAG Demo

## このデモの目的
本デモは、**OCR 前処理に依存しない PDF 画像ベース RAG** を、単一 GPU 環境で検証することを目的としています。

- 検索: `TomoroAI/tomoro-ai-colqwen3-embed-4b-awq`（transformers 実装）
- ベクトルストア: Qdrant（multivector）
- 回答: Qwen3-VL（llama.cpp server）

## 概要
1. 起動時に `./docs/*.pdf` をページ画像化して Ingest
2. ColQwen3 でクエリとページ画像を multi-vector 埋め込み
3. Qdrant から上位ページを取得
4. 取得ページ画像を Qwen3-VL に渡して回答生成

GUI: `http://localhost:8501`

## 従来の OCR 前提 RAG との違い
従来のスキャン PDF RAG は、事前に OCR でテキスト化してから検索する方式が一般的です。
本デモは OCR を前提にせず、**ページを画像のまま検索・回答に利用**します。

- OCR 誤認識やレイアウト崩れの影響を受けにくい
- 図表・罫線・注記など、視覚情報を含めた検索が可能
- 反面、画像解像度や VLM の視認性能に精度が左右される

## シングルベクトル RAG との違い
一般的なシングルベクトル RAG は、文書を 1 ベクトル/チャンクに圧縮して検索します。
本デモは ColQwen3 + Qdrant multivector により、**トークンレベルに近い細粒度の照合**を行います。

- 局所的な記述や細かい語句に強い
- 複雑レイアウト文書（規格書・図表混在）で有利
- 計算/ストレージコストはシングルベクトルより重い

## 動作要件
- NVIDIA GPU（VRAM >= 12GB）
- NVIDIA Driver
- nvidia-container-toolkit
- Docker / Docker Compose

## セットアップ
```bash
cp ./.env.example ./.env
```

主な設定:
- `MODEL_ID=TomoroAI/tomoro-ai-colqwen3-embed-4b-awq`
- `VLM_GGUF_REPO=unsloth/Qwen3-VL-4B-Instruct-GGUF`
- `VLM_GGUF_FILE=Qwen3-VL-4B-Instruct-Q5_K_M.gguf`
- `VLM_MMPROJ_FILE=mmproj-BF16.gguf`
- `RAG_RENDER_SCALE=2.0`（小さい文字の視認性向上に推奨）

## 起動
```bash
docker compose up -d colqwen3-embed qdrant vlm-llamacpp rag-gui
```

## 動作確認
```bash
curl -sS http://localhost:8000/v1/models
curl -sS http://localhost:8001/health
curl -sS http://localhost:6333/collections
curl -sS http://localhost:8501
```

## 補足
- モデルファイルは `col/llamacpp/start.sh` で初回起動時に自動ダウンロードします。
- `models/` は Git 管理対象外です。
