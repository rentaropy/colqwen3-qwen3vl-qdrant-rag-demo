#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from rag_pipeline import RagPipeline, page_image_bytes


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


def load_env_file_text() -> str:
    path = Path(env("ENV_FILE_PATH", "/app/.env"))
    if not path.exists():
        return f"{path} が見つかりません。"
    masked_keys = {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw:
            lines.append(raw)
            continue
        key, value = raw.split("=", 1)
        if key.strip() in masked_keys and value.strip():
            lines.append(f"{key}=***REDACTED***")
        else:
            lines.append(raw)
    return "\n".join(lines)


@st.cache_resource
def get_pipeline() -> RagPipeline:
    return RagPipeline(
        model_id=env("MODEL_ID", "TomoroAI/tomoro-ai-colqwen3-embed-4b-awq"),
        vlm_model_id=env("VLM_MODEL_ID", "Qwen3-VL-4B-Instruct-Q5_K_M.gguf"),
        vllm_base_url=env("VLLM_BASE_URL", "http://colqwen3-embed:8000"),
        vlm_base_url=env("VLM_BASE_URL", "http://vlm-llamacpp:8000"),
        qdrant_url=env("QDRANT_URL", "http://qdrant:6333"),
        collection=env("RAG_COLLECTION", "jaxa_pdf_pages"),
        docs_dir=env("RAG_DOCS_DIR", "/app/docs"),
        render_scale=float(env("RAG_RENDER_SCALE", "1.5")),
    )


def load_startup_ingest_report() -> dict[str, Any] | None:
    path = Path(env("STARTUP_INGEST_REPORT_PATH", "/app/.rag_gui_startup_report.json"))
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"message": f"invalid ingest report json: {path}"}


def render_report(report: dict[str, Any] | None) -> None:
    if not report:
        return
    st.header("ファイル投入状況")
    if "message" in report:
        st.info(report["message"])
        return
    ingested = report.get("ingested", [])
    skipped = report.get("skipped", [])
    failed = report.get("failed", [])

    def _normalize_item(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return item
        return {"pdf": str(item), "pages": None}

    st.subheader("今回投入")
    if ingested:
        for raw in [_normalize_item(x) for x in ingested]:
            pdf = raw.get("pdf", "-")
            pages = raw.get("pages", "-")
            st.write(f"- {pdf} ({pages} ページ)")
    else:
        st.write("- なし")

    st.subheader("すでに投入済み")
    if skipped:
        for raw in [_normalize_item(x) for x in skipped]:
            pdf = raw.get("pdf", "-")
            pages = raw.get("pages", "-")
            st.write(f"- {pdf} ({pages} ページ)")
    else:
        st.write("- なし")

    st.subheader("投入エラー")
    if failed:
        for raw in failed:
            item = _normalize_item(raw)
            st.write(
                f"- {item.get('pdf', '-')} ({item.get('pages', '-')} ページ): "
                f"{item.get('error', '-')}"
            )
    else:
        st.write("- なし")


def main() -> None:
    st.set_page_config(page_title="ColQwen3 RAG GUI", layout="wide")
    st.title("ColQwen3 (Embedding) + Qwen3-VL (VLM) + Qdrant (Vector DB) RAG")

    pipeline = get_pipeline()

    with st.sidebar:
        st.subheader("設定")
        st.code(load_env_file_text())

    render_report(load_startup_ingest_report())

    st.header("チャット")
    st.subheader("質問を入力")
    query = st.text_input(
        "質問を入力",
        value="タンタル電解コンデンサの電極構成材料は？",
        label_visibility="collapsed",
    )
    st.subheader("VLMに渡す検索結果上位nページ")
    top_k = st.number_input(
        "VLMに渡す検索結果上位nページ",
        min_value=1,
        max_value=10,
        value=int(env("RAG_TOP_K", "2")),
        label_visibility="collapsed",
    )

    if st.button("検索して回答", type="primary"):
        if not query.strip():
            st.warning("クエリを入力してください。")
            return
        with st.spinner("検索中..."):
            pages = pipeline.search(query, int(top_k))
        if not pages:
            st.warning("検索結果がありません。")
            return
        with st.spinner("VLM回答生成中..."):
            answer = pipeline.answer(query, pages)

        st.subheader("回答")
        st.write(answer)

        st.subheader("検索結果")
        cols = st.columns(len(pages))
        for i, page in enumerate(pages):
            with cols[i]:
                st.write(f"ファイル名: {page.doc_id}")
                st.write(f"ページ番号: {page.page_no}")
                st.write(f"検索スコア: {page.score:.4f}")
                st.write(f"検索順位: {i+1}")
                st.image(page_image_bytes(page), use_container_width=True)


if __name__ == "__main__":
    main()
