#!/usr/bin/env python3
import json
import os
from pathlib import Path

from rag_pipeline import RagPipeline


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


def build_pipeline() -> RagPipeline:
    return RagPipeline(
        model_id=env("MODEL_ID", "TomoroAI/tomoro-ai-colqwen3-embed-4b-awq"),
        vlm_model_id=env("VLM_MODEL_ID", "Qwen3-VL-4B-Instruct-Q5_K_M.gguf"),
        vllm_base_url=env("VLLM_BASE_URL", "http://colqwen3-embed:8000"),
        vlm_base_url=env("VLM_BASE_URL", "http://vlm-llamacpp:8000"),
        qdrant_url=env("QDRANT_URL", "http://qdrant:6333"),
        collection=env("RAG_COLLECTION", "jaxa_pdf_pages"),
        docs_dir=env("RAG_DOCS_DIR", "/app/docs"),
        render_scale=float(env("RAG_RENDER_SCALE", "1.5")),
        manifest_path=env("RAG_MANIFEST_PATH", "/app/.rag_gui_manifest.json"),
    )


def write_report(report: dict) -> None:
    report_path = Path(env("STARTUP_INGEST_REPORT_PATH", "/app/.rag_gui_startup_report.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ingest] startup report written: {report_path}", flush=True)


def main() -> None:
    auto_ingest = env("AUTO_INGEST_ON_START", "1") == "1"
    if not auto_ingest:
        report = {"message": "AUTO_INGEST_ON_START=0"}
        write_report(report)
        print("[ingest] auto ingest disabled", flush=True)
        return

    pipeline = build_pipeline()
    report = pipeline.auto_ingest()
    write_report(report)


if __name__ == "__main__":
    main()
