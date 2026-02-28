#!/usr/bin/env python3
import base64
import hashlib
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import pypdfium2 as pdfium
import requests
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models


@dataclass
class RetrievedPage:
    score: float
    doc_id: str
    page_no: int
    source_pdf: str
    image_data_url: str


class RagPipeline:
    def __init__(
        self,
        model_id: str,
        vlm_model_id: str,
        vllm_base_url: str,
        vlm_base_url: str,
        qdrant_url: str,
        collection: str,
        docs_dir: str,
        render_scale: float = 1.5,
        manifest_path: str = "/app/.rag_gui_manifest.json",
    ) -> None:
        self.model_id = model_id
        self.vlm_model_id = vlm_model_id
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.vlm_base_url = vlm_base_url.rstrip("/")
        self.qdrant_url = qdrant_url
        self.collection = collection
        self.docs_dir = Path(docs_dir)
        self.render_scale = render_scale
        self.manifest_path = Path(manifest_path)
        self.qdrant = QdrantClient(url=self.qdrant_url, timeout=180)

    @staticmethod
    def _log(message: str) -> None:
        print(message, file=sys.stdout, flush=True)

    def wait_for_services(self, max_attempts: int = 60, sleep_sec: int = 3) -> None:
        for attempt in range(1, max_attempts + 1):
            try:
                self._log(
                    f"[startup] service check {attempt}/{max_attempts}: "
                    f"vllm={self.vllm_base_url} vlm={self.vlm_base_url} qdrant={self.qdrant_url}"
                )
                requests.get(f"{self.vllm_base_url}/v1/models", timeout=10).raise_for_status()
                requests.get(f"{self.vlm_base_url}/health", timeout=10).raise_for_status()
                requests.get(f"{self.qdrant_url}/collections", timeout=10).raise_for_status()
                self._log("[startup] all services are ready")
                return
            except requests.RequestException:
                if attempt == max_attempts:
                    raise
                time.sleep(sleep_sec)

    def auto_ingest(self) -> dict[str, Any]:
        self._log("[ingest] waiting for dependent services")
        self.wait_for_services()
        manifest = self._load_manifest()
        docs = sorted(self.docs_dir.glob("*.pdf"))
        report: dict[str, Any] = {"ingested": [], "skipped": [], "failed": []}

        if not docs:
            self._log(f"[ingest] no PDF found under {self.docs_dir}")
            return report

        self._log(f"[ingest] start: docs={len(docs)}")
        for pdf_path in docs:
            signature = self._pdf_signature(pdf_path)
            pages = self._pdf_page_count(pdf_path)
            prev = manifest.get(str(pdf_path))
            if prev == signature:
                self._log(f"[ingest] skip unchanged: {pdf_path.name}")
                report["skipped"].append({"pdf": pdf_path.name, "pages": pages})
                continue
            try:
                self._log(f"[ingest] begin: {pdf_path.name}")
                page_count = self._ingest_one_pdf(pdf_path)
                manifest[str(pdf_path)] = signature
                report["ingested"].append({"pdf": pdf_path.name, "pages": page_count})
                self._log(f"[ingest] done: {pdf_path.name} pages={page_count}")
            except Exception as exc:
                self._log(f"[ingest] failed: {pdf_path.name} error={exc}")
                report["failed"].append(
                    {"pdf": pdf_path.name, "pages": pages, "error": str(exc)}
                )

        self._save_manifest(manifest)
        self._log(
            "[ingest] summary: "
            f"ingested={len(report['ingested'])} skipped={len(report['skipped'])} failed={len(report['failed'])}"
        )
        return report

    def search(self, query: str, top_k: int = 3) -> List[RetrievedPage]:
        query_vectors = self._embed_query(query)
        results = self.qdrant.query_points(
            collection_name=self.collection,
            using="colqwen_multi",
            query=query_vectors,
            limit=top_k,
            with_payload=True,
        )

        pages: List[RetrievedPage] = []
        for p in results.points:
            payload = p.payload or {}
            source_pdf = str(payload.get("source_pdf", ""))
            page_no = int(payload.get("page_no", 0))
            if not source_pdf or page_no < 1:
                continue
            pages.append(
                RetrievedPage(
                    score=float(p.score),
                    doc_id=str(payload.get("doc_id", "")),
                    page_no=page_no,
                    source_pdf=source_pdf,
                    image_data_url=self._render_page_as_data_url(source_pdf, page_no),
                )
            )
        return pages

    def answer(self, query: str, pages: List[RetrievedPage], max_tokens: int = 512) -> str:
        retrieved_summary = "\n".join(
            [
                f"- [{i}] {p.doc_id} page={p.page_no} score={p.score:.4f}"
                for i, p in enumerate(pages, start=1)
            ]
        )
        # prompt = (
        #     "あなたは文書QAアシスタントです。"
        #     "検索で得た文書ページ画像だけを根拠に回答してください。"
        #     "根拠が不十分ならその旨を明記してください。\n\n"
        #     f"ユーザークエリ: {query}\n"
        #     "検索結果:\n"
        #     f"{retrieved_summary}\n"
        #     "回答では、参照したページ番号を末尾に列挙してください。"
        # )
        prompt = (
            "あなたは文書QAアシスタントです。"
            "検索で得た文書ページ画像を根拠に回答してください。"
            f"ユーザークエリ: {query}\n"
            "検索結果:\n"
            f"{retrieved_summary}\n"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for i, p in enumerate(pages, start=1):
            content.append(
                {"type": "text", "text": f"Retrieved page [{i}] {p.doc_id} page={p.page_no}"}
            )
            content.append({"type": "image_url", "image_url": {"url": p.image_data_url}})

        payload = {
            "model": self.vlm_model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "top_k": 20,
            "max_tokens": max_tokens,
        }
        res = requests.post(
            f"{self.vlm_base_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

    def _ingest_one_pdf(self, pdf_path: Path) -> int:
        doc_id = pdf_path.name
        doc = pdfium.PdfDocument(str(pdf_path))
        vector_size = None
        total_pages = len(doc)
        ingested_pages = 0
        self._log(f"[ingest] document={doc_id} total_pages={total_pages}")

        for idx in range(1, total_pages + 1):
            self._log(f"[ingest] page {idx}/{total_pages} document={doc_id}")
            data_url = self._render_page_as_data_url(str(pdf_path), idx)
            token_vectors = self._embed_page_image(data_url)
            self._log(
                f"[ingest] embedded page {idx}/{total_pages} document={doc_id} "
                f"tokens={len(token_vectors)} dim={len(token_vectors[0]) if token_vectors else 0}"
            )
            if vector_size is None:
                vector_size = len(token_vectors[0])
                self._ensure_collection(vector_size)
            point = models.PointStruct(
                id=int.from_bytes(
                    hashlib.sha1(f"{doc_id}:{idx}".encode("utf-8")).digest()[:8],
                    "big",
                ),
                vector={"colqwen_multi": token_vectors},
                payload={
                    "doc_id": doc_id,
                    "page_no": idx,
                    "source_pdf": str(pdf_path),
                    "render_scale": self.render_scale,
                },
            )
            self._log(
                f"[ingest] qdrant upsert begin document={doc_id} page={idx}/{total_pages} points=1"
            )
            self.qdrant.upsert(collection_name=self.collection, points=[point], wait=True)
            self._log(f"[ingest] qdrant upsert done document={doc_id} page={idx}/{total_pages}")
            ingested_pages += 1
        return ingested_pages

    def _ensure_collection(self, vector_size: int) -> None:
        vector_params = models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
        )
        if self.qdrant.collection_exists(self.collection):
            return
        self.qdrant.create_collection(
            collection_name=self.collection,
            vectors_config={"colqwen_multi": vector_params},
        )

    def _embed_query(self, query: str) -> List[List[float]]:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            "task": "token_embed",
        }
        res = requests.post(f"{self.vllm_base_url}/pooling", json=payload, timeout=120)
        res.raise_for_status()
        return res.json()["data"][0]["data"]

    def _embed_page_image(self, image_data_url: str) -> List[List[float]]:
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": "Represent this document page for retrieval."},
                    ],
                }
            ],
            "task": "token_embed",
        }
        res = requests.post(f"{self.vllm_base_url}/pooling", json=payload, timeout=180)
        res.raise_for_status()
        return res.json()["data"][0]["data"]

    def _render_page_as_data_url(self, pdf_path: str, page_no: int) -> str:
        doc = pdfium.PdfDocument(pdf_path)
        page = doc[page_no - 1]
        bitmap = page.render(scale=self.render_scale)
        image = bitmap.to_pil().convert("RGB")
        return self._image_to_data_url(image)

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _pdf_signature(self, pdf_path: Path) -> dict[str, Any]:
        stat = pdf_path.stat()
        sha = hashlib.sha1()
        with pdf_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        return {
            "sha1": sha.hexdigest(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "render_scale": self.render_scale,
            "collection": self.collection,
        }

    @staticmethod
    def _pdf_page_count(pdf_path: Path) -> int | None:
        try:
            return len(pdfium.PdfDocument(str(pdf_path)))
        except Exception:
            return None

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def page_image_bytes(page: RetrievedPage) -> bytes:
    header, b64 = page.image_data_url.split(",", 1)
    _ = header
    return base64.b64decode(b64)
