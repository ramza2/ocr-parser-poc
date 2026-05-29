"""
VLM 원격 워커 프록시 엔진.

Ubuntu 서버(VLM_WORKER_URL 설정)에서 GPU PC의 VLM 워커 API를 호출한다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import httpx

from app.ocr.engines.vlm_base import VlmEngine
from app.schemas.vlm import (
    QaResponse,
    SchemaExtractResponse,
    SchemaField,
    VlmModelInfo,
    VlmModelsResponse,
    VlmOcrResponse,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(connect=30.0, read=600.0, write=120.0, pool=30.0)


class RemoteVlmEngine(VlmEngine):
    """GPU PC VLM 워커로 요청을 전달하는 프록시 엔진."""

    def __init__(self, worker_url: str, info: VlmModelInfo) -> None:
        self._worker = worker_url.rstrip("/")
        self.engine_id = info.model_id
        self.name = info.name
        self.model_id = info.description.removeprefix("HuggingFace: ").split(" (")[0]
        self.vram_gb = info.vram_gb

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self._worker, timeout=_DEFAULT_TIMEOUT)

    def is_loaded(self) -> bool:
        try:
            with self._client() as client:
                resp = client.get("/api/vlm/models")
                resp.raise_for_status()
                data = VlmModelsResponse(**resp.json())
                return data.current_model == self.engine_id
        except Exception as exc:
            logger.warning("VLM worker 상태 조회 실패: %s", exc)
            return False

    def load(self) -> None:
        with self._client() as client:
            resp = client.post("/api/vlm/load", data={"model_id": self.engine_id})
            resp.raise_for_status()
            body = resp.json()
            if not body.get("success", True):
                raise RuntimeError(body.get("error", "VLM worker load 실패"))

    def unload(self) -> None:
        # 워커 측 싱글턴 매니저가 모델 전환 시 해제한다.
        pass

    def ocr(self, image_path: str, options: dict | None = None) -> VlmOcrResponse:
        return self._post_file("/api/vlm/ocr", image_path, VlmOcrResponse)

    def extract_schema(
        self,
        image_path: str,
        schema: list[SchemaField],
        options: dict | None = None,
    ) -> SchemaExtractResponse:
        import json

        fields = [f.model_dump() for f in schema]
        with self._client() as client, open(image_path, "rb") as fh:
            resp = client.post(
                "/api/vlm/extract",
                data={
                    "model_id": self.engine_id,
                    "schema_fields_json": json.dumps(fields, ensure_ascii=False),
                },
                files={
                    "file": (Path(image_path).name, fh, "application/octet-stream"),
                },
            )
            resp.raise_for_status()
            return SchemaExtractResponse(**resp.json())

    def ask(
        self, image_path: str, question: str, options: dict | None = None
    ) -> QaResponse:
        with self._client() as client, open(image_path, "rb") as fh:
            resp = client.post(
                "/api/vlm/ask",
                data={"model_id": self.engine_id, "question": question},
                files={
                    "file": (Path(image_path).name, fh, "application/octet-stream"),
                },
            )
            resp.raise_for_status()
            return QaResponse(**resp.json())

    def _post_file(self, path: str, image_path: str, model_cls):
        with self._client() as client, open(image_path, "rb") as fh:
            resp = client.post(
                path,
                data={"model_id": self.engine_id},
                files={
                    "file": (Path(image_path).name, fh, "application/octet-stream"),
                },
            )
            resp.raise_for_status()
            return model_cls(**resp.json())


def fetch_remote_engines(worker_url: str) -> dict[str, VlmEngine]:
    """워커 /api/vlm/models 로부터 프록시 엔진 목록 생성."""
    base = worker_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=_DEFAULT_TIMEOUT) as client:
        resp = client.get("/api/vlm/models")
        resp.raise_for_status()
        data = VlmModelsResponse(**resp.json())

    engines: dict[str, VlmEngine] = {}
    for info in data.models:
        engines[info.model_id] = RemoteVlmEngine(base, info)
    logger.info("VLM worker 연결: %s (%d models)", base, len(engines))
    return engines
