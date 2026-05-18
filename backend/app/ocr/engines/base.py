from __future__ import annotations

from abc import ABC, abstractmethod


class OcrEngine(ABC):
    engine_id: str
    name: str

    @abstractmethod
    def recognize(self, image_path: str, options: dict | None = None) -> tuple[str, list[dict]]:
        """Returns (text, blocks for metadata)."""
