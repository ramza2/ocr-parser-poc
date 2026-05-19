#!/usr/bin/env python3
"""컨테이너/서버에서 Paddle GPU·PaddleOCR 3.x 동작 확인."""
from __future__ import annotations

import json
import sys


def main() -> int:
    out: dict = {}
    try:
        import paddle

        out["paddle_version"] = paddle.__version__
        out["cuda_compiled"] = paddle.device.is_compiled_with_cuda()
        if out["cuda_compiled"]:
            paddle.device.set_device("gpu:0")
            t = paddle.to_tensor([1.0], place="gpu:0")
            out["gpu_tensor"] = str(t.place)
    except Exception as exc:
        out["paddle_error"] = str(exc)

    try:
        import paddleocr

        out["paddleocr_version"] = paddleocr.__version__
        major = int(str(paddleocr.__version__).split(".")[0])
        if major >= 3:
            from paddleocr import PaddleOCR

            device = "gpu:0" if out.get("cuda_compiled") else "cpu"
            ocr = PaddleOCR(
                lang="korean",
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            out["paddleocr_device"] = device
            out["paddleocr_init"] = "ok"
    except Exception as exc:
        out["paddleocr_error"] = str(exc)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    ok = out.get("paddleocr_init") == "ok" and out.get("cuda_compiled")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
