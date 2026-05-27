from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import requests

from app.types import AppResult
from app.utils import ensure_dir, make_job_dir, safe_text, sanitize_filename


def get_worker_url() -> str:
    return safe_text(os.getenv("WORKER_URL")).rstrip("/")


def worker_enabled() -> bool:
    if safe_text(os.getenv("DISABLE_WORKER")).lower() in {"1", "true", "yes", "on"}:
        return False
    return bool(get_worker_url())


def allow_local_fallback() -> bool:
    return safe_text(os.getenv("WORKER_ALLOW_LOCAL_FALLBACK")).lower() in {"1", "true", "yes", "on"}


def process_image_on_worker(
    *,
    filename: str,
    content: bytes,
    output_dir: Path,
    params: dict[str, Any],
    timeout: int | None = None,
) -> AppResult | None:
    worker_url = get_worker_url()
    if not worker_url:
        return None

    request_timeout = timeout or int(os.getenv("WORKER_TIMEOUT", "900") or "900")
    endpoint = f"{worker_url}/process"
    safe_name = sanitize_filename(filename)
    print(f"[WORKER] enviando processamento para {endpoint}")

    try:
        response = requests.post(
            endpoint,
            files={"file": (safe_name, content, "application/octet-stream")},
            data={key: str(value) for key, value in params.items() if value is not None},
            timeout=request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"[WORKER] falha: {exc}")
        if not allow_local_fallback():
            raise RuntimeError(f"worker indisponivel; processamento local bloqueado para nao travar VPS principal: {exc}") from exc
        print("[WORKER] usando processamento local por WORKER_ALLOW_LOCAL_FALLBACK=true")
        return None

    image_b64 = safe_text(payload.get("image_base64"))
    if not image_b64:
        print("[WORKER] resposta sem imagem")
        if not allow_local_fallback():
            raise RuntimeError("worker respondeu sem imagem; processamento local bloqueado para nao travar VPS principal")
        print("[WORKER] usando processamento local por WORKER_ALLOW_LOCAL_FALLBACK=true")
        return None

    job_dir = make_job_dir(ensure_dir(output_dir))
    translated_path = job_dir / f"worker_translated_{Path(safe_name).stem}.png"
    original_path = job_dir / safe_name
    try:
        original_path.write_bytes(content)
        translated_path.write_bytes(base64.b64decode(image_b64))
    except Exception as exc:
        print(f"[WORKER] falha ao salvar resposta: {exc}")
        if not allow_local_fallback():
            raise RuntimeError(f"falha ao salvar resposta do worker: {exc}") from exc
        print("[WORKER] usando processamento local por WORKER_ALLOW_LOCAL_FALLBACK=true")
        return None

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata["processed_by_worker"] = True
    metadata["worker_url"] = worker_url
    print(f"[WORKER] processamento concluido: {translated_path}")
    return AppResult(
        original_image_path=original_path,
        translated_image_path=translated_path,
        bubbles=[],
        output_dir=job_dir,
        metadata=metadata,
    )
