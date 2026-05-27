from __future__ import annotations

import os


def get_hf_token() -> str | None:
    token = _clean(os.getenv("HF_TOKEN")) or _clean(os.getenv("HUGGINGFACE_HUB_TOKEN"))
    return token or None


def hf_auth_kwargs() -> dict[str, str]:
    token = get_hf_token()
    if not token:
        return {}
    return {"token": token}


def is_hf_auth_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {401, 403}:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) in {401, 403}:
        return True
    text = _clean(str(exc)).lower()
    return "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text


def hf_auth_help_message(model_ref: str) -> str:
    return (
        f"Acesso negado ao modelo '{model_ref}' (401/403). "
        "Se o repositorio for privado/restrito, configure HF_TOKEN ou HUGGINGFACE_HUB_TOKEN no ambiente."
    )


def _clean(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
