from __future__ import annotations

import json
import secrets
import unicodedata
from pathlib import Path
from typing import Iterable, Any


INVISIBLE_CHARS = {
    "\ufeff",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
}
DASH_VARIANTS = {
    "\u2010",
    "\u2011",
    "\u2012",
    "\u2013",
    "\u2014",
    "\u2212",
}


def get_keys_file() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "keys.json"


def ensure_keys_file() -> Path:
    keys_file = get_keys_file()
    keys_file.parent.mkdir(parents=True, exist_ok=True)

    if not keys_file.exists():
        keys_file.write_text(
            json.dumps(
                {
                    "keys": ["ADMIN-2026", "MANGA-ACCESS", "TESTE-KEY"],
                    "admin_keys": ["ADMIN-2026"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return keys_file

    payload = _read_payload(keys_file)
    keys = _normalize_keys(_extract_keys(payload))
    admin_keys = _normalize_keys(_extract_admin_keys(payload))

    if not isinstance(payload, dict) or "admin_keys" not in payload:
        if "ADMIN-2026" in keys and "ADMIN-2026" not in admin_keys:
            admin_keys.append("ADMIN-2026")
        _write_payload(keys_file, keys, admin_keys)

    return keys_file


def _normalize_keys(keys: Iterable[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_key in keys:
        key = _clean_key(raw_key)
        if key and key not in seen:
            normalized.append(key)
            seen.add(key)

    return normalized


def _extract_keys(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("keys"), list):
        return payload["keys"]
    return []


def _extract_admin_keys(payload: Any) -> list[Any]:
    if isinstance(payload, dict) and isinstance(payload.get("admin_keys"), list):
        return payload["admin_keys"]
    return []


def _clean_key(key: Any) -> str:
    text = unicodedata.normalize("NFKC", str(key or ""))
    for invisible_char in INVISIBLE_CHARS:
        text = text.replace(invisible_char, "")
    for dash_variant in DASH_VARIANTS:
        text = text.replace(dash_variant, "-")

    text = text.strip().strip(",")
    text = text.strip("\"'` ")
    return text.strip().upper()


def _read_payload(keys_file: Path) -> Any:
    try:
        return json.loads(keys_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Erro ao carregar data/keys.json: {exc}")
        return {}


def _write_payload(keys_file: Path, keys: Iterable[Any], admin_keys: Iterable[Any]) -> dict[str, list[str]]:
    normalized_keys = _normalize_keys(keys)
    normalized_admin_keys = [
        key for key in _normalize_keys(admin_keys)
        if key in normalized_keys
    ]
    payload = {
        "keys": normalized_keys,
        "admin_keys": normalized_admin_keys,
    }
    keys_file.parent.mkdir(parents=True, exist_ok=True)
    keys_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def load_keys() -> list[str]:
    keys_file = ensure_keys_file()
    payload = _read_payload(keys_file)

    keys = _normalize_keys(_extract_keys(payload))
    return keys


def save_keys(keys: Iterable[Any]) -> list[str]:
    keys_file = get_keys_file()
    payload = _read_payload(ensure_keys_file())
    admin_keys = _normalize_keys(_extract_admin_keys(payload))
    saved_payload = _write_payload(keys_file, keys, admin_keys)
    normalized = saved_payload["keys"]
    return normalized


def generate_key() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

    while True:
        chunks = [
            "".join(secrets.choice(alphabet) for _ in range(4))
            for _ in range(3)
        ]
        key = f"MANGA-{'-'.join(chunks)}"
        if key not in load_keys():
            return key


def add_key(key: str) -> bool:
    clean_key = _clean_key(key)
    if not clean_key:
        return False

    keys = load_keys()
    if clean_key in keys:
        return False

    keys.append(clean_key)
    save_keys(keys)
    return True


def remove_key(key: str) -> bool:
    clean_key = _clean_key(key)
    keys = load_keys()
    updated_keys = [existing_key for existing_key in keys if existing_key != clean_key]

    if len(updated_keys) == len(keys):
        return False

    save_keys(updated_keys)
    return True


def is_valid_key(key: str) -> bool:
    raw_key = str(key or "")
    clean_key = _clean_key(raw_key)
    keys = load_keys()
    return clean_key in keys


def list_admin_keys() -> list[str]:
    keys_file = ensure_keys_file()
    payload = _read_payload(keys_file)
    keys = load_keys()
    return [
        admin_key for admin_key in _normalize_keys(_extract_admin_keys(payload))
        if admin_key in keys
    ]


def is_admin_key(key: str) -> bool:
    clean_key = _clean_key(key)
    admin_keys = list_admin_keys()
    return clean_key in admin_keys


def list_keys() -> list[str]:
    return load_keys()
