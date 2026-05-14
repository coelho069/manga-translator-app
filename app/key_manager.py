from __future__ import annotations

import json
import re
import secrets
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


DEFAULT_KEYS = ("ADMIN-2026", "MANGA-ACCESS", "TESTE-KEY")
ADMIN_KEY = "ADMIN-2026"
DEFAULT_EXPIRATION_DAYS = 365

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
        save_keys(_default_entries())
        return keys_file

    payload = _read_payload(keys_file)
    entries = _normalize_entries(_extract_raw_keys(payload))

    if not entries:
        entries = _default_entries()

    normalized_payload = {"keys": entries}
    if payload != normalized_payload:
        _write_payload(keys_file, entries)

    return keys_file


def load_keys() -> list[dict[str, Any]]:
    keys_file = ensure_keys_file()
    payload = _read_payload(keys_file)
    return _normalize_entries(_extract_raw_keys(payload))


def save_keys(keys: Iterable[Any]) -> list[dict[str, Any]]:
    keys_file = get_keys_file()
    entries = _normalize_entries(keys)
    _write_payload(keys_file, entries)
    return entries


def generate_key(expiration_hours: int | None = None, expiration_days: int | None = None) -> str:
    _resolve_expiration(_now(), expiration_hours, expiration_days)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

    existing_keys = {entry["key"] for entry in load_keys()}
    while True:
        chunks = [
            "".join(secrets.choice(alphabet) for _ in range(4))
            for _ in range(3)
        ]
        key = f"MANGA-{'-'.join(chunks)}"
        if key not in existing_keys:
            return key


def add_key(key: str, expiration_hours: int | None = None, expiration_days: int | None = None) -> bool:
    clean_key = _clean_key(key)
    if not clean_key:
        return False

    entries = load_keys()
    if any(entry["key"] == clean_key for entry in entries):
        return False

    created_at = _now()
    expires_at = _resolve_expiration(created_at, expiration_hours, expiration_days)
    entries.append(_make_entry(clean_key, created_at, expires_at, True))
    save_keys(entries)
    return True


def remove_key(key: str) -> bool:
    clean_key = _clean_key(key)
    entries = load_keys()
    updated_entries = [entry for entry in entries if entry["key"] != clean_key]

    if len(updated_entries) == len(entries):
        return False

    save_keys(updated_entries)
    return True


def is_valid_key(key: str) -> bool:
    deactivate_expired_keys()
    status = get_key_status(key)
    return bool(status["exists"] and status["active"] and not status["expired"])


def is_key_expired(key_or_entry: str | dict[str, Any]) -> bool:
    entry = _resolve_entry(key_or_entry)
    if not entry:
        return True

    expires_at = _parse_datetime(entry.get("expires_at"))
    if expires_at is None:
        return True

    return _now() >= expires_at


def deactivate_expired_keys() -> int:
    entries = load_keys()
    changed_count = 0

    for entry in entries:
        if bool(entry.get("active", True)) and is_key_expired(entry):
            entry["active"] = False
            changed_count += 1

    if changed_count:
        save_keys(entries)

    return changed_count


def get_remaining_time(key_or_entry: str | dict[str, Any]) -> str:
    entry = _resolve_entry(key_or_entry)
    if not entry:
        return "inexistente"

    expires_at = _parse_datetime(entry.get("expires_at"))
    if expires_at is None:
        return "expirada"

    remaining = expires_at - _now()
    if remaining.total_seconds() <= 0:
        return "expirada"

    total_seconds = int(remaining.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)

    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def get_key_status(key: str) -> dict[str, Any]:
    clean_key = _clean_key(key)
    entry = _find_entry(clean_key)

    if not entry:
        return {
            "key": clean_key,
            "exists": False,
            "created_at": "",
            "expires_at": "",
            "active": False,
            "expired": False,
            "remaining_time": "inexistente",
            "status": "inexistente",
        }

    expired = is_key_expired(entry)
    active = bool(entry.get("active", True))
    status = "ativa" if active and not expired else "expirada"

    return {
        "key": entry["key"],
        "exists": True,
        "created_at": entry["created_at"],
        "expires_at": entry["expires_at"],
        "active": active,
        "expired": expired,
        "remaining_time": get_remaining_time(entry),
        "status": status,
    }


def list_admin_keys() -> list[str]:
    return [
        entry["key"]
        for entry in load_keys()
        if entry["key"] == ADMIN_KEY
    ]


def is_admin_key(key: str) -> bool:
    clean_key = _clean_key(key)
    return clean_key == ADMIN_KEY and is_valid_key(clean_key)


def list_keys() -> list[dict[str, Any]]:
    deactivate_expired_keys()
    return load_keys()


def _now() -> datetime:
    return datetime.now().replace(microsecond=0)


def _default_entries() -> list[dict[str, Any]]:
    created_at = _now()
    expires_at = created_at + timedelta(days=DEFAULT_EXPIRATION_DAYS)
    return [_make_entry(key, created_at, expires_at, True) for key in DEFAULT_KEYS]


def _make_entry(key: Any, created_at: datetime, expires_at: datetime, active: bool) -> dict[str, Any]:
    return {
        "key": _clean_key(key),
        "created_at": _format_datetime(created_at),
        "expires_at": _format_datetime(expires_at),
        "active": bool(active),
    }


def _normalize_entries(keys: Iterable[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_entry in keys:
        entry = _entry_from_raw(raw_entry)
        if not entry:
            continue

        key = entry["key"]
        if key and key not in seen:
            normalized.append(entry)
            seen.add(key)

    return normalized


def _entry_from_raw(raw_entry: Any) -> dict[str, Any] | None:
    created_at = _now()
    expires_at = created_at + timedelta(days=DEFAULT_EXPIRATION_DAYS)

    if isinstance(raw_entry, dict):
        key = _clean_key(raw_entry.get("key"))
        if not key:
            return None

        parsed_created_at = _parse_datetime(raw_entry.get("created_at"))
        parsed_expires_at = _parse_datetime(raw_entry.get("expires_at"))
        if parsed_created_at is not None:
            created_at = parsed_created_at
        if parsed_expires_at is not None:
            expires_at = parsed_expires_at

        return _make_entry(
            key,
            created_at,
            expires_at,
            bool(raw_entry.get("active", True)),
        )

    legacy_entry = _entry_from_legacy_dict_string(raw_entry)
    if legacy_entry is not None:
        return legacy_entry

    key = _clean_key(raw_entry)
    if not key:
        return None

    return _make_entry(key, created_at, expires_at, True)


def _entry_from_legacy_dict_string(raw_entry: Any) -> dict[str, Any] | None:
    if not isinstance(raw_entry, str) or "{" not in raw_entry or "}" not in raw_entry:
        return None

    key_match = re.search(r"['\"]?key['\"]?\s*:\s*['\"]([^'\"]+)['\"]", raw_entry, re.IGNORECASE)
    if not key_match:
        return None

    created_match = re.search(
        r"['\"]?created_at['\"]?\s*:\s*['\"]([^'\"]+)['\"]",
        raw_entry,
        re.IGNORECASE,
    )
    expires_match = re.search(
        r"['\"]?expires_at['\"]?\s*:\s*['\"]([^'\"]+)['\"]",
        raw_entry,
        re.IGNORECASE,
    )
    active_match = re.search(
        r"['\"]?active['\"]?\s*:\s*(true|false|1|0)",
        raw_entry,
        re.IGNORECASE,
    )

    now = _now()
    created_at = _parse_datetime(created_match.group(1)) if created_match else None
    expires_at = _parse_datetime(expires_match.group(1)) if expires_match else None
    active = True
    if active_match:
        active = active_match.group(1).lower() in {"true", "1"}

    return _make_entry(
        key_match.group(1),
        created_at or now,
        expires_at or now + timedelta(days=DEFAULT_EXPIRATION_DAYS),
        active,
    )


def _extract_raw_keys(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("keys"), list):
        return payload["keys"]
    return []


def _resolve_entry(key_or_entry: str | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(key_or_entry, dict):
        return _entry_from_raw(key_or_entry)
    return _find_entry(str(key_or_entry or ""))


def _find_entry(key: str) -> dict[str, Any] | None:
    clean_key = _clean_key(key)
    for entry in load_keys():
        if entry["key"] == clean_key:
            return entry
    return None


def _resolve_expiration(
    created_at: datetime,
    expiration_hours: int | None = None,
    expiration_days: int | None = None,
) -> datetime:
    hours = int(expiration_hours or 0)
    days = int(expiration_days or 0)

    if hours <= 0 and days <= 0:
        days = DEFAULT_EXPIRATION_DAYS

    return created_at + timedelta(hours=hours, days=days)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _format_datetime(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat()


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


def _write_payload(keys_file: Path, entries: Iterable[dict[str, Any]]) -> None:
    keys_file.parent.mkdir(parents=True, exist_ok=True)
    keys_file.write_text(
        json.dumps({"keys": list(entries)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
