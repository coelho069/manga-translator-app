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


def get_key_record(key: str) -> dict[str, Any] | None:
    deactivate_expired_keys()
    entry = _find_entry(key)
    return dict(entry) if entry else None


def redeem_key_for_ip(key: str, ip: str) -> bool:
    clean_key = _clean_key(key)
    clean_ip = _clean_ip(ip)
    entries = load_keys()

    for entry in entries:
        if entry["key"] != clean_key:
            continue

        if is_key_expired(entry) or not bool(entry.get("active", True)):
            if bool(entry.get("active", True)):
                entry["active"] = False
                save_keys(entries)
            return False

        redeemed_ips = _get_redeemed_ips(entry)
        if clean_ip in redeemed_ips:
            return True

        redeemed_ips.append(clean_ip)
        entry["redeemed"] = True
        entry["redeemed_at"] = _format_datetime(_now())
        entry["redeemed_ip"] = clean_ip
        entry["redeemed_ips"] = redeemed_ips
        save_keys(entries)
        return True

    return False


def is_key_allowed_for_ip(key: str, ip: str) -> bool:
    entry = get_key_record(key)
    if not entry:
        return False
    if is_key_expired(entry) or not bool(entry.get("active", True)):
        return False
    return _clean_ip(ip) in _get_redeemed_ips(entry)


def validate_key_for_ip(key: str, ip: str, allow_used_ip: bool = True) -> dict[str, Any]:
    deactivate_expired_keys()
    clean_key = _clean_key(key)
    clean_ip = _clean_ip(ip)
    entry = _find_entry(clean_key)

    if not entry:
        return {
            "valid": False,
            "message": "KEY inválida.",
            "key": clean_key,
            "ip": clean_ip,
            "record": None,
        }

    if is_key_expired(entry) or not bool(entry.get("active", True)):
        if bool(entry.get("active", True)):
            _set_key_fields(clean_key, active=False)
            entry = _find_entry(clean_key) or entry
        return {
            "valid": False,
            "message": "KEY expirada.",
            "key": clean_key,
            "ip": clean_ip,
            "record": dict(entry),
        }

    redeemed_ips = _get_redeemed_ips(entry)
    if clean_ip in redeemed_ips:
        if allow_used_ip:
            return {
                "valid": True,
                "message": "",
                "key": clean_key,
                "ip": clean_ip,
                "record": dict(entry),
            }
        return {
            "valid": False,
            "message": "Esta KEY já foi usada por este IP.",
            "key": clean_key,
            "ip": clean_ip,
            "record": dict(entry),
        }

    redeem_key_for_ip(clean_key, clean_ip)
    entry = _find_entry(clean_key) or entry
    return {
        "valid": True,
        "message": "",
        "key": clean_key,
        "ip": clean_ip,
        "record": dict(entry),
    }


def reset_key_redemption(key: str) -> bool:
    return _set_key_fields(
        key,
        redeemed=False,
        redeemed_at=None,
        redeemed_ip=None,
        redeemed_ips=[],
    )


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
            "redeemed": False,
            "redeemed_at": None,
            "redeemed_ip": None,
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
        "redeemed": bool(entry.get("redeemed", False)),
        "redeemed_at": entry.get("redeemed_at"),
        "redeemed_ip": entry.get("redeemed_ip"),
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


def _make_entry(
    key: Any,
    created_at: datetime,
    expires_at: datetime,
    active: bool,
    redeemed: bool = False,
    redeemed_at: Any = None,
    redeemed_ip: Any = None,
    redeemed_ips: Any = None,
) -> dict[str, Any]:
    clean_redeemed_at = _format_optional_datetime(redeemed_at) if redeemed else None
    clean_redeemed_ips = _clean_ip_list(redeemed_ips)
    clean_redeemed_ip = _clean_ip(redeemed_ip) if redeemed and redeemed_ip else None
    if clean_redeemed_ip and clean_redeemed_ip not in clean_redeemed_ips:
        clean_redeemed_ips.append(clean_redeemed_ip)
    return {
        "key": _clean_key(key),
        "created_at": _format_datetime(created_at),
        "expires_at": _format_datetime(expires_at),
        "active": bool(active),
        "redeemed": bool(redeemed),
        "redeemed_at": clean_redeemed_at,
        "redeemed_ip": clean_redeemed_ip,
        "redeemed_ips": clean_redeemed_ips if redeemed else [],
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
            bool(raw_entry.get("redeemed", False)),
            raw_entry.get("redeemed_at"),
            raw_entry.get("redeemed_ip"),
            raw_entry.get("redeemed_ips"),
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
    redeemed_match = re.search(
        r"['\"]?redeemed['\"]?\s*:\s*(true|false|1|0)",
        raw_entry,
        re.IGNORECASE,
    )
    redeemed_at_match = re.search(
        r"['\"]?redeemed_at['\"]?\s*:\s*['\"]([^'\"]+)['\"]",
        raw_entry,
        re.IGNORECASE,
    )
    redeemed_ip_match = re.search(
        r"['\"]?redeemed_ip['\"]?\s*:\s*['\"]([^'\"]+)['\"]",
        raw_entry,
        re.IGNORECASE,
    )

    now = _now()
    created_at = _parse_datetime(created_match.group(1)) if created_match else None
    expires_at = _parse_datetime(expires_match.group(1)) if expires_match else None
    active = True
    if active_match:
        active = active_match.group(1).lower() in {"true", "1"}
    redeemed = False
    if redeemed_match:
        redeemed = redeemed_match.group(1).lower() in {"true", "1"}

    return _make_entry(
        key_match.group(1),
        created_at or now,
        expires_at or now + timedelta(days=DEFAULT_EXPIRATION_DAYS),
        active,
        redeemed,
        redeemed_at_match.group(1) if redeemed_at_match else None,
        redeemed_ip_match.group(1) if redeemed_ip_match else None,
    )


def _get_redeemed_ips(entry: dict[str, Any]) -> list[str]:
    redeemed_ips = _clean_ip_list(entry.get("redeemed_ips"))
    legacy_ip = _clean_ip(entry.get("redeemed_ip"))
    if bool(entry.get("redeemed", False)) and legacy_ip and legacy_ip not in redeemed_ips:
        redeemed_ips.append(legacy_ip)
    return redeemed_ips


def _clean_ip_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    clean_ips: list[str] = []
    for raw_ip in value:
        clean_ip = _clean_ip(raw_ip)
        if clean_ip and clean_ip not in clean_ips:
            clean_ips.append(clean_ip)
    return clean_ips


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


def _set_key_fields(key: str, **fields: Any) -> bool:
    clean_key = _clean_key(key)
    entries = load_keys()

    for entry in entries:
        if entry["key"] != clean_key:
            continue
        entry.update(fields)
        save_keys(entries)
        return True

    return False


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


def _format_optional_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _format_datetime(value)
    parsed = _parse_datetime(value)
    if parsed is not None:
        return _format_datetime(parsed)
    text = str(value).strip()
    return text or None


def _clean_key(key: Any) -> str:
    text = unicodedata.normalize("NFKC", str(key or ""))
    for invisible_char in INVISIBLE_CHARS:
        text = text.replace(invisible_char, "")
    for dash_variant in DASH_VARIANTS:
        text = text.replace(dash_variant, "-")

    text = text.strip().strip(",")
    text = text.strip("\"'` ")
    return text.strip().upper()


def _clean_ip(ip: Any) -> str:
    text = str(ip or "").strip()
    return text or "local"


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
