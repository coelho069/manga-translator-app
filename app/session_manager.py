from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


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


def get_sessions_file() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "sessions.json"


def ensure_sessions_file() -> Path:
    sessions_file = get_sessions_file()
    sessions_file.parent.mkdir(parents=True, exist_ok=True)

    if not sessions_file.exists():
        _write_payload(sessions_file, [])
        return sessions_file

    payload = _read_payload(sessions_file)
    sessions = _normalize_sessions(_extract_raw_sessions(payload))
    normalized_payload = {"sessions": sessions}
    if payload != normalized_payload:
        _write_payload(sessions_file, sessions)

    return sessions_file


def load_sessions() -> list[dict[str, Any]]:
    sessions_file = ensure_sessions_file()
    payload = _read_payload(sessions_file)
    return _normalize_sessions(_extract_raw_sessions(payload))


def save_sessions(sessions: Iterable[Any]) -> list[dict[str, Any]]:
    sessions_file = get_sessions_file()
    entries = _normalize_sessions(sessions)
    _write_payload(sessions_file, entries)
    return entries


def register_login(key: str, ip: str) -> dict[str, Any] | None:
    clean_key = _clean_key(key)
    if not clean_key:
        return None

    clean_ip = _clean_ip(ip)
    deactivate_stale_sessions()
    sessions = load_sessions()

    for session in sessions:
        if _matches_session(session, clean_key, clean_ip) and bool(session.get("active", False)):
            session["active"] = False

    now = _format_datetime(_now())
    entry = {
        "key": clean_key,
        "ip": clean_ip,
        "login_at": now,
        "last_seen": now,
        "active": True,
    }
    sessions.append(entry)
    save_sessions(sessions)
    return dict(entry)


def update_last_seen(key: str, ip: str) -> bool:
    clean_key = _clean_key(key)
    if not clean_key:
        return False

    clean_ip = _clean_ip(ip)
    deactivate_stale_sessions()
    sessions = load_sessions()
    now = _format_datetime(_now())

    for index in range(len(sessions) - 1, -1, -1):
        session = sessions[index]
        if _matches_session(session, clean_key, clean_ip) and bool(session.get("active", False)):
            session["last_seen"] = now
            session["active"] = True
            save_sessions(sessions)
            return True

    sessions.append(
        {
            "key": clean_key,
            "ip": clean_ip,
            "login_at": now,
            "last_seen": now,
            "active": True,
        }
    )
    save_sessions(sessions)
    return True


def logout_session(key: str, ip: str) -> bool:
    clean_key = _clean_key(key)
    if not clean_key:
        return False

    clean_ip = _clean_ip(ip)
    sessions = load_sessions()
    now = _format_datetime(_now())

    for index in range(len(sessions) - 1, -1, -1):
        session = sessions[index]
        if _matches_session(session, clean_key, clean_ip) and bool(session.get("active", False)):
            session["last_seen"] = now
            session["active"] = False
            save_sessions(sessions)
            return True

    return False


def list_sessions() -> list[dict[str, Any]]:
    deactivate_stale_sessions()
    return load_sessions()


def list_online_sessions() -> list[dict[str, Any]]:
    return [
        session
        for session in list_sessions()
        if bool(session.get("active", False))
    ]


def deactivate_stale_sessions(timeout_minutes: int = 5) -> int:
    sessions = load_sessions()
    cutoff = _now() - timedelta(minutes=int(timeout_minutes or 5))
    changed_count = 0

    for session in sessions:
        if not bool(session.get("active", False)):
            continue

        last_seen = _parse_datetime(session.get("last_seen"))
        if last_seen is None or last_seen < cutoff:
            session["active"] = False
            changed_count += 1

    if changed_count:
        save_sessions(sessions)

    return changed_count


def _matches_session(session: dict[str, Any], key: str, ip: str) -> bool:
    return session.get("key") == key and session.get("ip") == ip


def _normalize_sessions(sessions: Iterable[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for raw_session in sessions:
        entry = _session_from_raw(raw_session)
        if entry:
            normalized.append(entry)

    return normalized


def _session_from_raw(raw_session: Any) -> dict[str, Any] | None:
    if not isinstance(raw_session, dict):
        return None

    key = _clean_key(raw_session.get("key"))
    if not key:
        return None

    login_at = _format_optional_datetime(raw_session.get("login_at")) or _format_datetime(_now())
    last_seen = _format_optional_datetime(raw_session.get("last_seen")) or login_at

    return {
        "key": key,
        "ip": _clean_ip(raw_session.get("ip")),
        "login_at": login_at,
        "last_seen": last_seen,
        "active": bool(raw_session.get("active", False)),
    }


def _extract_raw_sessions(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
        return payload["sessions"]
    return []


def _now() -> datetime:
    return datetime.now().replace(microsecond=0)


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


def _read_payload(sessions_file: Path) -> Any:
    try:
        return json.loads(sessions_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Erro ao carregar data/sessions.json: {exc}")
        return {}


def _write_payload(sessions_file: Path, sessions: Iterable[dict[str, Any]]) -> None:
    sessions_file.parent.mkdir(parents=True, exist_ok=True)
    sessions_file.write_text(
        json.dumps({"sessions": list(sessions)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
