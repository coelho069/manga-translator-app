from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import streamlit as st

from app.key_manager import (
    add_key,
    deactivate_expired_keys,
    generate_key,
    get_key_status,
    get_remaining_time,
    is_admin_key,
    list_admin_keys,
    list_keys,
    remove_key,
    reset_key_redemption,
    validate_key_for_ip,
)
from app.session_manager import (
    deactivate_stale_sessions,
    ensure_sessions_file,
    list_online_sessions,
    list_sessions,
    logout_session,
    register_login,
    update_last_seen,
)


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def is_authenticated() -> bool:
    deactivate_expired_keys()
    ensure_sessions_file()
    deactivate_stale_sessions()
    key = str(st.session_state.get("user_key") or st.session_state.get("auth_key", "")).strip()
    current_ip = get_client_ip()
    authenticated = bool(st.session_state.get("authenticated")) and bool(key)

    if not authenticated:
        return False

    validation = validate_key_for_ip(key, current_ip)
    if validation["valid"]:
        st.session_state["auth_key"] = validation["key"]
        st.session_state["user_key"] = validation["key"]
        st.session_state["user_ip"] = current_ip
        update_last_seen(validation["key"], validation["ip"])
        return True

    logout()
    return False


def login(key: str) -> bool:
    raw_key = str(key or "")
    deactivate_expired_keys()
    current_ip = get_client_ip()
    validation = validate_key_for_ip(raw_key, current_ip)

    if not validation["valid"]:
        st.session_state["login_error"] = validation["message"]
        return False

    clean_key = validation["key"]
    st.session_state["authenticated"] = True
    st.session_state["auth_key"] = clean_key
    st.session_state["user_key"] = clean_key
    st.session_state["user_ip"] = current_ip
    register_login(clean_key, validation["ip"])
    st.session_state.pop("login_error", None)
    return True


def logout() -> None:
    current_key = str(st.session_state.get("user_key") or st.session_state.get("auth_key", "")).strip()
    current_ip = str(st.session_state.get("user_ip") or get_client_ip()).strip()
    if current_key:
        logout_session(current_key, current_ip)

    st.session_state.pop("authenticated", None)
    st.session_state.pop("auth_key", None)
    st.session_state.pop("user_key", None)
    st.session_state.pop("user_ip", None)
    st.session_state.pop("login_error", None)


def login_screen() -> None:
    deactivate_expired_keys()
    st.markdown("## Login por KEY")

    with st.form("key_login_form"):
        key = st.text_input("Digite sua KEY de acesso", type="password")
        submitted = st.form_submit_button("Entrar", use_container_width=True)

    if submitted:
        if login(key):
            st.success("KEY validada.")
            _rerun()
        else:
            st.error(st.session_state.get("login_error", "KEY inválida."))


def require_auth() -> None:
    deactivate_expired_keys()
    ensure_sessions_file()
    if is_authenticated():
        render_key_panel()
        return

    login_screen()
    st.stop()


def render_key_panel() -> None:
    with st.sidebar:
        current_key = st.session_state.get("user_key") or st.session_state.get("auth_key", "")
        st.markdown("### Acesso")
        st.write(f"KEY ativa: `{current_key}`")
        st.write(f"IP: `{st.session_state.get('user_ip', get_client_ip())}`")

        if st.button("Sair", use_container_width=True):
            logout()
            _rerun()

        admin_keys = set(list_admin_keys())
        if admin_keys and not is_admin_key(current_key):
            return

        st.divider()
        if not admin_keys:
            st.warning("Controle admin não encontrado. Painel exibido temporariamente.")

        render_online_users_panel()

        st.divider()
        st.markdown("### Painel de keys")
        if admin_keys:
            st.caption("Disponivel apenas para a key especial.")
        else:
            st.caption("Exibido temporariamente porque nenhuma key admin foi encontrada.")

        expiration_unit = st.selectbox("Expiração", ["horas", "dias"])
        expiration_value = st.number_input(
            "Quantidade",
            min_value=1,
            value=24 if expiration_unit == "horas" else 7,
            step=1,
        )
        st.caption("Exemplos: 1 hora, 24 horas, 7 dias, 30 dias.")

        if st.button("Gerar nova key", use_container_width=True):
            expiration_hours = int(expiration_value) if expiration_unit == "horas" else None
            expiration_days = int(expiration_value) if expiration_unit == "dias" else None
            new_key = generate_key(
                expiration_hours=expiration_hours,
                expiration_days=expiration_days,
            )
            add_key(
                new_key,
                expiration_hours=expiration_hours,
                expiration_days=expiration_days,
            )
            st.session_state["last_generated_key"] = new_key
            st.success(f"Nova key: {new_key}")
            _rerun()

        last_generated_key = st.session_state.get("last_generated_key")
        if last_generated_key:
            st.info(f"Ultima key gerada: {last_generated_key}")

        keys = list_keys()
        if not keys:
            st.warning("Nenhuma key cadastrada.")
            return

        st.markdown("#### Keys cadastradas")
        for entry in keys:
            key = entry["key"]
            status = get_key_status(key)
            key_col, reset_col, remove_col = st.columns([0.52, 0.24, 0.24])
            key_col.code(key, language="text")
            if reset_col.button("Resetar", key=f"reset_key_{key}", use_container_width=True):
                reset_key_redemption(key)
                if key == st.session_state.get("user_key"):
                    st.session_state.pop("user_ip", None)
                _rerun()
            if key in admin_keys:
                remove_col.caption("Especial")
            else:
                if remove_col.button("Remover", key=f"remove_key_{key}", use_container_width=True):
                    removed = remove_key(key)
                    if removed and key == (st.session_state.get("user_key") or st.session_state.get("auth_key")):
                        logout()
                    _rerun()

            st.caption(
                f"Status: {status['status']} | "
                f"Expira: {entry['expires_at']} | "
                f"Restante: {get_remaining_time(entry)} | "
                f"Resgatada: {entry.get('redeemed', False)} | "
                f"IP: {entry.get('redeemed_ip') or '-'} | "
                f"Resgatada em: {entry.get('redeemed_at') or '-'}"
            )


def render_online_users_panel() -> None:
    sessions = list_sessions()
    online_sessions = list_online_sessions()
    offline_sessions = [
        session
        for session in sessions
        if not bool(session.get("active", False))
    ]

    with st.expander("Usuários Online", expanded=False):
        online_col, total_col = st.columns(2)
        online_col.metric("Usuários online", len(online_sessions))
        total_col.metric("Sessões registradas", len(sessions))

        st.markdown("#### Sessões ativas")
        _render_session_table(online_sessions, "Nenhum usuário online.")

        st.markdown("#### Sessões offline")
        _render_session_table(offline_sessions, "Nenhuma sessão offline.")


def _render_session_table(sessions: list[dict[str, Any]], empty_message: str) -> None:
    if not sessions:
        st.caption(empty_message)
        return

    st.dataframe(
        [_session_display_row(session) for session in sessions],
        use_container_width=True,
        hide_index=True,
    )


def _session_display_row(session: dict[str, Any]) -> dict[str, str]:
    active = bool(session.get("active", False))
    return {
        "KEY": str(session.get("key", "")),
        "IP": str(session.get("ip", "")),
        "login_at": str(session.get("login_at", "")),
        "last_seen": str(session.get("last_seen", "")),
        "status": "online" if active else "offline",
        "tempo online": _format_online_time(session),
        "ultima atividade": _format_since(session.get("last_seen")),
    }


def _format_online_time(session: dict[str, Any]) -> str:
    login_at = _parse_session_datetime(session.get("login_at"))
    if login_at is None:
        return "-"

    if bool(session.get("active", False)):
        end_at = datetime.now().replace(microsecond=0)
    else:
        end_at = _parse_session_datetime(session.get("last_seen")) or login_at

    return _format_duration(end_at - login_at)


def _format_since(value: Any) -> str:
    parsed = _parse_session_datetime(value)
    if parsed is None:
        return "-"
    return _format_duration(datetime.now().replace(microsecond=0) - parsed)


def _format_duration(delta) -> str:
    total_seconds = max(0, int(delta.total_seconds()))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _parse_session_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def get_client_ip() -> str:
    header_names = (
        "x-forwarded-for",
        "x-real-ip",
        "cf-connecting-ip",
        "true-client-ip",
        "x-client-ip",
        "forwarded",
    )

    headers = _get_streamlit_headers()
    for header_name in header_names:
        value = _get_header(headers, header_name)
        if not value:
            continue
        ip = _extract_ip_from_header(header_name, value)
        if ip:
            return ip

    return "local"


def _get_streamlit_headers():
    try:
        return st.context.headers
    except Exception:
        pass

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        context = get_script_run_ctx()
        request = getattr(context, "request", None) if context else None
        return getattr(request, "headers", None)
    except Exception:
        return None


def _get_header(headers, name: str) -> str:
    if not headers:
        return ""

    candidates = (name, name.lower(), name.upper(), "-".join(part.capitalize() for part in name.split("-")))
    for candidate in candidates:
        try:
            value = headers.get(candidate)
        except AttributeError:
            try:
                value = headers[candidate]
            except (KeyError, TypeError):
                value = None
        if value:
            return str(value).strip()
    return ""


def _extract_ip_from_header(header_name: str, value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    if header_name == "forwarded":
        match = re.search(r"for=\"?([^;,\"\s]+)", text, flags=re.IGNORECASE)
        if match:
            text = match.group(1)

    if "," in text:
        text = text.split(",", 1)[0]

    return text.strip().strip("\"[]") or ""
