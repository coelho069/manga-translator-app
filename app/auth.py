from __future__ import annotations

import re

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


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def is_authenticated() -> bool:
    deactivate_expired_keys()
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
    st.session_state.pop("login_error", None)
    return True


def logout() -> None:
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
    if is_authenticated():
        render_key_panel()
        return

    login_screen()
    st.stop()


def render_key_panel() -> None:
    with st.sidebar:
        st.markdown("### Acesso")
        st.write(f"KEY ativa: `{st.session_state.get('user_key') or st.session_state.get('auth_key', '')}`")
        st.write(f"IP: `{st.session_state.get('user_ip', get_client_ip())}`")

        if st.button("Sair", use_container_width=True):
            logout()
            _rerun()

        if not is_admin_key(st.session_state.get("user_key") or st.session_state.get("auth_key", "")):
            return

        st.divider()
        st.markdown("### Painel de keys")
        st.caption("Disponivel apenas para a key especial.")

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
        admin_keys = set(list_admin_keys())
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
