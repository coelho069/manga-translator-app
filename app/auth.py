from __future__ import annotations

import streamlit as st

from app.key_manager import (
    add_key,
    deactivate_expired_keys,
    generate_key,
    get_key_status,
    get_remaining_time,
    is_admin_key,
    is_valid_key,
    list_admin_keys,
    list_keys,
    remove_key,
    _clean_key,
)


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def is_authenticated() -> bool:
    deactivate_expired_keys()
    key = str(st.session_state.get("auth_key", "")).strip()
    authenticated = bool(st.session_state.get("authenticated")) and bool(key)

    if not authenticated:
        return False

    if is_valid_key(key):
        return True

    logout()
    return False


def login(key: str) -> bool:
    raw_key = str(key or "")
    clean_key = _clean_key(raw_key)
    deactivate_expired_keys()
    status = get_key_status(raw_key)
    st.session_state["last_key_debug"] = {
        "key_digitada": raw_key,
        "expires_at": status["expires_at"],
        "tempo_restante": status["remaining_time"],
        "status": status["status"],
    }

    if status["exists"] and (status["expired"] or not status["active"]):
        st.session_state["login_error"] = "KEY expirada."
        return False

    valid = is_valid_key(raw_key)

    if not valid:
        st.session_state["login_error"] = "KEY invalida."
        return False

    st.session_state["authenticated"] = True
    st.session_state["auth_key"] = clean_key
    st.session_state.pop("login_error", None)
    return True


def logout() -> None:
    st.session_state.pop("authenticated", None)
    st.session_state.pop("auth_key", None)
    st.session_state.pop("login_error", None)
    st.session_state.pop("last_key_debug", None)


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
            st.error(st.session_state.get("login_error", "KEY invalida."))

    debug = st.session_state.get("last_key_debug")
    if debug:
        with st.expander("Debug temporario", expanded=True):
            st.json(debug)


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
        st.write(f"KEY ativa: `{st.session_state.get('auth_key', '')}`")

        if st.button("Sair", use_container_width=True):
            logout()
            _rerun()

        if not is_admin_key(st.session_state.get("auth_key", "")):
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
            key_col, remove_col = st.columns([0.72, 0.28])
            key_col.code(key, language="text")
            if key in admin_keys:
                remove_col.caption("Especial")
            else:
                if remove_col.button("Remover", key=f"remove_key_{key}", use_container_width=True):
                    removed = remove_key(key)
                    if removed and key == st.session_state.get("auth_key"):
                        logout()
                    _rerun()

            st.caption(
                f"Criada: {entry['created_at']} | "
                f"Expira: {entry['expires_at']} | "
                f"Restante: {get_remaining_time(entry)} | "
                f"Status: {status['status']}"
            )
