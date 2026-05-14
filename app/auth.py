from __future__ import annotations

import streamlit as st

from app.key_manager import (
    add_key,
    generate_key,
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
    valid = is_valid_key(raw_key)

    if not valid:
        return False

    st.session_state["authenticated"] = True
    st.session_state["auth_key"] = clean_key
    return True


def logout() -> None:
    st.session_state.pop("authenticated", None)
    st.session_state.pop("auth_key", None)


def login_screen() -> None:
    st.markdown("## Login por KEY")

    with st.form("key_login_form"):
        key = st.text_input("Digite sua KEY de acesso", type="password")
        submitted = st.form_submit_button("Entrar", use_container_width=True)

    if submitted:
        if login(key):
            st.success("KEY validada.")
            _rerun()
        else:
            st.error("KEY invalida.")


def require_auth() -> None:
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

        if st.button("Gerar nova key", use_container_width=True):
            new_key = generate_key()
            add_key(new_key)
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
        for key in keys:
            key_col, remove_col = st.columns([0.72, 0.28])
            key_col.code(key, language="text")
            if key in admin_keys:
                remove_col.caption("Especial")
                continue
            if remove_col.button("Remover", key=f"remove_key_{key}", use_container_width=True):
                removed = remove_key(key)
                if removed and key == st.session_state.get("auth_key"):
                    logout()
                _rerun()
