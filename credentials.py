"""Secrets (the optional Discogs token) in the operating system's credential store.

Windows Credential Manager or the macOS Keychain, through the `keyring`
package. Secrets are never written to settings.json. If no credential store
is available (keyring missing or no usable backend), nothing is saved: the
token works for the current session only.
"""

from __future__ import annotations

SERVICE = "RekordboxOrganizer"
DISCOGS_TOKEN = "discogs_token"


def _keyring():
    try:
        import keyring
        from keyring.backends import fail
        backend = keyring.get_keyring()
        if isinstance(backend, fail.Keyring) or getattr(backend, "priority", 1) <= 0:
            return None
        return keyring
    except Exception:          # not installed, or a broken backend
        return None


def available() -> bool:
    return _keyring() is not None


def get_secret(name: str) -> str:
    kr = _keyring()
    if kr is None:
        return ""
    try:
        return kr.get_password(SERVICE, name) or ""
    except Exception:
        return ""


def set_secret(name: str, value: str) -> bool:
    """Store (or, for an empty value, delete) a secret.

    True only if reading it back shows the store now holds exactly that value
    (so a caller can safely delete any other copy).
    """
    kr = _keyring()
    if kr is None:
        return False
    try:
        if value:
            kr.set_password(SERVICE, name, value)
        else:
            try:
                kr.delete_password(SERVICE, name)
            except Exception:
                pass            # nothing stored
        return (kr.get_password(SERVICE, name) or "") == value
    except Exception:
        return False
