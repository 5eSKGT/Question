"""Secure storage for Bitget API credentials.

Design contract:
  * Credentials NEVER appear in the main window after entry. The
    configuration dialog masks all three fields, the inputs are wiped
    on dialog close, and the main window shows only a "configured /
    not configured" status indicator.
  * Persistence uses the OS-native secret store via the ``keyring``
    package — Windows Credential Manager, macOS Keychain, or Linux
    Secret Service (libsecret / KWallet via DBus).
  * If the OS keyring is unavailable, credentials live only in
    process memory until the user explicitly enters them again.
    They are not persisted to disk in plain form.
  * The store is loaded on demand at engine-start time and cleared
    from memory when the engine stops.
"""
from __future__ import annotations

from dataclasses import dataclass

SERVICE = "AlphaPulse-Bitget"
USER_KEY = "api_key"
USER_SECRET = "api_secret"
USER_PASS = "api_passphrase"


@dataclass
class Credentials:
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""

    @property
    def is_complete(self) -> bool:
        return bool(self.api_key and self.api_secret and self.api_passphrase)

    def wipe(self) -> None:
        # Best-effort overwrite to shorten the lifetime of secrets
        self.api_key = ""
        self.api_secret = ""
        self.api_passphrase = ""


class CredentialStore:
    """Facade over the OS keyring with an in-memory fallback.

    All instance methods are no-throw: failures are swallowed and the
    GUI falls back to the in-memory copy.  This keeps the user from
    being locked out because of a transient keyring issue.
    """

    def __init__(self) -> None:
        # We catch BaseException, not just Exception, because some platforms
        # (notably broken sandboxes with pyo3-based cryptography) can panic
        # *during* the keyring import. We must not let that crash the GUI.
        self._kr = None
        try:
            import keyring                                   # type: ignore
            keyring.get_keyring()
            self._kr = keyring
        except BaseException:                                 # noqa: BLE001
            self._kr = None
        self._mem: dict[str, str] = {}

    @property
    def keyring_available(self) -> bool:
        return self._kr is not None

    @property
    def backend_name(self) -> str:
        if self._kr is None:
            return "in-memory (OS keyring unavailable)"
        try:
            return type(self._kr.get_keyring()).__name__
        except Exception:                                     # noqa: BLE001
            return "keyring"

    # ------------------------------------------------------------------ #
    def load(self) -> Credentials:
        if self._kr is not None:
            try:
                return Credentials(
                    api_key=self._kr.get_password(SERVICE, USER_KEY) or "",
                    api_secret=self._kr.get_password(SERVICE, USER_SECRET) or "",
                    api_passphrase=self._kr.get_password(SERVICE, USER_PASS) or "",
                )
            except Exception:                                 # noqa: BLE001
                pass
        return Credentials(
            api_key=self._mem.get(USER_KEY, ""),
            api_secret=self._mem.get(USER_SECRET, ""),
            api_passphrase=self._mem.get(USER_PASS, ""),
        )

    def save(self, c: Credentials) -> None:
        if self._kr is not None:
            try:
                self._kr.set_password(SERVICE, USER_KEY, c.api_key)
                self._kr.set_password(SERVICE, USER_SECRET, c.api_secret)
                self._kr.set_password(SERVICE, USER_PASS, c.api_passphrase)
                return
            except Exception:                                 # noqa: BLE001
                pass
        self._mem = {
            USER_KEY: c.api_key,
            USER_SECRET: c.api_secret,
            USER_PASS: c.api_passphrase,
        }

    def clear(self) -> None:
        if self._kr is not None:
            for u in (USER_KEY, USER_SECRET, USER_PASS):
                try:
                    self._kr.delete_password(SERVICE, u)
                except Exception:                             # noqa: BLE001
                    pass
        self._mem = {}

    def is_configured(self) -> bool:
        return self.load().is_complete
