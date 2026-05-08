"""Modal dialog that captures API credentials and immediately closes.

The dialog is the only place where credentials are visible in plaintext,
and even there each field starts in masked mode. Inputs are cleared on
close so they cannot be recovered through Qt's clipboard scrape or
window-screenshot tools later.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout,
                                QLabel, QLineEdit, QPushButton, QVBoxLayout)

from .credentials import Credentials, CredentialStore
from .theme import ACCENT, GRAY, RED, SUBTEXT, SURFACE


class CredentialsDialog(QDialog):
    def __init__(self, parent, store: CredentialStore,
                 prefill: Credentials | None = None):
        super().__init__(parent)
        self.setWindowTitle("Bitget API 자격증명")
        self.setMinimumWidth(460)
        self.setModal(True)
        self.store = store

        v = QVBoxLayout(self)
        v.setContentsMargins(24, 22, 24, 22)
        v.setSpacing(10)

        title = QLabel("🔐 자격증명 설정")
        title.setStyleSheet(f"font-size:18px; font-weight:600; color:#1f2329;")
        v.addWidget(title)

        warn = QLabel(
            f"입력하신 키는 OS 보안 저장소"
            f"({store.backend_name})에 저장되며, 메인 화면에는 절대 표시되지 "
            f"않습니다. 이 창을 닫는 즉시 입력 필드는 비워집니다."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet(f"color:{SUBTEXT}; font-size:12px; "
                            f"background:#f3f6fa; padding:10px; border-radius:8px;")
        v.addWidget(warn)

        self.key_edit = self._secret_edit("Bitget API Key")
        self.secret_edit = self._secret_edit("API Secret")
        self.pass_edit = self._secret_edit("API Passphrase")

        for label_text, w in [("API Key", self.key_edit),
                               ("API Secret", self.secret_edit),
                               ("API Passphrase", self.pass_edit)]:
            lbl = QLabel(label_text)
            lbl.setStyleSheet(f"color:{GRAY}; font-size:12px;")
            v.addWidget(lbl)
            v.addWidget(w)

        if prefill is not None and prefill.is_complete:
            self.key_edit.setText(prefill.api_key)
            self.secret_edit.setText(prefill.api_secret)
            self.pass_edit.setText(prefill.api_passphrase)

        # Show / hide toggle (off by default)
        self.show_btn = QPushButton("👁 표시")
        self.show_btn.setObjectName("ghost")
        self.show_btn.setCheckable(True)
        self.show_btn.toggled.connect(self._toggle_visible)

        self.btn_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.btn_box.button(QDialogButtonBox.Save).setText("저장")
        self.btn_box.button(QDialogButtonBox.Cancel).setText("취소")
        self.btn_box.accepted.connect(self._on_save)
        self.btn_box.rejected.connect(self.reject)

        bar = QHBoxLayout()
        bar.addWidget(self.show_btn)
        bar.addStretch(1)
        bar.addWidget(self.btn_box)
        v.addLayout(bar)

        self.error_lbl = QLabel("")
        self.error_lbl.setStyleSheet(f"color:{RED}; font-size:12px;")
        self.error_lbl.hide()
        v.addWidget(self.error_lbl)

    # ------------------------------------------------------------------ #
    def _secret_edit(self, placeholder: str) -> QLineEdit:
        e = QLineEdit()
        e.setEchoMode(QLineEdit.Password)
        e.setPlaceholderText(placeholder)
        e.setMinimumHeight(32)
        return e

    def _toggle_visible(self, on: bool) -> None:
        mode = QLineEdit.Normal if on else QLineEdit.Password
        for w in (self.key_edit, self.secret_edit, self.pass_edit):
            w.setEchoMode(mode)

    # ------------------------------------------------------------------ #
    def _on_save(self) -> None:
        c = Credentials(
            api_key=self.key_edit.text().strip(),
            api_secret=self.secret_edit.text().strip(),
            api_passphrase=self.pass_edit.text().strip(),
        )
        if not c.is_complete:
            self.error_lbl.setText("세 필드 모두 입력해 주세요.")
            self.error_lbl.show()
            return
        self.store.save(c)
        c.wipe()
        self.accept()

    # ------------------------------------------------------------------ #
    def closeEvent(self, event):                                       # noqa: N802
        # Defense in depth: clear input fields before destruction.
        for w in (self.key_edit, self.secret_edit, self.pass_edit):
            w.clear()
            w.setEchoMode(QLineEdit.Password)
        super().closeEvent(event)
