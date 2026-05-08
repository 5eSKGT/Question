"""Modal dialog that captures API credentials and immediately closes.

The dialog is the only place where credentials are visible in plaintext,
and even there each field starts in masked mode. Inputs are cleared on
close so they cannot be recovered through Qt's clipboard scrape or
window-screenshot tools later.

Each field carries inline help text — particularly important for the
Passphrase, which is a Bitget-specific concept that users frequently
confuse with their account login password.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFrame,
                                QHBoxLayout, QLabel, QLineEdit,
                                QPushButton, QVBoxLayout)

from .credentials import Credentials, CredentialStore
from .theme import ACCENT, ACCENT_DEEP, GRAY, RED, SUBTEXT


# --------------------------------------------------------------------------- #
# Field-level help text — kept here as constants so future tweaks live in one
# place. Each entry is (label, placeholder, help_html).
# --------------------------------------------------------------------------- #


HELP_KEY = (
    "API Key",
    "Bitget API 키 (영문/숫자)",
    "Bitget 에서 API 키를 만들 때 함께 발급되는 <b>공개 식별자</b>입니다. "
    "이 값만으로는 거래가 불가능하므로 Secret 보다 약간 덜 민감하지만, "
    "그래도 외부에 노출하지 마세요."
)

HELP_SECRET = (
    "API Secret",
    "API 키와 함께 발급된 비밀 키",
    "API 키와 한 쌍으로 발급되는 <b>요청 서명용 비밀 키</b>입니다. "
    "Bitget 화면에서 키를 만든 직후 한 번만 표시되고 다시 볼 수 없습니다. "
    "유출 시 즉시 Bitget 에서 해당 API 키를 삭제하고 새로 발급하세요."
)

HELP_PASSPHRASE = (
    "API Passphrase",
    "Bitget API 키 생성 시 본인이 직접 입력한 패스프레이즈",
    "<b>Bitget 계정 로그인 비밀번호와는 완전히 다른 별도 값입니다.</b><br>"
    "API 키를 처음 만들 때 폼에 직접 입력해 두는 <b>API 전용 패스프레이즈</b>로, "
    "Key·Secret 와 함께 매 요청 헤더에 동봉되어 <i>3-요소 인증</i>을 구성합니다.<br><br>"
    "• 위치 : Bitget 웹 → 우측 상단 프로필 → API 관리 → 키 생성 시<br>"
    "• 형식 : 영문/숫자, Bitget 정책상 8 ~ 32자<br>"
    "• 분실 : 복구 불가. 잊었다면 키를 삭제하고 새로 발급해야 합니다."
)


# --------------------------------------------------------------------------- #
class CredentialsDialog(QDialog):
    def __init__(self, parent, store: CredentialStore,
                 prefill: Credentials | None = None):
        super().__init__(parent)
        self.setWindowTitle("Bitget API 자격증명")
        self.setMinimumWidth(560)
        self.setModal(True)
        self.store = store

        v = QVBoxLayout(self)
        v.setContentsMargins(26, 22, 26, 22)
        v.setSpacing(10)

        title = QLabel("🔐 자격증명 설정")
        title.setStyleSheet("font-size:18px; font-weight:600; color:#1f2329;")
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

        # ---- three field blocks ------------------------------------- #
        self.key_edit, key_block = self._field_block(*HELP_KEY)
        self.secret_edit, secret_block = self._field_block(*HELP_SECRET)
        self.pass_edit, pass_block = self._field_block(
            *HELP_PASSPHRASE, emphasized=True)

        v.addWidget(key_block)
        v.addWidget(secret_block)
        v.addWidget(pass_block)

        if prefill is not None and prefill.is_complete:
            self.key_edit.setText(prefill.api_key)
            self.secret_edit.setText(prefill.api_secret)
            self.pass_edit.setText(prefill.api_passphrase)

        # ---- buttons ------------------------------------------------ #
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
    def _field_block(self, label: str, placeholder: str, help_html: str,
                     emphasized: bool = False) -> tuple[QLineEdit, QFrame]:
        """Build (input, container) for one credential field with inline help."""
        container = QFrame()
        # Highlight Passphrase block since it's the most-misunderstood field.
        if emphasized:
            container.setStyleSheet(
                "QFrame {"
                f"  background:#eef3ff;"
                f"  border:1px solid #c8d8ff;"
                "  border-radius:8px;"
                "  padding:8px 10px;"
                "}")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0 if not emphasized else 4,
                                   0 if not emphasized else 4,
                                   0 if not emphasized else 4,
                                   0 if not emphasized else 4)
        layout.setSpacing(4)

        title = QLabel(label)
        title.setStyleSheet(
            f"color:{ACCENT_DEEP if emphasized else GRAY}; "
            f"font-size:12px; font-weight:{'700' if emphasized else '600'};")
        layout.addWidget(title)

        edit = QLineEdit()
        edit.setEchoMode(QLineEdit.Password)
        edit.setPlaceholderText(placeholder)
        edit.setMinimumHeight(32)
        layout.addWidget(edit)

        help_lbl = QLabel(help_html)
        help_lbl.setWordWrap(True)
        help_lbl.setTextFormat(Qt.RichText)
        help_lbl.setStyleSheet(
            f"color:{SUBTEXT}; font-size:11.5px; padding:2px 0 0 0;")
        layout.addWidget(help_lbl)

        return edit, container

    # ------------------------------------------------------------------ #
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
