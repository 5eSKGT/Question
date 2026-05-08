# AlphaPulse · Crypto Trend Following — Bitget

VWAP / TITAN TRADING 스택과 **완전히 분리된** 별도 자동매매 시스템.
암호화폐 USDT 무기한 선물 시장에서 **급등(WINNER) / 급락(LOSER)** 종목을
전체 상장 심볼 중 스크리닝하여, **CVaR(기대 부족손실) 하한 강제** 아래에서
추세추종으로 수익을 추구합니다. **OOS 데이터로 자가 보정**하며, 보정
실패 시 자동으로 매매를 중단하고 사용자에게 경고를 띄웁니다.

이름 "**AlphaPulse**" — sudden directional pulse(WINNER/LOSER)에서 알파를
추출한다는 전략 토픽에서 따왔습니다.

---

## 실행하는 세 가지 방법 — 가장 빠른 것부터

| 방법 | 첫 설정 | 매번 업데이트 | 시작 속도 | 언제 쓰는가 |
| --- | --- | --- | --- | --- |
| **A. 바탕화면 바로가기 + `git pull`** ⭐ | 30초 | **`git pull` 5초** | 즉시 | **개인 사용 / 일상 개발** |
| B. PyInstaller `--onedir` (폴더형 .exe) | 10–15분 | 1–2분 (캐시) | 즉시 | 다른 PC에 배포할 때 |
| C. PyInstaller `--onefile` (단일 .exe) | 10–15분 | **매번 10–15분** | 5–10초 압축해제 | 한 파일로 보내야 할 때만 |

### A. 바탕화면 바로가기 (가장 권장)

```powershell
cd C:\Users\82102\AlphaPulse
pip install -r requirements.txt                        # 한 번만
powershell -ExecutionPolicy Bypass -File tools\install_shortcut.ps1
```

바탕화면에 **AlphaPulse** 아이콘이 생깁니다. 더블클릭 → GUI 즉시 실행
(콘솔 창 안 뜸, AlphaPulse 아이콘 사용).

업데이트는:
```powershell
cd C:\Users\82102\AlphaPulse
git pull
```
끝. 5초. 더블클릭 다시 누르면 새 코드로 실행됩니다.

### B. 폴더형 빌드 (배포용, 빠른 증분)

```powershell
python build\build_exe.py             # 기본값 = onedir
```

`dist\AlphaPulse\AlphaPulse.exe` 가 생기고, 이 폴더를 통째로 다른 PC에
복사하면 Python 없이 실행됩니다. 두 번째 빌드부터는 PyInstaller 캐시
덕분에 1–2분만에 끝납니다.

### C. 단일 파일 빌드 (한 파일로 전달해야 할 때만)

```powershell
python build\build_exe.py --onefile   # 10-15분, 매번 풀 리빌드
```
용량이 큰 라이브러리(PySide6 QtWebEngine, scipy, numpy …) 를 매번 다시
압축하므로 시간이 오래 걸립니다. 이메일·USB 한 파일 전달용으로만 권장.

---

## 실행 스크립트 (수동)

| 파일 | 동작 |
| --- | --- |
| `run_alphapulse.bat` | `pythonw -m crypto_trend` (콘솔 숨김) — 바로가기가 가리키는 파일 |
| `run_alphapulse_debug.bat` | `python -m crypto_trend` (콘솔 표시) — 트러블슈팅용 |
| `run_alphapulse.ps1` | PowerShell 버전. `-Update` 시 git pull, `-Install` 시 pip install |

```powershell
.\run_alphapulse.ps1 -Update           # 받고 실행
.\run_alphapulse.ps1 -Update -Install  # 받고 deps 설치 후 실행
.\run_alphapulse.ps1 -Debug            # 콘솔 보면서 실행
```

---

## GUI 동작 — 한 창에서 전부

CLI 의존 없이 GUI 안에서 모든 동작이 가능합니다 — 모드 전환, API 키 입력,
전략 파라미터 조정, 엔진 시작/정지, 매매 중단/재개, 심볼 선택, 차트 보기,
거래 메시지 확인 모두 한 창에서. 헤더의 ⓘ 버튼으로 상세 설명/SVG 다이어그램
문서를 호출할 수 있습니다.

* 창 제목 : `AlphaPulse · Crypto Trend Following — Bitget`
* 아이콘  : `crypto_trend/desktop/assets/icon.ico` (인디고 + 캔들스틱 + 골든 라이트닝)
* 배경    : `crypto_trend/desktop/assets/background.png` (라이트 그라디언트 + 펄스 웨이브)

---

## 데스크톱 UI 구성

| 영역 | 내용 |
| --- | --- |
| 헤더 | 아이콘 + AlphaPulse 타이틀 + 모드 뱃지 (PAPER 청록 / LIVE 빨강) |
| 상단 카드 3종 | 자산 USDT(±색상 규칙) · OOS 게이트 상태(SR/PSR/DSR) · 매매 제어(중단/재개) |
| 좌측 컨트롤 | 모드 콤보, API 키/시크릿/패스프레이즈, 전략 파라미터(z·CVaR·레버리지·기준자본·사이클 주기), **엔진 시작/정지** |
| 중앙 차트 | 심볼 드롭다운 + 신호 차트 (A/B/C 일관 랜더링) |
| 우측 메시지 | 거래 메시지 색상 코드: +녹색 / -빨강 / 평소 회색 / halt 빨강 / 경고 황색 |
| 상태바 | 최근 사이클 결과 / 오류 메시지 |

수동 매매 버튼은 일부러 두지 않았습니다 — 모든 거래는 전략 트리거에 의해
자동 발생합니다 (요구 스펙 준수). 사용자가 누를 수 있는 것은 **엔진
시작/정지** 와 **매매 중단/재개** 뿐입니다.

---

## 핵심 설계 (학술적 근거)

| 영역 | 사용된 학술적 근거 |
| --- | --- |
| 스크리너 | Robust z-score (Huber 1981 / Rousseeuw & Croux 1993), Hurst R/S (Mandelbrot) |
| 변동성 | Yang–Zhang (2000) 드리프트 독립 OHLC 추정 |
| 진입 | Donchian 돌파 + 파동 밴드 내부 조건 |
| 청산 | Chandelier (Le Beau) ATR 트레일링 + 시간 스톱 + CVaR 위반 |
| 사이징 | Acerbi & Tasche (2002) CVaR + Cornish-Fisher 4-moment 확장 |
| OOS 게이트 | Probabilistic Sharpe Ratio & Deflated Sharpe Ratio (Bailey & López de Prado, 2012/2014) |

---

## 디렉토리

```
crypto_trend/
├── config.py                # .env 기반 설정 (런타임 변경 가능 — GUI에서 주입)
├── exchange/bitget_client.py# BitgetClient (live, ccxt) + PaperBroker (모의)
├── screener/winner_loser.py # 급등/급락 횡단면 스크리너
├── risk/cvar.py             # CVaR 추정 + 사이징 상한
├── strategy/trend_following.py
├── oos/adaptive.py          # OOS 모니터·자가보정·자동중단
├── execution/engine.py      # 사이클 오케스트레이션
├── portfolio/state.py       # 자산/포지션/메시지 공유 상태
├── ui/charts.py             # 단일 SIGNAL_STYLE 테이블 → A/B/C 일관 랜더링
├── ui/app.py                # (옵션) Dash 진단용 UI
└── desktop/                 # ← 데스크톱 GUI (배포용)
    ├── app.py               # QApplication 부팅
    ├── main_window.py       # 메인 윈도우 — E2E 컨트롤 패널
    ├── controls.py / chart_view.py / theme.py / workers.py
    └── assets/icon.ico icon.png background.png

build/
├── AlphaPulse.spec          # PyInstaller 스펙 (icon, datas, hiddenimports)
└── build_exe.py             # 한 명령으로 .exe 빌드

tools/
└── generate_assets.py       # 아이콘·배경 재생성
```

---

## 모드 분기

GUI에서 **거래 모드** 콤보로 선택. 내부적으로는 `config.apply(mode=...)` 가
싱글턴 `SETTINGS`를 변경하고, 엔진 시작 시 `build_broker()` 가 그 값을 보고
`BitgetClient`(live, ccxt) 또는 `PaperBroker`(모의)를 반환합니다. 엔진/UI는
모드 차이를 인지하지 않습니다.

`live` 선택 시 GUI는 즉시 **확인 다이얼로그**를 띄우고, API 자격증명 누락
시 시작을 차단합니다.

---

## A / B / C 신호 차트 일관성

`crypto_trend/ui/charts.py` 의 `SIGNAL_STYLE` 단일 테이블이 `(side, type) →
marker(symbol, color, size)` 를 결정합니다. Source(HIST/LIVE/OOS)는
legend group과 OOS 강조용 외곽 링만 결정하며, 내부 마커는 어떤 그룹이든
동일합니다. `tests/test_charts.py` 가 이 invariant를 잠굼.

* A · 과거 신호 + 실제 매매 기록 → `SignalSource.HIST`
* B · 현재 매매 중 → `SignalSource.LIVE`
* C · OOS 입력으로 실시간 신규 → `SignalSource.OOS` (외곽 링 강조)

데스크톱 GUI는 `ui/charts.build_signal_chart` 를 그대로 호출해 `QWebEngineView`
안에 plotly HTML로 띄우므로 — Dash UI와 **완전히 같은 픽셀**이 나옵니다.

---

## 자산 / 메시지 색상 규칙 (스펙 준수)

* 자산 텍스트 / 거래 메시지: `+` → 녹색(#1faa59), `−` → 빨강(#d2474d), 평소 → 회색(#7a8085).
* 모드 뱃지: live는 빨간 톤, paper는 청록 톤.
* OOS 상태: OK 녹색 / RECALIBRATED 황색 / HALTED 빨강.

---

## OOS 자가보정 워크플로우

1. 매 사이클 → `_run_oos_check()` → 최근 OHLCV로 시뮬레이션 손익 재구성.
2. PSR ≥ 0.80 그리고 SR ≥ 0.5 만족 → `OK`.
3. 실패 시 `breakout_n × atr_n × chandelier_mult` 격자 탐색.
4. `RECALIBRATION_MAX_ATTEMPTS` 안에 통과 → 새 파라미터로 교체 + 메시지.
5. 끝까지 실패 → `RecalibrationFailed` → `portfolio.halt()` →
   GUI 상단 알림 + 메시지 로그에 `⚠ TRADING HALTED` + 에러 다이얼로그.

---

## 테스트

```bash
pytest -q                    # 15 passed
```

* `test_cvar.py`     CVaR + Cornish-Fisher + 사이즈 상한
* `test_screener.py` robust-z, Hurst, 횡단면 스크리닝
* `test_strategy.py` Donchian 진입 신호
* `test_oos.py`      PSR / DSR / 자가보정·중단
* `test_charts.py`   A/B/C 마커 스타일 invariant
