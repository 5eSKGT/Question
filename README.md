# Crypto Trend Following · Bitget

VWAP / TITAN TRADING 스택과 **완전히 분리된** 별도 자동매매 시스템.
암호화폐 USDT 무기한 선물 시장에서 **급등(WINNER) / 급락(LOSER)** 종목을 전체
상장 심볼 중 스크리닝하여, **CVaR(기대 부족손실) 하한 강제** 아래에서 추세추종으로
수익을 추구합니다. **OOS 데이터로 자가 보정**하며, 보정 실패 시 자동으로 매매를
중단하고 사용자에게 경고를 띄웁니다.

---

## 핵심 설계

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
├── config.py                # .env 기반 설정 (paper/live 분기 포함)
├── exchange/bitget_client.py# BitgetClient (live) + PaperBroker (모의)
├── screener/winner_loser.py # 급등/급락 횡단면 스크리너
├── risk/cvar.py             # CVaR 추정 + 사이징 상한
├── strategy/trend_following.py   # 진입/청산 로직 + 신호 생성
├── oos/adaptive.py          # OOS 모니터·자가보정·중단
├── execution/engine.py      # 사이클 오케스트레이션
├── portfolio/state.py       # 자산/포지션/메시지 공유 상태
├── ui/charts.py             # 단일 SIGNAL_STYLE 테이블 → A/B/C 일관 랜더링
└── ui/app.py                # 밝은 Dash UI · 자동매매 전용
```

---

## 모드 분기

`.env`의 `TRADING_MODE` 값으로 선택합니다.

* `TRADING_MODE=paper` → `PaperBroker` 사용. Bitget의 시장 데이터는 그대로
  쓰지만 체결은 메모리 내 시뮬레이션. 수수료 6 bps 반영.
* `TRADING_MODE=live` → `BitgetClient`(ccxt) 사용. API 키 누락 시 부팅 자체 실패.

`build_broker()`가 둘 중 하나를 반환하므로 엔진/UI는 모드를 구분하지 않습니다.

---

## A / B / C 신호 차트 일관성 보장

`crypto_trend/ui/charts.py` 의 `SIGNAL_STYLE` 단일 테이블이 (side, type) →
marker(symbol, color, size) 를 결정합니다. Source(HIST/LIVE/OOS)는 legend
group과 OOS 강조용 외곽 링만 결정하며, 내부 마커는 어떤 그룹이든
동일합니다. 단위 테스트 `tests/test_charts.py` 가 이 invariant를 잠급니다.

* A · 과거에 매매했더라면 발생했을 + 실제로 진입/청산했던 신호 → `SignalSource.HIST`
* B · 현재 매매 중인 심볼의 진입/청산 → `SignalSource.LIVE`
* C · OOS 입력으로 인해 실시간 신규 트리거 → `SignalSource.OOS` (외곽 링 강조)

---

## UI 색상 규칙 (스펙 준수)

* 자산 / 거래 메시지: `+` → 녹색(#1faa59), `−` → 빨강(#d2474d), 평소 → 회색(#7a8085).
* 모드 뱃지: live는 빨간 톤, paper는 청록 톤.
* 모든 거래는 전략 트리거 자동 발생 — UI에는 수동 주문 버튼이 없습니다.
  사용자가 누를 수 있는 버튼은 **매매 중단 / 재개** 두 가지 뿐.

---

## 실행

```bash
pip install -r requirements.txt
cp .env.example .env          # 키 채우기
python -m crypto_trend.main   # 엔진(백그라운드 스레드) + UI(127.0.0.1:8050)
```

테스트:
```bash
pytest -q
```

---

## OOS 자가보정 워크플로우

1. 매 사이클 → `_run_oos_check()` → 최근 OHLCV로 시뮬레이션 손익을 재구성.
2. PSR ≥ 0.80 그리고 SR ≥ 0.5 만족 → `OK`.
3. 실패 시 `breakout_n × atr_n × chandelier_mult` 격자에서 후보 파라미터 탐색.
4. `RECALIBRATION_MAX_ATTEMPTS` 회 안에 통과하면 새 파라미터로 교체 + 메시지 표시.
5. 끝까지 실패하면 `RecalibrationFailed` → 엔진이 `portfolio.halt()` 호출 →
   UI 상단에 빨간 경고 배너 표시 + 거래 메시지에 `⚠ TRADING HALTED`.
