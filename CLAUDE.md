# AlphaPulse — North Star (Absolute Design Criterion)

> **궁극적 목표 (절대 기준)**: 오버피팅 없이 타겟 대상(Bitget USDT-Perp WINNER/LOSER)에서 OOS 데이터에 따른 재보정 로직과 함께 수익률 최대화.

이 한 줄이 프로젝트가 완료될 때까지 모든 설계 결정의 절대적 기준이다. 다른 모든 지표·실험·튜닝은 이 기준에 *복무할 때만* 채택한다. 이 기준에 어긋나면 기술적으로 매력적이어도 reject 한다.

## 운영 원칙 (이 기준에서 직접 파생됨)

### 1. 모든 변경은 학술적 근거를 가져야 한다
- 새로운 필터·게이트·파라미터를 추가할 땐 *왜 이것이 데이터의 통계 구조에 의해 정당화되는지* 인용 가능한 논문/정리로 설명하고, 코드 옆 주석에 reference 박는다.
- "거래 수가 늘어서", "Sharpe가 보기 좋게 나와서" 같은 *결과 매칭* 정당화는 overfitting 의 정의 그 자체이므로 거부한다.

### 2. 검증은 OOS only
- production_validation.py 의 5-criterion gate (Sharpe, P95-MDD, Sortino vs Naive, trade count, win-rate) 와 ±20% perturbation robustness 가 PASS 라인이다.
- 이 게이트들은 *고정* — 게이트를 못 통과해서 게이트 자체를 낮추는 행위는 overfitting.
- 어떤 보정도 OOS에서 ±20% 파라미터 변동에 대해 robust 해야 promote 가능.

### 3. 정보 한계 (information-theoretic ceiling) 를 인정한다
- Grinold Fundamental Law: `IR = IC × √BR × TC`
- IC (정보 계수) 는 *데이터의* 성질이지 전략 튜닝으로 늘릴 수 없다.
- 우리가 늘릴 수 있는 건 BR (독립 베팅 수, Hawkes 클러스터 비중첩) 과 TC (실행/사이징 효율) 뿐.
- `tools/upper_bound_analysis.py` 가 OOS 측정한 IC × √BR × TC 가 도달 가능한 천장이다. 그 위는 fitting 이지 alpha 가 아니다.

### 4. 백테스팅 = 라이브 (parity invariant)
- backtest/simulator.py 와 strategy/trend_following.py + execution/engine.py 는 같은 게이트, 같은 사이즈 공식, 같은 OOS 재보정 로직을 호출해야 한다.
- 한쪽만 바꾸는 건 금지. parity 가 깨지면 OOS 재보정의 의미가 없어진다.

### 5. OOS 재보정 (HALT) 은 안전장치이지 성능 부스터가 아니다
- live/PSR/SR 게이트가 깨지면 HALT 가 트리거되어 모든 포지션을 닫는다.
- 이 로직 자체를 *완화* 해서 (예: PSR 기준을 0.55 → 0.40) "더 자주 거래하게" 만드는 것은 절대 금지 — 안전장치를 부수면서 수익률을 끌어올리는 것은 사용자의 본질적 자본 보호 의도와 정면 충돌한다.

## 진단 우선순위 (실측: reports/upper_bound_diagnostic.json)

`tools/upper_bound_analysis.py` 가 313심볼 1년 실데이터에서 측정한 결과:

| 측정 | raw screener | post-filter |
|---|---|---|
| n events/yr (BR) | 6,800 | 4,675 |
| **IC (purged 5-fold OOS)** | **−0.130** | **−0.086** |
| mean PnL / event | +0.025% | **+0.237%** (9× ↑) |
| PnL skew | +4.09 | **+4.61** |
| win rate | 36.7% | 37.8% |

**결정적 진단**: IC 가 음수인데 (점프 후 평균 회귀, Lo-MacKinlay 1990; De Bondt-Thaler 1985) 필터 통과 후 mean PnL 는 양수 → **수익의 원천은 시그널 검출이 아니라 execution mechanics** (3-ATR chandelier 의 비대칭 청산 + Conviction-Power Kelly 사이즈 비선형). 매우 두꺼운 오른쪽 꼬리(skew +4.6) 가 expectancy 를 양으로 만든다.

이 측정에 의해 못 박힌 보강 우선순위 (이 순서로만 진행):

### P1 — Exit management 정교화
chandelier_mult=3.0 고정값. 변동성/Hawkes-decay 조건부 적응형 chandelier 가 두꺼운 꼬리를 더 길게 잡을 수 있다. 학술 근거: Bandy (2014) §5; Aït-Sahalia-Cacho-Diaz-Laeven (2014) Hawkes 클러스터 종료 검출. **반드시 walk-forward OOS 게이트로 검증** — chandelier 튜닝은 가장 흔한 overfitting 경로.

### P2 — Position-sizing 비선형성 검증
`confidence_exponent=2` (k=2 quadratic). MacLean-Thorp-Ziemba (2011) §3 가 right-skewed PnL 분포에서 k∈[1.5, 2.5] 가 robust 함을 보임. 현재 값이 그 범위 안에 있어 위험은 작지만 OOS 측정으로 k 의 적정 범위를 *확인* 만 (조정은 신중히).

### P3 — Breadth utilisation
필터 통과 이벤트 4,675/yr 중 실제 trade = 332/yr → **7% 활용률**. Position-blocking + CVaR floor 가 4,343 이벤트를 reject. Faber (2007) pyramiding within cluster 같은 학술 정합 방안만 고려. 단, 동일 방향 누적 노출은 risk_per_trade 를 같은 비율로 분할해야 (Kelly 일치).

### P4 — DO NOT ADD NEW SIGNAL CLASSES
IC 가 음수이므로 새 시그널 클래스 (multi-timeframe LM, order-flow 등) 추가 = 데이터의 negative IC 를 우회하기 위한 사후 적합 = overfitting 의 정의. **금지**.

## 절대 금지 행동
- OOS 게이트 자체 완화
- 파라미터 grid search 로 단일 metric 극대화
- 학술적 근거 없는 ad-hoc 게이트/필터 추가
- "거래 수 N회 이상" 같은 외부 quota 를 위한 진짜 게이트의 약화
- 백테스트와 라이브의 parity 파괴
