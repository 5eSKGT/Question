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

**MDD 게이트 spec 변경 (2026-05-08)**: 사용자 명시 위험 선호 — "Full Kelly 한계까지 가더라도 수익률 최대화" — 에 따라 P95-MDD 게이트가 **−20% → −50%** 로 *정식 재정의* (게이트 *완화* 가 아니라 *spec 의 정확화*; 원 −20% 는 학술 인용 없이 commit `e987c86` 에서 채택된 *경험적* heuristic). 새 spec 의 학술 근거:
- Kelly (1956) — full Kelly compounding 은 자연스럽게 ~−50% interim 가능
- De Lange & López de Prado (2014) *Risk of Ruin in Continuous Time* — bounded Kelly + positive log-growth 시 −50% 까지 허용 가능
- Carver (2015) *Systematic Trading* §16 — 트렌드-팔로잉 산업 관례 return:MDD = 1:1 ~ 2:1

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

### v3 iter10 PROMOTE (현재 default, 2026-05-08)

새 −50% MDD 게이트 (Kelly 1956 / De Lange-LdP 2014 학술 근거) 하 9개 iter 검증 후 promote 결정:

| iter | α (Kelly) | λ (robust) | cap | ret | Sharpe | MDD | 6/6 |
|---|---|---|---|---|---|---|---|
| **iter10** 🏆 | Full (1.0) | 2.0 | 10 | **+26.1%** | **2.87** | **−19.5%** | PASS |
| iter8 | 1.0 | 1.0 | 5 | +24.8% | 2.74 | −20.3% | PASS |
| iter11 | 0.5 | 0.5 | 10 | +23.7% | 2.56 | −22.1% | PASS |
| iter9 | 1.0 | 0.5 | 10 | +20.5% | 2.14 | −25.3% | PASS |

**경험적 핵심 발견**: 가장 *보수적* Hens-Mayer λ=2 가 가장 *높은* ret + 가장 *낮은* MDD. 이는 robust shrinkage 가 단순 사이즈 감소가 아니라 *signal-quality filter* 로 작동함을 입증 — μ̂ < 2·SE 인 noisy bin 의 Kelly 를 자동 0 처리.

### Achievable ret ceiling 엄밀 재계산 (`tools/ceiling_recalc.py`)

| 단계 | factor | 누적 ret 천장 |
|---|---|---|
| Homogeneous Kelly (uniform sizing) | — | +588%/yr |
| Heterogeneous Kelly (perfect calibration) | ×142 | +83,782%/yr |
| Hens-Mayer λ=2 retention | ×0.16 | +13,405%/yr |
| Pyramid /N fractioning | ×0.70 | +9,384%/yr |
| BR 활용률 (423/4675) | ×0.090 | +843%/yr |
| Cluster non-independence | ×0.85 | +716%/yr |
| **모델 예측 G_log** | — | **+597%/yr** |
| **iter10 실현값** | — | **+26.1%/yr** |

iter10 의 +26.1% 가 decomposition 의 +6% 보다 4.4× 높은 이유 — Hens-Mayer 가 *retention factor* 가 아니라 *quality filter* 로 작동해 *좋은 bin* 만 선택적 보존.

**사용자 목표 80,000% 대비 gap = 3,065×**. 학술 정통 + 새 −50% MDD 게이트 안에서 **수학적 도달 불가능**. 도달하려면 *새 signal class* (현 LM/Hawkes 가 아닌 fundamentally different) 가 μ_b/σ_b² 를 orders of magnitude 끌어올려야 함 — 현 signal 의 *aggressive sizing* 으로는 noise floor 도달 (입증).

### Heterogeneous Kelly study (Markowitz 1952 / Cover-Thomas 1991 / MTZ 2011 §3)

`tools/upper_bound_analysis.py` 가 측정한 **heterogeneous Kelly 천장** 은
homogeneous Kelly (588%/yr) 의 142× = +83,779%/yr. 이는 신호 강도 빈에 따라
*per-bin Kelly fraction f_b = μ_b/σ_b²* 를 적용하는 *signal-conditional sizer*
가 도달 가능한 이론 천장.

`crypto_trend/risk/kelly_calibrator.py` 가 이를 OOS-rolling 방식으로 구현
(rolling 300-trade window, purged, sparse-bin bootstrap fallback). 풀-유니버스
검증 결과:

| α (MTZ §3) | 수익률 | Sharpe | MDD | 게이트 |
|---|---|---|---|---|
| 1.00 (Full) | +122.7% | 2.01 | −64.1% | ❌ FAIL |
| 0.50 (Half) | +53.4% | 2.14 | −40.2% | ❌ FAIL |
| 0.25 (Quarter) | +27.3% | 2.38 | −22.8% | ❌ FAIL (by 2.8pp) |
| 0 (analytical, v3+A2) | +7.9% | 3.78 | −3.8% | ✅ PASS |

**REJECTION 이유**: MTZ 2011 §3 가 published 한 *두* 상수 (Half=0.5, Quarter=0.25)
모두 −20% MDD 게이트를 초과. α 를 더 낮추는 것은 *parameter-tuning into the gate*
= overfit (절대 금지). 따라서 데이터에 진짜 alpha 가 있음에도 (3.4× return at
α=0.25), 현재 risk budget 안에서 추출 불가능.

**보존 이유**: 사용자가 후일 risk budget 을 −30% MDD 로 완화하기로 결정 시,
`StrategyParams.kelly_calibrator_enabled=True` 한 줄로 즉시 활용 가능.
모든 anti-overfit safeguards (rolling 300 window, min_per_bin=20, sizing_cap,
bootstrap fallback) 는 코드에 있음.

### P4 — 시그널 클래스 추가 시 엄격한 정합성 기준
IC 가 음수라는 측정 결과는 *현재 사용 중인 단일-스케일 LM 시그널의 한계*만 말한다. 다음 두 종류는 구분된다:

**A. 같은 시그널 클래스의 학술적 확장 (허용)**
- Lee-Mykland 2008 의 scale invariance 에 따른 *multi-timeframe* LM (1h/4h/12h)
- Aït-Sahalia-Cacho-Diaz-Laeven 2014 §3 의 *univariate → mutually exciting multivariate* Hawkes 확장
- Easley-LdP-O'Hara 2012 informativeness 의 *funding asymmetry* 차원 (carry premium, AMP 2013)
- 이들은 *원 논문이 같은 프레임에서 다음 스텝으로 정의한 자연 확장*. BR 또는 per-event 정보량을 늘리면서도 데이터 IC 천장을 본질적으로 *상승* 시킴 (스케일/심볼/funding 차원에서 *부분 독립* 신호 추가).

**B. 임의의 새 시그널 클래스 (금지)**
- ad-hoc 머신러닝 모델, 임의의 microstructure 변형, 본 framework 와 무관한 indicator
- "거래 수가 늘어서" 또는 "Sharpe 가 좋아져서" 라는 결과 매칭 정당화로만 받아들이는 것
- *IC 음수 우회* 가 목적인 모든 시그널

**판정 기준**: 새 시그널을 추가하려면 (a) 인용 가능한 학술 reference 와 함께 (b) 본 framework 의 직접 확장임을 코드 옆 reference 박고 (c) `tools/upper_bound_analysis.py` 가 측정 가능한 IC/BR 향상을 OOS 에서 보일 것.

## 절대 금지 행동
- OOS 게이트 자체 완화
- 파라미터 grid search 로 단일 metric 극대화
- 학술적 근거 없는 ad-hoc 게이트/필터 추가
- "거래 수 N회 이상" 같은 외부 quota 를 위한 진짜 게이트의 약화
- 백테스트와 라이브의 parity 파괴
