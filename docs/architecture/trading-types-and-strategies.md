# Trading Types and Strategies - Complete Reference

## Table of Contents
1. [Stock Trading Types](#stock-trading-types)
2. [Options Trading Fundamentals](#options-trading-fundamentals)
3. [Options Strategies](#options-strategies)
4. [Options Pricing Models](#options-pricing-models)
5. [The Greeks - Risk Metrics](#the-greeks)
6. [Profit/Loss Calculations](#profitloss-calculations)
7. [Risk Evaluation](#risk-evaluation)
8. [Tax Implications](#tax-implications)

---

## Stock Trading Types

### 1. Market Orders
- **Definition:** Execute immediately at best available price
- **Use Case:** High liquidity stocks, urgent execution needed
- **Risks:** Slippage in volatile markets
- **Implementation:** `OrderType::MARKET`

### 2. Limit Orders
- **Definition:** Execute only at specified price or better
- **Use Case:** Price-sensitive trades, illiquid stocks
- **Risks:** May not execute if price never reaches limit
- **Implementation:** `OrderType::LIMIT` with `limit_price`

### 3. Stop-Loss Orders
- **Definition:** Market order triggered when price reaches stop price
- **Use Case:** Risk management, protect profits
- **Risks:** Gap risk, slippage on execution
- **Implementation:** `OrderType::STOP_LOSS` with `stop_price`

### 4. Stop-Limit Orders
- **Definition:** Limit order triggered at stop price
- **Use Case:** Better price control than stop-loss
- **Risks:** May not execute if price gaps through limits
- **Implementation:** `OrderType::STOP_LIMIT` with `stop_price` and `limit_price`

### 5. Trailing Stop Orders
- **Definition:** Stop price trails market price by fixed percentage/amount
- **Use Case:** Lock in profits while allowing upside
- **Calculation:**
  ```
  stop_price = current_price - trailing_amount  (long position)
  stop_price = current_price + trailing_amount  (short position)
  ```

### 6. Time-in-Force Options
- **DAY:** Order expires end of trading day
- **GTC (Good-Till-Canceled):** Stays active until filled or canceled
- **IOC (Immediate-or-Cancel):** Fill immediately, cancel remainder
- **FOK (Fill-or-Kill):** Fill entire order immediately or cancel
- **GTD (Good-Till-Date):** Active until specified date

---

## Options Trading Fundamentals

### Option Basics

**Call Option:**
- Right (not obligation) to BUY stock at strike price
- Profitable when: Stock Price > Strike + Premium
- Maximum Loss: Premium paid
- Maximum Gain: Unlimited (stock can rise indefinitely)

**Put Option:**
- Right (not obligation) to SELL stock at strike price
- Profitable when: Stock Price < Strike - Premium
- Maximum Loss: Premium paid
- Maximum Gain: Strike - Premium (stock can't go below $0)

### Option Terminology

**Strike Price (K):** Price at which option can be exercised
**Premium (P):** Price paid for option contract
**Expiration:** Date when option expires
**Moneyness:**
- **ITM (In-The-Money):** Call: S > K, Put: S < K
- **ATM (At-The-Money):** S ≈ K
- **OTM (Out-of-The-Money):** Call: S < K, Put: S > K

**Intrinsic Value:**
```
Call: max(S - K, 0)
Put:  max(K - S, 0)
```

**Time Value:**
```
Time Value = Premium - Intrinsic Value
```

---

## Options Strategies

### 1. Single-Leg Strategies

#### Long Call
- **Construction:** Buy 1 call option
- **Max Profit:** Unlimited
- **Max Loss:** Premium paid
- **Break-Even:** Strike + Premium
- **When to Use:** Bullish outlook, limited capital
- **P/L Formula:**
  ```
  P/L = max(S_T - K, 0) - Premium
  ```

#### Long Put
- **Construction:** Buy 1 put option
- **Max Profit:** Strike - Premium
- **Max Loss:** Premium paid
- **Break-Even:** Strike - Premium
- **When to Use:** Bearish outlook, downside protection
- **P/L Formula:**
  ```
  P/L = max(K - S_T, 0) - Premium
  ```

#### Short Call (Covered or Naked)
- **Construction:** Sell 1 call option
- **Max Profit:** Premium received
- **Max Loss:** Unlimited (naked), Limited (covered)
- **Break-Even:** Strike + Premium
- **When to Use:** Neutral to bearish, generate income
- **Risk:** Assignment risk if ITM

#### Short Put (Cash-Secured)
- **Construction:** Sell 1 put option
- **Max Profit:** Premium received
- **Max Loss:** Strike - Premium
- **Break-Even:** Strike - Premium
- **When to Use:** Want to buy stock at lower price, generate income

### 2. Vertical Spreads

#### Bull Call Spread
- **Construction:**
  - Buy call at strike K1 (lower)
  - Sell call at strike K2 (higher), K2 > K1
- **Max Profit:** K2 - K1 - Net Debit
- **Max Loss:** Net Debit (Premium paid)
- **Break-Even:** K1 + Net Debit
- **When to Use:** Moderately bullish, reduce cost
- **P/L Formula:**
  ```
  P/L = min(max(S_T - K1, 0) - max(S_T - K2, 0), K2 - K1) - Net_Debit
  ```

#### Bear Put Spread
- **Construction:**
  - Buy put at strike K2 (higher)
  - Sell put at strike K1 (lower), K1 < K2
- **Max Profit:** K2 - K1 - Net Debit
- **Max Loss:** Net Debit
- **Break-Even:** K2 - Net Debit
- **When to Use:** Moderately bearish, reduce cost

#### Bull Put Spread (Credit Spread)
- **Construction:**
  - Sell put at K2 (higher)
  - Buy put at K1 (lower), K1 < K2
- **Max Profit:** Net Credit
- **Max Loss:** K2 - K1 - Net Credit
- **When to Use:** Neutral to bullish, generate income

#### Bear Call Spread (Credit Spread)
- **Construction:**
  - Sell call at K1 (lower)
  - Buy call at K2 (higher), K2 > K1
- **Max Profit:** Net Credit
- **Max Loss:** K2 - K1 - Net Credit
- **When to Use:** Neutral to bearish, generate income

### 3. Volatility Strategies

#### Long Straddle
- **Construction:**
  - Buy call at strike K
  - Buy put at same strike K
- **Max Profit:** Unlimited
- **Max Loss:** Total premiums paid
- **Break-Even:** K ± Total Premium
- **When to Use:** Expect large move, direction unknown
- **Volatility:** Benefits from increased IV
- **P/L Formula:**
  ```
  P/L = max(S_T - K, 0) + max(K - S_T, 0) - Total_Premium
      = |S_T - K| - Total_Premium
  ```

#### Short Straddle
- **Construction:**
  - Sell call at K
  - Sell put at K
- **Max Profit:** Total premiums received
- **Max Loss:** Unlimited
- **When to Use:** Expect low volatility
- **Risk:** HIGH - undefined risk

#### Long Strangle
- **Construction:**
  - Buy call at K2 (higher)
  - Buy put at K1 (lower), K1 < K2
- **Max Profit:** Unlimited
- **Max Loss:** Total premiums
- **Break-Even:** K1 - Premium, K2 + Premium
- **When to Use:** Expect large move, cheaper than straddle
- **P/L Formula:**
  ```
  P/L = max(S_T - K2, 0) + max(K1 - S_T, 0) - Total_Premium
  ```

#### Short Strangle
- **Construction:**
  - Sell call at K2
  - Sell put at K1
- **Max Profit:** Total premiums received
- **Max Loss:** Unlimited
- **When to Use:** Expect stock to stay range-bound
- **Risk:** HIGH - undefined risk both directions

#### Iron Condor
- **Construction:**
  - Bull put spread: Sell put K2, Buy put K1 (K1 < K2)
  - Bear call spread: Sell call K3, Buy call K4 (K3 < K4)
  - All different strikes: K1 < K2 < K3 < K4
- **Max Profit:** Net credit received
- **Max Loss:** Width of wider spread - Net credit
- **Break-Even:** K2 - Net Credit, K3 + Net Credit
- **When to Use:** Expect low volatility, range-bound
- **Sweet Spot:** Stock stays between K2 and K3

#### Iron Butterfly
- **Construction:**
  - Sell call and put at K (center strike)
  - Buy call at K2 (higher)
  - Buy put at K1 (lower)
- **Max Profit:** Net credit
- **Max Loss:** Strike width - Net credit
- **When to Use:** Expect stock to pin at strike K
- **Advantage:** Higher credit than iron condor

### 4. Calendar (Time) Spreads

#### Calendar Call Spread
- **Construction:**
  - Sell near-term call at K
  - Buy longer-term call at K (same strike)
- **Max Profit:** Depends on volatility changes
- **Max Loss:** Net debit paid
- **When to Use:** Neutral outlook, volatility play
- **Theta:** Benefits from time decay differential

#### Diagonal Spread
- **Construction:** Different strikes AND different expirations
- **Variation:** Diagonal call or diagonal put
- **When to Use:** Directional bias + time decay

### 5. Ratio Spreads

#### Call Ratio Spread
- **Construction:**
  - Buy 1 call at K1 (lower)
  - Sell 2+ calls at K2 (higher)
- **Max Profit:** Limited
- **Max Loss:** Unlimited upside
- **When to Use:** Moderately bullish, generate extra credit
- **Risk:** Naked upside exposure

#### Put Ratio Spread
- **Construction:**
  - Buy 1 put at K2 (higher)
  - Sell 2+ puts at K1 (lower)
- **Max Profit:** Limited
- **Max Loss:** Unlimited downside
- **When to Use:** Moderately bearish

### 6. Advanced Strategies

#### Butterfly Spread
- **Construction:**
  - Buy 1 call at K1 (lower)
  - Sell 2 calls at K2 (middle)
  - Buy 1 call at K3 (higher)
  - Equal spacing: K2 - K1 = K3 - K2
- **Max Profit:** K2 - K1 - Net Debit
- **Max Loss:** Net debit
- **When to Use:** Expect stock to stay near K2

#### Condor Spread
- **Construction:**
  - Buy call K1, Sell call K2, Sell call K3, Buy call K4
  - K1 < K2 < K3 < K4
- **Max Profit:** K2 - K1 - Net Debit
- **Max Loss:** Net debit
- **When to Use:** Expect stock between K2 and K3

#### Box Spread (Arbitrage)
- **Construction:**
  - Bull call spread + Bear put spread at same strikes
- **Theoretical Value:** K2 - K1 (discounted to present)
- **Use Case:** Riskless arbitrage if mispriced
- **Profit:** Difference from theoretical value

#### Collar
- **Construction:**
  - Own stock
  - Buy put at K1 (protection)
  - Sell call at K2 (K2 > K1)
- **Purpose:** Protect downside, cap upside
- **Cost:** Often zero or small debit (protective strategy)

#### Married Put
- **Construction:**
  - Own stock
  - Buy put for protection
- **Purpose:** Portfolio insurance
- **Cost:** Put premium

#### Protective Call
- **Construction:**
  - Short stock
  - Buy call for protection
- **Purpose:** Limit loss on short position

---

## Options Pricing Models

### 1. Black-Scholes Model

**Assumptions:**
- European options only
- No dividends
- Constant volatility
- Constant risk-free rate
- Log-normal price distribution

**Call Option Price:**
```
C = S₀ * N(d₁) - K * e^(-rT) * N(d₂)

where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T

N(x) = Cumulative standard normal distribution
S₀  = Current stock price
K   = Strike price
r   = Risk-free interest rate
T   = Time to expiration (years)
σ   = Volatility (annualized)
```

**Put Option Price:**
```
P = K * e^(-rT) * N(-d₂) - S₀ * N(-d₁)
```

**Put-Call Parity:**
```
C - P = S₀ - K * e^(-rT)
```

**Implementation:**
- File: `src/correlation_engine/black_scholes.cpp`
- Class: `BlackScholesModel`
- Methods: `callPrice()`, `putPrice()`

### 2. Black-Scholes-Merton (with Dividends)

**Call Price with Continuous Dividend:**
```
C = S₀ * e^(-qT) * N(d₁) - K * e^(-rT) * N(d₂)

where:
d₁ = [ln(S₀/K) + (r - q + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
q  = Continuous dividend yield
```

**Discrete Dividends:**
- Subtract PV of dividends from S₀
- `S₀_adjusted = S₀ - Σ(D_i * e^(-r*t_i))`

### 3. Binomial Tree Model

**Advantages:**
- American options (early exercise)
- Discrete dividends
- Time-varying parameters

**Algorithm:**
```
1. Build stock price tree:
   u = e^(σ√Δt)           # Up factor
   d = 1/u = e^(-σ√Δt)    # Down factor
   p = (e^(rΔt) - d)/(u - d)  # Risk-neutral probability

2. Calculate terminal payoffs:
   Call: max(S_T - K, 0)
   Put:  max(K - S_T, 0)

3. Backward induction:
   For each node from T-1 to 0:
     Option_value = e^(-rΔt) * [p * Value_up + (1-p) * Value_down]

     For American options:
     Option_value = max(Intrinsic_value, Continuation_value)
```

**Implementation:**
- File: `src/correlation_engine/binomial_tree.cpp`
- Class: `BinomialTreeModel`
- Steps: Typically 100-1000 for accuracy

### 4. Trinomial Tree Model

**Improvement over Binomial:**
- Three branches: up, middle, down
- Better convergence for barrier options
- More stable for American options

**Parameters:**
```
u = e^(λσ√Δt)
d = e^(-λσ√Δt)
m = 1

p_u = [(e^(rΔt/2) - e^(-λσ√Δt/2)) / (e^(λσ√Δt/2) - e^(-λσ√Δt/2))]²
p_d = [(e^(λσ√Δt/2) - e^(rΔt/2)) / (e^(λσ√Δt/2) - e^(-λσ√Δt/2))]²
p_m = 1 - p_u - p_d

λ = √(3/2) typically
```

**Implementation:**
- File: `src/correlation_engine/trinomial_tree.cpp`

### 5. Monte Carlo Simulation

**Use Cases:**
- Path-dependent options
- Multi-asset options
- Complex payoffs

**Algorithm:**
```
1. Generate N random price paths:
   S(t+Δt) = S(t) * exp((r - σ²/2)Δt + σ√Δt * Z)
   where Z ~ N(0,1)

2. Calculate payoff for each path
3. Average payoffs and discount:
   Option_price = e^(-rT) * mean(payoffs)

4. Standard error:
   SE = std_dev(payoffs) / √N
```

**Implementation:**
- File: `src/risk_management/monte_carlo.cpp`
- Paths: 10,000 to 1,000,000

---

## The Greeks - Risk Metrics

### Delta (Δ)

**Definition:** Rate of change of option price with respect to stock price

**Formulas:**
```
Call Delta: Δ_C = N(d₁) * e^(-qT)
Put Delta:  Δ_P = -N(-d₁) * e^(-qT) = (Δ_C - e^(-qT))

Range:
- Call: 0 to 1
- Put: -1 to 0
```

**Interpretation:**
- Delta = 0.50: Option moves $0.50 for $1 stock move
- Delta = 1.00: Option moves dollar-for-dollar with stock
- **Hedge Ratio:** Number of shares to hedge one option

**Position Delta:**
```
Portfolio_Delta = Σ(position_i * delta_i * shares_per_contract)
```

**Delta-Neutral Portfolio:** Total delta = 0

**Implementation:** `src/correlation_engine/greeks.cpp::calculateDelta()`

### Gamma (Γ)

**Definition:** Rate of change of delta with respect to stock price

**Formula:**
```
Γ = N'(d₁) / (S₀ * σ * √T) * e^(-qT)

where N'(x) = (1/√(2π)) * e^(-x²/2)

Same for calls and puts
```

**Range:** Always positive, 0 to ∞

**Interpretation:**
- Gamma = 0.05: Delta increases by 0.05 for $1 stock move
- **Highest at ATM near expiration**
- **Measures delta risk**

**Gamma Scalping:**
- Rebalance delta-neutral portfolio as stock moves
- Profit from realized volatility vs implied volatility

### Theta (Θ)

**Definition:** Rate of change of option price with respect to time (time decay)

**Formulas:**
```
Call Theta:
Θ_C = -[S₀ * N'(d₁) * σ * e^(-qT)] / (2√T)
      - r * K * e^(-rT) * N(d₂)
      + q * S₀ * N(d₁) * e^(-qT)

Put Theta:
Θ_P = -[S₀ * N'(d₁) * σ * e^(-qT)] / (2√T)
      + r * K * e^(-rT) * N(-d₂)
      - q * S₀ * N(-d₁) * e^(-qT)
```

**Units:** Dollars per day

**Interpretation:**
- Theta = -0.05: Option loses $0.05 per day (all else equal)
- **Always negative for long options**
- **Accelerates as expiration approaches**

**Theta Decay Pattern:**
```
Days to Expiration | Approximate Theta Acceleration
90+ days           | Slow decay
45-90 days         | Moderate decay
30 days            | Accelerating decay
< 7 days           | Rapid decay
```

### Vega (ν)

**Definition:** Rate of change of option price with respect to volatility

**Formula:**
```
ν = S₀ * √T * N'(d₁) * e^(-qT)

Same for calls and puts
```

**Units:** Dollars per 1% change in IV

**Interpretation:**
- Vega = 0.20: Option gains $0.20 if IV increases by 1%
- **Highest for ATM options**
- **Decreases as expiration approaches**

**Vega and IV:**
```
New_Price ≈ Old_Price + Vega * (New_IV - Old_IV)
```

### Rho (ρ)

**Definition:** Rate of change of option price with respect to interest rate

**Formulas:**
```
Call Rho: ρ_C = K * T * e^(-rT) * N(d₂)
Put Rho:  ρ_P = -K * T * e^(-rT) * N(-d₂)
```

**Units:** Dollars per 1% change in interest rate

**Interpretation:**
- Usually small impact (< 0.01 for most options)
- More significant for long-dated options (LEAPS)

### Minor Greeks

**Vomma (Volga):** Second derivative of value with respect to volatility
```
Vomma = Vega * (d₁ * d₂) / σ
```

**Vanna:** Sensitivity to changes in both stock price and volatility
```
Vanna = -N'(d₁) * d₂ / σ
```

**Charm (Delta Decay):** Rate of change of delta over time
```
Charm = -N'(d₁) * [2rT - d₂*σ√T] / (2T*σ√T)
```

**Implementation:** `src/correlation_engine/greeks.cpp`

---

## Implied Volatility

### Definition
The volatility value that makes the Black-Scholes price equal to market price

### Calculation Methods

#### 1. Newton-Raphson Method
```
σ_new = σ_old - (BS_Price(σ_old) - Market_Price) / Vega(σ_old)

Iterate until |BS_Price - Market_Price| < tolerance
```

**Implementation:**
```cpp
double calculateImpliedVolatility(
    double market_price,
    double S, double K, double r, double T,
    OptionType type,
    double initial_guess = 0.3,
    double tolerance = 1e-6,
    int max_iterations = 100
);
```

#### 2. Bisection Method
- More stable but slower
- Guaranteed convergence
- Bracket IV between [0.01, 5.0]

### IV Surface

**Volatility Smile:**
- IV varies by strike (for same expiration)
- Typically higher IV for OTM puts (crash premium)

**Volatility Term Structure:**
- IV varies by expiration (for same strike)
- Front-month usually higher (event risk)

**IV Surface Interpolation:**
- Bilinear interpolation
- Cubic spline
- SABR model calibration

**Implementation:** `src/correlation_engine/iv_surface.cpp`

---

## Profit/Loss Calculations

### Stock Trades

**Long Stock:**
```
P/L = (Sale_Price - Purchase_Price) * Shares
P/L% = (Sale_Price / Purchase_Price - 1) * 100
```

**Short Stock:**
```
P/L = (Short_Price - Cover_Price) * Shares
P/L% = (Short_Price / Cover_Price - 1) * 100
```

**With Commissions:**
```
P/L = (Sale_Price * Shares) - (Purchase_Price * Shares)
      - Commission_Buy - Commission_Sell
```

### Options Trades

**Single Option:**
```
P/L = (Exit_Price - Entry_Price) * Contracts * 100
      - Commission_Entry - Commission_Exit

Per Contract: 100 shares typically (US options)
```

**Multi-Leg Spread:**
```
P/L = Σ(leg_i_exit - leg_i_entry) * contracts * multiplier - commissions
```

**At Expiration:**
```
Call: P/L = max(S_T - K, 0) * 100 - Premium_Paid * 100 - Commissions
Put:  P/L = max(K - S_T, 0) * 100 - Premium_Paid * 100 - Commissions
```

### Return on Capital

**ROC for Options:**
```
ROC = (P/L / Capital_at_Risk) * 100

Capital at Risk:
- Debit spreads: Net debit
- Credit spreads: Max loss (width - credit)
- Naked positions: Margin requirement
```

**Annualized Return:**
```
Annualized_ROC = ROC * (365 / Days_in_Trade)
```

### Expected Value

**For probability-based strategies:**
```
EV = Σ(P(outcome_i) * P/L_i) - Costs

Example Iron Condor:
EV = P(stay_in_range) * Max_Profit
     + P(break_lower) * Loss_lower
     + P(break_upper) * Loss_upper
     - Commissions - Slippage
```

---

## Risk Evaluation

### 1. Position-Level Risk Metrics

**Maximum Loss:**
```
Debit Spreads: Net debit paid
Credit Spreads: Spread width - net credit
Naked Short: Unlimited (calls) or Strike price (puts)
```

**Maximum Profit:**
```
Debit Spreads: Spread width - net debit
Credit Spreads: Net credit received
Long Options: Unlimited (calls) or Strike price (puts)
```

**Break-Even Analysis:**
```
Find stock price(s) where P/L = 0
Multiple break-evens for complex strategies
```

**Probability of Profit (POP):**
```
Using delta as proxy:
POP_approx = 1 - |Delta|

More accurate: Use Monte Carlo or distribution analysis
```

### 2. Portfolio-Level Risk

**Greeks Aggregation:**
```
Portfolio_Delta = Σ(position_delta_i)
Portfolio_Gamma = Σ(position_gamma_i)
Portfolio_Theta = Σ(position_theta_i)
Portfolio_Vega  = Σ(position_vega_i)
```

**Beta-Weighted Greeks:**
```
Beta_weighted_delta = Σ(delta_i * beta_i * shares_i)

Accounts for correlation with market (SPX)
```

**Value at Risk (VaR):**
```
VaR_α = -Quantile(P/L_distribution, α)

α = confidence level (typically 95% or 99%)

Example: VaR_95% = $10,000
Means: 95% confident daily loss won't exceed $10,000
```

**Conditional VaR (CVaR/Expected Shortfall):**
```
CVaR_α = E[Loss | Loss > VaR_α]

Average loss when VaR is exceeded
```

### 3. Kelly Criterion

**Optimal Position Sizing:**
```
f* = (p * b - q) / b

where:
f* = Fraction of capital to risk
p  = Probability of winning
q  = Probability of losing (1 - p)
b  = Odds (profit/loss ratio)
```

**Example:**
```
Strategy: 60% win rate, 2:1 reward/risk
f* = (0.60 * 2 - 0.40) / 2 = 0.40 (40% of capital)

Conservative: Use f*/2 or f*/4 to reduce volatility
```

**Implementation:** `src/risk_management/kelly_criterion.cpp`

### 4. Sharpe Ratio

**Definition:** Risk-adjusted return

```
Sharpe = (R_p - R_f) / σ_p

where:
R_p = Portfolio return
R_f = Risk-free rate
σ_p = Portfolio standard deviation
```

**Interpretation:**
- < 1.0: Poor risk-adjusted returns
- 1.0-2.0: Good
- 2.0-3.0: Very good
- > 3.0: Excellent

### 5. Maximum Drawdown

```
MDD = max(Peak_value - Trough_value) / Peak_value

Example: Portfolio $100k → $70k → $110k
MDD = ($100k - $70k) / $100k = 30%
```

### 6. Buying Power Reduction (BPR)

**Margin Requirements:**

**Long Stock:**
```
BPR = Stock_Value * Margin_Rate
Typical: 50% for Reg T margin
```

**Short Stock:**
```
BPR = Stock_Value * (1 + Margin_Rate)
Typical: 150% of stock value
```

**Short Put (Cash-Secured):**
```
BPR = Strike * 100 - Premium_Received
```

**Short Put (Margin):**
```
BPR = max(
    Strike * 100 * 0.20 - OTM_amount + Premium,
    Strike * 100 * 0.10 + Premium
)
```

**Credit Spreads:**
```
BPR = (Spread_Width * 100) - Net_Credit
```

**Iron Condor:**
```
BPR = max(Put_Spread_Width, Call_Spread_Width) * 100 - Net_Credit
```

**Portfolio Margin (PM):**
- Risk-based calculation
- Typically 15-25% of notional for defined-risk strategies
- Requires $125k+ account

### 7. Correlation Risk

**Portfolio Correlation:**
```
ρ(X,Y) = Cov(X,Y) / (σ_X * σ_Y)

For options portfolio:
Consider correlations between:
- Underlying assets
- Volatility movements (VIX)
- Sector exposures
```

**Diversification Benefit:**
```
σ_portfolio = √(Σ Σ w_i * w_j * σ_i * σ_j * ρ_ij)

where:
w_i = weight of asset i
σ_i = volatility of asset i
ρ_ij = correlation between assets i and j
```

---

## Tax Implications (US Tax Code)

### Stock Trades

**Short-Term Capital Gains (< 1 year):**
- Taxed as ordinary income
- Rates: 10%, 12%, 22%, 24%, 32%, 35%, 37% (2025 brackets)

**Long-Term Capital Gains (≥ 1 year):**
- Preferential rates: 0%, 15%, 20%
- Additional 3.8% Net Investment Income Tax (NIIT) for high earners

**Calculation:**
```
Holding Period = Sale_Date - Purchase_Date

Short-term if < 365 days
Long-term if ≥ 365 days
```

### Options Trades

**General Rules:**
- Treated as capital assets
- Holding period starts when option is purchased/sold
- Cash settlement: realized on settlement date

**Long Options (Held to Expiration):**
```
If Exercised:
  Call: Stock basis = Strike + Premium
        Holding period starts at exercise

  Put:  Sale price = Strike - Premium
        Holding period from stock purchase to exercise

If Expires Worthless:
  Loss = Premium paid
  Type: Short-term (regardless of holding period)

If Sold Before Expiration:
  Gain/Loss = Sale - Purchase
  Type: Based on holding period
```

**Short Options:**
```
Premium received is not income until:
1. Option expires worthless → Short-term gain
2. Option is closed → Gain/loss based on closing price
3. Option is assigned → Adjusts stock basis
```

**Straddles/Strangles (Identified Straddle Rule):**
- If held > 30 days: Losses are deferred if offsetting position
- Mark-to-market at year-end for some strategies
- Complex rules - consult tax advisor

**Wash Sale Rule:**
```
If sell stock at loss and buy substantially identical:
- Within 30 days before sale, OR
- Within 30 days after sale

Then: Loss is disallowed, added to basis of new position

Applies to options on same underlying
```

### Section 1256 Contracts

**Broad-Based Index Options (SPX, RUT, NDX):**
- 60/40 treatment: 60% long-term, 40% short-term
- Mark-to-market at year-end
- More favorable tax treatment

**Tax Calculation:**
```
Profit on SPX option = $10,000

Tax (assuming 37% ordinary, 20% LT rate):
= (0.60 * $10,000 * 0.20) + (0.40 * $10,000 * 0.37)
= $1,200 + $1,480
= $2,680 effective tax

Vs. regular options (all short-term):
= $10,000 * 0.37 = $3,700

Savings: $1,020
```

### Trader Tax Status (TTS)

**Qualifications:**
- Trade substantially full-time
- Seek to profit from short-term price movements
- Trading is primary income source
- Significant trading activity (750+ trades/year guideline)

**Benefits if Qualified:**
- Mark-to-market accounting (elect Section 475(f))
- Deduct trading losses against ordinary income
- No wash sale rules
- Business expense deductions

**Requirements:**
- Must elect by April 15 of prior year
- File with CPA familiar with TTS
- Document trading as business

### Tax-Loss Harvesting

**Strategy:**
```
1. Sell losing positions before year-end
2. Realize losses to offset gains
3. Wait 31 days or buy different security
4. Re-enter position

Benefits:
- Offset up to $3,000 ordinary income
- Carry forward unlimited losses
```

**Optimization:**
```
Net short-term gains: Harvest short-term losses first
Net long-term gains: Harvest long-term losses first

Maximum tax benefit when:
loss_type_rate == gain_type_rate
```

### Record Keeping Requirements

**Must Track:**
1. Date acquired and date sold
2. Purchase price and sale price
3. Commissions and fees
4. Adjustments (splits, dividends, etc.)
5. Wash sales
6. Mark-to-market elections

**Form 8949:** Report all capital transactions
**Schedule D:** Summarize capital gains/losses

---

## Implementation in BigBrotherAnalytics

### Pricing Engine
```
src/correlation_engine/
├── options_pricing.cpp      # Main pricing interface
├── black_scholes.cpp        # BS model implementation
├── binomial_tree.cpp        # Binomial model (American options)
├── trinomial_tree.cpp       # Trinomial model
├── greeks.cpp               # All Greeks calculations
├── implied_volatility.cpp   # IV solver (Newton-Raphson)
└── iv_surface.cpp           # IV surface modeling
```

### Risk Management
```
src/risk_management/
├── position_sizer.cpp       # Kelly criterion, fixed fractional
├── stop_loss.cpp            # Stop-loss algorithms
├── portfolio_constraints.cpp # Position limits, concentration
├── kelly_criterion.cpp      # Optimal position sizing
└── monte_carlo.cpp          # MC simulation for VaR, CVaR
```

### Trading Strategies
```
src/trading_decision/
├── strategy_base.cpp        # Base class for all strategies
├── strategy_straddle.cpp    # Straddle/strangle strategies
├── strategy_strangle.cpp    # Optimized strangle entry/exit
├── strategy_volatility_arb.cpp # IV rank-based strategies
├── signal_aggregator.cpp    # Combine multiple signals
└── portfolio_optimizer.cpp  # Portfolio-level optimization
```

### Tax Engine
```
src/tax_engine/              # To be implemented
├── trade_classifier.cpp     # ST vs LT classification
├── wash_sale_detector.cpp   # Identify wash sales
├── tax_lot_optimizer.cpp    # FIFO, LIFO, specific ID
└── tax_calculator.cpp       # Estimate tax liability
```

---

## Volatility Trading Strategies (BigBrotherAnalytics Focus)

### IV Rank-Based Entry

**IV Rank Formula:**
```
IV_Rank = (Current_IV - IV_Low_52wk) / (IV_High_52wk - IV_Low_52wk) * 100
```

**Entry Rules:**
```
High IV Rank (> 50): Sell premium (strangles, iron condors)
  - Reason: IV likely to contract → theta/vega profit

Low IV Rank (< 20): Buy options (straddles, long calls/puts)
  - Reason: IV likely to expand → vega profit
```

### IV Percentile
```
IV_Percentile = (Days_IV_below_current / Total_days) * 100

More robust than IV Rank for trending volatility
```

### Expected Move Calculation

**1 Standard Deviation Move:**
```
Expected_Move = Stock_Price * IV * √(DTE/365)

Example:
Stock: $100, IV: 30%, DTE: 30 days
EM = $100 * 0.30 * √(30/365) = $8.59

~68% probability stock stays within $91.41 - $108.59
```

**Strangle Strike Selection:**
```
Lower_Strike = Stock_Price - Expected_Move
Upper_Strike = Stock_Price + Expected_Move

Typically choose strikes at 1 SD for ~16% POP per side
```

---

## Real-World Adjustments

### Slippage
```
Estimated_Slippage = Spread * Fill_Percentage

Wide spreads → higher slippage
Use mid-price or limit orders to control
```

### Commission Impact

**Example:**
```
Iron Condor:
Entry: 4 legs * $0.65 = $2.60
Exit:  4 legs * $0.65 = $2.60
Total: $5.20 per contract

If credit = $50, commissions = 10.4% of profit
```

**Break-Even Adjustment:**
```
Adjusted_Breakeven = Theoretical_BE ± (Commissions / 100)
```

### Assignment Risk

**Early Assignment Probability:**
- American options can be assigned anytime
- Most likely: Deep ITM, near ex-dividend date
- **Monitor:** Options with intrinsic value > time value

**Pin Risk:**
- Stock closes exactly at short strike on expiration
- Uncertain assignment
- **Mitigation:** Close positions before 3:30 PM ET on expiration

---

## Implementation Priority for BigBrotherAnalytics

### Phase 1: Core Pricing (IMPLEMENTED)
- ✅ Black-Scholes model
- ✅ Greeks calculation
- ✅ Implied volatility solver
- ✅ Binomial/Trinomial trees (stub)

### Phase 2: Risk Management (IN PROGRESS)
- ✅ Kelly criterion
- ✅ Monte Carlo simulation (stub)
- ⏳ Position sizing
- ⏳ Portfolio constraints

### Phase 3: Strategy Implementation (IN PROGRESS)
- ✅ Base strategy framework
- ⏳ Straddle/Strangle strategies
- ⏳ Volatility arbitrage
- ⏳ Signal aggregation

### Phase 4: Advanced Features (PLANNED)
- ⏳ IV surface modeling
- ⏳ Multi-asset correlation
- ⏳ Tax optimization
- ⏳ Transaction cost analysis

---

## References

1. **Options Pricing:**
   - Hull, John. "Options, Futures, and Other Derivatives" (11th Edition)
   - Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"

2. **Volatility Trading:**
   - Natenberg, Sheldon. "Option Volatility and Pricing"
   - Sinclair, Euan. "Volatility Trading"

3. **Risk Management:**
   - Taleb, Nassim. "Dynamic Hedging"
   - Thorp, Edward O. "The Kelly Capital Growth Investment Criterion"

4. **Tax:**
   - IRS Publication 550 (Investment Income and Expenses)
   - Green, Robert A. "Green's Trader Tax Guide"

---

## Toolchain Note

**This documentation supports the C++ implementation using:**
- **Clang 21.1.5** for C++23 features
- **OpenMP 21** for parallel Greeks calculations
- **OpenMPI 5.0.7** for distributed backtesting
- **Flang 21** for Fortran numerical libraries integration
- **BLAS/LAPACK** for matrix operations (correlation, optimization)
