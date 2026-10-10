"""
Pricing a call option on Brent crude oil expressed in euros.

A European buyer (airline, refiner) pays its oil in USD but earns EUR, so it
is exposed to two risks at once: the oil price and the dollar.

    Brent_EUR(t) = Brent_USD(t) * Z(t),      Z = value of 1 USD in EUR

Brent_USD and Z are two correlated geometric Brownian motions. The call on
Brent_EUR is priced by Monte Carlo and checked against Black-Scholes, since
the product of two lognormals is lognormal with volatility

    sigma_EUR = sqrt(sigma_oil^2 + sigma_fx^2 + 2 * rho * sigma_oil * sigma_fx)

Usage:  python brent_eur_option.py
"""
import numpy as np
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------- illustrative parameters
BRENT_USD = 80.0          # Brent spot, USD per barrel
EURUSD = 1.10             # USD per EUR  ->  Z0 = 1 / EURUSD
SIGMA_OIL = 0.35          # annual volatility of Brent in USD
SIGMA_FX = 0.08           # annual volatility of the dollar against the euro
RHO = -0.30               # correlation between Brent (USD) and the dollar's value
R_USD, R_EUR = 0.040, 0.025
MATURITY = 0.5            # 6 months
N_PATHS = 200_000
SEED = 42


def black_scholes_call(spot, strike, rate, sigma, maturity):
    """Black-Scholes price of a European call (no dividend)."""
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)


def sigma_eur(rho):
    """Volatility of the Brent price expressed in euros."""
    return np.sqrt(SIGMA_OIL**2 + SIGMA_FX**2 + 2 * rho * SIGMA_OIL * SIGMA_FX)


def monte_carlo_call(strike_eur, rho, rng):
    """Monte Carlo price in EUR of a call on Brent_EUR. Returns (price, std error).

    Simulated under the euro risk-neutral measure:
      - Z (a dollar, seen from Europe) drifts at r_eur - r_usd;
      - Brent_USD drifts at r_usd - rho * sigma_oil * sigma_fx (quanto correction),
    so that Brent_EUR = Brent_USD * Z drifts at r_eur, as any asset priced in euros.
    """
    t = MATURITY
    z1 = rng.standard_normal(N_PATHS)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(N_PATHS)   # correlated shocks

    drift_oil = R_USD - rho * SIGMA_OIL * SIGMA_FX
    oil = BRENT_USD * np.exp((drift_oil - 0.5 * SIGMA_OIL**2) * t + SIGMA_OIL * np.sqrt(t) * z1)
    fx = (1 / EURUSD) * np.exp((R_EUR - R_USD - 0.5 * SIGMA_FX**2) * t + SIGMA_FX * np.sqrt(t) * z2)

    payoff = np.exp(-R_EUR * t) * np.maximum(oil * fx - strike_eur, 0.0)
    return payoff.mean(), payoff.std(ddof=1) / np.sqrt(N_PATHS)


def main():
    rng = np.random.default_rng(SEED)
    spot_eur = BRENT_USD / EURUSD
    strike = round(spot_eur, 2)                      # at-the-money call, in EUR
    print(f"Brent = {BRENT_USD:.2f} USD = {spot_eur:.2f} EUR   strike = {strike:.2f} EUR   T = {MATURITY}y")

    mc, se = monte_carlo_call(strike, RHO, rng)
    bs = black_scholes_call(spot_eur, strike, R_EUR, sigma_eur(RHO), MATURITY)
    print(f"\nrho = {RHO:+.2f}   vol of Brent in EUR = {sigma_eur(RHO):.2%}")
    print(f"Monte Carlo   = {mc:.4f} EUR  (std error {se:.4f})")
    print(f"Black-Scholes = {bs:.4f} EUR  (difference {abs(mc - bs) / bs:.2%})")

    print("\n--- Effect of the oil / dollar correlation ---")
    rhos = np.array([-0.8, -0.5, -0.3, 0.0, 0.3, 0.5, 0.8])
    base = black_scholes_call(spot_eur, strike, R_EUR, sigma_eur(0.0), MATURITY)
    mc_prices, bs_prices = [], []
    for rho in rhos:
        p_mc, _ = monte_carlo_call(strike, rho, rng)
        p_bs = black_scholes_call(spot_eur, strike, R_EUR, sigma_eur(rho), MATURITY)
        mc_prices.append(p_mc)
        bs_prices.append(p_bs)
        print(f"rho = {rho:+.1f}   vol EUR = {sigma_eur(rho):6.2%}   MC = {p_mc:6.3f}   "
              f"BS = {p_bs:6.3f}   vs rho=0: {p_bs / base - 1:+6.1%}")

    fine = np.linspace(-0.9, 0.9, 50)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(fine, [black_scholes_call(spot_eur, strike, R_EUR, sigma_eur(r), MATURITY) for r in fine],
            label="Black-Scholes (closed form)")
    ax.scatter(rhos, mc_prices, color="tab:red", zorder=3, label="Monte Carlo")
    ax.set_xlabel("Correlation between Brent (USD) and the dollar")
    ax.set_ylabel("Call price (EUR per barrel)")
    ax.set_title("6-month ATM call on Brent in euros")
    ax.legend()
    fig.tight_layout()
    fig.savefig("price_vs_correlation.png", dpi=150)
    print("\nSaved price_vs_correlation.png")


if __name__ == "__main__":
    main()
