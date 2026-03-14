import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

os.makedirs("output_plots", exist_ok=True)

S0 = 100
K = 100
T = 1.0
r = 0.05
sigma = 0.20

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price, d1, d2

def bs_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))- r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    return {"Delta": delta, "Gamma": gamma, "Vega": vega,"Theta": theta, "Rho": rho}

bs_call, d1, d2 = black_scholes(S0, K, T, r, sigma, "call")
bs_put,  _,  _  = black_scholes(S0, K, T, r, sigma, "put")
greeks = bs_greeks(S0, K, T, r, sigma)
print("=" * 55)
print(" BLACK-SCHOLES ANALYTICAL RESULTS")
print("=" * 55)
print(f" Parameters: S={S0}, K={K}, T={T}yr, r={r*100}%, σ={sigma*100}%")
print(f" d1 = {d1:.4f}   d2 = {d2:.4f}")
print(f"\n Call Price : ${bs_call:.4f}")
print(f" Put Price : ${bs_put:.4f}")
print(f"\n Put-Call Parity Check:")
pcp = bs_call - bs_put
pcp_theory = S0 - K * np.exp(-r * T)
print(f" C - P (simulation): {pcp:.4f}")
print(f" S - K*e^(-rT): {pcp_theory:.4f} CORRECT" if abs(pcp - pcp_theory) < 0.001 else "MISMATCH")
print(f"\n Greeks (Call):")
for g, v in greeks.items():
    print(f"{g:6s}: {v:.6f}")

def monte_carlo_option(S0, K, T, r, sigma, n_paths=100000, option_type="call"):
    Z = np.random.standard_normal(n_paths)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    se = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    return price, se, S_T

print("\n" + "=" * 55)
print("MONTE CARLO RESULTS  (n = 100,000 paths)")
print("=" * 55)
mc_call, se_call, S_T= monte_carlo_option(S0, K, T, r, sigma, n_paths=100000, option_type="call")
mc_put,  se_put,  _  = monte_carlo_option(S0, K, T, r, sigma, n_paths=100000, option_type="put")
print(f" MC Call Price : ${mc_call:.4f} ±{1.96*se_call:.4f}  (95% CI)")
print(f" MC Put  Price : ${mc_put:.4f} ±{1.96*se_put:.4f}  (95% CI)")
print(f"\n Comparison (Call):")
print(f"Black-Scholes : ${bs_call:.4f}")
print(f"Monte Carlo   : ${mc_call:.4f}")
print(f"Error: ${abs(bs_call - mc_call):.4f} ({abs(bs_call - mc_call)/bs_call*100:.3f}%)")

steps = 252
dt = T / steps
plt.figure(figsize=(10, 5))
for _ in range(50):
    path = [S0]
    for _ in range(steps):
        dW = np.random.normal(0, np.sqrt(dt))
        path.append(path[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW))
    plt.plot(path, alpha=0.3, linewidth=0.7)
plt.axhline(K, color='red', linestyle='--', linewidth=1.5, label=f"Strike K = ${K}")
plt.title("Simulated GBM Price Paths (50 Trajectories)")
plt.xlabel("Trading Days")
plt.ylabel("Stock Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("output_plots/1_gbm_price_paths.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
Z = np.random.standard_normal(100000)
S_T_all = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
in_money = S_T_all[S_T_all > K]
out_money = S_T_all[S_T_all <= K]
plt.hist(out_money, bins=100, color='salmon', alpha=0.7, label="Out-of-the-money")
plt.hist(in_money,  bins=100, color='steelblue', alpha=0.8, label="In-the-money (Call payoff)")
plt.axvline(K, color='black', linestyle='--', linewidth=1.5, label=f"Strike K = ${K}")
plt.axvline(S_T_all.mean(), color='darkblue', linestyle=':', linewidth=1.5,label=f"Mean S_T = ${S_T_all.mean():.2f}")
plt.title("Terminal Stock Price Distribution at Expiry")
plt.xlabel("Stock Price at Expiry S_T ($)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("output_plots/2_terminal_distribution.png", dpi=150)
plt.close()

trial_sizes = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
mc_estimates = []
mc_errors    = []
for n in trial_sizes:
    est, se, _ = monte_carlo_option(S0, K, T, r, sigma, n_paths=n, option_type="call")
    mc_estimates.append(est)
    mc_errors.append(1.96 * se)

plt.figure(figsize=(10, 5))
plt.semilogx(trial_sizes, mc_estimates, 'o-', color='steelblue', label="MC Estimate")
plt.fill_between(trial_sizes,[e - err for e, err in zip(mc_estimates, mc_errors)],[e + err for e, err in zip(mc_estimates, mc_errors)],alpha=0.2, color='steelblue', label="95% Confidence Interval")
plt.axhline(bs_call, color='red', linestyle='--', linewidth=2,label=f"Black-Scholes = ${bs_call:.4f}")
plt.title("Monte Carlo Convergence to Black-Scholes Price (Call Option)")
plt.xlabel("Number of Simulated Paths (log scale)")
plt.ylabel("Estimated Call Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("output_plots/3_convergence_plot.png", dpi=150)
plt.close()

spot_range = np.linspace(60, 140, 200)
call_prices = [black_scholes(s, K, T, r, sigma, "call")[0] for s in spot_range]
put_prices  = [black_scholes(s, K, T, r, sigma, "put")[0]  for s in spot_range]
intrinsic_call = np.maximum(spot_range - K, 0)
intrinsic_put  = np.maximum(K - spot_range, 0)

plt.figure(figsize=(10, 5))
plt.plot(spot_range, call_prices, color='steelblue', linewidth=2, label="Call (BS Price)")
plt.plot(spot_range, put_prices,  color='coral', linewidth=2, label="Put (BS Price)")
plt.plot(spot_range, intrinsic_call,'--', color='steelblue', alpha=0.5, linewidth=1, label="Call Intrinsic")
plt.plot(spot_range, intrinsic_put, '--', color='coral', alpha=0.5, linewidth=1, label="Put Intrinsic")
plt.axvline(K, color='black', linestyle=':', linewidth=1, label=f"Strike K={K}")
plt.axvline(S0, color='gray', linestyle=':', linewidth=1, label=f"Current S={S0}")
plt.title("Black-Scholes Option Price vs Stock Price")
plt.xlabel("Stock Price ($)")
plt.ylabel("Option Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("output_plots/4_price_vs_spot.png", dpi=150)
plt.close()

spot_range2 = np.linspace(60, 140, 300)
deltas = []
gammas = []
vegas  = []
for s in spot_range2:
    g = bs_greeks(s, K, T, r, sigma)
    deltas.append(g["Delta"])
    gammas.append(g["Gamma"])
    vegas.append(g["Vega"])
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(spot_range2, deltas, color='steelblue', linewidth=2)
axes[0].axvline(K, color='red', linestyle='--', linewidth=1)
axes[0].set_title("Delta (Call)")
axes[0].set_xlabel("Stock Price ($)")
axes[0].set_ylabel("Delta")

axes[1].plot(spot_range2, gammas, color='mediumseagreen', linewidth=2)
axes[1].axvline(K, color='red', linestyle='--', linewidth=1)
axes[1].set_title("Gamma (Call)")
axes[1].set_xlabel("Stock Price ($)")
axes[1].set_ylabel("Gamma")

axes[2].plot(spot_range2, vegas, color='coral', linewidth=2)
axes[2].axvline(K, color='red', linestyle='--', linewidth=1)
axes[2].set_title("Vega (Call, per 1% vol)")
axes[2].set_xlabel("Stock Price ($)")
axes[2].set_ylabel("Vega")

plt.suptitle("Option Greeks vs Stock Price", fontsize=13)
plt.tight_layout()
plt.savefig("output_plots/5_greeks.png", dpi=150)
plt.close()

strikes = np.linspace(70, 130, 200)
for vol in [0.10, 0.20, 0.30, 0.40]:
    prices = [black_scholes(S0, k, T, r, vol, "call")[0] for k in strikes]
    plt.plot(strikes, prices, linewidth=1.8, label=f"σ = {int(vol*100)}%")
plt.axvline(S0, color='black', linestyle='--', linewidth=1, label=f"ATM (S={S0})")
plt.title("Call Option Price vs Strike — Sensitivity to Volatility")
plt.xlabel("Strike Price K ($)")
plt.ylabel("Call Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("output_plots/6_volatility_sensitivity.png", dpi=150)
plt.close()
