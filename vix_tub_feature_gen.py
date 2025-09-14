import pandas as pd
import matplotlib.pyplot as plt

# Reload files after reset
spx = pd.read_csv("./output/SPX.csv", parse_dates=["date"]).set_index("date").sort_index()
probs = pd.read_csv("./outputs/daily_probabilities.csv", parse_dates=["date"]).set_index("date").sort_index()


# Load SPX_VIX file
spx_vix = pd.read_csv("./outputs/spx_vix.csv", parse_dates=["date"]).set_index("date").sort_index()


# Load Turbulence file
turb = pd.read_csv("./outputs/turbulence_sector.csv", parse_dates=["date"]).set_index("date").sort_index()
# Align time range 2013-01-01 to 2024-12-31

start, end = pd.Timestamp("2013-01-01"), pd.Timestamp("2024-12-31")
spx = spx.loc[start:end]
probs = probs.loc[start:end]
spx_vix = spx_vix.loc[start:end]
turb = turb.loc[start:end]

# ========== Feature Engineering for VIX and Turbulence ==========
# ΔTurbulence, ΔVIX
turb["delta"] = turb["turbulence"].diff()
spx_vix["delta"] = spx_vix["close"].diff()

# ZScore (rolling mean/std, e.g., 252d = 1 year)
window = 252
turb["zscore"] = (turb["turbulence"] - turb["turbulence"].rolling(window).mean()) / turb["turbulence"].rolling(window).std()
spx_vix["zscore"] = (spx_vix["close"] - spx_vix["close"].rolling(window).mean()) / spx_vix["close"].rolling(window).std()

# ΔZScore
turb["delta_zscore"] = (turb["delta"] - turb["delta"].rolling(window).mean()) / turb["delta"].rolling(window).std()
spx_vix["delta_zscore"] = (spx_vix["delta"] - spx_vix["delta"].rolling(window).mean()) / spx_vix["delta"].rolling(window).std()


# ========== Export ==========
# align SPX
spx_close = spx[["close"]].rename(columns={"close":"spx_close"})


# Turbulence output
turb_out = turb[["turbulence","delta","zscore","delta_zscore"]].join(spx_close, how="left")
turb_out.to_csv("./outputs/turb_z.csv")

# VIX output
vix_out = spx_vix[["close","delta","zscore","delta_zscore"]].join(spx_close, how="left")
vix_out.to_csv("./outputs/vix_z.csv")

# Compute cumulative log return for SPX (if close price is given)
if "close" in spx.columns:
    spx_ret = spx["close"].pct_change().fillna(0)
    spx_cum = (1 + spx_ret).cumprod()
else:
    spx_cum = spx.iloc[:,0]  # fallback first column

# Plot
fig, ax1 = plt.subplots(figsize=(14,6))

# SPX on left axis
ax1.plot(spx_cum.index, spx_cum.values, color="orange", label="SPX (Cumulative Return)")
ax1.set_ylabel("SPX Cumulative Return", color="orange")
ax1.tick_params(axis="y", labelcolor="orange")

# Probabilities on right axis
ax2 = ax1.twinx()
if "p_xgb" in probs.columns:
    ax2.plot(probs.index, probs["p_xgb"], color="blue", alpha=0.7, label="P(Event) - XGB")
#if "p_lgb" in probs.columns:
#    ax2.plot(probs.index, probs["p_lgb"], color="green", alpha=0.7, label="P(Event) - LGB")
ax2.set_ylabel("Probability of Event", color="black")
ax2.tick_params(axis="y", labelcolor="black")

# Title and legend
fig.suptitle("SPX vs Probability of Event (2013-2024)", fontsize=14)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

# ---------- Figure 2: SPX + VIX ----------
fig, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(spx.index, spx["close"], color="orange", label="SPX")
ax1.set_ylabel("SPX Close", color="orange")
ax1.tick_params(axis="y", labelcolor="orange")

ax2 = ax1.twinx()
ax2.plot(spx_vix.index, spx_vix["close"], color="red", alpha=0.7, label="VIX")
ax2.set_ylabel("VIX Close", color="red")
ax2.tick_params(axis="y", labelcolor="red")

fig.suptitle("SPX vs VIX (2013-2024)", fontsize=14)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ---------- Figure 3: SPX + Turbulence ----------
fig, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(spx.index, spx["close"], color="orange", label="SPX")
ax1.set_ylabel("SPX Close", color="orange")
ax1.tick_params(axis="y", labelcolor="orange")

ax2 = ax1.twinx()
ax2.plot(turb.index, turb["zscore"], color="purple", alpha=0.7, label="Turbulence ZScore")
ax2.plot(turb.index, turb["delta"], color="violet", alpha=0.5, label="ΔTurbulence")
ax2.plot(turb.index, turb["delta_zscore"], color="blue", alpha=0.6, label="ΔTurbulence ZScore")
ax2.set_ylabel("Turbulence Features", color="purple")
ax2.tick_params(axis="y", labelcolor="purple")

fig.suptitle("SPX vs Turbulence (ZScore, Δ, ΔZScore)", fontsize=14)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.show()


# ---------- Figure 4: SPX + VIX ----------
fig, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(spx.index, spx["close"], color="orange", label="SPX")
ax1.set_ylabel("SPX Close", color="orange")
ax1.tick_params(axis="y", labelcolor="orange")

ax2 = ax1.twinx()
ax2.plot(spx_vix.index, spx_vix["zscore"], color="red", alpha=0.7, label="VIX ZScore")
ax2.plot(spx_vix.index, spx_vix["delta"], color="pink", alpha=0.5, label="ΔVIX")
ax2.plot(spx_vix.index, spx_vix["delta_zscore"], color="blue", alpha=0.6, label="ΔVIX ZScore")
ax2.set_ylabel("VIX Features", color="red")
ax2.tick_params(axis="y", labelcolor="red")

fig.suptitle("SPX vs VIX (ZScore, Δ, ΔZScore)", fontsize=14)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.show()