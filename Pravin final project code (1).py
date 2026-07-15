import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
np.random.seed(42)

NEAR_FAULT_HOURS = 96

base = r"C:\Users\inp_pravin\OneDrive - Ashok Leyland Ltd\Documents\Revised project code"

# ---------------- INPUT FILES ----------------
files = [
    f"{base}\\MB1PREHD4PEDM0107_2026-03-08_2026-03-31.xlsx",
    f"{base}\\MB1KJCHD9PRNC9276_2026-03-08_2026-03-31.xlsx",
    f"{base}\\MB1H3GHD6RRBB2165_2026-03-08_2026-03-31.xlsx",
    f"{base}\\MB1CWCHD4PPNU7285_2026-03-08_2026-03-31.xlsx",
    f"{base}\\MB1CWCHD1NRGY6892_2026-03-08_2026-03-31.xlsx",
    f"{base}\\MB1A5EHDXPADP9437_2026-03-08_2026-03-31.xlsx",
    f"{base}\\MB1A5CHD6PEMK4674_2026-03-08_2026-03-31.xlsx"
]

df = pd.concat([pd.read_excel(f, engine="openpyxl") for f in files], ignore_index=True)
df.columns = df.columns.str.lower().str.strip()

df["local_timestamp"] = pd.to_datetime(df["local_timestamp"])
df = df.sort_values(["vin", "local_timestamp"])

# ---------------- FAULT TIME (ONLY CHANGE) ----------------
alert_df = pd.read_excel(f"{base}\\ALERT-TYPE-9(April-1).xlsx", engine="openpyxl")
alert_df.columns = alert_df.columns.str.lower().str.strip()

# ✅ Use correct column from file
alert_df["fault_time"] = pd.to_datetime(alert_df["start_timestamp_local"])

# ✅ Get FIRST fault occurrence per VIN
alert_df = alert_df.sort_values("fault_time").drop_duplicates("vin", keep="first")

# ✅ Merge
df = df.merge(alert_df[["vin", "fault_time"]], on="vin", how="left")

# ✅ Validation (important)
if df["fault_time"].isna().sum() > 0:
    raise ValueError("Missing fault_time for some VINs. Check alert file.")

# ---------------- TARGET ----------------
df["ttf"] = (df["fault_time"] - df["local_timestamp"]).dt.total_seconds() / 3600
df["y"] = (df["ttf"] <= NEAR_FAULT_HOURS).astype(int)

# ---------------- SCR CORE (UNCHANGED) ----------------
df["nox_in"] = df.groupby("vin")["aft1_intake_nox"].transform(lambda x: x.rolling(5,3).median())
df["nox_out"] = df.groupby("vin")["aftertreatment1_outlet_nox"].transform(lambda x: x.rolling(5,3).median())

df["nox_in"] = df["nox_in"].clip(lower=10)
ratio = (df["nox_out"] / df["nox_in"]).clip(0, 1.2)

valid_scr = (
    df["ambient_air_temperature"].between(-7,50) &
    df["aft1_scr_catalyst_intake_gas_temp"].between(250,350) &
    df["aft1_exhaust_gas_mass_flow_rate"].between(215,900) &
    df["aft1_diesel_pf_intake_temp"].between(225,400) &
    df["aft1_scr_act_dosing_reagent_qty"].between(250,1500)
)

df["scr_efficiency"] = np.where(valid_scr, (1 - ratio) * 100, np.nan)

fallback = 70 - (df["aft1_scr_catalyst_intake_gas_temp"] - 250) * 0.2
df["scr_efficiency"] = df["scr_efficiency"].fillna(fallback).clip(10,95)

df["scr_efficiency"] = df.groupby("vin")["scr_efficiency"].transform(
    lambda x: x.rolling(8,4).mean()
)

# ---------------- SCR FEATURES (UNCHANGED) ----------------
df["scr_degradation_%"] = 100 - df["scr_efficiency"]
df["scr_effect"] = (df["scr_degradation_%"] / 100)
df["scr_trend"] = df.groupby("vin")["scr_effect"].transform(lambda x: x.rolling(25,8).mean())

# ---------------- MODEL ----------------
FEATURES = [
    "engine_speed","aft1_intake_nox","aftertreatment1_outlet_nox",
    "aft1_scr_catalyst_intake_gas_temp","aft1_scr_act_dosing_reagent_qty",
    "aft1_exhaust_gas_mass_flow_rate","scr_effect","scr_trend","scr_degradation_%"
]

X = StandardScaler().fit_transform(SimpleImputer().fit_transform(df[FEATURES]))

rf_probs = np.zeros(len(df))
xgb_probs = np.zeros(len(df))

for tr, te in GroupKFold(3).split(X, df["y"], df["vin"]):

    rf = RandomForestClassifier(n_estimators=120, max_depth=7, random_state=42)
    xgb = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.05, eval_metric="logloss")

    rf.fit(X[tr], df["y"].iloc[tr])
    xgb.fit(X[tr], df["y"].iloc[tr])

    rf_probs[te] = rf.predict_proba(X[te])[:,1]
    xgb_probs[te] = xgb.predict_proba(X[te])[:,1]

# ---------------- FINAL OUTPUT ----------------
df["prob"] = (rf_probs + xgb_probs)/2

res = []

for v, sub in df.groupby("vin"):

    recent = sub.tail(520)
    smooth_prob = recent["prob"].rolling(4, min_periods=2).mean().dropna()

    idx = smooth_prob.idxmax()

    pred = df.loc[idx, "local_timestamp"]
    actual = sub["fault_time"].iloc[0]

    gap = (actual - pred).total_seconds()/86400

    likelihood = smooth_prob.iloc[-1] * 100
    confidence = (1 - recent["prob"].std()) * 100

    scr_eff = recent["scr_efficiency"].tail(10).mean()
    scr_deg = recent["scr_degradation_%"].tail(10).mean()

    res.append([
        v, pred, actual,
        round(gap,2),
        round(likelihood,2),
        round(confidence,2),
        round(scr_eff,2),
        round(scr_deg,2)
    ])

res = pd.DataFrame(res, columns=[
    "VIN","Predicted","Actual","Gap(days)",
    "Likelihood(%)","Confidence(%)",
    "SCR_Efficiency(%)","SCR_Degradation(%)"
])

print("\n✅ FINAL RESULTS\n")
print(res.to_string(index=False))

res.to_excel(f"{base}\\SCR_Final_Output.xlsx", index=False)
