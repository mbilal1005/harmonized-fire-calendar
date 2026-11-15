"""ML forecast test: harmonized monthly hotspots 1-3 months ahead, per region.

Gradient boosting (sklearn HistGradientBoostingRegressor) against the climatology
baseline (same-month historical median), in a rolling forward test: at each month t
the model is trained only on data up to t and predicts t+1..t+3.

Runs offline (needs scikit-learn); the app only reads data/agg/forecast.json.
Usage:  python forecast.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

AGG = "data/agg"
HORIZONS = (1, 2, 3)
MIN_TRAIN_YEARS = 10          # first forecast origin = 10 years after the series starts
MIN_MAPE_COUNT = 10           # months with fewer actual hotspots are skipped in MdAPE
# Fixed before looking at test results; not tuned.
PARAMS = dict(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=10,
              l2_regularization=1.0, random_state=0)
FEATURES = ["target_month", "horizon", "anom_t", "anom_t1", "anom_t2", "anom_same_month_last_year",
            "normal_log"]


def samples(logy, cm, upto):
    """Feature rows for every (origin, horizon) with origin <= upto.

    Anomaly = log1p(count) - median log1p(count) of the same calendar month, with the
    medians computed from months <= upto only (no look-ahead).
    """
    clim = pd.Series(logy[:upto + 1]).groupby(cm[:upto + 1]).median()
    normal = np.array([clim.get(c, np.nan) for c in cm_all(cm, len(logy) + max(HORIZONS))])
    anom = logy - normal[:len(logy)]
    rows = []
    for i in range(12, upto + 1):
        for h in HORIZONS:
            j = i + h                       # target index (may be beyond the data for the live forecast)
            rows.append((i, h, j, (cm[i] - 1 + h) % 12 + 1, h, anom[i], anom[i - 1], anom[i - 2],
                         anom[j - 12], normal[j]))
    df = pd.DataFrame(rows, columns=["origin", "h", "target"] + FEATURES)
    df["y"] = [logy[j] - normal[j] if j < len(logy) else np.nan for j in df["target"]]
    return df, normal


def cm_all(cm, n):
    """Calendar months continued past the end of the data."""
    return [(cm[0] - 1 + k) % 12 + 1 for k in range(n)]


def run_region(s):
    y = s["harmonized"].to_numpy(float)
    months = list(s["month"])
    cm = np.array([int(m[5:]) for m in months])
    logy = np.log1p(y)
    n = len(y)
    preds = []
    for t in range(MIN_TRAIN_YEARS * 12, n - 1):
        df, normal = samples(logy, cm, t)
        train = df[df["target"] <= t]
        test = df[(df["origin"] == t) & (df["target"] < n)]
        model = HistGradientBoostingRegressor(**PARAMS).fit(train[FEATURES], train["y"])
        p = model.predict(test[FEATURES])
        for (_, r), pa in zip(test.iterrows(), p):
            j = int(r["target"])
            preds.append(dict(origin=months[t], target=months[j], h=int(r["h"]), actual=y[j],
                              ml=float(np.expm1(normal[j] + pa)), baseline=float(np.expm1(normal[j])),
                              err_ml=abs(normal[j] + pa - logy[j]), err_base=abs(normal[j] - logy[j])))
    P = pd.DataFrame(preds)
    res = {}
    for h in HORIZONS:
        d = P[P["h"] == h]
        big = d[d["actual"] >= MIN_MAPE_COUNT]
        res[str(h)] = dict(
            n=int(len(d)), n_mdape=int(len(big)),
            ml_mdape_pct=round(float(100 * ((big["ml"] - big["actual"]).abs() / big["actual"]).median()), 1),
            baseline_mdape_pct=round(float(100 * ((big["baseline"] - big["actual"]).abs() / big["actual"]).median()), 1),
            ml_log_mae=round(float(d["err_ml"].mean()), 3), baseline_log_mae=round(float(d["err_base"].mean()), 3),
            ml_better_share=round(float((d["err_ml"] < d["err_base"]).mean()), 2))
    h1 = res["1"]
    beats = bool(h1["ml_mdape_pct"] < h1["baseline_mdape_pct"] and h1["ml_log_mae"] < h1["baseline_log_mae"])
    # live forecast from the latest full month
    df, normal = samples(logy, cm, n - 1)
    model = HistGradientBoostingRegressor(**PARAMS).fit(df[df["target"] < n][FEATURES], df[df["target"] < n]["y"])
    live = df[df["origin"] == n - 1]
    last = pd.Period(months[-1], "M")
    fc = [dict(month=str(last + int(r["h"])), horizon=int(r["h"]),
               ml=round(float(np.expm1(normal[int(r["target"])] + p)), 1),
               normal=round(float(np.expm1(normal[int(r["target"])])), 1))
          for (_, r), p in zip(live.iterrows(), model.predict(live[FEATURES]))]
    return dict(test_period=f"{P['target'].min()}..{P['target'].max()}", origin=months[-1], metrics=res,
                ml_beats_baseline=beats, forecast=fc), P


def main():
    meta = json.load(open(os.path.join(AGG, "meta.json")))
    ser = pd.read_parquet(os.path.join(AGG, "region_monthly.parquet"))
    ser = ser[~ser["is_partial"]]
    out = dict(
        model="sklearn HistGradientBoostingRegressor", params=PARAMS, features=FEATURES,
        target="log1p(harmonized hotspots) minus the same-month median (anomaly), per region",
        baseline="climatology: same-month historical median, from data up to the forecast origin",
        test="rolling forward: train on data up to month t, predict t+1..t+3; first origin after "
             f"{MIN_TRAIN_YEARS} years",
        rule="ML counts as better for a region only if both its 1-month-ahead MdAPE and its log-scale "
             "mean absolute error are lower than the baseline's",
        last_data_day=meta["calendar"]["last_data_day"], regions={})
    for r, s in ser.groupby("region"):
        res, P = run_region(s.sort_values("month").reset_index(drop=True))
        out["regions"][r] = res
        print(f"\n{r}  test {res['test_period']}  ML beats baseline (1 month ahead): {res['ml_beats_baseline']}")
        print(pd.DataFrame(res["metrics"]).T.to_string())
        print("forecast:", res["forecast"])
    with open(os.path.join(AGG, "forecast.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
