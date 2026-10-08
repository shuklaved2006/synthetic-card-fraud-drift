"""Synthetic card-transaction generator for the fraud-drift challenge.
Licensed CC0 1.0. Deterministic: fixed seed, numpy/pandas only.
Usage: python generate_data.py --out raw/
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SEED, DAYS, N_CUST, N_MERCH, N_TOTAL = 42, 180, 20000, 3000, 400000
START = pd.Timestamp("2025-01-01")
COUNTRIES = np.array(["US", "GB", "DE", "FR", "IN", "BR", "JP", "NG"])
CATS = np.array(["grocery", "fuel", "restaurant", "travel", "electronics", "apparel",
                 "gift_cards", "utilities", "health", "entertainment", "marketplace", "other"])
HIGH_VALUE = ["electronics", "travel", "gift_cards", "marketplace"]


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    c_country = rng.choice(len(COUNTRIES), N_CUST, p=[.4, .12, .1, .08, .12, .08, .06, .04])
    c_avg = rng.lognormal(3.4, 0.5, N_CUST)
    c_w = rng.gamma(1.5, 1, N_CUST); c_w /= c_w.sum()
    c_ndev = rng.integers(1, 3, N_CUST)
    m_cat = rng.integers(0, len(CATS), N_MERCH)
    m_w = rng.pareto(1.5, N_MERCH) + 1
    risky = np.where(np.isin(CATS[m_cat], ["electronics", "gift_cards", "marketplace"]))[0]
    mule = rng.choice(risky, 60, replace=False)
    m_w[mule] = np.median(m_w)              # mule merchants look ordinary in volume
    small = np.argsort(m_w)[: N_MERCH // 2]  # low-volume merchants
    hv = np.where(np.isin(CATS[m_cat], HIGH_VALUE))[0]
    m_w = m_w / m_w.sum()

    F = []  # fraud frames

    def frame(cust, t, merch, amt, dev, ctry, delay_lo, delay_hi):
        n = len(cust)
        F.append(pd.DataFrame(dict(customer_id=cust, t=t, merchant_id=merch, amount=amt,
                                   channel="online", device_id=dev, country_idx=ctry,
                                   is_fraud=1, label_delay_days=rng.integers(delay_lo, delay_hi + 1, n))))

    # Pattern A: card-testing bursts (common early, fades)
    nb = 140
    w = np.exp(-np.arange(DAYS) / 50.0); w /= w.sum()
    bday = rng.choice(DAYS, nb, p=w)
    k = rng.integers(5, 16, nb)
    b = np.repeat(np.arange(nb), k)
    cust = np.repeat(rng.integers(0, N_CUST, nb), k)
    t0 = np.repeat(bday * 86400 + rng.uniform(0, 86400, nb), k)
    frame(cust, t0 + rng.uniform(0, 600, len(b)), rng.choice(small, len(b)),
          rng.uniform(0.5, 5, len(b)), 900000 + b,
          np.where(rng.random(len(b)) < .6, c_country[cust], rng.integers(0, 8, len(b))), 3, 10)

    # Pattern B: account takeover (all periods)
    na = 900
    k = rng.integers(1, 4, na)
    cust = np.repeat(rng.integers(0, N_CUST, na), k)
    hr = np.repeat(rng.normal(3, 3, na) % 24, k)
    day = np.repeat(rng.integers(0, DAYS, na), k)
    n = len(cust)
    foreign = (c_country[cust] + rng.integers(1, 8, n)) % 8
    frame(cust, day * 86400 + hr * 3600 + rng.uniform(0, 3600, n), rng.choice(hv, n),
          c_avg[cust] * rng.uniform(3, 8, n), 800000 + rng.integers(0, 5000, n), foreign, 5, 21)

    # Pattern C: mule-merchant ring, appears only from day 130 (mostly test period)
    nr = 1100
    cust = rng.integers(0, N_CUST, nr)
    day = rng.integers(130, DAYS, nr)
    hr = np.clip(rng.normal(14, 4, nr), 6, 23)
    frame(cust, day * 86400 + hr * 3600 + rng.uniform(0, 3600, nr), rng.choice(mule, nr),
          rng.lognormal(4.6, 0.5, nr), 700000 + rng.integers(0, 150, nr), c_country[cust], 7, 21)

    fr = pd.concat(F, ignore_index=True)

    # Legitimate traffic
    n = N_TOTAL - len(fr)
    cust = rng.choice(N_CUST, n, p=c_w)
    dow_w = np.array([1, 1, 1, 1, 1.15, 1.3, 1.2]); dd = np.arange(DAYS)
    dp = dow_w[dd % 7]; dp = dp / dp.sum()
    day = rng.choice(DAYS, n, p=dp)
    hp = np.array([1, .6, .4, .3, .3, .5, 1, 2, 3, 3.5, 3.5, 4, 4.5, 4, 3.5, 3.5, 4, 4.5, 5, 4.5, 3.5, 2.5, 1.8, 1.2])
    hr = rng.choice(24, n, p=hp / hp.sum())
    ch = rng.choice(["online", "pos", "atm"], n, p=[.45, .45, .10])
    merch = rng.choice(N_MERCH, n, p=m_w)
    amt = rng.lognormal(np.log(c_avg[cust]) - 0.3, 0.8)
    amt = np.where(ch == "atm", np.maximum(20, np.round(amt / 20) * 20), amt)
    dev = (cust * 3 + (rng.random(n) * c_ndev[cust]).astype(int)).astype(float)
    newdev = rng.random(n) < 0.02
    dev = np.where(newdev, 500000 + rng.integers(0, 400000, n), dev)
    dev = np.where(ch == "online", dev, np.nan)
    ctry = np.where(rng.random(n) < .97, c_country[cust], rng.integers(0, 8, n))
    leg = pd.DataFrame(dict(customer_id=cust, t=day * 86400 + hr * 3600 + rng.uniform(0, 3600, n),
                            merchant_id=merch, amount=amt, channel=ch, device_id=dev,
                            country_idx=ctry, is_fraud=0, label_delay_days=np.nan))

    df = pd.concat([leg, fr], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["transaction_id"] = np.arange(len(df))
    df["timestamp"] = START + pd.to_timedelta(df.t.astype(int), unit="s")
    df["merchant_category"] = CATS[m_cat[df.merchant_id.values]]
    df["country"] = COUNTRIES[df.country_idx.values]
    df["amount"] = df.amount.round(2)
    df["device_id"] = df.device_id.astype("Int64")
    df.loc[rng.random(len(df)) < .02, "merchant_category"] = np.nan   # missingness
    df.loc[rng.random(len(df)) < .01, "country"] = np.nan
    cols = ["transaction_id", "timestamp", "customer_id", "merchant_id", "merchant_category",
            "amount", "channel", "device_id", "country", "is_fraud", "label_delay_days"]
    return df[cols]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="raw")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    generate().to_csv(Path(a.out) / "data.csv", index=False)
