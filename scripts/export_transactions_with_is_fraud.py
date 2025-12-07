import json
import re
from pathlib import Path
import pandas as pd


FALLBACK_COLS = [
    "ssn","cc_num","first","last","gender","street","city","state","zip",
    "lat","long","zip2","occupation","dob","card_num","profile_json",
    "trans_num","trans_date","trans_time","trans_epoch","category","amt",
    "is_fraud","merchant","merch_lat","merch_long"
]

CANDIDATES = {
    "transaction_id": ["trans_num","transaction_id","trans_id","id"],
    "amount": ["amt","amount","transaction_amt"],
    "category": ["category","trans_category"],
    "lat": ["lat","latitude"],
    "long": ["long","longitude","lon"],
    "merchant": ["merchant","merchant_name"],
    "trans_date": ["trans_date","date"],
    "trans_time": ["trans_time","time"],
    "cc_num": ["cc_num","user_id","card_num"],
    "merch_lat": ["merch_lat","merchant_lat"],
    "merch_long": ["merch_long","merchant_long"],
    "is_fraud": ["is_fraud","fraud","label"]
}


def clean_merchant(s):
    if pd.isna(s) or s is None:
        return ""
    return re.sub(r"^fraud_", "", str(s)).strip()


def find_col(cols, names):
    for n in names:
        if n in cols:
            return n
    return None


def is_floatable(val):
    try:
        if val is None:
            return False
        # treat pandas NA / nan as non-floatable
        if pd.isna(val):
            return False
        float(val)
        return True
    except Exception:
        return False


def safe_float(val):
    try:
        return float(pd.to_numeric(val, errors="coerce"))
    except Exception:
        return None


def drop_header_like_rows(df):
    if df.shape[0] == 0:
        return df
    matches = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        # compare cell string to column name (trimmed)
        matches[col] = df[col].astype(str).str.strip() == str(col).strip()
    match_counts = matches.sum(axis=1)
    to_drop = match_counts > 2
    if to_drop.any():
        df = df[~to_drop].reset_index(drop=True)
    return df


def row_to_sample(row, colmap):
    def g(k):
        key = colmap.get(k)
        if not key:
            return None
        # Series access: prefer .get for safety
        return row.get(key) if key in row.index else None

    # location
    lat = g("lat"); lon = g("long")
    location = None
    try:
        if is_floatable(lat) and is_floatable(lon):
            location = (float(lat), float(lon))
    except Exception:
        location = None

    # merchant coords
    merch_lat = g("merch_lat"); merch_long = g("merch_long")
    merch_lat_f = float(merch_lat) if is_floatable(merch_lat) else None
    merch_long_f = float(merch_long) if is_floatable(merch_long) else None

    trans_date = g("trans_date"); trans_time = g("trans_time")
    timestamp = None
    if pd.notna(trans_date) and pd.notna(trans_time):
        timestamp = f"{trans_date}T{trans_time}Z"

    # is_fraud: prefer explicit column, else infer from merchant prefix
    label = None
    if colmap.get("is_fraud"):
        v = g("is_fraud")
        if pd.notna(v):
            try:
                label = int(v)
            except Exception:
                label = None
    if label is None:
        m = str(g("merchant") or "")
        label = 1 if m.startswith("fraud_") else 0

    amt_raw = g("amount")
    amt_val = safe_float(amt_raw) if amt_raw is not None else None

    return {
        "transaction_id": str(g("transaction_id") or ""),
        "amount": amt_val,
        "category": g("category") if g("category") is not None and not pd.isna(g("category")) else None,
        "location": list(location) if location is not None else None,
        "merchant": clean_merchant(g("merchant") or ""),
        "timestamp": timestamp,
        "user_id": str(g("cc_num") or ""),
        "merch_lat": merch_lat_f,
        "merch_long": merch_long_f,
        "trans_time": trans_time if trans_time is not None and not pd.isna(trans_time) else None,
        "is_fraud": str(int(label))
    }


def build(input_csv: str, out_json: str = "data/transactions_all_with_is_fraud.json"):
    # read CSV (try header, else fallback)
    try:
        df_try = pd.read_csv(input_csv, sep="|", low_memory=False)
        if set(FALLBACK_COLS).issubset(set(df_try.columns)):
            df = df_try
        else:
            df = pd.read_csv(input_csv, sep="|", header=None, names=FALLBACK_COLS, low_memory=False)
    except Exception:
        df = pd.read_csv(input_csv, sep="|", header=None, names=FALLBACK_COLS, low_memory=False)

    # normalize columns and drop header-like rows
    df.columns = [c.strip() for c in df.columns]
    df = drop_header_like_rows(df)
    cols = df.columns.tolist()

    colmap = {}
    for k, cand in CANDIDATES.items():
        found = find_col(cols, cand)
        if found:
            colmap[k] = found

    samples = [row_to_sample(row, colmap) for _, row in df.iterrows()]

    outp = Path(out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf8") as f:
        json.dump(samples, f, indent=2)
    print(f"Wrote {len(samples)} transactions to {out_json}")
    return samples


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default="data/customer_transaction_history.csv")
    p.add_argument("-o", "--out", default="data/transactions_all_with_is_fraud.json")
    args = p.parse_args()
    build(args.input, args.out)