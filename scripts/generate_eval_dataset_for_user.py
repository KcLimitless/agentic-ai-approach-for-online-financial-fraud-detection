import json
from pathlib import Path
import pandas as pd

# inferred column order from your CSV excerpt
COLUMNS = [
    "ssn","cc_num","first","last","gender","street","city","state","zip",
    "lat","long","zip2","occupation","dob","card_num","profile_json",
    "trans_num","trans_date","trans_time","trans_epoch","category","amt",
    "is_fraud","merchant","merch_lat","merch_long"
]

def clean_merchant(s):
    if pd.isna(s):
        return ""
    s = str(s)
    return s.replace("fraud_", "").strip()

def row_to_sample(row):
    lat = row.get("lat")
    lon = row.get("long")
    try:
        location = (float(lat), float(lon)) if pd.notna(lat) and pd.notna(lon) else None
    except Exception:
        location = None
    merch_lat = row.get("merch_lat")
    merch_long = row.get("merch_long")
    try:
        merch_lat_f = float(merch_lat) if pd.notna(merch_lat) else None
        merch_long_f = float(merch_long) if pd.notna(merch_long) else None
    except Exception:
        merch_lat_f = merch_long_f = None
    timestamp = None
    if pd.notna(row.get("trans_date")) and pd.notna(row.get("trans_time")):
        timestamp = f"{row.get('trans_date')}T{row.get('trans_time')}Z"
    label = None
    if "is_fraud" in row and pd.notna(row.get("is_fraud")):
        try:
            label = int(row.get("is_fraud"))
        except Exception:
            label = None
    # fallback infer label from merchant prefix
    if label is None:
        m = str(row.get("merchant") or "")
        label = 1 if m.startswith("fraud_") else 0

    return {
        "transaction_id": str(row.get("trans_num") or ""),
        "amount": float(row.get("amt")) if pd.notna(row.get("amt")) else None,
        "category": row.get("category"),
        "location": location,
        "merchant": clean_merchant(row.get("merchant", "")),
        "timestamp": timestamp,
        "user_id": str(row.get("cc_num") or ""),
        "merch_lat": merch_lat_f,
        "merch_long": merch_long_f,
        "trans_time": row.get("trans_time"),
        "label": int(label) if label is not None else None
    }

def build(input_csv, out_json, user_id=None, limit=100):
    df = pd.read_csv(input_csv, sep="|", header=None, names=COLUMNS, low_memory=False)
    if user_id:
        df = df[df["cc_num"].astype(str) == str(user_id)]
    samples = []
    for _, row in df.iterrows():
        samples.append(row_to_sample(row))
        if len(samples) >= limit:
            break
    p = Path(out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf8") as f:
        json.dump(samples, f, indent=2)
    print(f"Wrote {len(samples)} samples to {out_json}")
    return samples

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default="data/customer_transaction_history.csv")
    p.add_argument("-o", "--out", default="data/eval_dataset_user.json")
    p.add_argument("-u", "--user", default="4065133387262473")
    p.add_argument("-n", "--limit", type=int, default=100)
    args = p.parse_args()
    build(args.input, args.out, args.user, args.limit)