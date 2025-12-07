# preprocessing/schema_mapper.py
import pandas as pd
from datetime import datetime, timezone

def map_sparkov_csv_to_transactions(csv_path: str):
    df = pd.read_csv(csv_path, sep='|')
    mapped = []
    for _, row in df.iterrows():
        try:
            dt_str = f"{row['trans_date']} {row['trans_time']}"
            dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            timestamp = dt_obj.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            mapped.append({
                "transaction_id": str(row["trans_num"]),
                "amount": float(row["amt"]),
                "category": str(row["category"]),
                "location": (float(row["lat"]), float(row["long"])),
                "merchant": str(row["merchant"]),
                "timestamp": timestamp,
                "user_id": str(row["cc_num"]),
                "merch_lat": float(row["merch_lat"]),
                "merch_long": float(row["merch_long"]),
                "trans_time": row["trans_time"]
            })
        except Exception as e:
            print(f"⚠️ Skipping row due to error: {e}")
            continue
    return mapped