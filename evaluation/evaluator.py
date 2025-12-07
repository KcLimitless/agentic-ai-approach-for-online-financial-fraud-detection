import json
import re
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Optional, Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def compute_latency_stats(latencies: List[float]):
    if not latencies:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "mean": mean(latencies),
        "median": median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "std": stdev(latencies) if len(latencies) > 1 else 0.0,
    }

def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf8"))

def build_map(arr: List[Dict], id_fields=("transaction_id","trans_num","id")) -> Dict[str, Dict]:
    m = {}
    for o in arr:
        tid = next((o.get(f) for f in id_fields if o.get(f)), None)
        if tid:
            m[tid] = o
    return m

def to_int_label(v) -> int:
    try:
        return int(v)
    except Exception:
        s = str(v).lower() if v is not None else ""
        if s in ("true","fraud","high","1"): return 1
        if s in ("false","legit","low","0"): return 0
        return 0

def _metrics_from_confusion(TN:int, FP:int, FN:int, TP:int) -> Dict:
    total = TN + FP + FN + TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    conf_matrix = [[int(TN), int(FP)], [int(FN), int(TP)]]
    counts = {"total_evaluated": int(total), "total_positive": int(TP + FN), "total_predicted_positive": int(TP + FP)}
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix,
        "counts": counts
    }

def _plot_confusion_matrix(conf_matrix: List[List[int]], out_png: Path):
    arr = np.array(conf_matrix)
    plt.figure(figsize=(4,3))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix [[TN,FP],[FN,TP]]")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def _plot_pr_curve(y_true: List[int], scores: List[Optional[float]], metrics: Dict, out_png: Path):
    plt.figure(figsize=(6,5))
    # if we have continuous scores for every sample, draw full curve
    if scores and all(s is not None for s in scores) and len(set(scores))>1:
        y_true_arr = np.array(y_true)
        scores_arr = np.array([float(s) for s in scores])
        precision, recall, _ = precision_recall_curve(y_true_arr, scores_arr)
        ap = None
        try:
            ap = average_precision_score(y_true_arr, scores_arr)
        except Exception:
            ap = None
        plt.plot(recall, precision, label=f"PR curve (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
    else:
        # plot single point using computed precision/recall
        p = metrics.get("precision", 0.0)
        r = metrics.get("recall", 0.0)
        plt.scatter([r], [p], c="red", label=f"Point (P={p:.3f}, R={r:.3f})")
        plt.xlim(0,1); plt.ylim(0,1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (single point)")
        plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def _write_md(metrics: Dict, md_path: Path, pr_png: Path, cm_png: Path):
    lines = []
    lines.append("# Evaluation Metrics\n")
    lines.append(f"- Precision: {metrics.get('precision')}")
    lines.append(f"- Recall: {metrics.get('recall')}")
    lines.append(f"- F1 score: {metrics.get('f1_score')}")
    if metrics.get("aucpr") is not None:
        lines.append(f"- AUC-PR: {metrics.get('aucpr')}")
    lines.append("\n## Confusion Matrix\n")
    lines.append(f"![confusion]({cm_png.name})\n")
    lines.append("\n## Precision-Recall\n")
    lines.append(f"![pr]({pr_png.name})\n")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf8")

# New helper: compute aucpr and latency using available score and latency fields (including latency_seconds)
def _try_compute_aucpr_and_latency_from_files(results_path: Optional[str], ground_truth_path: Optional[str]) -> Tuple[Optional[float], Dict]:
    """Attempt to compute AUC-PR and latency by aligning results + ground-truth if files exist.
       For AUC-PR we compute over only samples that have a numeric score.
       For latency we look for latency_seconds, latency_ms or latency fields.
    """
    if not results_path:
        return None, {"mean": None, "median": None, "min": None, "max": None, "std": None}
    rp = Path(results_path)
    if not rp.exists():
        return None, {"mean": None, "median": None, "min": None, "max": None, "std": None}

    try:
        results = load_json(results_path)
    except Exception:
        return None, {"mean": None, "median": None, "min": None, "max": None, "std": None}

    # ground_truth optional; only needed to align y_true when available
    gmap = {}
    if ground_truth_path and Path(ground_truth_path).exists():
        try:
            ground_truth = load_json(ground_truth_path)
            gmap = build_map(ground_truth)
        except Exception:
            gmap = {}

    rmap = build_map(results)
    ids = sorted(rmap.keys() if not gmap else (set(rmap.keys()) & set(gmap.keys())))
    if not ids:
        # if no ids matched, fallback: use all results entries in order if they contain scores/latencies
        ids = list(rmap.keys())

    pairs = []  # (y_true, score)
    latencies = []
    for tid in ids:
        res = rmap.get(tid)
        if res is None:
            continue

        # latency: accept latency_seconds, latency_ms, latency
        lat = None
        if "latency_seconds" in res:
            lat = safe_float(res.get("latency_seconds"))
        elif "latency_ms" in res:
            # convert ms to seconds for consistency (keep unit seconds)
            ms = safe_float(res.get("latency_ms"))
            lat = ms/1000.0 if ms is not None else None
        elif "latency" in res:
            lat = safe_float(res.get("latency"))
        if lat is not None:
            latencies.append(lat)

        # score extraction
        score = None
        for s in ("score","prob","probability","confidence"):
            if s in res and res[s] is not None:
                score = safe_float(res[s]); break
        # fallback: maybe result text contains "Risk Score: 30" -> extract numeric token
        if score is None and isinstance(res.get("result"), str):
            m = re.search(r"Risk Score[:\s]*([0-9]+(?:\.[0-9]+)?)", res["result"], re.IGNORECASE)
            if m:
                score = safe_float(m.group(1))

        # y_true: try to get from ground truth map when available, else skip
        y = None
        if gmap:
            gt = gmap.get(tid)
            if gt:
                y = to_int_label(gt.get("is_fraud"))
        # If no ground truth mapping but results include an explicit is_fraud field, use it
        if y is None and "is_fraud" in res:
            y = to_int_label(res.get("is_fraud"))

        if score is not None and y is not None:
            pairs.append((int(y), float(score)))

    aucpr = None
    if pairs:
        ys, ss = zip(*pairs)
        # need at least one positive and one negative to compute meaningful AP
        if len(set(ys)) > 1 and len(ss) > 1:
            try:
                aucpr = float(average_precision_score(list(ys), list(ss)))
            except Exception:
                aucpr = None

    latency_stats = compute_latency_stats([s for s in latencies if s is not None])
    return aucpr, latency_stats

def evaluate_results(all_results_path="all_results.json",
                     ground_truth_path="evaluation/ground_truth1.json",
                     out_path="evaluation/evaluation_metrics.json",
                     pr_png="evaluation/evaluation_aucpr.png",
                     cm_png="evaluation/evaluation_confusion_matrix.png",
                     md_out="evaluation/evaluation_metrics.md",
                     manual_conf: Optional[Tuple[int,int,int,int]] = None):
    out_json = Path(out_path)
    pr_png = Path(pr_png)
    cm_png = Path(cm_png)
    md_out = Path(md_out)

    if manual_conf:
        # use manual confusion numbers; attempt to get aucpr+latency from provided results file
        TN, FP, FN, TP = manual_conf
        metrics = _metrics_from_confusion(TN, FP, FN, TP)

        # try compute aucpr/latency from provided results file (user should pass --results reports/all_results.json)
        aucpr, latency_stats = _try_compute_aucpr_and_latency_from_files(all_results_path, ground_truth_path)
        metrics["aucpr"] = aucpr
        metrics["latency"] = latency_stats

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf8")

        # plotting - prepare arrays when possible (scores + y_true)
        y_true_plot = []
        scores_plot = []
        try:
            if Path(all_results_path).exists():
                results = load_json(all_results_path)
                rmap = build_map(results)
                if Path(ground_truth_path).exists():
                    ground_truth = load_json(ground_truth_path)
                    gmap = build_map(ground_truth)
                else:
                    gmap = {}
                ids = sorted(set(rmap.keys()) & set(gmap.keys())) if gmap else sorted(rmap.keys())
                for tid in ids:
                    res = rmap.get(tid)
                    if res is None:
                        continue
                    score = None
                    for s in ("score","prob","probability","confidence"):
                        if s in res and res[s] is not None:
                            score = safe_float(res[s]); break
                    # fallback risk score parse
                    if score is None and isinstance(res.get("result"), str):
                        m = re.search(r"Risk Score[:\s]*([0-9]+(?:\.[0-9]+)?)", res["result"], re.IGNORECASE)
                        if m:
                            score = safe_float(m.group(1))
                    if gmap and tid in gmap:
                        y_true_plot.append(to_int_label(gmap[tid].get("is_fraud")))
                        scores_plot.append(score)
        except Exception:
            y_true_plot = []
            scores_plot = []

        _plot_confusion_matrix(metrics["confusion_matrix"], cm_png)
        _plot_pr_curve(y_true_plot, scores_plot, metrics, pr_png)
        _write_md(metrics, md_out, pr_png, cm_png)
        print(f"Wrote metrics (manual_conf) to {out_json}, md to {md_out}, images: {pr_png}, {cm_png}")
        return metrics

    # else compute from files (unchanged behavior)
    results = load_json(all_results_path)
    ground_truth = load_json(ground_truth_path)
    rmap = build_map(results)
    gmap = build_map(ground_truth)
    ids = sorted(set(rmap.keys()) & set(gmap.keys()))
    if not ids:
        raise SystemExit("No overlapping transaction_ids found between results and ground-truth.")

    y_true = []
    y_pred = []
    scores = []
    latencies = []

    for tid in ids:
        gt = gmap[tid]
        res = rmap[tid]
        truth = to_int_label(gt.get("is_fraud"))
        y_true.append(truth)

        # extract explicit prediction if present
        pred = None
        for f in ("is_fraud","predicted_label","prediction","label","pred"):
            if f in res and res[f] is not None:
                try:
                    pred = int(float(res[f])); break
                except Exception:
                    pass
        score = None
        if pred is None:
            for s in ("score","prob","probability","confidence"):
                if s in res and res[s] is not None:
                    score = safe_float(res[s]); break
        if pred is None and score is None:
            txt = res.get("result") or res.get("output") or res.get("text") or ""
            t = str(txt).upper()
            if any(k in t for k in ("HIGH","FRAUD","TRUE","POSITIVE")):
                pred = 1
            elif any(k in t for k in ("LOW","LEGIT","FALSE","NEGATIVE")):
                pred = 0
        if pred is None and score is not None:
            pred = 1 if score >= 0.5 else 0
        if pred is None:
            pred = 0
        y_pred.append(int(pred))
        scores.append(score if score is not None else None)
        # collect latency: support latency_seconds as well
        if "latency_seconds" in res:
            latencies.append(safe_float(res["latency_seconds"]))
        elif "latency_ms" in res:
            ms = safe_float(res["latency_ms"])
            latencies.append(ms/1000.0 if ms is not None else None)
        elif "latency" in res:
            latencies.append(safe_float(res["latency"]))

    # compute confusion from aligned arrays
    arr_true = np.array(y_true, dtype=int)
    arr_pred = np.array(y_pred, dtype=int)
    TP = int(((arr_true == 1) & (arr_pred == 1)).sum())
    TN = int(((arr_true == 0) & (arr_pred == 0)).sum())
    FP = int(((arr_true == 0) & (arr_pred == 1)).sum())
    FN = int(((arr_true == 1) & (arr_pred == 0)).sum())
    metrics = _metrics_from_confusion(TN, FP, FN, TP)
    metrics["latency"] = compute_latency_stats([s for s in latencies if s is not None])

    # AUC-PR if we have valid continuous scores (compute using only samples that have a score)
    idxs = [i for i, s in enumerate(scores) if s is not None]
    aucpr = None
    if idxs:
        y_for_scores = [y_true[i] for i in idxs]
        s_for_scores = [float(scores[i]) for i in idxs]
        if len(set(y_for_scores)) > 1 and len(s_for_scores) > 1:
            try:
                aucpr = float(average_precision_score(y_for_scores, s_for_scores))
            except Exception:
                aucpr = None
    metrics["aucpr"] = aucpr

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf8")
    # plots
    _plot_confusion_matrix(metrics["confusion_matrix"], cm_png)
    _plot_pr_curve(y_true, scores, metrics, pr_png)
    _write_md(metrics, md_out, pr_png, cm_png)
    print(f"Wrote metrics to {out_json}, md to {md_out}, images: {pr_png}, {cm_png}")
    return metrics

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="reports/all_results.json", help="Model results file (ignored if --manual-conf provided).")
    p.add_argument("--ground-truth", default="evaluation/ground_truth1.json", help="Ground-truth file (ignored if --manual-conf provided).")
    p.add_argument("--out", default="evaluation/evaluation_metrics.json")
    p.add_argument("--pr-png", default="evaluation/evaluation_aucpr.png")
    p.add_argument("--cm-png", default="evaluation/evaluation_confusion_matrix.png")
    p.add_argument("--md-out", default="evaluation/evaluation_metrics.md")
    p.add_argument("--manual-conf", nargs=4, type=int, help="Optional manual confusion counts: TN FP FN TP. If provided, --results/--ground-truth are used to extract aucpr/latency if available.")
    args = p.parse_args()

    manual = tuple(args.manual_conf) if args.manual_conf else None
    evaluate_results(all_results_path=args.results, ground_truth_path=args.ground_truth,
                     out_path=args.out, pr_png=args.pr_png, cm_png=args.cm_png, md_out=args.md_out,
                     manual_conf=manual)



'''
import json
import re
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Optional, Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def compute_latency_stats(latencies: List[float]):
    if not latencies:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "mean": mean(latencies),
        "median": median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "std": stdev(latencies) if len(latencies) > 1 else 0.0,
    }

def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf8"))

def build_map(arr: List[Dict], id_fields=("transaction_id","trans_num","id")) -> Dict[str, Dict]:
    m = {}
    for o in arr:
        tid = next((o.get(f) for f in id_fields if o.get(f)), None)
        if tid:
            m[tid] = o
    return m

def to_int_label(v) -> int:
    try:
        return int(v)
    except Exception:
        s = str(v).lower() if v is not None else ""
        if s in ("true","fraud","high","1"): return 1
        if s in ("false","legit","low","0"): return 0
        return 0

def _metrics_from_confusion(TN:int, FP:int, FN:int, TP:int) -> Dict:
    total = TN + FP + FN + TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    conf_matrix = [[int(TN), int(FP)], [int(FN), int(TP)]]
    counts = {"total_evaluated": int(total), "total_positive": int(TP + FN), "total_predicted_positive": int(TP + FP)}
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix,
        "counts": counts
    }

def _plot_confusion_matrix(conf_matrix: List[List[int]], out_png: Path):
    arr = np.array(conf_matrix)
    plt.figure(figsize=(4,3))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix [[TN,FP],[FN,TP]]")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def _plot_pr_curve(y_true: List[int], scores: List[Optional[float]], metrics: Dict, out_png: Path):
    plt.figure(figsize=(6,5))
    # if we have continuous scores for every sample, draw full curve
    if scores and all(s is not None for s in scores) and len(set(scores))>1:
        y_true_arr = np.array(y_true)
        scores_arr = np.array([float(s) for s in scores])
        precision, recall, _ = precision_recall_curve(y_true_arr, scores_arr)
        ap = None
        try:
            ap = average_precision_score(y_true_arr, scores_arr)
        except Exception:
            ap = None
        plt.plot(recall, precision, label=f"PR curve (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
    else:
        # plot single point using computed precision/recall
        p = metrics.get("precision", 0.0)
        r = metrics.get("recall", 0.0)
        plt.scatter([r], [p], c="red", label=f"Point (P={p:.3f}, R={r:.3f})")
        plt.xlim(0,1); plt.ylim(0,1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (single point)")
        plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def _write_md(metrics: Dict, md_path: Path, pr_png: Path, cm_png: Path):
    lines = []
    lines.append("# Evaluation Metrics\n")
    lines.append(f"- Precision: {metrics.get('precision')}")
    lines.append(f"- Recall: {metrics.get('recall')}")
    lines.append(f"- F1 score: {metrics.get('f1_score')}")
    if metrics.get("aucpr") is not None:
        lines.append(f"- AUC-PR: {metrics.get('aucpr')}")
    lines.append("\n## Confusion Matrix\n")
    lines.append(f"![confusion]({cm_png.name})\n")
    lines.append("\n## Precision-Recall\n")
    lines.append(f"![pr]({pr_png.name})\n")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf8")

def evaluate_results(all_results_path="all_results.json",
                     ground_truth_path="evaluation/ground_truth1.json",
                     out_path="evaluation/evaluation_metrics.json",
                     pr_png="evaluation/evaluation_aucpr.png",
                     cm_png="evaluation/evaluation_confusion_matrix.png",
                     md_out="evaluation/evaluation_metrics.md",
                     manual_conf: Optional[Tuple[int,int,int,int]] = None):
    out_json = Path(out_path)
    pr_png = Path(pr_png)
    cm_png = Path(cm_png)
    md_out = Path(md_out)

    if manual_conf:
        # use manual confusion numbers; do not require results/ground-truth
        TN, FP, FN, TP = manual_conf
        metrics = _metrics_from_confusion(TN, FP, FN, TP)
        # AUC-PR unknown with no scores
        metrics["aucpr"] = None
        # no latencies/scores available
        metrics["latency"] = {"mean": None, "median": None, "min": None, "max": None, "std": None}
        # write JSON
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf8")
        # produce plots: confusion matrix and PR single point
        _plot_confusion_matrix(metrics["confusion_matrix"], cm_png)
        _plot_pr_curve([], [], metrics, pr_png)
        _write_md(metrics, md_out, pr_png, cm_png)
        print(f"Wrote metrics (manual_conf) to {out_json}, md to {md_out}, images: {pr_png}, {cm_png}")
        return metrics

    # else compute from files
    results = load_json(all_results_path)
    ground_truth = load_json(ground_truth_path)
    rmap = build_map(results)
    gmap = build_map(ground_truth)
    ids = sorted(set(rmap.keys()) & set(gmap.keys()))
    if not ids:
        raise SystemExit("No overlapping transaction_ids found between results and ground-truth.")

    y_true = []
    y_pred = []
    scores = []
    latencies = []

    for tid in ids:
        gt = gmap[tid]
        res = rmap[tid]
        truth = to_int_label(gt.get("is_fraud"))
        y_true.append(truth)

        # extract explicit prediction if present
        pred = None
        for f in ("is_fraud","predicted_label","prediction","label","pred"):
            if f in res and res[f] is not None:
                try:
                    pred = int(float(res[f])); break
                except Exception:
                    pass
        score = None
        if pred is None:
            for s in ("score","prob","probability","confidence"):
                if s in res and res[s] is not None:
                    score = safe_float(res[s]); break
        if pred is None and score is None:
            txt = res.get("result") or res.get("output") or res.get("text") or ""
            t = str(txt).upper()
            if any(k in t for k in ("HIGH","FRAUD","TRUE","POSITIVE")):
                pred = 1
            elif any(k in t for k in ("LOW","LEGIT","FALSE","NEGATIVE")):
                pred = 0
        if pred is None and score is not None:
            pred = 1 if score >= 0.5 else 0
        if pred is None:
            pred = 0
        y_pred.append(int(pred))
        scores.append(score if score is not None else None)
        if "latency_ms" in res:
            latencies.append(safe_float(res["latency_ms"]))
        elif "latency" in res:
            latencies.append(safe_float(res["latency"]))

    # compute confusion from aligned arrays
    arr_true = np.array(y_true, dtype=int)
    arr_pred = np.array(y_pred, dtype=int)
    TP = int(((arr_true == 1) & (arr_pred == 1)).sum())
    TN = int(((arr_true == 0) & (arr_pred == 0)).sum())
    FP = int(((arr_true == 0) & (arr_pred == 1)).sum())
    FN = int(((arr_true == 1) & (arr_pred == 0)).sum())
    metrics = _metrics_from_confusion(TN, FP, FN, TP)
    metrics["latency"] = compute_latency_stats(latencies) if latencies else {"mean": None, "median": None, "min": None, "max": None, "std": None}

    # AUC-PR if we have valid continuous scores for every sample
    valid_scores = [s for s in scores if s is not None]
    aucpr = None
    if valid_scores and len(valid_scores) == len(scores) and len(set(valid_scores))>1:
        try:
            aucpr = float(average_precision_score(arr_true, [float(s) for s in scores]))
        except Exception:
            aucpr = None
    metrics["aucpr"] = aucpr

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf8")
    # plots
    _plot_confusion_matrix(metrics["confusion_matrix"], cm_png)
    _plot_pr_curve(y_true, scores, metrics, pr_png)
    _write_md(metrics, md_out, pr_png, cm_png)
    print(f"Wrote metrics to {out_json}, md to {md_out}, images: {pr_png}, {cm_png}")
    return metrics

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="all_results.json", help="Model results file (ignored if --manual-conf provided).")
    p.add_argument("--ground-truth", default="evaluation/ground_truth1.json", help="Ground-truth file (ignored if --manual-conf provided).")
    p.add_argument("--out", default="evaluation/evaluation_metrics.json")
    p.add_argument("--pr-png", default="evaluation/evaluation_aucpr.png")
    p.add_argument("--cm-png", default="evaluation/evaluation_confusion_matrix.png")
    p.add_argument("--md-out", default="evaluation/evaluation_metrics.md")
    p.add_argument("--manual-conf", nargs=4, type=int, help="Optional manual confusion counts: TN FP FN TP. If provided, --results/--ground-truth are not required.")
    args = p.parse_args()

    manual = tuple(args.manual_conf) if args.manual_conf else None
    evaluate_results(all_results_path=args.results, ground_truth_path=args.ground_truth,
                     out_path=args.out, pr_png=args.pr_png, cm_png=args.cm_png, md_out=args.md_out,
                     manual_conf=manual)



import json
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)
from datetime import datetime
from statistics import mean, median, stdev


def extract_classification(result_text: str) -> str:
    """Parse the classification (LOW, MEDIUM, HIGH) from model output text."""
    match = re.search(r"Classification[:\s\-]*([A-Z]+)", result_text, re.IGNORECASE)
    if match:
        return match.group(1).upper().strip()
    match = re.search(r"Fraud Classification[:\s\-]*([A-Z]+)", result_text, re.IGNORECASE)
    if match:
        return match.group(1).upper().strip()
    return "UNKNOWN"


def classification_to_binary(label: str) -> int:
    """Convert HIGH/MEDIUM → 1, LOW → 0."""
    if label.upper() in ["HIGH", "MEDIUM"] :
        return 1
    elif label.upper() == "LOW":
        return 0
    return 0


def evaluate_results(all_results_path="all_results.json", ground_truth_path="ground_truth1.json"):
    # --- Load files ---
    with open(all_results_path, "r", encoding="utf-8") as f:
        model_results = json.load(f)
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    gt_map = {x["transaction_id"]: x for x in ground_truth}

    y_true, y_pred, latencies, categories = [], [], [], []

    # --- Extract classifications & truth ---
    for entry in model_results:
        txn_id = entry.get("transaction_id")
        latency = entry.get("latency_seconds", 0)
        latencies.append(latency)

        classification = extract_classification(entry.get("result", ""))
        pred = classification_to_binary(classification)
        y_pred.append(pred)

        if txn_id in gt_map:
            truth_label = 1 if gt_map[txn_id].get("is_fraud", False) else 0
            y_true.append(truth_label)
            categories.append(gt_map[txn_id].get("fraud_category", "Unknown"))
        else:
            y_true.append(0)
            categories.append("Unknown")

    # --- Compute core metrics ---
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    aucpr = average_precision_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    # --- Latency ---
    latency_stats = {
        "mean": round(mean(latencies), 3),
        "median": round(median(latencies), 3),
        "min": round(min(latencies), 3),
        "max": round(max(latencies), 3),
        "std": round(stdev(latencies), 3),
    }

    # --- Per-category ---
    category_metrics = {}
    for cat in sorted(set(categories)):
        idxs = [i for i, c in enumerate(categories) if c == cat]
        if not idxs:
            continue
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        category_metrics[cat] = {
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
        }

    # --- Save JSON ---
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "aucpr": aucpr,
        "latency": latency_stats,
        "per_category": category_metrics,
        "confusion_matrix": cm.tolist(),
    }
    with open("evaluation_metrics.json", "w", encoding="utf-8") as jf:
        json.dump(metrics, jf, indent=4)

    # --- Generate Markdown report ---
    with open("evaluation_report.md", "w", encoding="utf-8") as mf:
        mf.write(f"# Fraud Detection Evaluation Report\n\n")
        mf.write(f"**Generated:** {datetime.utcnow().isoformat()} UTC\n\n")
        mf.write(f"## Overall Performance\n")
        mf.write(f"- Precision: {precision:.3f}\n")
        mf.write(f"- Recall: {recall:.3f}\n")
        mf.write(f"- F1-Score: {f1:.3f}\n")
        mf.write(f"- AUCPR: {aucpr:.3f}\n\n")

        mf.write(f"## Latency Statistics (seconds)\n")
        for k, v in latency_stats.items():
            mf.write(f"- {k.capitalize()}: {v}\n")
        mf.write("\n")

        mf.write("## Confusion Matrix\n")
        mf.write(f"{cm.tolist()}\n\n")

        mf.write("## Per-Category Performance\n")
        for cat, vals in category_metrics.items():
            mf.write(f"### {cat}\n")
            for k, v in vals.items():
                mf.write(f"- {k.capitalize()}: {v:.3f}\n")
            mf.write("\n")

    # --- Plot AUCPR curve ---
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, color="blue", lw=2)
    plt.title("Precision-Recall (AUCPR)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.text(0.6, 0.1, f"AUCPR = {aucpr:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig("aucpr_curve.png", dpi=300)
    plt.close()

    # --- Plot Confusion Matrix ---
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Pred: Legit (0)", "Pred: Fraud (1)"],
        yticklabels=["True Legit (0)", "True Fraud (1)"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    print("✅ Evaluation complete — files generated:")
    print("• evaluation_metrics.json")
    print("• evaluation_report.md")
    print("• aucpr_curve.png")
    print("• confusion_matrix.png")


if __name__ == "__main__":
    evaluate_results()



import json
import re
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)


# --- parsing utilities -----------------------------------------------------
def extract_classification(result_text: str) -> Optional[str]:
    """Extract HIGH/MEDIUM/LOW (or similar) from model output text."""
    if not result_text:
        return None
    # common patterns
    patterns = [
        r"Fraud Classification[:\s\-]*([A-Z]+)",
        r"Classification[:\s\-]*([A-Z]+)",
        r"Label[:\s\-]*([A-Z]+)",
    ]
    for p in patterns:
        m = re.search(p, result_text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # fallback: look for the words anywhere
    m = re.search(r"\b(HIGH|MEDIUM|LOW|LEGIT|FRAUD|TRUE|FALSE)\b", result_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def classification_to_binary(label: Optional[str]) -> Optional[int]:
    """Map HIGH/MEDIUM -> 1, LOW/LEGIT -> 0. Return None if unknown."""
    if label is None:
        return None
    label = label.upper()
    if label in ("HIGH", "MEDIUM", "FRAUD", "TRUE"):
        return 1
    if label in ("LOW", "LEGIT", "LEGITIMATE", "FALSE"):
        return 0
    return None


# --- loading helpers ------------------------------------------------------
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf8"))


def build_ground_truth_map(ground_truth: List[Dict]) -> Dict[str, Dict]:
    """Return dict: txn_id -> ground_truth_record (with is_fraud int)."""
    mapping = {}
    for obj in ground_truth:
        tid = obj.get("transaction_id") or obj.get("trans_num") or obj.get("id")
        if not tid:
            continue
        # coerce is_fraud to int if possible
        is_fraud = obj.get("is_fraud")
        try:
            is_fraud = int(is_fraud)
        except Exception:
            is_fraud = 1 if str(is_fraud).lower().startswith("t") else 0 if str(is_fraud).lower().startswith("f") else None
        mapping[tid] = {**obj, "is_fraud": is_fraud}
    return mapping


def build_model_results_map(model_results: List[Dict]) -> Dict[str, Dict]:
    """Return dict: txn_id -> result_record (with optional score/prob and text)."""
    mapping = {}
    for r in model_results:
        tid = r.get("transaction_id") or r.get("trans_num") or r.get("id")
        if not tid:
            # try to infer from nested structure
            tid = r.get("meta", {}).get("transaction_id")
        if not tid:
            continue
        mapping[tid] = r
    return mapping


# --- metric computations --------------------------------------------------
def compute_latency_stats(latencies: List[float]) -> Dict:
    if not latencies:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "mean": mean(latencies),
        "median": median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "std": stdev(latencies) if len(latencies) > 1 else 0.0,
    }


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def evaluate_results(all_results_path="all_results.json", ground_truth_path="ground_truth1.json", out_path="evaluation_metrics.json"):
    # Load files
    model_results = load_json(all_results_path)
    ground_truth = load_json(ground_truth_path)

    gt_map = build_ground_truth_map(ground_truth)
    res_map = build_model_results_map(model_results)

    # Align by transaction_id (intersection)
    txn_ids = sorted(set(gt_map.keys()) & set(res_map.keys()))
    if not txn_ids:
        raise SystemExit("No overlapping transaction IDs between model results and ground truth.")

    y_true = []
    y_pred = []
    scores = []  # optional continuous scores for AUC-PR
    latencies = []
    per_category_records = {}

    for tid in txn_ids:
        gt = gt_map[tid]
        res = res_map[tid]

        truth = gt.get("is_fraud")
        # skip if truth unknown
        if truth is None:
            continue

        # determine prediction label
        pred_label = None
        # prefer explicit numeric pred field
        for candidate in ("is_fraud", "predicted_label", "prediction", "label", "pred"):
            if candidate in res and res[candidate] is not None:
                try:
                    pred_label = int(res[candidate])
                except Exception:
                    # maybe text like "HIGH"
                    pred_label = classification_to_binary(str(res[candidate]))
                break

        # if not explicit, parse textual output
        if pred_label is None:
            # common text fields
            txt = res.get("result") or res.get("output") or res.get("prediction_text") or res.get("text") or res.get("response")
            pred_label = classification_to_binary(extract_classification(str(txt))) if txt is not None else None

        # if still None but a numeric score exists, threshold at 0.5
        score = None
        for sfield in ("score", "prob", "probability", "confidence"):
            if sfield in res and res[sfield] is not None:
                score = safe_float(res[sfield])
                break

        if pred_label is None and score is not None:
            pred_label = 1 if score >= 0.5 else 0

        # default conservative if still unknown
        if pred_label is None:
            pred_label = 0

        # collect per-sample
        y_true.append(int(truth))
        y_pred.append(int(pred_label))
        scores.append(score if score is not None else None)

        # latency (if present)
        if "latency_ms" in res:
            latencies.append(safe_float(res["latency_ms"]))
        elif "latency" in res:
            latencies.append(safe_float(res["latency"]))

        # per-category collect
        cat = gt.get("fraud_category") or gt.get("category") or "unknown"
        per_category_records.setdefault(cat, {"y_true": [], "y_pred": []})
        per_category_records[cat]["y_true"].append(int(truth))
        per_category_records[cat]["y_pred"].append(int(pred_label))

    # compute core metrics
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # confusion matrix with explicit label order [0,1] -> [[TN, FP],[FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    conf_matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]

    # AUC-PR (average_precision) only if at least one non-None score exists
    valid_scores = [s for s in scores if s is not None]
    aucpr = None
    if valid_scores and len(valid_scores) == len(scores):
        try:
            aucpr = float(average_precision_score(y_true, [float(s) for s in scores]))
        except Exception:
            aucpr = None

    # per-category metrics
    per_category = {}
    for cat, rec in per_category_records.items():
        yt = rec["y_true"]
        yp = rec["y_pred"]
        per_category[cat] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        }

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "aucpr": aucpr,
        "latency": compute_latency := (compute_latency_stats(latencies) if latencies else {"mean": None, "median": None, "min": None, "max": None, "std": None}),
        "per_category": per_category,
        "confusion_matrix": conf_matrix,
        "counts": {"total_evaluated": len(y_true), "total_positive": int(sum(y_true)), "total_predicted_positive": int(sum(y_pred))},
    }

    # write metrics
    Path(out_path).write_text(json.dumps(metrics, indent=2), encoding="utf8")
    print(f"Wrote evaluation metrics to {out_path}")
    return metrics


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--results", default="all_results.json")
    p.add_argument("--ground-truth", default="evaluation/ground_truth1.json")
    p.add_argument("--out", default="evaluation/evaluation_metrics.json")
    args = p.parse_args()

    evaluate_results(all_results_path=args.results, ground_truth_path=args.ground_truth, out_path=args.out)

'''