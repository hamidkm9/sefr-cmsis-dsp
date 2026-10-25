"""Accuracy and agreement tables from results/cc18_c.jsonl."""
import json, os
import pandas as pd

df = pd.DataFrame([json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "..", "results", "cc18_c.jsonl"))])
print("failed folds:", df.groupby("method").status.apply(lambda s: int((s != 0).sum())).to_dict())
g = df.groupby("method")
print("\nagreement of C float32 predictions with the float64 reference")
print(pd.DataFrame({"mean": g.agree.mean(), "worst fold": g.agree.min()}).round(4).to_string())
d = df.groupby(["ds", "method"]).agg(acc=("acc_c", "mean"), K=("K", "first")).reset_index()
P, K = d.pivot(index="ds", columns="method", values="acc"), d.groupby("ds").K.first()
base = "Gaussian NB (CMSIS predictor)"
for name, m in [("binary", K == 2), ("multiclass", K > 2)]:
    s = P[m]
    print(f"\n{name} ({len(s)} data sets): mean accuracy %")
    for col in s.columns:
        line = f"  {col:32s} {100 * s[col].mean():5.1f}"
        if col != base:
            diff = s[col] - s[base]
            line += f"   vs GNB: better on {(diff > 0).sum()}, worse on {(diff < 0).sum()}, mean {100 * diff.mean():+.1f}"
        print(line)
