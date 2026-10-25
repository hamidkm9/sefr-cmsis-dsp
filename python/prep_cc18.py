"""Download the 71 OpenML-CC18 datasets and store each as a dense numeric matrix in data/<id>.npz.

Categorical columns with at most 30 levels are one-hot encoded, others become integer codes; missing numbers
become the column median; constant columns are dropped; data sets above 25,000 rows are subsampled (stratified).
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

MAX_ROWS = 25000
MAX_ONEHOT = 30


def prep(did):
    out = f"data/{did}.npz"
    if os.path.exists(out):
        return did, "cached"
    warnings.filterwarnings("ignore")
    from sklearn.datasets import fetch_openml
    try:
        b = fetch_openml(data_id=int(did), as_frame=True, parser="auto", data_home="openml_cache")
    except Exception as e:
        return did, f"download failed: {e!r}"[:200]
    X, y = b.data, b.target
    keep = y.notna().values
    X, y = X[keep], y[keep]
    cols = []
    for c in X.columns:
        s = X[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or s.dtype == bool:
            s = s.astype("category")
            if len(s.cat.categories) <= MAX_ONEHOT:
                cols.append(pd.get_dummies(s, prefix=str(c), dummy_na=s.isna().any(), dtype=float))
            else:
                cols.append(pd.DataFrame({c: s.cat.codes.astype(float)}))
        else:
            s = pd.to_numeric(s, errors="coerce").astype(float)
            cols.append(pd.DataFrame({c: s.fillna(s.median() if s.notna().any() else 0.0)}))
    Xn = pd.concat(cols, axis=1).to_numpy(dtype=np.float64)
    Xn = Xn[:, Xn.std(0) > 0]
    classes, yi = np.unique(y.astype(str).values, return_inverse=True)
    if len(yi) > MAX_ROWS:
        from sklearn.model_selection import train_test_split
        idx, _ = train_test_split(np.arange(len(yi)), train_size=MAX_ROWS, stratify=yi, random_state=0)
        Xn, yi = Xn[idx], yi[idx]
    np.savez_compressed(out, X=Xn.astype(np.float32), y=yi.astype(np.int32), name=b.details.get("name", str(did)), n_orig=keep.sum())
    return did, f"{b.details.get('name')} {Xn.shape} K={len(classes)}"


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    ids = sys.argv[1:] or json.load(open(os.path.join(os.path.dirname(__file__), "cc18_ids.json")))
    with ProcessPoolExecutor(6) as ex:
        for did, msg in ex.map(prep, ids):
            print(did, msg, flush=True)
