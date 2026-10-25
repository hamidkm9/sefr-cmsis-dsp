"""C f32 SEFR / SEFR + LDA / GNB (CMSIS predictor) against the float64 Python references, OpenML-CC18, 5-fold CV."""
import ctypes, glob, json, os, sys, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LIB = next(p for p in (os.path.join(HERE, "..", "host", "build", f) for f in ["libsefrhost.dylib", "libsefrhost.so"])
           if os.path.exists(p))
F, U = np.ctypeslib.ndpointer(np.float32, flags="C"), np.ctypeslib.ndpointer(np.uint32, flags="C")


class Inst(ctypes.Structure):
    _fields_ = [("d", ctypes.c_uint32), ("K", ctypes.c_uint32), ("W", ctypes.c_void_p), ("b", ctypes.c_void_p)]


def lib():
    L = ctypes.CDLL(LIB)
    L.arm_sefr_train_f32.argtypes = [ctypes.POINTER(Inst), F, U, ctypes.c_uint32, F]
    L.arm_sefr_lda_f32.argtypes = [ctypes.POINTER(Inst), F, U, ctypes.c_uint32, F]
    L.arm_sefr_predict_f32.argtypes = [ctypes.POINTER(Inst), F, F]
    L.arm_sefr_predict_f32.restype = ctypes.c_uint32
    L.gnb_train_f32.argtypes = [F, U, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, F, F, F, ctypes.POINTER(ctypes.c_float)]
    L.gnb_predict_f32.argtypes = [F, ctypes.c_uint32, ctypes.c_uint32, F, F, F, ctypes.c_float, F]
    L.gnb_predict_f32.restype = ctypes.c_uint32
    return L


def c_sefr(L, X, y, K, lda):
    N, d = X.shape
    W, b = np.zeros(K * d, np.float32), np.zeros(K, np.float32)
    S = Inst(d, K, W.ctypes.data, b.ctypes.data)
    st = L.arm_sefr_train_f32(ctypes.byref(S), X, y, N, np.zeros(2 * (d + K), np.float32))
    if st == 0 and lda:
        st = L.arm_sefr_lda_f32(ctypes.byref(S), X, y, N, np.zeros(5 * K * K + 2 * K, np.float32))
    return st, S, W, b


def run(path):
    warnings.filterwarnings("ignore")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.naive_bayes import GaussianNB
    from sefr_ref import SEFRMulti
    L = lib()
    z = np.load(path, allow_pickle=True)
    X64, y0, name = z["X"].astype(np.float64), z["y"], str(z["name"])
    classes, y = np.unique(y0, return_inverse=True)
    K = len(classes)
    X32, y = np.ascontiguousarray(X64, np.float32), y.astype(np.uint32)
    rows = []
    for fold, (tr, te) in enumerate(StratifiedKFold(min(5, np.bincount(y).min()), shuffle=True, random_state=0).split(X64, y)):
        Xtr, ytr = np.ascontiguousarray(X32[tr]), np.ascontiguousarray(y[tr])
        for lda in [False, True]:
            ref = SEFRMulti(lda=lda).fit(X64[tr], y[tr]).predict(X64[te])
            st, S, W, b = c_sefr(L, Xtr, ytr, K, lda)
            sc = np.zeros(K, np.float32)
            p = np.array([L.arm_sefr_predict_f32(ctypes.byref(S), np.ascontiguousarray(X32[i]), sc) for i in te]) if st == 0 else None
            rows.append(dict(ds=name, K=K, d=X64.shape[1], fold=fold, method="SEFR + LDA" if lda else "SEFR", status=int(st),
                             acc_c=float((p == y[te]).mean()) if p is not None else None,
                             acc_ref=float((ref == y[te]).mean()),
                             agree=float((p == ref).mean()) if p is not None else None))
        ref = GaussianNB().fit(X64[tr], y[tr]).predict(X64[te])
        th, sg, pr = np.zeros(K * X32.shape[1], np.float32), np.zeros(K * X32.shape[1], np.float32), np.zeros(K, np.float32)
        eps = ctypes.c_float()
        L.gnb_train_f32(Xtr, ytr, len(tr), X32.shape[1], K, th, sg, pr, ctypes.byref(eps))
        buf = np.zeros(2 * K, np.float32)
        p = np.array([L.gnb_predict_f32(np.ascontiguousarray(X32[i]), X32.shape[1], K, th, sg, pr, eps.value, buf) for i in te])
        rows.append(dict(ds=name, K=K, d=X64.shape[1], fold=fold, method="Gaussian NB (CMSIS predictor)", status=0,
                         acc_c=float((p == y[te]).mean()), acc_ref=float((ref == y[te]).mean()), agree=float((p == ref).mean())))
    return rows


if __name__ == "__main__":
    paths = sorted(glob.glob("data/*.npz"))
    with ProcessPoolExecutor(os.cpu_count()) as ex, open(os.path.join(HERE, "..", "results", "cc18_c.jsonl"), "w") as f:
        for fut in as_completed([ex.submit(run, p) for p in paths]):
            try:
                for r in fut.result():
                    f.write(json.dumps(r) + "\n")
            except Exception as e:
                print("ERR", repr(e), flush=True)
