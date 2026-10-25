"""float64 reference: one-vs-rest SEFR, optionally with the LDA second pass on the K class scores."""
import numpy as np


class SEFRMulti:
    def __init__(self, lda=False, ridge=1e-6):
        self.lda, self.ridge = lda, ridge

    def _scores(self, X):
        return X @ self.W_.T + self.b_

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_, yi = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        lo, hi = X.min(0), X.max(0)
        rng = np.where(hi - lo > 0, hi - lo, 1.0)
        Z = (X - lo) / rng
        n = np.bincount(yi, minlength=K).astype(float)
        sums = np.zeros((K, X.shape[1]))
        np.add.at(sums, yi, Z)
        mp = sums / n[:, None]
        rest = (sums.sum(0)[None, :] - sums) / (len(yi) - n)[:, None]
        W = (mp - rest) / (mp + rest + 1e-7)
        t = ((len(yi) - n) * (W * mp).sum(1) + n * (W * rest).sum(1)) / len(yi)
        self.W_ = W / rng
        self.b_ = -(t + self.W_ @ lo)
        if self.lda and K > 1:
            S = self._scores(X)
            means = np.zeros((K, K))
            np.add.at(means, yi, S)
            means /= n[:, None]
            R = S - means[yi]
            C = R.T @ R / len(yi)
            C += (self.ridge * np.trace(C) / K + 1e-12) * np.eye(K)
            A = np.linalg.solve(C, means.T).T
            self.b_ = np.log(n / n.sum()) + A @ self.b_ - 0.5 * np.einsum("ck,ck->c", A, means)
            self.W_ = A @ self.W_
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self._scores(np.asarray(X, dtype=np.float64)), 1)]
