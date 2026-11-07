"""A small neural network (pure NumPy, no deep-learning library needed) that learns the torque the controller's model gets wrong.

    missing torque  =  torque really applied  -  (M_nominal * measured acceleration + h_nominal)

Inputs  x = [q (6), q_dot (6), commanded acceleration a (6), gripper closed (1)]  -> 19 numbers
Output  y = 6 joint torques (N m) to ADD to the computed-torque command.

Network: 19 -> 128 -> 128 -> 6, tanh units, trained with Adam on mean squared error of standardised targets.
The last layer starts at zero, so an untrained network adds exactly nothing and the controller behaves like the baseline.
"""
import numpy as np


def features(q, qd, a, grip, hist=None):
    """19 numbers; with `hist` (the controller's running averages of the recent model error, 18 numbers) 37."""
    x = np.concatenate([q, qd, a, [float(grip)]])
    return x if hist is None else np.concatenate([x, hist])


class ResidualNet:
    def __init__(self, n_in=19, n_out=6, hidden=(128, 128), seed=0, lr=1e-3, wd=1e-5):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [rng.normal(0, np.sqrt(1.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.W[-1] = np.zeros_like(self.W[-1])                       # untrained network outputs exactly zero
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.lr, self.wd, self.t = lr, wd, 0
        self.m = [np.zeros_like(p) for p in self.W + self.b]; self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.ready = False                                           # becomes True when input/target scales are set
        self.xm = np.zeros(n_in); self.xs = np.ones(n_in); self.ys = np.ones(n_out)

    # ---- scaling (frozen after the first batch of data, so the network never sees a moving target)
    def set_norm(self, X, Y):
        self.xm, self.xs = X.mean(0), np.maximum(X.std(0), 1e-3)
        self.ys = np.maximum(Y.std(0), 1e-2); self.ready = True

    def nx(self, X): return (X - self.xm) / self.xs
    def ny(self, Y): return Y / self.ys

    # ---- forward / predict
    def _forward(self, Xn):
        acts = [Xn]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            acts.append(np.tanh(acts[-1] @ W + b))
        return acts, acts[-1] @ self.W[-1] + self.b[-1]

    def predict(self, x):
        """One sample, raw units in -> torques in N m out."""
        if not self.ready:
            return np.zeros(self.b[-1].shape)
        return self._forward(self.nx(x)[None, :])[1][0] * self.ys

    def predict_batch(self, X):
        return self._forward(self.nx(X))[1] * self.ys

    # ---- one Adam step on a batch (raw units); returns the standardised mean squared error before the step
    def train_step(self, X, Y):
        Xn, Yn = self.nx(X), self.ny(Y)
        acts, out = self._forward(Xn)
        err = out - Yn
        loss = float(np.mean(err ** 2))
        g = 2.0 * err / err.size                                       # dLoss/dout
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ g + self.wd * self.W[i]
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (1.0 - acts[i] ** 2)         # tanh'
        self.t += 1
        params, grads = self.W + self.b, gW + gb
        c1, c2 = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        for k, (p, gr) in enumerate(zip(params, grads)):
            self.m[k] = 0.9 * self.m[k] + 0.1 * gr
            self.v[k] = 0.999 * self.v[k] + 0.001 * gr * gr
            p -= self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + 1e-8)
        return loss

    def r2(self, X, Y):
        """Share of the variance of the (standardised) target that the network explains, averaged over joints."""
        P, Yn = self.predict_batch(X) / self.ys, self.ny(Y)
        return float(np.mean(1.0 - ((P - Yn) ** 2).sum(0) / np.maximum(((Yn - Yn.mean(0)) ** 2).sum(0), 1e-12)))

    def rmse(self, X, Y):
        return np.sqrt(np.mean((self.predict_batch(X) - Y) ** 2, axis=0))

    # ---- files
    def save(self, path):
        np.savez(path, **{f"W{i}": w for i, w in enumerate(self.W)}, **{f"b{i}": b for i, b in enumerate(self.b)},
                 xm=self.xm, xs=self.xs, ys=self.ys)

    @classmethod
    def load(cls, path):
        z = np.load(path); n = len([k for k in z.files if k.startswith("W")])
        net = cls(n_in=z["W0"].shape[0], n_out=z[f"W{n-1}"].shape[1], hidden=tuple(z[f"W{i}"].shape[1] for i in range(n - 1)))
        net.W = [z[f"W{i}"] for i in range(n)]; net.b = [z[f"b{i}"] for i in range(n)]
        net.xm, net.xs, net.ys, net.ready = z["xm"], z["xs"], z["ys"], True
        return net