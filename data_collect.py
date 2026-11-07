"""Record (state, command, missing torque) samples while the arm works.

For every `every`-th simulation step the Collector reads, from the simulation that is running:
    x = [q, q_dot, commanded acceleration a, gripper closed]            (what a controller can know)
    y = torque applied  -  (M_nominal * measured acceleration + h_nominal)   (the torque the nominal model misses)
This label is exact in simulation because the true acceleration is known.  On a real robot it would have to be estimated
from noisy joint measurements, which is much harder.  The identity holds for any controller, so samples from the baseline
and from the network-assisted runs can be mixed.
"""
import numpy as np
import mujoco
from residual_net import features


class Buffer:
    """Growing training set plus a separate validation set (whole tasks are assigned to one or the other)."""
    def __init__(self, capacity=400_000, n_in=19, n_out=6):
        self.X = np.zeros((capacity, n_in), np.float32); self.Y = np.zeros((capacity, n_out), np.float32); self.n = 0
        self.Xv = np.zeros((60_000, n_in), np.float32); self.Yv = np.zeros((60_000, n_out), np.float32); self.nv = 0

    def add(self, x, y, validation=False):
        if validation:
            if self.nv < len(self.Xv): self.Xv[self.nv], self.Yv[self.nv] = x, y; self.nv += 1
        elif self.n < len(self.X):
            self.X[self.n], self.Y[self.n] = x, y; self.n += 1

    def batch(self, rng, size=256):
        i = rng.integers(0, self.n, size); return self.X[i].astype(np.float64), self.Y[i].astype(np.float64)

    def val(self, cap=6000):
        k = min(self.nv, cap); return self.Xv[:k].astype(np.float64), self.Yv[:k].astype(np.float64)


class Collector:
    """Attach with  pp.recorder = Collector(nominal_model, sink, every=10).  sink(x, y) receives each sample."""
    def __init__(self, nominal_dynamics, sink, every=10, history=False):
        self.model, self.sink, self.every, self.k, self.history = nominal_dynamics, sink, every, 0, history

    def __call__(self, pp):
        self.k += 1
        a = getattr(pp.c, "last_a", None)
        if self.k % self.every or a is None:
            return
        m, d = pp.m, pp.d
        mujoco.mj_forward(m, d)                          # acceleration produced by the torque that was just commanded
        M, h = self.model.dynamics(d)
        y = d.ctrl[:6] - M @ d.qacc[:6] - h
        q_seen = getattr(pp.c, "last_q", d.qpos[:6]); qd_seen = getattr(pp.c, "last_qd", d.qvel[:6])      # the readings the controller had (noisy if sensors are)
        self.sink(features(q_seen, qd_seen, a, d.ctrl[6] > 0.01, pp.c.last_hist if self.history else None), y)