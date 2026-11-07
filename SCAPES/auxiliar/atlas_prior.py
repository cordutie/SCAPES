"""The prior of an Atlas: a small network that predicts the next disk point of a sequence of CLAP embeddings.

It takes the last N disk points (newest first) and the one-hot cluster, and gives the next disk point. Sampling adds
noise (c * sigma * z) to the prediction, so a GUI can move the disk point by itself with the dynamics of the data.
Fitted by Atlas.add_names_and_prior (dataprep with GUI = True); saved inside atlas.npz with the prefix `prior_`.

    prior = AtlasPrior.fit(codes_per_file, clusters_per_file, K)   # lists of [T, 2] and [T] arrays, in time order
    u = prior.sample(history, cluster, rng)                        # history [N, 2] newest first -> [2]
"""
import numpy as np
import torch
from torch import nn

N = 10              # points of history
HIDDEN = 64
EPOCHS = 40
BATCH = 256
LR = 1e-3
C_GRID = (0.5, 0.75, 1.0, 1.5)   # the noise factor c is the one of these whose free run matches the data best
PREFIX = "prior_"


def _net(n_in):
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 2))


# the network output scaled into the unit disk: m / max(1, |m|)
def _into_disk(m):
    return m / m.norm(dim=1, keepdim=True).clamp_min(1.0)


# inputs and targets of all windows (none crosses a file): X [n, 2N + K], y [n, 2]
def _windows(codes, clusters, K):
    X, y = [], []
    for u, k in zip(codes, clusters):
        u, k = np.asarray(u, dtype=np.float64), np.asarray(k)
        n = len(u) - N
        if n <= 0:
            continue
        idx = np.arange(N, len(u))[:, None] - 1 - np.arange(N)[None, :]     # newest first
        X.append(np.concatenate([u[idx].reshape(n, 2 * N), np.eye(K)[k[N:]]], axis=1))
        y.append(u[N:])
    if not X:
        raise ValueError(f"no file has more than {N} rows: the prior needs one window")
    return np.concatenate(X), np.concatenate(y)


class AtlasPrior:
    def __init__(self, net, sigma, c, K):
        self.net, self.sigma, self.c, self.K = net.eval(), np.asarray(sigma, dtype=np.float64), float(c), int(K)

    # the prediction without noise for histories [B, N, 2] and clusters [B] -> [B, 2]
    def _predict(self, hist, cluster):
        hist = np.asarray(hist, dtype=np.float64)
        x = np.concatenate([hist.reshape(len(hist), -1), np.eye(self.K)[np.asarray(cluster)]], axis=1)
        with torch.no_grad():
            return _into_disk(self.net(torch.as_tensor(x, dtype=torch.float32))).numpy().astype(np.float64)

    # the prediction plus c * sigma * z, into the unit disk. History [N, 2] newest first, one cluster -> [2]
    def sample(self, hist, cluster, rng):
        return self._sample(np.asarray(hist)[None], [cluster], rng)[0]

    def _sample(self, hist, cluster, rng):
        out = self._predict(hist, cluster)
        out = out + self.c * self.sigma * rng.standard_normal(out.shape)
        return out / np.maximum(1.0, np.linalg.norm(out, axis=1, keepdims=True))

    @classmethod
    def fit(cls, codes, clusters, K, epochs=EPOCHS):
        torch.manual_seed(0)
        X, y = _windows(codes, clusters, K)
        Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
        net = _net(X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        gen = torch.Generator().manual_seed(0)
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), BATCH):
                b = perm[i:i + BATCH]
                loss = torch.mean((_into_disk(net(Xt[b])) - yt[b]) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred = _into_disk(net(Xt)).numpy().astype(np.float64)
        prior = cls(net, np.std(y - pred, axis=0), 1.0, K)
        prior.c = prior._choose_c(codes, clusters)
        return prior

    # c from C_GRID by a free run: every file rolled out from its first N points, feeding the samples back; per cluster
    # and axis the variance of the free run over the variance of the data; the c with the smallest mean |log ratio|
    def _choose_c(self, codes, clusters):
        files = [(np.asarray(u, dtype=np.float64), np.asarray(k)) for u, k in zip(codes, clusters) if len(u) > N]
        T = max(len(u) for u, _ in files)
        U = np.zeros((len(files), T, 2))
        L = np.zeros((len(files), T), dtype=np.int64)
        valid = np.zeros((len(files), T - N), dtype=bool)
        for i, (u, k) in enumerate(files):
            U[i, :len(u)], L[i, :len(u)], valid[i, :len(u) - N] = u, k, True
        best = None
        for c in C_GRID:
            self.c = c
            rng = np.random.default_rng(0)
            hist, free = U[:, N - 1::-1, :].copy(), np.zeros((len(files), T - N, 2))
            for j, t in enumerate(range(N, T)):
                free[:, j] = self._sample(hist, L[:, t], rng)
                hist = np.concatenate([free[:, j][:, None, :], hist[:, :-1, :]], axis=1)
            logs = []
            for k in range(self.K):
                m = (L[:, N:] == k) & valid
                if m.sum() >= 2:
                    logs += list(np.abs(np.log(free[m].var(axis=0) / U[:, N:][m].var(axis=0))))
            score = float(np.mean(logs)) if logs else np.inf
            if best is None or score < best[0]:
                best = (score, c)
        return best[1]

    def to_arrays(self):
        out = {PREFIX + k.replace(".", "_"): v.detach().cpu().numpy().astype(np.float32)
               for k, v in self.net.state_dict().items()}
        out.update({PREFIX + "sigma": self.sigma, PREFIX + "c": np.float64(self.c)})
        return out

    @classmethod
    def from_arrays(cls, z, K):
        net = _net(np.asarray(z[PREFIX + "0_weight"]).shape[1])
        net.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(z[PREFIX + k.replace(".", "_")], dtype=np.float32))
                             for k in net.state_dict()})
        return cls(net, z[PREFIX + "sigma"], float(z[PREFIX + "c"]), K)
