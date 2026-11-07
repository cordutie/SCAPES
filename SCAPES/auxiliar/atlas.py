"""Atlas: the shallow model that re-represents CLAP embeddings (used when the config has GUI = True).

An Atlas holds K clusters of unit CLAP vectors. Cluster k has a unit mean mu_k; the errors v - mu_k of its rows share
one 2-D basis P (PCA) and one scale s. A CLAP vector becomes a cluster and a point u of the unit disk, and goes back:

    cluster, u = atlas.encode(v)          # nearest centre; u = (P^T (v - mu_k) / s), shrunk into the unit disk
    v_hat      = atlas.decode(cluster, u) # normalise(mu_k + P (s u))
    v_hat      = atlas.autoencode(v)      # decode(encode(v)): the vector that is on the atlas

Dataprep with GUI = True fits one Atlas on all the CLAP embeddings, saves it as <dataset>/atlas.npz and replaces the
embeddings by their autoencode. Training copies it to <model folder>/atlas.npz, and inference uses it to rewrite the
CLAP embeddings that come from audio, so that they look like the training ones.

Use it from anywhere (torch and numpy only):

    atlas = Atlas.load("atlas.npz")
    v_hat = atlas.autoencode(v)                       # v: [n, 1024] or [1024], torch or numpy
    v = atlas.decode(cluster=3, u=[0.2, -0.1])        # a point of the 2-D plane of cluster 3 -> [1024]

The 3-D view of the GUI is kept too, so the embeddings are not needed later: `atlas.to3d(v)` projects any vector, and
`atlas.cloud` [m, 3] with `atlas.cloud_labels` [m] are m points of the original embeddings (at most CLOUD_ROWS, evenly
spaced) for the dots of the plot.

The file is a plain .npz with `centres` [K, d], `mu` [K, d], `P` [d, 2], `s` (a scalar), `view_mean` [d],
`view_P3` [d, 3], `view_scale` (a scalar), `cloud` [m, 3] and `cloud_labels` [m]. After add_names_and_prior (dataprep does
it) the file also has `names` [K] (the sound of each cluster, from the audio files) and the arrays of the prior
(atlas_prior.py: `atlas.prior.sample(history, cluster, rng)` gives the next disk point of a sequence).
"""
from pathlib import Path

import numpy as np
import torch

TRIM = 0.1          # share of the rows of each cluster, farthest from the mean, left out of the PCA and the scale
SCALE_QUANTILE = 0.9
K_RANGE = (8, 32)   # K is the one of this range with the best cosine silhouette (fewer clusters if fewer rows)
FIT_ROWS = 20000    # rows used to choose K and the centres (all rows are used for the means, PCA and scale)
SILHOUETTE_ROWS = 2000
CLOUD_ROWS = 4000   # the 3-D plot keeps at most this many points (the GUI ran well with 3968)


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# Spherical k-means with a fixed seed: unit rows x [n, d], k clusters -> centres [k, d]
def _kmeans(x, k, generator, iters=30):
    centres = x[torch.randperm(len(x), generator=generator)[:k]].clone()
    for _ in range(iters):
        labels = (x @ centres.T).argmax(dim=1)
        for j in range(k):
            members = x[labels == j]
            if len(members):
                centres[j] = _unit(members.mean(dim=0))
    return centres


# Mean cosine silhouette of the first rows of x [n, d] for the labels [n]
def _silhouette(x, labels, k):
    dist = 1 - x @ x.T
    own = (labels[:, None] == labels[None, :])
    size = own.sum(dim=1)
    a = (dist * own).sum(dim=1) / (size - 1).clamp_min(1)
    b = torch.full_like(a, float("inf"))
    for j in range(k):
        mask = labels == j
        if mask.any():
            other = dist[:, mask].mean(dim=1)
            b = torch.where((labels != j) & (other < b), other, b)
    s = (b - a) / torch.maximum(a, b).clamp_min(1e-12)
    return float(torch.where(size > 1, s, torch.zeros_like(s)).mean())


class Atlas:
    def __init__(self, centres, mu, P, s, view_mean, view_P3, view_scale, cloud, cloud_labels, names=None, prior=None):
        self.centres = torch.as_tensor(centres, dtype=torch.float64)   # [K, d] assignment centres
        self.mu = torch.as_tensor(mu, dtype=torch.float64)             # [K, d] unit cluster means
        self.P = torch.as_tensor(P, dtype=torch.float64)               # [d, 2] orthonormal error basis
        self.s = float(s)                                              # scale of the disk
        self.view_mean = torch.as_tensor(view_mean, dtype=torch.float64)    # [d] mean of the embeddings
        self.view_P3 = torch.as_tensor(view_P3, dtype=torch.float64)        # [d, 3] top 3 principal directions
        self.view_scale = float(view_scale)                                 # to3d divides by it
        self.cloud = torch.as_tensor(cloud, dtype=torch.float32)            # [m, 3] points of the original rows
        self.cloud_labels = torch.as_tensor(cloud_labels, dtype=torch.int64)  # [m] their cluster
        self.names = list(names) if names is not None else [f"cluster {k}" for k in range(len(self.mu))]
        self.prior = prior                                                  # AtlasPrior, or None

    @property
    def K(self):
        return len(self.mu)

    # Fits the atlas on rows [N, d] (torch or numpy, normalised here). Clusters: spherical k-means, K by silhouette.
    @classmethod
    def fit(cls, rows, k_range=K_RANGE, seed=0):
        v = _unit(torch.as_tensor(rows).double())
        n = len(v)
        generator = torch.Generator().manual_seed(seed)
        sub = v[torch.randperm(n, generator=generator)[:FIT_ROWS]].float()
        k_max = min(k_range[1], len(sub) - 1)
        k_min = min(k_range[0], k_max)
        if k_max < 2:
            centres = _unit(v.mean(dim=0, keepdim=True))
        else:
            best = None
            for k in range(k_min, k_max + 1):
                c = _kmeans(sub, k, generator)
                labels = (sub @ c.T).argmax(dim=1)
                score = _silhouette(sub[:SILHOUETTE_ROWS], labels[:SILHOUETTE_ROWS], k)
                if best is None or score > best[0]:
                    best = (score, c)
            centres = best[1].double()
        labels = torch.cdist(v, centres.double()).argmin(dim=1)
        # drop the clusters that ended up empty
        used = torch.unique(labels)
        centres = centres[used]
        labels = torch.searchsorted(used, labels)
        mu = torch.stack([_unit(v[labels == k].mean(dim=0)) for k in range(len(used))])
        w = v - mu[labels]
        # trim: the farthest rows of each cluster do not enter the PCA and the scale
        keep = torch.ones(n, dtype=torch.bool)
        dist = w.norm(dim=1)
        for k in range(len(used)):
            idx = (labels == k).nonzero().squeeze(1)
            cut = min(int(np.ceil(round(TRIM * len(idx), 9))), len(idx) - 1)   # a cluster keeps at least one row
            keep[idx[dist[idx].argsort(descending=True)[:cut]]] = False
        if int(keep.sum()) < 2:
            raise ValueError("fewer than 2 rows left after trimming: too few CLAP embeddings for an atlas")
        wk = w[keep]
        _, vecs = torch.linalg.eigh(wk.T @ wk / len(wk))
        P = vecs[:, -2:].flip(1)
        s = float(torch.quantile((wk @ P).norm(dim=1), SCALE_QUANTILE))
        if not s > 0:
            raise ValueError("the scale is zero: all CLAP embeddings lie on their cluster means")
        # 3-D view: PCA-3 of all rows, scaled so that the farthest row has norm 1; the cloud is a thin sample of them
        mean = v.mean(dim=0)
        xc = v - mean
        P3 = torch.linalg.eigh(xc.T @ xc / n)[1][:, -3:].flip(1)
        scale = float((xc @ P3).norm(dim=1).max())
        pick = torch.arange(0, n, -(-n // CLOUD_ROWS))
        return cls(centres, mu, P, s, mean, P3, scale, (xc[pick] @ P3 / scale).float(), labels[pick])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {} if self.prior is None else self.prior.to_arrays()
        np.savez(path, centres=self.centres.numpy(), mu=self.mu.numpy(), P=self.P.numpy(), s=np.float64(self.s),
                 view_mean=self.view_mean.numpy(), view_P3=self.view_P3.numpy(), view_scale=np.float64(self.view_scale),
                 cloud=self.cloud.numpy(), cloud_labels=self.cloud_labels.numpy(), names=np.array(self.names), **extra)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            names = [str(n) for n in z["names"]] if "names" in z.files else None
            prior = None
            if "prior_0_weight" in z.files:
                from SCAPES.auxiliar.atlas_prior import AtlasPrior
                prior = AtlasPrior.from_arrays(z, len(z["mu"]))
            return cls(z["centres"], z["mu"], z["P"], float(z["s"]), z["view_mean"], z["view_P3"],
                       float(z["view_scale"]), z["cloud"], z["cloud_labels"], names, prior)

    # rows [n, d] or [d] (torch or numpy) -> unit float64 rows [n, d] and whether the input was a single vector
    def _rows(self, v):
        v = torch.as_tensor(v).detach().cpu().double()
        single = v.dim() == 1
        return _unit(v[None] if single else v), single

    # Nearest cluster of each row (Euclidean distance on the unit rows) and its disk point: (cluster [n], u [n, 2])
    def encode(self, v):
        v, single = self._rows(v)
        cluster = torch.cdist(v, self.centres).argmin(dim=1)
        y = (v - self.mu[cluster]) @ self.P / self.s
        u = y / y.norm(dim=1, keepdim=True).clamp_min(1.0)
        return (cluster[0], u[0]) if single else (cluster, u)

    # Unit vectors [n, d] from clusters [n] (or one int) and disk points [n, 2] (or [2]); float32
    def decode(self, cluster, u):
        cluster = torch.as_tensor(cluster).long()
        single = cluster.dim() == 0
        u = torch.as_tensor(u).double().reshape(-1, 2)
        x = self.mu[cluster.reshape(-1)] + (self.s * u) @ self.P.T
        out = _unit(x).float()
        return out[0] if single else out

    # Names the clusters after the audio files and fits the prior. arrays: one [T, d] array of CLAP rows per audio file,
    # in time order; stems: the file names without extension. A cluster is named after the file that holds most of its
    # rows; a file that is the majority of several clusters gives stem_1, stem_2, ... (largest cluster first).
    # No prior (None) when no file has more than 10 rows.
    def add_names_and_prior(self, arrays, stems):
        from SCAPES.auxiliar.atlas_prior import N, AtlasPrior
        clusters, codes = zip(*[self.encode(a) for a in arrays])
        labels = torch.cat(clusters).numpy()
        file_of_row = np.repeat(np.arange(len(arrays)), [len(a) for a in arrays])
        size = np.bincount(labels, minlength=self.K)
        majority = [int(np.bincount(file_of_row[labels == k], minlength=len(stems)).argmax()) if size[k] else -1
                    for k in range(self.K)]
        names = []
        for k, f in enumerate(majority):
            same = sorted((j for j in range(self.K) if majority[j] == f), key=lambda j: (-size[j], j))
            names.append(f"cluster {k}" if f < 0 else stems[f] if len(same) == 1 else f"{stems[f]}_{same.index(k) + 1}")
        self.names = names
        self.prior = None
        if any(len(a) > N for a in arrays):
            self.prior = AtlasPrior.fit([c.numpy() for c in codes], [c.numpy() for c in clusters], self.K)

    # 3-D coordinates [n, 3] (or [3]) of CLAP vectors, in the view of the cloud: (v - view_mean) @ view_P3 / view_scale
    def to3d(self, v):
        v, single = self._rows(v)
        out = ((v - self.view_mean) @ self.view_P3 / self.view_scale).float()
        return out[0] if single else out

    # decode(encode(v)), in the shape and the torch dtype of the input (float32 for numpy input)
    def autoencode(self, v):
        out = self.decode(*self.encode(v))
        return out.numpy() if isinstance(v, np.ndarray) else out.to(dtype=v.dtype, device=v.device)
