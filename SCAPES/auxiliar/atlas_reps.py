"""
Representatives of the clusters of an atlas: for the score view of the demo, which needs more sources than the clusters give.

Inside each cluster the rows are split in 1, 2 or 3 groups (spherical k-means), and each group is represented by one real row of the
corpus: the one nearest to the group's centre. So a representative is never an average: it is a row that exists, and the audio it
comes from can be played.

When to split: a split into 2 or 3 groups is kept (the one with the better cosine silhouette) when
  - the cluster has at least MIN_ROWS_TO_SPLIT rows,
  - that silhouette is at least SPLIT_SILHOUETTE, and
  - no group has fewer than MIN_GROUP_SHARE of the rows of the cluster (nor MIN_GROUP_ROWS rows).
Otherwise the cluster has one representative, the row nearest to its centre.
SPLIT_SILHOUETTE was chosen on real CLAP embeddings of three datasets (microtex, glitchi, tape_echo; 10 to 33 thousand rows each): at 0.15
almost every cluster splits (21 of 21, 31 of 33, 8 of 9), at 0.20 about two thirds do (21, 23 and 5 clusters) and at 0.25 about half.

pick(rows, labels, K) -> [(cluster, row), ...] with row an index into rows.
"""
import torch

SPLIT_SILHOUETTE = 0.20
MAX_GROUPS = 3
MIN_ROWS_TO_SPLIT = 24
MIN_GROUP_SHARE = 0.10
MIN_GROUP_ROWS = 8
FIT_ROWS = 2000          # rows of a cluster used to find the groups (all its rows are used to find the representatives)


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _kmeans(x, k, generator, iters=30):
    from SCAPES.auxiliar.atlas import _kmeans as kmeans
    return kmeans(x, k, generator, iters)


def _silhouette(x, labels, k):
    from SCAPES.auxiliar.atlas import _silhouette as silhouette
    return silhouette(x, labels, k)


# The groups of one cluster: unit rows x [n, d] -> centres [g, d] (g = 1, 2 or 3)
def groups(x, generator):
    n = len(x)
    centre = _unit(x.mean(dim=0, keepdim=True))
    if n < MIN_ROWS_TO_SPLIT:
        return centre
    sub = x[torch.randperm(n, generator=generator)[:FIT_ROWS]]
    best = None
    for g in range(2, MAX_GROUPS + 1):
        c = _kmeans(sub, g, generator)
        labels = (sub @ c.T).argmax(dim=1)
        sizes = torch.bincount(labels, minlength=g)
        if int(sizes.min()) < max(MIN_GROUP_ROWS * len(sub) / n, MIN_GROUP_SHARE * len(sub)):
            continue
        score = _silhouette(sub[:1000], labels[:1000], g)
        if score >= SPLIT_SILHOUETTE and (best is None or score > best[0]):
            best = (score, c)
    return centre if best is None else best[1]


def pick(rows, labels, K, seed=0):
    rows = _unit(torch.as_tensor(rows).float())
    labels = torch.as_tensor(labels)
    generator = torch.Generator().manual_seed(seed)
    out = []
    for k in range(K):
        idx = (labels == k).nonzero().squeeze(1)
        if not len(idx):
            continue
        x = rows[idx]
        centres = groups(x, generator)
        owner = (x @ centres.T).argmax(dim=1)
        taken = set()
        for g in range(len(centres)):
            members = (owner == g).nonzero().squeeze(1)
            if not len(members):
                continue
            row = int(idx[members[(x[members] @ centres[g]).argmax()]])
            if row not in taken:
                taken.add(row)
                out.append((k, row))
    return out
