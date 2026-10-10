# SPDX-License-Identifier: Apache-2.0
"""Quick label sanity: WER of target greedy transcripts vs dataset references (simple normaliser)."""
import collections
import glob
import re
import sys

import pyarrow.parquet as pq


def norm(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        nd = [i] + [0] * len(h)
        for j, b in enumerate(h, 1):
            nd[j] = min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (a != b))
        d = nd
    return d[-1]


def main(pattern):
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for f in sorted(glob.glob(pattern)):
        t = pq.read_table(f)
        if t.num_rows == 0:
            continue
        for r in t.select(["dataset", "split", "ref_text", "hyp_text", "finished", "n_resp"]).to_pylist():
            k = (r["dataset"], r["split"])
            a, b = norm(r["ref_text"]), norm(r["hyp_text"])
            g = agg[k]
            g[0] += edits(a, b); g[1] += len(a); g[2] += 1; g[3] += int(r["finished"]); g[4] = max(g[4], r["n_resp"])
    for k, (e, n, c, fin, mx) in sorted(agg.items()):
        print(f"{k[0]:32s} {k[1]:24s} clips {c:6d} finished {fin/c:.3%}  max_resp {mx:4d}  WER {100*e/max(n,1):.2f}%")


if __name__ == "__main__":
    main(sys.argv[1])
