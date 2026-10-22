#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Greedy labels for the short eval + train sets, one process per core / device (default 4 =
# one trn2.3xlarge at LNC=2). Usage: sh labels/run_labels_all.sh   (env: see train/_env.sh, NPROC)
set -e
. "$(dirname "$0")/../train/_env.sh"
NPROC=${NPROC:-4}
mkdir -p "$VXDF_WORK/labels"
for r in $(seq 0 $((NPROC-1))); do
  ( pin $r
    for s in eval train; do
      python labels/generate_labels.py --device "$DEVICE" --rank $r --world $NPROC --batch 32 --max-new 192 \
        --manifest "$VXDF_CORPUS/shards/${s}_manifest.parquet" --root "$VXDF_CORPUS/shards" \
        --out "$VXDF_WORK/labels/$s" > "$VXDF_WORK/labels/${s}_rank$r.out" 2>&1
    done ) &
  # let the first process populate the compile cache before the others start
  [ $r = 0 ] && sleep ${FIRST_WAIT:-600}
done
wait
