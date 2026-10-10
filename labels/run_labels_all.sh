#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Launch greedy label generation on all 4 logical cores (trn2.3xlarge, LNC=2). Host-side script.
set -e
C=${CONTAINER:-vxdf}; H=${HOST_ROOT:-/host}   # container name / host mount inside the container
for r in 0 1 2 3; do
  sudo docker exec -d -e NEURON_RT_VISIBLE_CORES=$r -e OMP_NUM_THREADS=3 $C sh -c "cd $H/vxdf && \
    python labels/generate_labels_neuron.py --rank $r --world 4 --batch 32 --max-new 192 \
      --manifest $H/vxdf_corpus/shards/eval_manifest.parquet --root $H/vxdf_corpus/shards --out $H/vxdf_labels/eval \
      > $H/vxdf_labels/eval_rank$r.out 2>&1 && \
    python labels/generate_labels_neuron.py --rank $r --world 4 --batch 32 --max-new 192 \
      --manifest $H/vxdf_corpus/shards/train_manifest.parquet --root $H/vxdf_corpus/shards --out $H/vxdf_labels/train \
      > $H/vxdf_labels/train_rank$r.out 2>&1"
  [ $r = 0 ] && sleep 600
done
