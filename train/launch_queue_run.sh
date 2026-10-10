#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Host-side launcher: 3 feature producers (cores 0-2) + 1 draft trainer (core 3), trn2.3xlarge LNC=2.
# Usage: launch_queue_run.sh <run_name> <total_steps> [extra trainer args...]
# Env: SUBSET (producer --subset-frac, default 1.0), BS (block size, default 16), EVAL (eval batch dir)
set -e
RUN=$1; STEPS=$2; shift 2
SUBSET=${SUBSET:-1.0}; BS=${BS:-16}; LAYERS=${LAYERS:-}
LARG=""; [ -n "$LAYERS" ] && LARG="--layer-ids $LAYERS"
C=${CONTAINER:-vxdf}; H=${HOST_ROOT:-/host}   # container name / host mount inside the container
Q=/dev/shm/vxdf_q_$RUN
OUT=$H/vxdf_runs/$RUN
EVAL=${EVAL:-$H/vxdf_eval_batches/b4_per64}
sudo docker exec $C sh -c "mkdir -p $Q $OUT"
for r in 0 1 2; do
  sudo docker exec -d -e NEURON_RT_VISIBLE_CORES=$r -e OMP_NUM_THREADS=3 $C sh -c "cd $H/vxdf && \
    python train/feature_producer.py --labels '$H/vxdf_labels/train/*.labels.parquet' \
      --audio-root $H/vxdf_corpus/shards --queue $Q --rank $r --world 3 --batch 4 --max-queue 8 --subset-frac $SUBSET --block-size $BS $LARG \
      > $OUT/producer$r.out 2>&1"
done
sudo docker exec -d -e NEURON_RT_VISIBLE_CORES=3 -e OMP_NUM_THREADS=3 $C sh -c "cd $H/vxdf && \
  python train/train_draft_queue.py --queue $Q --eval-dir $EVAL --out $OUT --total-steps $STEPS $LARG $* \
    > $OUT/trainer.out 2>&1"
echo "launched $RUN"
