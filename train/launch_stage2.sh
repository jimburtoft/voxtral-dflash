#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Stage-2 (long-form adaptation) launcher, trn2.3xlarge LNC=2:
#   cores 0-1: short-clip feature producers (B=4, 1-window prompts)
#   core  2  : long-form feature producer (B=1, 2-10 windows, chunked target features)
#   core  3  : draft trainer, one long item every LONG_EVERY steps, init from INIT
# Usage: launch_stage2.sh <run> <total_steps> <init_train_state.pt> [extra trainer args]
set -e
RUN=$1; STEPS=$2; INIT=$3; shift 3
LONG_EVERY=${LONG_EVERY:-2}; N_SHORT=${N_SHORT:-2}; N_LONG=${N_LONG:-1}
C=${CONTAINER:-vxdf}; H=${HOST_ROOT:-/host}   # container name / host mount inside the container
Q=/dev/shm/vxdf_q_$RUN
LQ=/dev/shm/vxdf_lq_$RUN
OUT=$H/vxdf_runs/$RUN
sudo docker exec $C sh -c "mkdir -p $Q $LQ $OUT"
core=${CORE0:-0}
for r in $(seq 0 $((N_SHORT-1))); do
  sudo docker exec -d -e NEURON_RT_VISIBLE_CORES=$core -e OMP_NUM_THREADS=3 $C sh -c "cd $H/vxdf && \
    python train/feature_producer.py --labels '$H/vxdf_labels/train/*.labels.parquet' \
      --audio-root $H/vxdf_corpus/shards --queue $Q --rank $r --world $N_SHORT --batch 4 --max-queue 8 \
      > $OUT/producer$r.out 2>&1"
  core=$((core+1))
done
for r in $(seq 0 $((N_LONG-1))); do
  sudo docker exec -d -e NEURON_RT_VISIBLE_CORES=$core -e OMP_NUM_THREADS=3 $C sh -c "cd $H/vxdf && \
    python train/feature_producer_longform.py --labels ${LF_LABELS:-$H/vxdf_labels/train_longform} \
      --audio-root $H/vxdf_corpus/shards/train_longform --queue $LQ --rank $r --world $N_LONG --n-blocks 64 \
      --max-queue 4 > $OUT/lf_producer$r.out 2>&1"
  core=$((core+1))
done
sudo docker exec -d -e NEURON_RT_VISIBLE_CORES=${TRAIN_CORE:-3} -e OMP_NUM_THREADS=3 $C sh -c "cd $H/vxdf && \
  python train/train_draft_queue.py --queue $Q --long-queue $LQ --long-every $LONG_EVERY \
    --eval-dir $H/vxdf_eval_batches/b4_per64 --long-eval-dir ${LONG_EVAL:-$H/vxdf_eval_batches/longform_eval} \
    --init $INIT --out $OUT --total-steps $STEPS $* > $OUT/trainer.out 2>&1"
echo "launched $RUN"
