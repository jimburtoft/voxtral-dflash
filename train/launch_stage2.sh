#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Stage 2 (long-form adaptation). Default = one trn2.3xlarge at LNC=2:
#   N_SHORT short-clip producers (B=4, 1-window prompts), then N_LONG long-form producers
#   (B=1, 2-10 windows, chunked target features), trainer on TRAIN_CORE (3);
#   one long item every LONG_EVERY steps, init from <init_train_state.pt>.
# Usage: sh train/launch_stage2.sh <run> <total_steps> <init_train_state.pt> [extra trainer args]
set -e
. "$(dirname "$0")/_env.sh"
RUN=$1; STEPS=$2; INIT=$3; shift 3
LONG_EVERY=${LONG_EVERY:-2}; N_SHORT=${N_SHORT:-2}; N_LONG=${N_LONG:-1}; TRAIN_CORE=${TRAIN_CORE:-3}
Q=${QUEUE_ROOT:-/dev/shm}/vxdf_q_$RUN
LQ=${QUEUE_ROOT:-/dev/shm}/vxdf_lq_$RUN
OUT=$VXDF_WORK/runs/$RUN
mkdir -p "$Q" "$LQ" "$OUT"
core=${CORE0:-0}
for r in $(seq 0 $((N_SHORT-1))); do
  ( pin $core; exec python train/feature_producer.py --device "$DEVICE" --labels "$VXDF_WORK/labels/train/*.labels.parquet" \
      --audio-root "$VXDF_CORPUS/shards" --queue "$Q" --rank $r --world $N_SHORT --batch 4 --max-queue 8 \
      > "$OUT/producer$r.out" 2>&1 ) &
  core=$((core+1))
done
for r in $(seq 0 $((N_LONG-1))); do
  ( pin $core; exec python train/feature_producer_longform.py --device "$DEVICE" \
      --labels "${LF_LABELS:-$VXDF_WORK/labels/train_longform}" --audio-root "$VXDF_CORPUS/shards/train_longform" \
      --queue "$LQ" --rank $r --world $N_LONG --n-blocks 64 --max-queue 4 > "$OUT/lf_producer$r.out" 2>&1 ) &
  core=$((core+1))
done
( pin $TRAIN_CORE; exec python train/train_draft_queue.py --device "$DEVICE" --queue "$Q" --long-queue "$LQ" \
    --long-every $LONG_EVERY --eval-dir "$VXDF_WORK/eval_batches/b4_per64" \
    --long-eval-dir "${LONG_EVAL:-$VXDF_WORK/eval_batches/longform_eval}" \
    --init "$INIT" --out "$OUT" --total-steps $STEPS "$@" > "$OUT/trainer.out" 2>&1 ) &
TP=$!
echo "launched $RUN (trainer pid $TP); logs in $OUT"
wait $TP
pkill -f "feature_producer.*$Q" || true
pkill -f "feature_producer_longform.*$LQ" || true
