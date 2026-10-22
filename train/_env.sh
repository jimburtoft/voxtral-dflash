# SPDX-License-Identifier: Apache-2.0
# Shared settings for the launch scripts. Run them from the repo root inside the Python
# environment that has torch (+ torch-neuronx on Trainium) and requirements-*.txt installed.
#   VXDF_CORPUS  corpus root written by data/prep (default ~/vxdf_corpus)
#   VXDF_WORK    labels / eval items / runs (default ~/vxdf_work)
#   DEVICE       auto | neuron | cuda | cpu (default auto)
VXDF_CORPUS=${VXDF_CORPUS:-$HOME/vxdf_corpus}
VXDF_WORK=${VXDF_WORK:-$HOME/vxdf_work}
DEVICE=${DEVICE:-auto}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}

#   NDEV         if set, process k goes to core / device k % NDEV (e.g. NDEV=1: everything shares one GPU)

# Pin one process to one accelerator core / device: NeuronCore on Trainium, GPU on CUDA.
pin() {
  c=$1; [ -n "$NDEV" ] && c=$((c % NDEV))
  NEURON_RT_VISIBLE_CORES=$c CUDA_VISIBLE_DEVICES=$c; export NEURON_RT_VISIBLE_CORES CUDA_VISIBLE_DEVICES
}
