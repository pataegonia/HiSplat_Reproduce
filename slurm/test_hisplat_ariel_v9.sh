#!/usr/bin/bash
#SBATCH -J hisplat-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v9
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-bootstrap-%A.out
#SBATCH -e slurm-bootstrap-%A.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_DIR="${REPO_DIR:-${SUBMIT_DIR}}"

EXPERIMENT="${1:-${EXPERIMENT:-re10k}}"
CKPT_PATH="${2:-${CKPT_PATH:-}}"
OUTPUT_NAME="${3:-${OUTPUT_NAME:-test_${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)}}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-hisplat}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-hisplat}"
WANDB_ENTITY="${WANDB_ENTITY:-placeholder}"
WANDB_NAME="${WANDB_NAME:-${EXPERIMENT}_test}"

DATASET_BASE="${DATASET_BASE:-/data3/local_datasets}"
DATASET_SUBDIR="${DATASET_SUBDIR:-}"
NUM_CONTEXT_VIEWS="${NUM_CONTEXT_VIEWS:-2}"
COMPUTE_SCORES="${COMPUTE_SCORES:-true}"
SAVE_IMAGE="${SAVE_IMAGE:-false}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-4}"
SH_DEGREE="${SH_DEGREE:-}"
RUN_DIR="${RUN_DIR:-${SUBMIT_DIR}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"

if [[ -z "${DATASET_SUBDIR}" ]]; then
    case "${EXPERIMENT}" in
        re10k)
            DATASET_SUBDIR="re10k"
            ;;
        acid)
            DATASET_SUBDIR="acid"
            ;;
        dtu)
            DATASET_SUBDIR="dtu"
            ;;
        replica)
            DATASET_SUBDIR="replica"
            ;;
        *)
            echo "Unsupported experiment: ${EXPERIMENT}"
            echo "Use one of: re10k, acid, dtu, replica"
            exit 1
            ;;
    esac
fi

DATASET_ROOT="${DATASET_ROOT:-${DATASET_BASE}/${DATASET_SUBDIR}}"
INDEX_PATH="${INDEX_PATH:-}"
if [[ -z "${INDEX_PATH}" ]]; then
    case "${EXPERIMENT}" in
        re10k)
            INDEX_PATH="assets/evaluation_index_re10k.json"
            ;;
        acid)
            INDEX_PATH="assets/evaluation_index_acid.json"
            ;;
        dtu)
            INDEX_PATH="assets/evaluation_index_dtu_nctx${NUM_CONTEXT_VIEWS}.json"
            ;;
        replica)
            INDEX_PATH="assets/evaluation_index_replica_nctx${NUM_CONTEXT_VIEWS}.json"
            ;;
    esac
fi

if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    NUM_GPUS="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
else
    NUM_GPUS=1
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${RUN_DIR}/outputs"

LOG_OUT="${LOG_DIR}/slurm-${SLURM_JOB_ID:-$$}.out"
LOG_ERR="${LOG_DIR}/slurm-${SLURM_JOB_ID:-$$}.err"
exec >"${LOG_OUT}" 2>"${LOG_ERR}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Dataset directory not found: ${DATASET_ROOT}"
    exit 1
fi

if [[ ! -f "${DATASET_ROOT}/test/index.json" ]]; then
    echo "Missing test/index.json under ${DATASET_ROOT}"
    exit 1
fi

if [[ ! -f "${REPO_DIR}/${INDEX_PATH}" ]]; then
    echo "Evaluation index not found: ${REPO_DIR}/${INDEX_PATH}"
    exit 1
fi

if [[ "${EXPERIMENT}" == "dtu" || "${EXPERIMENT}" == "replica" ]]; then
    if [[ "${NUM_CONTEXT_VIEWS}" != "2" && "${NUM_CONTEXT_VIEWS}" != "3" ]]; then
        echo "NUM_CONTEXT_VIEWS must be 2 or 3 for ${EXPERIMENT}"
        exit 1
    fi
fi

if [[ -z "${CKPT_PATH}" ]]; then
    shopt -s nullglob
    ckpts=("${RUN_DIR}/latest-run/checkpoints/"*.ckpt)
    shopt -u nullglob
    if [[ ${#ckpts[@]} -eq 0 ]]; then
        echo "Checkpoint path not provided and no ckpt found under ${RUN_DIR}/latest-run/checkpoints"
        exit 1
    fi
    IFS=$'\n' sorted_ckpts=($(ls -1t "${ckpts[@]}"))
    unset IFS
    CKPT_PATH="${sorted_ckpts[0]}"
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

# Resolve symlinked paths like latest-run/... before Python rewires latest-run.
if command -v readlink >/dev/null 2>&1; then
    CKPT_PATH="$(readlink -f "${CKPT_PATH}")"
elif command -v realpath >/dev/null 2>&1; then
    CKPT_PATH="$(realpath "${CKPT_PATH}")"
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Resolved checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "Conda initialization script not found."
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${RUN_DIR}"

CMD=(
    python -m src.main
    "+experiment=${EXPERIMENT}"
    "mode=test"
    "device=${NUM_GPUS}"
    "dataset.roots=[${DATASET_ROOT}]"
    "dataset/view_sampler=evaluation"
    "dataset.view_sampler.index_path=${INDEX_PATH}"
    "data_loader.test.batch_size=${TEST_BATCH_SIZE}"
    "data_loader.test.num_workers=${TEST_NUM_WORKERS}"
    "test.compute_scores=${COMPUTE_SCORES}"
    "test.save_image=${SAVE_IMAGE}"
    "test.save_video=${SAVE_VIDEO}"
    "output_dir=${OUTPUT_NAME}"
    "wandb.mode=${WANDB_MODE}"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.entity=${WANDB_ENTITY}"
    "wandb.name=${WANDB_NAME}"
    "checkpointing.load=${CKPT_PATH}"
)

if [[ "${EXPERIMENT}" == "dtu" || "${EXPERIMENT}" == "replica" ]]; then
    CMD+=("dataset.view_sampler.num_context_views=${NUM_CONTEXT_VIEWS}")
fi

if [[ -n "${SH_DEGREE}" ]]; then
    CMD+=("model.encoder.gaussian_adapter.sh_degree=${SH_DEGREE}")
fi

echo "Host: $(hostname)"
echo "Partition: batch_ugrad"
echo "QOS: qos_clue9986_2026_1"
echo "Repo dir: ${REPO_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Experiment: ${EXPERIMENT}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Evaluation index: ${INDEX_PATH}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Output dir: ${RUN_DIR}/outputs/${OUTPUT_NAME}"
echo "GPUs: ${NUM_GPUS}"
echo "SH degree override: ${SH_DEGREE:-<default>}"
echo "Compute scores: ${COMPUTE_SCORES}"
echo "Save image: ${SAVE_IMAGE}"
echo "Save video: ${SAVE_VIDEO}"
echo "Command: ${CMD[*]}"

srun "${CMD[@]}"
