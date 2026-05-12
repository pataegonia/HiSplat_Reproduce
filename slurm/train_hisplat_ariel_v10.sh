#!/usr/bin/bash
#SBATCH -J hisplat-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v9
#SBATCH -t 6-00:00:00
#SBATCH -o slurm-bootstrap-%A.out
#SBATCH -e slurm-bootstrap-%A.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_DIR="${REPO_DIR:-${SUBMIT_DIR}}"
SH_DEGREE="${SH_DEGREE:-3}"
NUM_NODES="${SLURM_NNODES:-1}"

EXPERIMENT="${1:-${EXPERIMENT:-re10k}}"
OUTPUT_NAME="${2:-${OUTPUT_NAME:-${EXPERIMENT}_sh${SH_DEGREE}_$(date +%Y%m%d_%H%M%S)}}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hisplat}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-hisplat}"
WANDB_ENTITY="${WANDB_ENTITY:-placeholder}"
WANDB_NAME="${WANDB_NAME:-${EXPERIMENT}}"
MAX_STEPS="${MAX_STEPS:-}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
CKPT_EVERY_N_TRAIN_STEPS="${CKPT_EVERY_N_TRAIN_STEPS:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-}"
TRAIN_PERSISTENT_WORKERS="${TRAIN_PERSISTENT_WORKERS:-}"
VAL_PERSISTENT_WORKERS="${VAL_PERSISTENT_WORKERS:-}"
LOAD_BACKBONE_PRETRAINED="${LOAD_BACKBONE_PRETRAINED:-}"
if [[ -z "${LOAD_BACKBONE_PRETRAINED}" ]]; then
    if [[ "${TRAIN_FROM_SCRATCH:-0}" == "1" ]]; then
        LOAD_BACKBONE_PRETRAINED=0
    else
        LOAD_BACKBONE_PRETRAINED=1
    fi
fi
DATASET_BASE="${DATASET_BASE:-/data3/local_datasets}"
DATASET_SUBDIR="${DATASET_SUBDIR:-}"
if [[ -z "${DATASET_SUBDIR}" ]]; then
    case "${EXPERIMENT}" in
        re10k)
            DATASET_SUBDIR="re10k"
            ;;
        acid)
            DATASET_SUBDIR="acid"
            ;;
        *)
            DATASET_SUBDIR="${EXPERIMENT}"
            ;;
    esac
fi
DATASET_ROOT="${DATASET_ROOT:-${DATASET_BASE}/${DATASET_SUBDIR}}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-}"
if [[ -z "${VAL_CHECK_INTERVAL}" ]]; then
    case "${DATASET_SUBDIR}" in
        *subset*)
            VAL_CHECK_INTERVAL=10
            ;;
        *)
            VAL_CHECK_INTERVAL=3000
            ;;
    esac
fi
RUN_DIR="${RUN_DIR:-${SUBMIT_DIR}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${RUN_DIR}/checkpoints}"
SOURCE_CKPT_DIR="${SOURCE_CKPT_DIR:-${REPO_DIR}/checkpoints}"
UNIMATCH_CKPT_NAME="gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth"
DINO_CKPT_NAME="dinov2_vitb14_pretrain.pth"

if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    NUM_GPUS="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
else
    NUM_GPUS=1
fi

if [[ "${EXPERIMENT}" != "re10k" && "${EXPERIMENT}" != "acid" ]]; then
    echo "Unsupported experiment: ${EXPERIMENT}"
    echo "Use one of: re10k, acid"
    exit 1
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${RUN_DIR}/outputs" "${PRETRAIN_DIR}"

LOG_OUT="${LOG_DIR}/slurm-${SLURM_JOB_ID:-$$}.out"
LOG_ERR="${LOG_DIR}/slurm-${SLURM_JOB_ID:-$$}.err"
exec >"${LOG_OUT}" 2>"${LOG_ERR}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Dataset directory not found: ${DATASET_ROOT}"
    exit 1
fi

if [[ ! -f "${DATASET_ROOT}/train/index.json" ]]; then
    echo "Missing train/index.json under ${DATASET_ROOT}"
    exit 1
fi

if [[ ! -f "${DATASET_ROOT}/test/index.json" ]]; then
    echo "Missing test/index.json under ${DATASET_ROOT}"
    exit 1
fi

if [[ "${LOAD_BACKBONE_PRETRAINED}" == "1" ]]; then
    if [[ ! -f "${PRETRAIN_DIR}/${UNIMATCH_CKPT_NAME}" ]]; then
        if [[ -f "${SOURCE_CKPT_DIR}/${UNIMATCH_CKPT_NAME}" ]]; then
            ln -sfn "${SOURCE_CKPT_DIR}/${UNIMATCH_CKPT_NAME}" "${PRETRAIN_DIR}/${UNIMATCH_CKPT_NAME}"
        else
            echo "Missing UniMatch checkpoint: ${PRETRAIN_DIR}/${UNIMATCH_CKPT_NAME}"
            echo "Set SOURCE_CKPT_DIR or copy the file into ${PRETRAIN_DIR}"
            exit 1
        fi
    fi

    if [[ ! -f "${PRETRAIN_DIR}/${DINO_CKPT_NAME}" ]]; then
        if [[ -f "${SOURCE_CKPT_DIR}/${DINO_CKPT_NAME}" ]]; then
            ln -sfn "${SOURCE_CKPT_DIR}/${DINO_CKPT_NAME}" "${PRETRAIN_DIR}/${DINO_CKPT_NAME}"
        else
            echo "Missing DINOv2 checkpoint: ${PRETRAIN_DIR}/${DINO_CKPT_NAME}"
            echo "Set SOURCE_CKPT_DIR or copy the file into ${PRETRAIN_DIR}"
            exit 1
        fi
    fi
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
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HISPLAT_NUM_NODES="${NUM_NODES}"

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
    export MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
fi

cd "${RUN_DIR}"

CMD=(
    python -m src.main
    "+experiment=${EXPERIMENT}"
    "data_loader.train.batch_size=${BATCH_SIZE_PER_GPU}"
    "device=${NUM_GPUS}"
    "dataset.roots=[${DATASET_ROOT}]"
    "output_dir=${OUTPUT_NAME}"
    "trainer.val_check_interval=${VAL_CHECK_INTERVAL}"
    "model.encoder.gaussian_adapter.sh_degree=${SH_DEGREE}"
    "wandb.mode=${WANDB_MODE}"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.entity=${WANDB_ENTITY}"
    "wandb.name=${WANDB_NAME}"
)

if [[ -n "${MAX_STEPS}" ]]; then
    CMD+=("trainer.max_steps=${MAX_STEPS}")
fi

if [[ -n "${OPTIMIZER_LR}" ]]; then
    CMD+=("optimizer.lr=${OPTIMIZER_LR}")
fi

if [[ -n "${CKPT_EVERY_N_TRAIN_STEPS}" ]]; then
    CMD+=("checkpointing.every_n_train_steps=${CKPT_EVERY_N_TRAIN_STEPS}")
fi

if [[ -n "${RESUME_CKPT}" ]]; then
    CMD+=("checkpointing.load=${RESUME_CKPT}")
else
    # Skip Lightning's strict ckpt resume by default.
    # Backbone weights are already loaded via UniMatch partial load (encoder_costvolume_pyramid),
    # so Lightning's ckpt_path resume would only force a strict state_dict load that breaks
    # whenever the pretrained ckpt shape differs (e.g. SH4 ckpt -> SH3 model).
    CMD+=("checkpointing.load=null")
fi

if [[ -n "${TRAIN_NUM_WORKERS}" ]]; then
    CMD+=("data_loader.train.num_workers=${TRAIN_NUM_WORKERS}")
fi

if [[ -n "${VAL_NUM_WORKERS}" ]]; then
    CMD+=("data_loader.val.num_workers=${VAL_NUM_WORKERS}")
fi

if [[ -n "${TRAIN_PERSISTENT_WORKERS}" ]]; then
    CMD+=("data_loader.train.persistent_workers=${TRAIN_PERSISTENT_WORKERS}")
fi

if [[ -n "${VAL_PERSISTENT_WORKERS}" ]]; then
    CMD+=("data_loader.val.persistent_workers=${VAL_PERSISTENT_WORKERS}")
fi

if [[ "${LOAD_BACKBONE_PRETRAINED}" == "1" ]]; then
    CMD+=("model.encoder.unimatch_weights_path=${PRETRAIN_DIR}/${UNIMATCH_CKPT_NAME}")
else
    CMD+=("model.encoder.unimatch_weights_path=null")
fi

echo "Host: $(hostname)"
echo "Partition: batch_ugrad"
echo "QOS: qos_clue9986_2026_1"
echo "Repo dir: ${REPO_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Experiment: ${EXPERIMENT}"
echo "Dataset subdir: ${DATASET_SUBDIR}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Pretrain dir: ${PRETRAIN_DIR}"
echo "Load backbone pretrained: ${LOAD_BACKBONE_PRETRAINED}"
echo "Output dir: ${RUN_DIR}/outputs/${OUTPUT_NAME}"
echo "GPUs: ${NUM_GPUS}"
echo "Nodes: ${NUM_NODES}"
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "SH degree override: ${SH_DEGREE}"
echo "Max steps override: ${MAX_STEPS:-<default>}"
echo "Optimizer lr override: ${OPTIMIZER_LR:-<default>}"
echo "Checkpoint interval override: ${CKPT_EVERY_N_TRAIN_STEPS:-<default>}"
echo "Train num_workers override: ${TRAIN_NUM_WORKERS:-<default>}"
echo "Val num_workers override: ${VAL_NUM_WORKERS:-<default>}"
echo "Train persistent_workers override: ${TRAIN_PERSISTENT_WORKERS:-<default>}"
echo "Val persistent_workers override: ${VAL_PERSISTENT_WORKERS:-<default>}"
echo "Command: ${CMD[*]}"

# Let Lightning launch local DDP workers inside the existing SLURM allocation.
"${CMD[@]}"
