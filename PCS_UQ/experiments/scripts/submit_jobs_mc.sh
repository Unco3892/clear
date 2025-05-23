#!/bin/bash
#SBATCH --job-name=class_%A_%a
#SBATCH --output=class_logs/slurm_output/slurm-%A_%a.out  # SLURM logs inside job-specific folder
#SBATCH --error=class_logs/slurm_output/slurm-%A_%a.err   # SLURM errors inside job-specific folder
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=64
#SBATCH --partition=jsteinhardt

# Create logs directory if it doesn't exist
mkdir -p class_logs

# Activate Conda environment correctly
conda init
source activate pcs_uq


#DATASETS=("data_chess" "data_cover_type" "data_dionis" "data_isolet" "data_walking")
#DATASETS=("data_language" "data_yeast" "data_chess" "data_cover_type" "data_dionis" "data_isolet")
#UQ_METHODS=("split_conformal_aps" "split_conformal_raps" "majority_vote" "pcs_oob" "split_conformal_topk")

DATASETS=("data_cover_type" "data_dionis")
UQ_METHODS=("pcs_oob")
ALL_ESTIMATORS=("LogisticRegression" "RandomForest" "ExtraTrees" "AdaBoost" "HistGradientBoosting" "MLP")
REDUCED_ESTIMATORS=("HistGradientBoosting")  # For majority_vote, pcs_uq, pcs_oob

SEEDS=(777 778 779 780 781 782 783 784 785 786)  # Modify as needed

TRAIN_SIZES=(0.8)

# Calculate total job count
TOTAL_JOBS=0
for uq in "${UQ_METHODS[@]}"; do
    if [[ "$uq" == "majority_vote" || "$uq" == "pcs_uq" || "$uq" == "pcs_oob" ]]; then
        TOTAL_JOBS=$(( TOTAL_JOBS + ${#DATASETS[@]} * ${#SEEDS[@]} * ${#REDUCED_ESTIMATORS[@]} ))
    else
        TOTAL_JOBS=$(( TOTAL_JOBS + ${#DATASETS[@]} * ${#SEEDS[@]} * ${#ALL_ESTIMATORS[@]} ))
    fi
done

# Subtract 1 since array jobs are 0-based
MAX_ARRAY_INDEX=$((TOTAL_JOBS - 1))

# **Debugging Statement**: Print Total Jobs
echo "Total Jobs: $TOTAL_JOBS"

# Ensure we have at least one job (avoid invalid array error)
if [[ "$TOTAL_JOBS" -le 0 ]]; then
    echo "Error: No jobs to submit. Check dataset, UQ method, or estimator definitions."
    exit 1
fi

# Submit the array job from within the script
if [[ "$SLURM_ARRAY_TASK_ID" == "" ]]; then
    # This is the initial submission
    echo "Submitting array job with indices 0-$MAX_ARRAY_INDEX"
    sbatch --array=0-$MAX_ARRAY_INDEX "$0"
    exit 0
fi

# Compute task index
TASK_ID=$SLURM_ARRAY_TASK_ID

# Compute dataset index
dataset_idx=0
uq_idx=0
estimator_idx=0
seed_idx=0
train_size_idx=0
job_counter=0

for uq in "${UQ_METHODS[@]}"; do
    if [[ "$uq" == "majority_vote" || "$uq" == "pcs_uq" || "$uq" == "pcs_oob" ]]; then
        estimators=("${REDUCED_ESTIMATORS[@]}")
    else
        estimators=("${ALL_ESTIMATORS[@]}")
    fi

    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for train_size in "${TRAIN_SIZES[@]}"; do
                for estimator in "${estimators[@]}"; do
                    if [[ "$job_counter" -eq "$TASK_ID" ]]; then
                        DATASET="$dataset"
                        UQ_METHOD="$uq"
                        ESTIMATOR="$estimator"
                        SEED="$seed"
                        TRAIN_SIZE="$train_size"
                    fi
                    job_counter=$((job_counter + 1))
                done
            done
        done
    done
done

# Create dataset and UQ method specific log directory with absolute path
LOG_DIR="class_logs/${DATASET}/${UQ_METHOD}/${ESTIMATOR}"
mkdir -p "$LOG_DIR"

# Set log file paths with absolute paths
LOG_FILE="$LOG_DIR/${DATASET}_${UQ_METHOD}_${ESTIMATOR}_${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERR_FILE="$LOG_DIR/${DATASET}_${UQ_METHOD}_${ESTIMATOR}_${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# Redirect output before any echo statements
exec 1>"$LOG_FILE"
exec 2>"$ERR_FILE"

echo "Starting job at $(date)"
echo "Job parameters:"
echo "Dataset: $DATASET"
echo "UQ Method: $UQ_METHOD"
echo "Estimator: $ESTIMATOR"
echo "Seed: $SEED"
echo "Train Size: $TRAIN_SIZE"

# Run the Python script
python experiments/scripts/run_classification_exp.py --dataset "$DATASET" --UQ_method "$UQ_METHOD" --seed "$SEED" --estimator "$ESTIMATOR" --train_size "$TRAIN_SIZE"

echo "Job completed at $(date)"
