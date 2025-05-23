#!/bin/bash
#SBATCH --job-name=reg_%A_%a
#SBATCH --output=logs/slurm_output/slurm-%A_%a.out  # SLURM logs inside job-specific folder
#SBATCH --error=logs/slurm_output/slurm-%A_%a.err   # SLURM errors inside job-specific folder
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=jsteinhardt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aa3797@berkeley.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate Conda environment correctly
conda init
source activate pcs_uq

# Define parameters
DATASETS=("data_ca_housing" "data_parkinsons" "data_airfoil" 
          "data_computer" "data_concrete" "data_powerplant" "data_miami_housing" 
          "data_insurance" "data_qsar" "data_allstate" "data_energy_efficiency" 
          "data_kin8nm" "data_naval_propulsion" "data_superconductor" 
          "data_elevator" "data_protein_structure" "data_debutanizer")

UQ_METHODS=("split_conformal" "studentized_conformal" "majority_vote" "pcs_uq" "pcs_oob")

ALL_ESTIMATORS=("XGBoost" "RandomForest" "ExtraTrees" "AdaBoost"
                "OLS" "Ridge" "Lasso" "ElasticNet" "MLP")

REDUCED_ESTIMATORS=("XGBoost")  # For majority_vote, pcs_uq, pcs_oob

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
LOG_DIR="logs/${DATASET}/${UQ_METHOD}/${ESTIMATOR}"
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
python experiments/scripts/run_regression_exp.py --dataset "$DATASET" --UQ_method "$UQ_METHOD" --seed "$SEED" --estimator "$ESTIMATOR" --train_size "$TRAIN_SIZE"

echo "Job completed at $(date)"
