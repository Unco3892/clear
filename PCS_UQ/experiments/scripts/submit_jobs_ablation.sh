#!/bin/bash
#SBATCH --job-name=abl3
#SBATCH --output=logs/abalation.out  # SLURM logs inside job-specific folder
#SBATCH --error=logs/abalation.err   # SLURM errors inside job-specific folder
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=jsteinhardt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=michaelxiao1999@berkeley.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# ------------------ ABLATION EXPERIMENT SETTING (uncomment one) ------------------

# # ABLATION EXPERIMENT 1: varying number of bootstraps
# DATASETS=("data_ca_housing" "data_parkinsons" "data_airfoil")
# UQ_METHODS=("pcs_oob")
# ALL_ESTIMATORS=("XGBoost" "RandomForest" "ExtraTrees" "AdaBoost"
#                 "OLS" "Ridge" "Lasso" "ElasticNet" "MLP")
# REDUCED_ESTIMATORS=("XGBoost")  # For majority_vote, pcs_uq, pcs_oob
# SEEDS=(777 778 779 780 781 782 783 784 785 786)  # Modify as needed
# NBOOTS=(10 25 50 100 500 1000) # Number of bootstrap samples for pcs_oob
# TRAIN_SIZES=(0.8)
# TRAIN_FRACS=(1.0)
# NMODELS=(1)  # Number of models in PCS
# CALIBRATIONS=("multiplicative")
# ABL="boot"

# # ABLATION EXPERIMENT 2: varying dataset size
# DATASETS=("data_ca_housing" "data_parkinsons" "data_airfoil")
# UQ_METHODS=("majority_vote")
# ALL_ESTIMATORS=("XGBoost" "RandomForest")
# REDUCED_ESTIMATORS=("XGBoost")  # For majority_vote, pcs_uq, pcs_oob
# SEEDS=(777 778 779 780 781 782 783 784 785 786)  # Modify as needed
# NBOOTS=(1000) # Number of bootstrap samples for pcs_oob
# TRAIN_SIZES=(0.8)
# TRAIN_FRACS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0) # Varying fraction of training set used
# NMODELS=(1)
# CALIBRATIONS=("multiplicative")
# ABL="split"

# ABLATION EXPERIMENT 3: varying number of models in PCS
DATASETS=("data_ca_housing" "data_parkinsons" "data_airfoil")
UQ_METHODS=("pcs_oob")
ALL_ESTIMATORS=("XGBoost" "RandomForest")
REDUCED_ESTIMATORS=("XGBoost")  # For majority_vote, pcs_uq, pcs_oob
SEEDS=(777 778 779 780 781 782 783 784 785 786)  # Modify as needed
NBOOTS=(1000) # Number of bootstrap samples for pcs_oob
TRAIN_SIZES=(0.8)
TRAIN_FRACS=(1.0) # Varying fraction of training set used
NMODELS=(1 2 3 4 5)
CALIBRATIONS=("multiplicative")
ABL="model"

# # ABLATION EXPERIMENT 4: ADDITIVE CALIBRATION OF PCS
# DATASETS=("data_parkinsons")
# UQ_METHODS=("pcs_oob")
# ALL_ESTIMATORS=("XGBoost" "RandomForest")
# REDUCED_ESTIMATORS=("XGBoost")  # For majority_vote, pcs_uq, pcs_oob
# SEEDS=(777 778 779 780 781 782 783 784 785 786)  # Modify as needed
# NBOOTS=(1000) # Number of bootstrap samples for pcs_oob
# TRAIN_SIZES=(0.8)
# TRAIN_FRACS=(1.0) # Varying fraction of training set used
# NMODELS=(1)
# CALIBRATIONS=("additive" "multiplicative")  # Different calibration methods
# ABL="calib"


# Calculate total job count
TOTAL_JOBS=0
for uq in "${UQ_METHODS[@]}"; do
    if [[ "$uq" == "majority_vote" || "$uq" == "pcs_uq" || "$uq" == "pcs_oob" ]]; then
        TOTAL_JOBS=$(( TOTAL_JOBS + ${#DATASETS[@]} * ${#SEEDS[@]} * ${#REDUCED_ESTIMATORS[@]} * ${#NBOOTS[@]} * ${#TRAIN_SIZES[@]} * ${#TRAIN_FRACS[@]} * ${#NMODELS[@]} * ${#CALIBRATIONS[@]} ))
    else
        TOTAL_JOBS=$(( TOTAL_JOBS + ${#DATASETS[@]} * ${#SEEDS[@]} * ${#ALL_ESTIMATORS[@]} * ${#NBOOTS[@]} * ${#TRAIN_SIZES[@]} * ${#TRAIN_FRACS[@]} * ${#NMODELS[@]} * ${#CALIBRATIONS[@]} ))
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
                    for nboot in "${NBOOTS[@]}"; do
                        for train_frac in "${TRAIN_FRACS[@]}"; do
                            for nmodels in "${NMODELS[@]}"; do
                                for calibs in "${CALIBRATIONS[@]}"; do
                                    if [[ "$job_counter" -eq "$TASK_ID" ]]; then
                                        DATASET="$dataset"
                                        UQ_METHOD="$uq"
                                        ESTIMATOR="$estimator"
                                        SEED="$seed"
                                        TRAIN_SIZE="$train_size"
                                        NBOOT="$nboot"
                                        TRAIN_FRAC="$train_frac"
                                        NMODEL="$nmodels"
                                        CALIB="$calibs"
                                    fi
                                    job_counter=$((job_counter + 1))
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# Create dataset and UQ method specific log directory with absolute path
LOG_DIR="logs/${DATASET}/${UQ_METHOD}/${ESTIMATOR}"
mkdir -p "$LOG_DIR"

# Set log file paths with absolute paths
LOG_FILE="$LOG_DIR/${DATASET}_${UQ_METHOD}_${ESTIMATOR}_${SEED}_${NBOOT}_${TRAIN_FRAC}_${NMODEL}_${CALIB}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERR_FILE="$LOG_DIR/${DATASET}_${UQ_METHOD}_${ESTIMATOR}_${SEED}_${NBOOT}_${TRAIN_FRAC}_${NMODEL}_${CALIB}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

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
echo "Number of Boots: $NBOOT"
echo "Train Fraction: $TRAIN_FRAC"
echo "Number of Models: $NMODEL"
echo "Calibration Method: $CALIB"

# Run the Python script
conda run -n pcs_uq python experiments/scripts/ablation_exp.py --dataset "$DATASET" --UQ_method "$UQ_METHOD" --seed "$SEED" --estimator "$ESTIMATOR" --train_size "$TRAIN_SIZE" --n_boot "$NBOOT" --abl_type "$ABL" --train_frac "$TRAIN_FRAC" --n_model "$NMODEL" --calibration_method "$CALIB"

echo "Job completed at $(date)"
