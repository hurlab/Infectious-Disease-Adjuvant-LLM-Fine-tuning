#!/bin/bash
set -e

source ~/py39/bin/activate

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="/home/hasin.rehana/Infectious disease adjuvant/Infectious-Disease-Adjuvant-LLM"
SCRIPT="Hyperparameter_tuning_LLM_V2.py"
BASE_MODEL="/home/hasin.rehana/models/Llama-3.3-70B-Instruct"
OUT_ROOT="results/hparam_runs_Llama_3.3_70B_Instruct_GRID1_V2"

mkdir -p "$OUT_ROOT"

#nohup ./Hyperparameter_tuning_Llama_3.3_70B_Instruct_V2.sh \
#  > nohup_Hyperparameter_tuning_Llama_3.3_70B_Instruct_V2_GRID1.out \
#  2>&1 &

RUN_ID=1

for LR in 5e-5 2e-4; do
for EPOCHS in 3 5; do
for WARMUP in 0.0 0.1; do


RUN_DIR=$(printf "%s/run_%d" "$OUT_ROOT" "$RUN_ID")
mkdir -p "$RUN_DIR"

CONFIG_JSON="$RUN_DIR/config.json"

cat > "$CONFIG_JSON" << EOF
{
  "lr": $LR,
  "epochs": $EPOCHS,
  "batch_size": 2,
  "grad_accum": 8,
  "warmup_ratio": $WARMUP,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05
}
EOF



echo "=============================="
echo "Starting run $RUN_ID"
echo "Config: $CONFIG_JSON"
echo "=============================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python "$SCRIPT" \
  --base_model "$BASE_MODEL" \
  --config "$CONFIG_JSON" \
  --output_dir "$RUN_DIR" \
  > "$RUN_DIR/stdout.log" \
  2> "$RUN_DIR/stderr.log"

echo "Finished run $RUN_ID"

# HARD CUDA RESET BETWEEN RUNS
sleep 10

RUN_ID=$((RUN_ID+1))

done
done
done
