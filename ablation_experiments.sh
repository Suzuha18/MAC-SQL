#!/bin/bash
# ==========================================
# MAC-SQL RAG消融实验脚本 (Linux/Mac)
# ==========================================
# 实验设置：
# 1. Baseline: 无RAG
# 2. RAG + keyword策略
# 3. RAG + embedding策略  
# 4. RAG + hybrid策略
# ==========================================

set -e

# 创建输出目录
mkdir -p ./outputs/ablation/bird
mkdir -p ./outputs/ablation/spider

# ==========================================
# BIRD数据集消融实验
# ==========================================

echo "=========================================="
echo "[1/4] BIRD Baseline (No RAG)"
echo "=========================================="
python ./run.py --dataset_name "bird" \
   --dataset_mode="dev" \
   --input_file "./data/bird/dev.json" \
   --db_path "./data/bird/dev_databases/" \
   --tables_json_path "./data/bird/dev_tables.json" \
   --output_file "./outputs/ablation/bird/output_baseline.json" \
   --log_file "./outputs/ablation/bird/log_baseline.txt"

echo "=========================================="
echo "[2/4] BIRD + RAG (Keyword Strategy)"
echo "=========================================="
python ./run.py --dataset_name "bird" \
   --dataset_mode="dev" \
   --input_file "./data/bird/dev.json" \
   --db_path "./data/bird/dev_databases/" \
   --tables_json_path "./data/bird/dev_tables.json" \
   --output_file "./outputs/ablation/bird/output_rag_keyword.json" \
   --log_file "./outputs/ablation/bird/log_rag_keyword.txt" \
   --enable_rag \
   --retrieval_strategy "keyword" \
   --decomposer_top_k 2 \
   --refiner_top_k 2

echo "=========================================="
echo "[3/4] BIRD + RAG (Embedding Strategy)"
echo "=========================================="
python ./run.py --dataset_name "bird" \
   --dataset_mode="dev" \
   --input_file "./data/bird/dev.json" \
   --db_path "./data/bird/dev_databases/" \
   --tables_json_path "./data/bird/dev_tables.json" \
   --output_file "./outputs/ablation/bird/output_rag_embedding.json" \
   --log_file "./outputs/ablation/bird/log_rag_embedding.txt" \
   --enable_rag \
   --retrieval_strategy "embedding" \
   --decomposer_top_k 2 \
   --refiner_top_k 2

echo "=========================================="
echo "[4/4] BIRD + RAG (Hybrid Strategy)"
echo "=========================================="
python ./run.py --dataset_name "bird" \
   --dataset_mode="dev" \
   --input_file "./data/bird/dev.json" \
   --db_path "./data/bird/dev_databases/" \
   --tables_json_path "./data/bird/dev_tables.json" \
   --output_file "./outputs/ablation/bird/output_rag_hybrid.json" \
   --log_file "./outputs/ablation/bird/log_rag_hybrid.txt" \
   --enable_rag \
   --retrieval_strategy "hybrid" \
   --decomposer_top_k 2 \
   --refiner_top_k 2

echo "=========================================="
echo "BIRD Experiments Complete!"
echo "=========================================="

# ==========================================
# BIRD评估
# ==========================================

echo "=========================================="
echo "Evaluating BIRD Results..."
echo "=========================================="

# Baseline评估
echo "[Evaluating] Baseline..."
python ./evaluation/evaluation_bird_ex.py \
    --db_root_path "./data/bird/dev_databases/" \
    --predicted_sql_json_path "./outputs/ablation/bird/predict_dev_baseline.json" \
    --data_mode "dev" \
    --ground_truth_sql_path "./data/bird/dev_gold.sql" \
    --num_cpus 12 \
    --mode_predict "gpt" \
    --diff_json_path "./data/bird/dev.json" \
    --meta_time_out 30.0

# RAG Keyword评估
echo "[Evaluating] RAG Keyword..."
python ./evaluation/evaluation_bird_ex.py \
    --db_root_path "./data/bird/dev_databases/" \
    --predicted_sql_json_path "./outputs/ablation/bird/predict_dev_rag_keyword.json" \
    --data_mode "dev" \
    --ground_truth_sql_path "./data/bird/dev_gold.sql" \
    --num_cpus 12 \
    --mode_predict "gpt" \
    --diff_json_path "./data/bird/dev.json" \
    --meta_time_out 30.0

# RAG Embedding评估
echo "[Evaluating] RAG Embedding..."
python ./evaluation/evaluation_bird_ex.py \
    --db_root_path "./data/bird/dev_databases/" \
    --predicted_sql_json_path "./outputs/ablation/bird/predict_dev_rag_embedding.json" \
    --data_mode "dev" \
    --ground_truth_sql_path "./data/bird/dev_gold.sql" \
    --num_cpus 12 \
    --mode_predict "gpt" \
    --diff_json_path "./data/bird/dev.json" \
    --meta_time_out 30.0

# RAG Hybrid评估
echo "[Evaluating] RAG Hybrid..."
python ./evaluation/evaluation_bird_ex.py \
    --db_root_path "./data/bird/dev_databases/" \
    --predicted_sql_json_path "./outputs/ablation/bird/predict_dev_rag_hybrid.json" \
    --data_mode "dev" \
    --ground_truth_sql_path "./data/bird/dev_gold.sql" \
    --num_cpus 12 \
    --mode_predict "gpt" \
    --diff_json_path "./data/bird/dev.json" \
    --meta_time_out 30.0

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
