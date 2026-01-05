@echo off
chcp 65001
REM ==========================================
REM MAC-SQL RAG消融实验脚本
REM ==========================================
REM 实验设置：
REM 1. Baseline: 无RAG
REM 2. RAG + keyword策略
REM 3. RAG + embedding策略  
REM 4. RAG + hybrid策略
REM ==========================================

REM 创建输出目录
if not exist "./outputs/ablation" mkdir "./outputs/ablation"
if not exist "./outputs/ablation/bird" mkdir "./outputs/ablation/bird"
if not exist "./outputs/ablation/spider" mkdir "./outputs/ablation/spider"

REM ==========================================
REM BIRD数据集消融实验
REM ==========================================

echo ==========================================
echo [1/4] BIRD Baseline (No RAG)
echo ==========================================
python ./run.py --dataset_name "bird" ^
   --dataset_mode="dev" ^
   --input_file "./data/bird/dev.json" ^
   --db_path "./data/bird/dev_databases/" ^
   --tables_json_path "./data/bird/dev_tables.json" ^
   --output_file "./outputs/ablation/bird/output_baseline.json" ^
   --log_file "./outputs/ablation/bird/log_baseline.txt"

echo ==========================================
echo [2/4] BIRD + RAG (Keyword Strategy)
echo ==========================================
python ./run.py --dataset_name "bird" ^
   --dataset_mode="dev" ^
   --input_file "./data/bird/dev.json" ^
   --db_path "./data/bird/dev_databases/" ^
   --tables_json_path "./data/bird/dev_tables.json" ^
   --output_file "./outputs/ablation/bird/output_rag_keyword.json" ^
   --log_file "./outputs/ablation/bird/log_rag_keyword.txt" ^
   --enable_rag ^
   --retrieval_strategy "keyword" ^
   --decomposer_top_k 2 ^
   --refiner_top_k 2

echo ==========================================
echo [3/4] BIRD + RAG (Embedding Strategy)
echo ==========================================
python ./run.py --dataset_name "bird" ^
   --dataset_mode="dev" ^
   --input_file "./data/bird/dev.json" ^
   --db_path "./data/bird/dev_databases/" ^
   --tables_json_path "./data/bird/dev_tables.json" ^
   --output_file "./outputs/ablation/bird/output_rag_embedding.json" ^
   --log_file "./outputs/ablation/bird/log_rag_embedding.txt" ^
   --enable_rag ^
   --retrieval_strategy "embedding" ^
   --decomposer_top_k 2 ^
   --refiner_top_k 2

echo ==========================================
echo [4/4] BIRD + RAG (Hybrid Strategy)
echo ==========================================
python ./run.py --dataset_name "bird" ^
   --dataset_mode="dev" ^
   --input_file "./data/bird/dev.json" ^
   --db_path "./data/bird/dev_databases/" ^
   --tables_json_path "./data/bird/dev_tables.json" ^
   --output_file "./outputs/ablation/bird/output_rag_hybrid.json" ^
   --log_file "./outputs/ablation/bird/log_rag_hybrid.txt" ^
   --enable_rag ^
   --retrieval_strategy "hybrid" ^
   --decomposer_top_k 2 ^
   --refiner_top_k 2

echo ==========================================
echo BIRD Experiments Complete!
echo ==========================================

REM ==========================================
REM BIRD评估 (需要为每个实验分别评估)
REM ==========================================

echo ==========================================
echo Evaluating BIRD Results...
echo ==========================================

REM Baseline评估
echo [Evaluating] Baseline...
python ./evaluation/evaluation_bird_ex.py ^
    --db_root_path "./data/bird/dev_databases/" ^
    --predicted_sql_json_path "./outputs/ablation/bird/predict_dev_baseline.json" ^
    --data_mode "dev" ^
    --ground_truth_sql_path "./data/bird/dev_gold.sql" ^
    --num_cpus 12 ^
    --mode_predict "gpt" ^
    --diff_json_path "./data/bird/dev.json" ^
    --meta_time_out 30.0

REM RAG Keyword评估
echo [Evaluating] RAG Keyword...
python ./evaluation/evaluation_bird_ex.py ^
    --db_root_path "./data/bird/dev_databases/" ^
    --predicted_sql_json_path "./outputs/ablation/bird/predict_dev_rag_keyword.json" ^
    --data_mode "dev" ^
    --ground_truth_sql_path "./data/bird/dev_gold.sql" ^
    --num_cpus 12 ^
    --mode_predict "gpt" ^
    --diff_json_path "./data/bird/dev.json" ^
    --meta_time_out 30.0

REM RAG Embedding评估
echo [Evaluating] RAG Embedding...
python ./evaluation/evaluation_bird_ex.py ^
    --db_root_path "./data/bird/dev_databases/" ^
    --predicted_sql_json_path "./outputs/ablation/bird/predict_dev_rag_embedding.json" ^
    --data_mode "dev" ^
    --ground_truth_sql_path "./data/bird/dev_gold.sql" ^
    --num_cpus 12 ^
    --mode_predict "gpt" ^
    --diff_json_path "./data/bird/dev.json" ^
    --meta_time_out 30.0

REM RAG Hybrid评估
echo [Evaluating] RAG Hybrid...
python ./evaluation/evaluation_bird_ex.py ^
    --db_root_path "./data/bird/dev_databases/" ^
    --predicted_sql_json_path "./outputs/ablation/bird/predict_dev_rag_hybrid.json" ^
    --data_mode "dev" ^
    --ground_truth_sql_path "./data/bird/dev_gold.sql" ^
    --num_cpus 12 ^
    --mode_predict "gpt" ^
    --diff_json_path "./data/bird/dev.json" ^
    --meta_time_out 30.0

echo ==========================================
echo All evaluations complete!
echo ==========================================

pause
