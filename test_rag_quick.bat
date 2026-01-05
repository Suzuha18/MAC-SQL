@echo off
chcp 65001
REM ==========================================
REM MAC-SQL RAG功能快速测试脚本
REM 使用BIRD dev数据集的前5条进行快速验证
REM ==========================================

echo ==========================================
echo RAG Quick Test on BIRD Dev Dataset
echo ==========================================

REM 创建输出目录
if not exist "./outputs/rag_test" mkdir "./outputs/rag_test"

echo ==========================================
echo [1/2] Baseline (No RAG)
echo ==========================================
python ./run.py --dataset_name "bird" ^
   --dataset_mode="dev" ^
   --input_file "./data/bird/bird_dev_20240627/dev.json" ^
   --db_path "./data/bird/bird_dev_20240627/dev_databases" ^
   --tables_json_path "./data/bird/bird_dev_20240627/dev_tables.json" ^
   --output_file "./outputs/rag_test/output_baseline.json" ^
   --log_file "./outputs/rag_test/log_baseline.txt"

echo ==========================================
echo [2/2] With RAG (Keyword Strategy)
echo ==========================================
python ./run.py --dataset_name "bird" ^
   --dataset_mode="dev" ^
   --input_file "./data/bird/bird_dev_20240627/dev.json" ^
   --db_path "./data/bird/bird_dev_20240627/dev_databases" ^
   --tables_json_path "./data/bird/bird_dev_20240627/dev_tables.json" ^
   --output_file "./outputs/rag_test/output_rag.json" ^
   --log_file "./outputs/rag_test/log_rag.txt" ^
   --enable_rag ^
   --retrieval_strategy "keyword" ^
   --decomposer_top_k 2 ^
   --refiner_top_k 2

echo ==========================================
echo Quick Test Complete!
echo Check outputs in ./outputs/rag_test/
echo ==========================================

pause


