work_path=$(dirname $0)
#conda activate v2xvit
export PYTHONPATH=/home/JJ_Group/cheny/v2x-vit/:$PYTHONPATH
srun --gres=gpu:a100:0 --cpus-per-task 16 --time 180  \
python -u -W ignore inference.py \
--model_dir $1 \
--fusion_method intermediate --save_npy \
2>&1 | tee $1/tee_cpu.log
