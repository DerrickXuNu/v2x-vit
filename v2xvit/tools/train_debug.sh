work_path=$(dirname $0)
#conda activate v2xvit
export PYTHONPATH=/home/JJ_Group/cheny/v2x-vit/:$PYTHONPATH
srun --gres=gpu:a100:1 --time 2800 \
python -u -W ignore train.py \
2>&1 | tee ./tee_logs/train_$(date +"%y%m%d%H%M%S").log
