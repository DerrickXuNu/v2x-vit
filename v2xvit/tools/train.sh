work_path=$(dirname $0)
#conda activate v2xvit
export PYTHONPATH=/home/JJ_Group/cheny/v2x-vit/:$PYTHONPATH
srun --gres=gpu:a100:1 --time 1800 \
python -u -W ignore train.py \
--hypes_yaml $1 \
2>&1 | tee $1/tee.log
