work_path=$(dirname $0)
#conda activate v2xvit
export PYTHONPATH=/home/JJ_Group/cheny/v2x-vit/:$PYTHONPATH
srun --gres=gpu:a100:1 --time 500 \
python -u -W ignore inference.py \
--hypes_yaml $1 \
--model_dir $2 \
--load_epoch $3 \
2>&1 | tee ./test_$(date +"%y%m%d%H%M%S").log
