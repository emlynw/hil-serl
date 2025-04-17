export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd_no_itvn.py "$@" \
    --exp_name=robomimic_can \
    --checkpoint_path=can_1 \
    --actor \