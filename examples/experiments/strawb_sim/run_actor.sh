export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rl.py "$@" \
    --exp_name=strawb_sim \
    --checkpoint_path=first_run \
    --actor \