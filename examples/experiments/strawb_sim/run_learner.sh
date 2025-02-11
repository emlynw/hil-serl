export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python ../../train_rl_2.py "$@" \
    --exp_name=strawb_sim \
    --checkpoint_path=feb_11 \
    --learner \