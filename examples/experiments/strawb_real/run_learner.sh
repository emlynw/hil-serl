export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python ../../train_rlpd_gamepad.py "$@" \
    --exp_name=strawb_real \
    --checkpoint_path=second_attempt \
    --demo_path=/home/emlyn/rl_franka/hil-serl/examples/demo_data/ \
    --learner \