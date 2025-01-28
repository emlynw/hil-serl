export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python ../../train_rlpd_gamepad.py "$@" \
    --exp_name=strawb_real \
    --checkpoint_path=learned_gripper \
    --demo_path=/home/emlyn/rl_franka/hil-serl/examples/demo_data/strawb_real_5_demos_2025-01-28_15-58-58.pkl \
    --learner \