export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd_gamepad.py "$@" \
    --exp_name=strawb_real \
    --checkpoint_path=learned_gripper \
    --actor \