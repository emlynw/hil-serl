export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../../train_rlpd_gamepad.py "$@" \
    --exp_name=strawb_real \
    --checkpoint_path=xirl_classifier_100 \
    --actor \