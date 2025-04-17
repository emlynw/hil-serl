export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python ../../train_rlpd_no_itvn.py "$@" \
    --exp_name=robomimic_can \
    --checkpoint_path=can_1 \
    --demo_path=/home/emlyn/rl_franka/hil-serl/examples/demo_data_128/ \
    --learner \