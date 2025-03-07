export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python ../../train_rl.py "$@" \
    --exp_name=strawb_sim \
    --checkpoint_path=fruit_gym_1 \
    --demo_path=/home/emlyn/rl_franka/hil-serl/examples/demo_data_sim_128/ \
    --learner \