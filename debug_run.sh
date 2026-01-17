#! /bin/bash

set -e

python src/main.py \
    dreamerv3.num_updates=30 \
    dreamerv3.num_worlds=32 \
    dreamerv3.batch_size=16 \
    dreamerv3.rollout_length=256 \
    dreamerv3.log_tensorboard=true \
    dreamerv3.save_checkpoints=true \
    dreamerv3.logging.log_dir=logs \
    dreamerv3.batch_length=64 \
    dreamerv3.imag_horizon=4 \
    dreamerv3.num_grad_steps=8 \
    dreamerv3.imag_last_states=8 \
    dreamerv3.replay_buffer.max_length=1024 \
    dreamerv3.logger.experiment_name=debug \
    dreamerv3.do_update=false
