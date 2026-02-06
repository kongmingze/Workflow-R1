set -x
export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1

# ============ Path Configuration ============
TRAIN_DATA_PATH=/path/to/train.parquet
VAL_DATA_PATH=/path/to/test.parquet
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
# =============================================

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${VAL_DATA_PATH} \
    data.train_batch_size=512 \
    data.val_batch_size=1024 \
    data.max_prompt_length=1280 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=workflow_r1 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=21 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=64 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gsspo \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    data.return_raw_chat=True \
    reward_model.reward_manager=workflow_r1 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_workflow_optimization' \
    trainer.experiment_name='GSsPO_7B' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=38 \
    trainer.test_freq=38 \
    trainer.val_before_train=False \
    trainer.total_epochs=2 $@
