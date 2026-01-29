#!/bin/bash
#SBATCH --job-name=test_cnn
#SBATCH -w inspur1
#SBATCH --gres=shard:3
#SBATCH -c 8
#SBATCH -t 144000

# test_meshage --config /home/guoqingzhang/meshage/resources/test_skae_cnn_condition_config_for_train.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/test_skae_cnn_condition_config_wo_skc_for_train.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/test_spae_cnn_condition_config_for_train.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/test_spae_cnn_condition_config_wo_lpc_for_train.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_skae_cnn_condition_config.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_skae_cnn_condition_config_512.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_skae_cnn_condition_config_512.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_skae_cnn_condition_config_for_train_512.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/test_skae_cnn_condition_config_wo_skc.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/test_spae_cnn_condition_config.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/test_spae_cnn_condition_config_wo_lpc.yaml

# test_meshage --config /home/guoqingzhang/meshage/resources/test_skspae_cnn_condition_config_for_train.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/test_skae_trans_condition_config_for_train.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/test_skspae_cnn_condition_config.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/test_skae_trans_condition_config.yaml

# CUDA_VISIBLE_DEVICES=3 test_meshage --config /home/guoqingzhang/meshage/resources/test_skspae_cnn_condition_config_wo_lpc_for_train.yaml
# CUDA_VISIBLE_DEVICES=4 test_meshage --config /home/guoqingzhang/meshage/resources/test_skae_trans_condition_config_wo_skc_for_train.yaml
# CUDA_VISIBLE_DEVICES=3 test_meshage --config /home/guoqingzhang/meshage/resources/test_skspae_cnn_condition_config_wo_lpc.yaml
# CUDA_VISIBLE_DEVICES=4 test_meshage --config /home/guoqingzhang/meshage/resources/test_skae_trans_condition_config_wo_skc.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_timer_ldm_skcnn_condition_config.yaml