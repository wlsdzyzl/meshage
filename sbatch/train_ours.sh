#!/bin/bash
#SBATCH --job-name=train_skspcnn
#SBATCH -w inspur1
#SBATCH --gres=shard:2
#SBATCH -c 8
#SBATCH -t 144000
# train_vcg --config /home/guoqingzhang/vcg/resources/skcnn/train_skae_cnn_condition_config.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/skcnn/train_skae_cnn_condition_config_with_latent_atten.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/skcnn/train_skae_cnn_condition_config_512_with_latent_atten.yaml
train_vcg --config /home/guoqingzhang/vcg/resources/skcnn/train_skae_cnn_condition_config_512_with_latent_atten_pre_skel.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/skcnn/train_skae_cnn_condition_config_wo_skc.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/spcnn/train_spae_cnn_condition_config.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/spcnn/train_spae_cnn_condition_config_wo_lpc.yaml

# train_vcg --config /home/guoqingzhang/vcg/resources/skspcnn/train_skspae_cnn_condition_config.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/skspcnn/train_skspae_cnn_condition_config_wo_lpc.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/skcnn/train_skae_cnn_condition_config_512.yaml

# train_vcg --config /home/guoqingzhang/vcg/resources/sktrans/train_skae_trans_condition_config.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/sktrans/train_skae_trans_condition_config_wo_skc.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/train_skspae_trans_condition_config_wo_lpc.yaml