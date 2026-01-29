#!/bin/bash
#SBATCH --job-name=train_skspcnn
#SBATCH -w inspur1
#SBATCH --gres=shard:2
#SBATCH -c 8
#SBATCH -t 144000
# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_skae_cnn_condition_config_wo_skc_with_os.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_skae_cnn_condition_config.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_skae_cnn_condition_config_with_latent_atten.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_skae_cnn_condition_config_512_with_latent_atten.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_skae_cnn_condition_config_512_with_latent_atten_pre_skel.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_skae_cnn_condition_config_wo_skc.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/spcnn/train_spae_cnn_condition_config.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/spcnn/train_spae_cnn_condition_config_wo_lpc.yaml

# train_meshage --config /home/guoqingzhang/meshage/resources/skspcnn/train_skspae_cnn_condition_config.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/skspcnn/train_skspae_cnn_condition_config_wo_lpc.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_skae_cnn_condition_config_512.yaml

# train_meshage --config /home/guoqingzhang/meshage/resources/sktrans/train_skae_trans_condition_config.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/sktrans/train_skae_trans_condition_config_wo_skc.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/train_skspae_trans_condition_config_wo_lpc.yaml
train_meshage --config /home/guoqingzhang/meshage/resources/spcnn/train_spae_cnn_condition_config_with_la.yaml