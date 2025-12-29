#!/bin/bash
#SBATCH --job-name=train_vessel_imagecas
#SBATCH -w inspur1
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH -t 144000

train_vcg --config /home/guoqingzhang/vcg/resources/vessel/imagecas_vessel_diff/train_skae_cnn_12_condition_config_dilated_with_skc_latent_atten.yaml
train_vcg --config /home/guoqingzhang/vcg/resources/vessel/imagecas_vessel_diff/retrain_skae_cnn_12_condition_config_dilated_with_skc_latent_atten.yaml
test_vcg --config /home/guoqingzhang/vcg/resources/vessel/imagecas_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_skc_latent_atten.yaml
train_vcg --config /home/guoqingzhang/vcg/resources/vessel/imagecas_vessel_diff/train_edm_condition_config_with_skc_latent_atten.yaml