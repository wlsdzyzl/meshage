#!/bin/bash
#SBATCH --job-name=train_edm_trans
#SBATCH -w inspur1
#SBATCH --gres=shard:3
#SBATCH -c 8
#SBATCH -t 144000

# train_vcg --config /home/guoqingzhang/vcg/resources/train_edm_skcnn_condition_config.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/train_edm_skcnn_condition_config_wo_skc.yaml

# train_vcg --config /home/guoqingzhang/vcg/resources/train_edm_spcnn_condition_config.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/train_edm_spcnn_condition_config_wo_lpc.yaml

# train_vcg --config /home/guoqingzhang/vcg/resources/train_edm_sktrans_condition_config.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/train_edm_skspcnn_condition_config.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/train_edm_skspcnn_condition_config_wo_lpc.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/sktrans/train_edm_sktrans_condition_config_wo_skc.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/sksptrans/test_skspae_trans_condition_config_wo_lpc.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/sksptrans/train_edm_sksptrans_condition_config.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/skcnn/train_edm_skcnn_condition_config_512.yaml
train_vcg --config /home/guoqingzhang/vcg/resources/skcnn/train_edm_skcnn_condition_config_512_with_la_os.yaml

