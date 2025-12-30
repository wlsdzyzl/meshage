#!/bin/bash
#SBATCH --job-name=test_ldm
#SBATCH -w inspur1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 144000

test_vcg --config /home/guoqingzhang/vcg/resources/skcnn/test_ldm_skcnn_condition_config.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/skcnn/test_ldm_skcnn_condition_config_wo_skc.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/spcnn/test_ldm_spcnn_condition_config.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/spcnn/test_ldm_spcnn_condition_config_wo_lpc.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/skspcnn/test_ldm_skspcnn_condition_config.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/skspcnn/test_ldm_skspcnn_condition_config_wo_lpc.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/sktrans/test_ldm_sktrans_condition_config.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/sktrans/test_ldm_sktrans_condition_config_wo_skc.yaml

