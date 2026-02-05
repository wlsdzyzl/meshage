#!/bin/bash
#SBATCH --job-name=test_ldm
#SBATCH -w inspur1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 144000

# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_ldm_skcnn_condition_config.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_ldm_skcnn_condition_config_wo_skc.yaml
test_meshage --config /home/guoqingzhang/meshage/resources/spcnn/test_ldm_spcnn_condition_config.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/spcnn/test_ldm_spcnn_condition_config_wo_lpc.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skspcnn/test_ldm_skspcnn_condition_config.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skspcnn/test_ldm_skspcnn_condition_config_wo_lpc.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/sktrans/test_ldm_sktrans_condition_config.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/sktrans/test_ldm_sktrans_condition_config_wo_skc.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_ldm_skcnn_condition_config_512.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_ldm_skcnn_condition_config_512_with_la_os.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_ldm_skcnn_condition_config_with_la.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_ldm_skcnn_condition_config_512_with_la.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/spcnn/test_ldm_spcnn_condition_config_with_la.yaml