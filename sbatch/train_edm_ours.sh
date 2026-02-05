#!/bin/bash
#SBATCH --job-name=train_edm_trans
#SBATCH -w inspur1
#SBATCH --gres=shard:3
#SBATCH -c 8
#SBATCH -t 144000

# train_meshage --config /home/guoqingzhang/meshage/resources/train_edm_skcnn_condition_config.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/train_edm_skcnn_condition_config_wo_skc.yaml

# train_meshage --config /home/guoqingzhang/meshage/resources/train_edm_spcnn_condition_config.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/train_edm_spcnn_condition_config_wo_lpc.yaml

# train_meshage --config /home/guoqingzhang/meshage/resources/train_edm_sktrans_condition_config.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/train_edm_skspcnn_condition_config.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/train_edm_skspcnn_condition_config_wo_lpc.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/sktrans/train_edm_sktrans_condition_config_wo_skc.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/sksptrans/test_skspae_trans_condition_config_wo_lpc.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/sksptrans/train_edm_sksptrans_condition_config.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_edm_skcnn_condition_config_512.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_edm_skcnn_condition_config_512_with_la_os.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_skae_cnn_condition_config_with_la_for_train.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_skae_cnn_condition_config_with_la.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_edm_skcnn_condition_config_with_la.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_skae_cnn_condition_config_with_la_512.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_skae_cnn_condition_config_with_la_512_for_train.yaml

# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_edm_skcnn_condition_config_512_with_la.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_skae_cnn_condition_config_wo_skc_with_os.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_skae_cnn_condition_config_wo_skc_with_os_for_train.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/skcnn/train_edm_skcnn_condition_config_wo_skc_with_os.yaml

test_meshage --config /home/guoqingzhang/meshage/resources/spcnn/test_spae_cnn_condition_config_with_la.yaml
test_meshage --config /home/guoqingzhang/meshage/resources/spcnn/test_spae_cnn_condition_config_with_la_for_train.yaml
train_meshage --config /home/guoqingzhang/meshage/resources/spcnn/train_edm_spcnn_condition_config_with_la.yaml