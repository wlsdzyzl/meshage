#!/bin/bash
# train_vcg --config /home/wlsdzyzl/project/vcg/resources/skspsdf/train_edm_condition_config.yaml
# train_vcg --config /home/wlsdzyzl/project/vcg/resources/spsdf/train_edm_condition_config.yaml
train_vcg --config /home/wlsdzyzl/project/vcg/resources/cow_vessel_diff/train_skae_cnn_12_condition_config_dilated_with_skc.yaml
train_vcg --config /home/wlsdzyzl/project/vcg/resources/imagecas_vessel_diff/train_skae_cnn_12_condition_config_dilated_with_skc.yaml
train_vcg --config /home/wlsdzyzl/project/vcg/resources/cow_vessel_diff/retrain_skae_cnn_12_condition_config_dilated_with_skc.yaml
train_vcg --config /home/wlsdzyzl/project/vcg/resources/imagecas_vessel_diff/retrain_skae_cnn_12_condition_config_dilated_with_skc.yaml
test_vcg --config /home/wlsdzyzl/project/vcg/resources/cow_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_skc.yaml
test_vcg --config /home/wlsdzyzl/project/vcg/resources/imagecas_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_skc.yaml
train_vcg --config /home/wlsdzyzl/project/vcg/resources/cow_vessel_diff/train_edm_condition_config_with_skc.yaml
train_vcg --config /home/wlsdzyzl/project/vcg/resources/imagecas_vessel_diff/train_edm_condition_config_with_skc.yaml

# train_vcg --config /home/wlsdzyzl/project/vcg/resources/cow_vessel_diff/train_tomesh_config.yaml
# train_vcg --config /home/wlsdzyzl/project/vcg/resources/imagecas_vessel_diff/train_tomesh_config.yaml