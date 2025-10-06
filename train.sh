#!/bin/bash

# train_vcg --config /home/wlsdzyzl/project/vcg/resources/train_skae_cnn_12_condition_config_dilated.yaml
test_vcg --config /home/wlsdzyzl/project/vcg/resources/test_skae_cnn_condition_config.yaml
train_vcg --config /home/wlsdzyzl/project/vcg/resources/train_edm_condition_config.yaml
test_vcg --config /home/wlsdzyzl/project/vcg/resources/test_ldm_condition_config.yaml