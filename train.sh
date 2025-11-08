#!/bin/bash
train_vcg --config /home/wlsdzyzl/project/vcg/resources/skspsdf/retrain_skspae_cnn_12_condition_config_dilated.yaml
train_vcg --config /home/wlsdzyzl/project/vcg/resources/sksdf/retrain_skae_cnn_12_condition_config_dilated.yaml
train_vcg --config /home/wlsdzyzl/project/vcg/resources/spsdf/retrain_spae_cnn_12_condition_config_dilated.yaml
train_vcg --config /home/wlsdzyzl/project/vcg/resources/skspsdf/retrain_skspae_cnn_12_condition_config_dilated.yaml
