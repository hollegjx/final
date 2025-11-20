#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
项目路径集中配置
允许通过环境变量或服务器前缀快速切换运行环境

使用示例:
    from config import feature_cache_dir, exp_root

环境变量:
    export SERVER_BASE_PATH=/data1/jiangzhen/gjx   # 或部署机器上的实际根路径
    export FEATURE_CACHE_DIR=/custom/cache/dir
"""

import os

# =============================================================================
# 服务器路径前缀
# =============================================================================

SERVER_BASE_PATH = os.getenv('SERVER_BASE_PATH', '/data1/jiangzhen/gjx')


# =============================================================================
# 数据集根目录
# =============================================================================

cifar_10_root = os.getenv('CIFAR10_ROOT', '/work/sagar/datasets/cifar10')
cifar_100_root = os.getenv('CIFAR100_ROOT', os.path.join(SERVER_BASE_PATH, 'data/cifar100'))
cub_root = os.getenv('CUB_ROOT', '/work/sagar/datasets/CUB')
aircraft_root = os.getenv('AIRCRAFT_ROOT', '/work/khan/datasets/aircraft/fgvc-aircraft-2013b')
herbarium_dataroot = os.getenv('HERBARIUM_ROOT', '/work/sagar/datasets/herbarium_19/')
imagenet_root = os.getenv('IMAGENET_ROOT', '/scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12')
stanford_cars_root = os.getenv('STANFORD_CARS_ROOT', '/work/sagar/datasets/stanford_car')
osr_split_dir = os.getenv('OSR_SPLIT_DIR', '/users/sagar/kai_collab/osr_novel_categories/data/ssb_splits')


# =============================================================================
# 预训练&模型保存
# =============================================================================

dino_pretrain_path = os.getenv('DINO_PRETRAIN_PATH',
                               os.path.join(SERVER_BASE_PATH, 'pretrained/dino_vitbase16_pretrain.pth'))
dino_resnet50_pretrain_path = os.getenv('DINO_RESNET50_PRETRAIN_PATH',
                                        os.path.join(SERVER_BASE_PATH, 'pretrained/dino_resnet50_pretrain.pth'))
feature_extract_dir = os.getenv('FEATURE_EXTRACT_DIR',
                                os.path.join(SERVER_BASE_PATH, 'exp/newgpc/final/extracted_features_public_impl'))
exp_root = os.getenv('EXP_ROOT', os.path.join(SERVER_BASE_PATH, 'exp/newgpc/final/'))
superclass_model_root = os.getenv('SUPERCLASS_MODEL_ROOT',
                                  os.path.join(SERVER_BASE_PATH, 'checkpoints/gcdsuperclass1'))
checkpoint_root = os.getenv('CHECKPOINT_ROOT', os.path.join(SERVER_BASE_PATH, 'checkpoints'))


# =============================================================================
# 聚类/特征缓存
# =============================================================================

feature_cache_dir = os.getenv('FEATURE_CACHE_DIR', os.path.join(SERVER_BASE_PATH, 'checkpoints/features1'))
feature_cache_dir_nol2 = os.getenv('FEATURE_CACHE_DIR_NOL2',
                                   os.path.join(SERVER_BASE_PATH, 'checkpoints/features_nol2'))
clustering_log_dir = os.getenv('CLUSTERING_LOG_DIR',
                               os.path.join(SERVER_BASE_PATH, 'checkpoints/log'))
error_analysis_log_dir = os.getenv('ERROR_ANALYSIS_LOG_DIR',
                                   os.path.join(SERVER_BASE_PATH, 'checkpoints/logs'))


# =============================================================================
# 网格搜索相关
# =============================================================================

grid_search_output_dir = os.getenv('GRID_SEARCH_DIR',
                                   os.path.join(SERVER_BASE_PATH, 'checkpoints/search'))
heatmap_output_dir = os.getenv('HEATMAP_DIR',
                               os.path.join(SERVER_BASE_PATH, 'checkpoints/heatmaps'))
l2_search_output_dir = os.getenv('L2_SEARCH_DIR',
                                 os.path.join(SERVER_BASE_PATH, 'checkpoints/l2_search'))
l1l2_search_output_dir = os.getenv('L1L2_SEARCH_DIR',
                                   os.path.join(SERVER_BASE_PATH, 'checkpoints/l1l2_search'))
weight_application_report_dir = os.getenv('WEIGHT_APP_REPORT_DIR',
                                          os.path.join(SERVER_BASE_PATH, 'checkpoints/weight_application'))
l1l2_region_report_dir = os.getenv('L1L2_REGION_REPORT_DIR',
                                   os.path.join(SERVER_BASE_PATH, 'checkpoints/findL'))


# =============================================================================
# 其他工具路径
# =============================================================================

slurm_summary_root = os.getenv('SLURM_SUMMARY_ROOT',
                               os.path.join(SERVER_BASE_PATH, 'open_set_recognition/sweep_summary_files/ensemble_pkls'))
slurm_log_base_pattern = os.getenv('SLURM_LOG_BASE',
                                   os.path.join(SERVER_BASE_PATH, 'osr_novel_categories/slurm_outputs/myLog-{}.out'))
