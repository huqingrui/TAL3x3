import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
import cv2
import glob
from collections import defaultdict, OrderedDict
from datetime import datetime
import csv
import multiprocessing
import time
import math
import warnings
import argparse
import psutil
import gc

from rku_evaluation import np_box_list, np_box_ops

mp.set_start_method('spawn', force=True)

class PerformanceMonitor:
    def __init__(self, rank=0):
        self.rank = rank
        self.training_start_time = None
        self.epoch_start_time = None
        self.batch_times = []
        self.memory_usage = []
        self.inference_times = []
        self.total_training_time = 0
        self.data_loading_times = []
        self.forward_times = []
        self.backward_times = []

    def start_training(self):
        self.training_start_time = time.time()
        if self.rank == 0:
            print(f"Starting performance monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def start_epoch(self):
        self.epoch_start_time = time.time()

    def record_batch_time(self, batch_time):
        self.batch_times.append(batch_time)

    def record_data_loading_time(self, data_time):
        self.data_loading_times.append(data_time)

    def record_forward_time(self, forward_time):
        self.forward_times.append(forward_time)

    def record_backward_time(self, backward_time):
        self.backward_times.append(backward_time)

    def record_memory_usage(self):
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024 ** 3
                gpu_max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
            else:
                gpu_memory = 0
                gpu_memory_cached = 0
                gpu_max_memory = 0

            cpu_memory = psutil.virtual_memory().used / 1024 ** 3

            self.memory_usage.append({
                'gpu_allocated': gpu_memory,
                'gpu_cached': gpu_memory_cached,
                'gpu_max': gpu_max_memory,
                'cpu_used': cpu_memory,
                'cpu_percent': psutil.virtual_memory().percent,
                'timestamp': time.time()
            })
        except Exception as e:
            if self.rank == 0:
                print(f"Memory monitoring error: {e}")

    def record_inference_time(self, inference_time, batch_size):
        try:
            fps = batch_size / inference_time if inference_time > 0 else 0
            self.inference_times.append({
                'time': inference_time,
                'batch_size': batch_size,
                'fps': fps,
                'timestamp': time.time()
            })
        except Exception as e:
            if self.rank == 0:
                print(f"Inference time recording error: {e}")

    def end_training(self):
        if self.training_start_time:
            self.total_training_time = time.time() - self.training_start_time

    def get_avg_batch_time(self):
        return np.mean(self.batch_times) if self.batch_times else 0

    def get_avg_data_loading_time(self):
        return np.mean(self.data_loading_times) if self.data_loading_times else 0

    def get_avg_forward_time(self):
        return np.mean(self.forward_times) if self.forward_times else 0

    def get_avg_backward_time(self):
        return np.mean(self.backward_times) if self.backward_times else 0

    def get_avg_fps(self):
        if self.inference_times:
            return np.mean([item['fps'] for item in self.inference_times])
        return 0

    def get_peak_memory(self):
        if self.memory_usage:
            peak_gpu = max([item['gpu_allocated'] for item in self.memory_usage])
            peak_cpu = max([item['cpu_used'] for item in self.memory_usage])
            return peak_gpu, peak_cpu
        return 0, 0

    def get_throughput(self):
        if self.total_training_time > 0 and len(self.batch_times) > 0:
            return len(self.batch_times) / self.total_training_time
        return 0

    def print_summary(self):
        if self.rank == 0:
            print(f"\n{'=' * 50}")
            print(f"{'Performance Summary':^50}")
            print(f"{'=' * 50}")

            print(f"Total training time: {self.total_training_time:.2f} seconds ({self.total_training_time / 3600:.2f} hours)")
            print(f"Average batch time: {self.get_avg_batch_time():.4f} seconds")

            avg_data_time = self.get_avg_data_loading_time()
            avg_forward_time = self.get_avg_forward_time()
            avg_backward_time = self.get_avg_backward_time()
            avg_batch_time = self.get_avg_batch_time()

            if avg_batch_time > 0:
                print(f"Average data loading time: {avg_data_time:.4f} seconds ({avg_data_time / avg_batch_time * 100:.1f}%)")
                print(f"Average forward time: {avg_forward_time:.4f} seconds ({avg_forward_time / avg_batch_time * 100:.1f}%)")
                print(f"Average backward time: {avg_backward_time:.4f} seconds ({avg_backward_time / avg_batch_time * 100:.1f}%)")

            print(f"Average inference FPS: {self.get_avg_fps():.2f}")
            print(f"Training throughput: {self.get_throughput():.2f} batches/second")

            peak_gpu, peak_cpu = self.get_peak_memory()
            print(f"Peak GPU memory usage: {peak_gpu:.2f} GB")
            print(f"Peak CPU memory usage: {peak_cpu:.2f} GB")

            print(f"{'=' * 50}")

def get_memory_info():
    memory_info = {}
    try:
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024 ** 3
            memory_info['gpu_cached'] = torch.cuda.memory_reserved() / 1024 ** 3
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024 ** 3
            memory_info['gpu_max_cached'] = torch.cuda.max_memory_reserved() / 1024 ** 3
        else:
            memory_info['gpu_allocated'] = 0
            memory_info['gpu_cached'] = 0
            memory_info['gpu_max_allocated'] = 0
            memory_info['gpu_max_cached'] = 0

        cpu_mem = psutil.virtual_memory()
        memory_info['cpu_used'] = cpu_mem.used / 1024 ** 3
        memory_info['cpu_total'] = cpu_mem.total / 1024 ** 3
        memory_info['cpu_percent'] = cpu_mem.percent

    except Exception as e:
        print(f"Error getting memory info: {e}")
        memory_info = {
            'gpu_allocated': 0, 'gpu_cached': 0, 'gpu_max_allocated': 0, 'gpu_max_cached': 0,
            'cpu_used': 0, 'cpu_total': 0, 'cpu_percent': 0
        }
    return memory_info

def print_memory_usage(prefix="", rank=0):
    if rank == 0:
        try:
            mem_info = get_memory_info()
            print(f"{prefix} - GPU: {mem_info['gpu_allocated']:.2f}GB allocated, "
                  f"{mem_info['gpu_cached']:.2f}GB cached | "
                  f"CPU: {mem_info['cpu_used']:.2f}GB ({mem_info['cpu_percent']:.1f}%)")
        except Exception as e:
            print(f"{prefix} - Memory info failed: {e}")

def print_system_info(rank=0):
    if rank == 0:
        print(f"\n{'=' * 40}")
        print(f"{'System Info':^40}")
        print(f"{'=' * 40}")

        print(f"PyTorch version: {torch.__version__}")
        print(f"Python version: {os.sys.version.split()[0]}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("CUDA not available")

        print(f"CPU cores: {multiprocessing.cpu_count()}")
        cpu_mem = psutil.virtual_memory()
        print(f"System memory: {cpu_mem.total / 1024 ** 3:.1f} GB")
        print(f"{'=' * 40}")

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

class DeConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: Union[int, Tuple[int]] = (1, 1, 1),
                 padding: Union[int, Tuple[int]] = 0,
                 bias: bool = False,
                 with_bn: bool = True,
                 with_relu: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.with_bn = with_bn
        self.with_relu = with_relu

        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 5
        N, C, T, H, W = x.shape
        out_shape = (N, self.out_channels, self.stride[0] * T,
                     self.stride[1] * H, self.stride[2] * W)
        x = self.conv(x, output_size=out_shape)
        if self.with_bn:
            x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 inflate=True):
        super(Bottleneck, self).__init__()
        self.inflate = inflate

        if isinstance(stride, int):
            temporal_stride = 1
            spatial_stride = stride
            self.stride_3d = (temporal_stride, spatial_stride, spatial_stride)
        else:
            self.stride_3d = stride

        if self.inflate:
            conv1_kernel_size = (3, 1, 1)
            conv1_padding = (1, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=conv1_kernel_size,
                               padding=conv1_padding, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=conv2_kernel_size,
                               stride=self.stride_3d, padding=conv2_padding, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ImprovedBasketballEventModel(nn.Module):
    def __init__(self, num_classes=9, hidden_dim=512, num_heads=8, num_layers=4, dropout=0.1,
                 use_bbox_feat=True, use_skeleton_feat=True, use_position_feat=True, use_trajectory_feat=True):
        super(ImprovedBasketballEventModel, self).__init__()

        self.use_bbox_feat = use_bbox_feat
        self.use_skeleton_feat = use_skeleton_feat
        self.use_position_feat = use_position_feat
        self.use_trajectory_feat = use_trajectory_feat

        self.bbox_projection = nn.Linear(2 * 1024, hidden_dim)
        self.skeleton_projection = nn.Linear(2 * 256, hidden_dim)
        self.position_projection = nn.Linear(256, hidden_dim)
        self.trajectory_projection = nn.Linear(2 * 256, hidden_dim)

        self.event_type_embedding = nn.Embedding(2, hidden_dim)

        self.single_player_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.num_features = sum([self.use_bbox_feat, self.use_skeleton_feat,
                                 self.use_position_feat, self.use_trajectory_feat]) + 1

        self.feature_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, bbox_feature, skeleton_feature, project_position_encoding, trajectory_feature, is_single_player):
        batch_size = bbox_feature.size(0)

        bbox_feature = bbox_feature.reshape(batch_size, -1)
        skeleton_feature = skeleton_feature.reshape(batch_size, -1)
        trajectory_feature = trajectory_feature.reshape(batch_size, -1)

        if self.use_bbox_feat:
            bbox_feat = self.bbox_projection(bbox_feature)
        else:
            bbox_feat = torch.zeros(batch_size, self.bbox_projection.out_features,
                                    device=bbox_feature.device, dtype=bbox_feature.dtype)

        if self.use_skeleton_feat:
            skeleton_feat = self.skeleton_projection(skeleton_feature)
        else:
            skeleton_feat = torch.zeros(batch_size, self.skeleton_projection.out_features,
                                        device=skeleton_feature.device, dtype=skeleton_feature.dtype)

        if self.use_position_feat:
            position_feat = self.position_projection(project_position_encoding)
        else:
            position_feat = torch.zeros(batch_size, self.position_projection.out_features,
                                        device=project_position_encoding.device, dtype=project_position_encoding.dtype)

        if self.use_trajectory_feat:
            trajectory_feat = self.trajectory_projection(trajectory_feature)
        else:
            trajectory_feat = torch.zeros(batch_size, self.trajectory_projection.out_features,
                                          device=trajectory_feature.device, dtype=trajectory_feature.dtype)

        event_type_ids = is_single_player.long()
        event_type_feat = self.event_type_embedding(event_type_ids)

        single_player_mask = (is_single_player == 1).unsqueeze(-1)

        if self.use_bbox_feat:
            bbox_feat_enhanced = torch.where(
                single_player_mask,
                self.single_player_processor(bbox_feat),
                bbox_feat
            )
        else:
            bbox_feat_enhanced = bbox_feat

        if self.use_skeleton_feat:
            skeleton_feat_enhanced = torch.where(
                single_player_mask,
                self.single_player_processor(skeleton_feat),
                skeleton_feat
            )
        else:
            skeleton_feat_enhanced = skeleton_feat

        features_list = []

        if self.use_bbox_feat:
            features_list.append(bbox_feat_enhanced)
        if self.use_skeleton_feat:
            features_list.append(skeleton_feat_enhanced)
        if self.use_position_feat:
            features_list.append(position_feat)
        if self.use_trajectory_feat:
            features_list.append(trajectory_feat)

        features_list.append(event_type_feat)

        if len(features_list) > 0:
            features = torch.stack(features_list, dim=1)
        else:
            features = event_type_feat.unsqueeze(1)

        features = self.feature_norm(features)

        transformed_features = self.transformer_encoder(features)

        seq_len = features.size(1)
        base_weights = torch.softmax(torch.randn(seq_len, device=features.device), dim=0)
        weights = base_weights.clone()
        if seq_len > 1:
            weights[-1] = weights[-1] * 2
            weights = weights / weights.sum()

        pooled_features = torch.sum(transformed_features * weights.unsqueeze(0).unsqueeze(-1), dim=1)

        pooled_features = self.dropout(pooled_features)
        logits = self.classifier(pooled_features)

        return logits

class SimpleResNet3D(nn.Module):
    def __init__(self, base_channels=64, depth=50):
        super().__init__()
        self.base_channels = base_channels
        self.inplanes = base_channels

        self.conv1 = nn.Conv3d(3, base_channels, kernel_size=(1, 7, 7),
                               stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if isinstance(stride, int):
            stride_3d = (1, stride, stride)
        else:
            stride_3d = stride

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride_3d, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class SimplifiedSlowFast(nn.Module):
    def __init__(self,
                 slow_channels=64,
                 fast_channels=8,
                 resample_rate=4,
                 speed_ratio=2):
        super().__init__()

        self.resample_rate = max(1, resample_rate)
        self.speed_ratio = max(1, speed_ratio)
        self.fast_downsample_rate = max(1, self.resample_rate // self.speed_ratio)

        self.slow_path = SimpleResNet3D(base_channels=slow_channels)
        self.fast_path = SimpleResNet3D(base_channels=fast_channels)

    def forward(self, x):
        try:
            if x is None or len(x.shape) != 5:
                raise ValueError(f"Invalid input shape: {x.shape if x is not None else None}")

            B, C, T, H, W = x.shape

            if T >= self.resample_rate:
                slow_temporal_size = T // self.resample_rate
            else:
                slow_temporal_size = 1

            x_slow = F.interpolate(x, size=(slow_temporal_size, H, W), mode='nearest')
            slow_features = self.slow_path(x_slow)

            if T >= self.fast_downsample_rate:
                fast_temporal_size = T // self.fast_downsample_rate
            else:
                fast_temporal_size = 1

            x_fast = F.interpolate(x, size=(fast_temporal_size, H, W), mode='nearest')
            fast_features = self.fast_path(x_fast)

            return slow_features, fast_features

        except Exception as e:
            print(f"SimplifiedSlowFast forward failed: {e}")
            B = x.size(0) if x is not None else 1
            device = x.device if x is not None else torch.device('cuda')

            dummy_slow = torch.zeros(B, 2048, 1, 7, 7).to(device)
            dummy_fast = torch.zeros(B, 32, 1, 7, 7).to(device)
            return dummy_slow, dummy_fast

class SlowFastFeatureExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super(SlowFastFeatureExtractor, self).__init__()

        self.slowfast_model = SimplifiedSlowFast(
            slow_channels=64,
            fast_channels=8,
            resample_rate=4,
            speed_ratio=2
        )

        self.slowfast_model = self.slowfast_model.to(device)
        self.slowfast_model.eval()

        self.device = device
        self._setup_feature_projection()

        self.total_frames = 8

    def _setup_feature_projection(self):
        try:
            test_input = torch.randn(1, 3, 8, 224, 224).to(self.device)
            with torch.no_grad():
                slow_features, fast_features = self.slowfast_model(test_input)

                slow_pooled = F.adaptive_avg_pool3d(slow_features, (1, 1, 1)).flatten(1)
                fast_pooled = F.adaptive_avg_pool3d(fast_features, (1, 1, 1)).flatten(1)

                total_feature_dim = slow_pooled.size(1) + fast_pooled.size(1)
                if dist.is_initialized() and dist.get_rank() == 0:
                    print(
                        f"SlowFast feature dimensions: slow={slow_pooled.size(1)}, fast={fast_pooled.size(1)}, total={total_feature_dim}")

                self.feature_projection = nn.Sequential(
                    nn.Linear(total_feature_dim, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ).to(self.device)

        except Exception as e:
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"Dynamic feature dimension calculation failed, using default: {e}")
            self.feature_projection = nn.Sequential(
                nn.Linear(2080, 1024),
                nn.ReLU(),
                nn.Dropout(0.1)
            ).to(self.device)

    def forward(self, x):
        try:
            with torch.no_grad():
                if x is None or len(x.shape) != 5:
                    return torch.zeros(1, 1024).to(self.device)

                B, C, T, H, W = x.shape
                if T == 0 or H == 0 or W == 0:
                    return torch.zeros(B, 1024).to(self.device)

                if torch.isnan(x).any() or torch.isinf(x).any():
                    return torch.zeros(B, 1024).to(self.device)

                if T < 2:
                    x = x.repeat(1, 1, 2, 1, 1)
                elif T % 2 != 0:
                    x = x[:, :, :T - 1, :, :]

                slow_features, fast_features = self.slowfast_model(x)

                if slow_features is None or fast_features is None:
                    return torch.zeros(B, 1024).to(self.device)

                if torch.isnan(slow_features).any() or torch.isinf(slow_features).any():
                    slow_features = torch.zeros_like(slow_features)
                if torch.isnan(fast_features).any() or torch.isinf(fast_features).any():
                    fast_features = torch.zeros_like(fast_features)

                slow_pooled = F.adaptive_avg_pool3d(slow_features, (1, 1, 1)).flatten(1)
                fast_pooled = F.adaptive_avg_pool3d(fast_features, (1, 1, 1)).flatten(1)

                combined_features = torch.cat([slow_pooled, fast_pooled], dim=1)

                expected_dim = self.feature_projection[0].in_features
                actual_dim = combined_features.size(1)
                if actual_dim != expected_dim:
                    if actual_dim > expected_dim:
                        combined_features = combined_features[:, :expected_dim]
                    else:
                        padding = torch.zeros(B, expected_dim - actual_dim).to(self.device)
                        combined_features = torch.cat([combined_features, padding], dim=1)

                output_features = self.feature_projection(combined_features)

                if torch.isnan(output_features).any() or torch.isinf(output_features).any():
                    output_features = torch.zeros(B, 1024).to(self.device)

                return output_features

        except Exception as e:
            B = x.size(0) if x is not None else 1
            return torch.zeros(B, 1024).to(self.device)

slowfast_extractor = None
device_global = None

CLASS_NAMES = ['Pass', 'DribbleSteal', 'Shot', 'InterShot', 'Rebound', 'Drive', 'Dribble', 'PassSteal', 'Background']
EVENT_TYPE_TO_LABEL = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}
LABEL_TO_EVENT_TYPE = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}

SINGLE_PLAYER_EVENTS = {2, 3, 4, 5, 7}
DOUBLE_PLAYER_EVENTS = {1, 6, 8}

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return -1

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    print(f"Rank {rank}, World Size {world_size}, Local Rank {local_rank}")
    return local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def initialize_slowfast(device='cuda'):
    global slowfast_extractor, device_global

    if slowfast_extractor is None:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Initializing SlowFast model...")
        try:
            slowfast_extractor = SlowFastFeatureExtractor(device)
            device_global = device
            if not dist.is_initialized() or dist.get_rank() == 0:
                print("SlowFast model initialized successfully")

            test_input = torch.randn(1, 3, 8, 224, 224).to(device)
            with torch.no_grad():
                test_output = slowfast_extractor(test_input)
                if test_output is None or torch.isnan(test_output).any() or torch.isinf(test_output).any():
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print("Warning: SlowFast model test forward returned invalid results")
                else:
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(f"SlowFast model test successful, output feature dimensions: {test_output.shape}")
        except Exception as e:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"SlowFast model initialization failed: {e}")
                import traceback
                traceback.print_exc()
            slowfast_extractor = None
            device_global = device
            raise e

    return slowfast_extractor

def extract_multi_frame_feature(frames, bboxes=None, device='cuda'):
    global slowfast_extractor, device_global

    if slowfast_extractor is None:
        initialize_slowfast(device)

    try:
        if isinstance(frames, np.ndarray):
            if len(frames.shape) == 4:
                frames = [frames[i] for i in range(frames.shape[0])]
            else:
                frames = [frames]

        target_frames = slowfast_extractor.total_frames
        if target_frames % 2 != 0:
            target_frames = target_frames + 1

        if len(frames) == 1:
            frames = frames * target_frames
        elif len(frames) < target_frames:
            while len(frames) < target_frames:
                frames.append(frames[-1])
        elif len(frames) > target_frames:
            indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]

        if bboxes is None:
            bboxes = [None] * len(frames)
        elif not isinstance(bboxes, list):
            bboxes = [bboxes] * len(frames)
        elif len(bboxes) == 1:
            bboxes = bboxes * len(frames)
        elif len(bboxes) != len(frames):
            if len(bboxes) < len(frames):
                bboxes.extend([bboxes[-1]] * (len(frames) - len(bboxes)))
            else:
                bboxes = bboxes[:len(frames)]

        processed_frames = []
        for frame_idx, (frame, bbox) in enumerate(zip(frames, bboxes)):
            try:
                if frame is None or len(frame.shape) != 3:
                    region = np.zeros((224, 224, 3), dtype=np.uint8)
                    processed_frames.append(region)
                    continue

                frame_height, frame_width = frame.shape[:2]

                if frame_height <= 0 or frame_width <= 0:
                    region = np.zeros((224, 224, 3), dtype=np.uint8)
                    processed_frames.append(region)
                    continue

                if bbox is not None and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]

                    if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                        region = cv2.resize(frame, (224, 224))
                        processed_frames.append(region)
                        continue

                    if any(not np.isfinite(coord) for coord in [x1, y1, x2, y2]):
                        region = cv2.resize(frame, (224, 224))
                        processed_frames.append(region)
                        continue

                    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
                        x1, y1, x2, y2 = x1 * frame_width, y1 * frame_height, x2 * frame_width, y2 * frame_height

                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    x1 = max(0, min(x1, frame_width - 1))
                    y1 = max(0, min(y1, frame_height - 1))
                    x2 = max(x1 + 1, min(x2, frame_width))
                    y2 = max(y1 + 1, min(y2, frame_height))

                    if x2 > x1 and y2 > y1:
                        region = frame[y1:y2, x1:x2]

                        if region.size > 0:
                            try:
                                region = cv2.resize(region, (224, 224))
                            except cv2.error:
                                region = np.zeros((224, 224, 3), dtype=np.uint8)
                        else:
                            region = np.zeros((224, 224, 3), dtype=np.uint8)
                    else:
                        region = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    try:
                        region = cv2.resize(frame, (224, 224))
                    except cv2.error:
                        region = np.zeros((224, 224, 3), dtype=np.uint8)

                processed_frames.append(region)

            except Exception as e:
                region = np.zeros((224, 224, 3), dtype=np.uint8)
                processed_frames.append(region)

        while len(processed_frames) < target_frames:
            processed_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        frame_tensors = []
        for region in processed_frames:
            try:
                if region.shape != (224, 224, 3):
                    region = np.zeros((224, 224, 3), dtype=np.uint8)

                frame_tensor = torch.from_numpy(region).float().permute(2, 0, 1)

                frame_tensor = torch.clamp(frame_tensor / 255.0, 0.0, 1.0)

                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

                std = torch.clamp(std, min=1e-8)
                frame_tensor = (frame_tensor - mean) / std

                if torch.isnan(frame_tensor).any() or torch.isinf(frame_tensor).any():
                    frame_tensor = torch.zeros(3, 224, 224)

                frame_tensors.append(frame_tensor)

            except Exception as e:
                frame_tensors.append(torch.zeros(3, 224, 224))

        if len(frame_tensors) == 0:
            return torch.zeros(1024)

        try:
            video_tensor = torch.stack(frame_tensors, dim=1)
            video_tensor = video_tensor.unsqueeze(0).to(device_global)

            if torch.isnan(video_tensor).any() or torch.isinf(video_tensor).any():
                return torch.zeros(1024)

            features = slowfast_extractor(video_tensor)

            if features is None or torch.isnan(features).any() or torch.isinf(features).any():
                return torch.zeros(1024)

            return features.cpu().squeeze(0)

        except Exception as e:
            return torch.zeros(1024)

    except Exception as e:
        return torch.zeros(1024)

class ImprovedBasketballEventDataset(Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 neg_pos_ratio=5,
                 cache_features=True,
                 bbox_temporal_size=8,
                 skeleton_temporal_size=5,
                 trajectory_temporal_size=30,
                 alpha=0.1):

        self.data_root = data_root
        self.split = split
        self.neg_pos_ratio = neg_pos_ratio
        self.cache_features = cache_features
        self.feature_cache = {}
        self.frame_width = 3840
        self.frame_height = 2160
        self.alpha = alpha

        self.bbox_temporal_size = bbox_temporal_size
        self.skeleton_temporal_size = skeleton_temporal_size
        self.trajectory_temporal_size = trajectory_temporal_size

        self.court_width = 1500
        self.court_height = 1400
        self.basket_position = np.array([750, 0])

        self.calculate_homography_matrix()

        self.videos_info = self.load_videos_info()
        self.frames = self.load_frames()
        self.bbox_data = self.load_bbox_data()
        self.skeleton_data = self.load_skeleton_data()
        self.event_data = self.load_event_data()

        self.proposals = self.generate_proposals()

        self.print_event_statistics()
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Dataset initialized with {len(self.proposals)} proposals (pos:neg ratio 1:{self.neg_pos_ratio})")

    def print_event_statistics(self):
        single_count = 0
        double_count = 0
        background_count = 0
        positive_count = 0
        negative_count = 0

        for proposal in self.proposals:
            event_type = proposal['event_type']
            original_type = LABEL_TO_EVENT_TYPE.get(event_type, 9)

            if proposal['is_positive']:
                positive_count += 1
            else:
                negative_count += 1

            if original_type in SINGLE_PLAYER_EVENTS:
                single_count += 1
            elif original_type in DOUBLE_PLAYER_EVENTS:
                double_count += 1
            else:
                background_count += 1

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"\n=== {self.split} Dataset Event Distribution (pos:neg = 1:{self.neg_pos_ratio}) ===")
            print(f"  Positive samples: {positive_count}")
            print(f"  Negative samples: {negative_count}")
            print(f"  Actual pos:neg ratio: 1:{negative_count / max(positive_count, 1):.1f}")
            print(f"  Single player events: {single_count} samples")
            print(f"  Double player events: {double_count} samples")
            print(f"  Background events: {background_count} samples")
            print(f"  Total: {len(self.proposals)} samples")

    def calculate_homography_matrix(self):
        court_points = np.array([
            [0, 0], [1500, 0], [0, 1400], [1500, 1400]
        ], dtype=np.float32)

        image_points = np.array([
            [869, 1239], [2693, 1170], [107, 1970], [3731, 1719]
        ], dtype=np.float32)

        self.homography_matrix = cv2.findHomography(image_points, court_points)[0]

    def load_videos_info(self):
        videos_info = {}
        rawframes_path = os.path.join(self.data_root, "rawframes")
        if os.path.exists(rawframes_path):
            for video_dir in glob.glob(os.path.join(rawframes_path, "*")):
                video_name = os.path.basename(video_dir)
                frame_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
                num_frames = len(frame_files)
                videos_info[video_name] = {
                    'path': video_dir,
                    'num_frames': num_frames
                }
        return videos_info

    def load_frames(self):
        frames = {}
        for video_name, video_info in self.videos_info.items():
            video_dir = video_info['path']
            frame_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
            for i, frame_file in enumerate(frame_files):
                frame_id = f"{video_name}_{i + 1:05d}"
                frames[frame_id] = frame_file
        return frames

    def get_video_name_from_frame_id(self, frame_id):
        parts = frame_id.split(',')
        return parts[0]

    def get_frame_number_from_frame_id(self, frame_id):
        parts = frame_id.split(',')
        return int(parts[1])

    def load_frame_by_id(self, frame_id):
        try:
            if frame_id in self.frames:
                frame_path = self.frames[frame_id]
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None and frame.size > 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            return frame

            video_name = self.get_video_name_from_frame_id(frame_id)
            frame_number = self.get_frame_number_from_frame_id(frame_id)

            if video_name in self.videos_info:
                frame_path = os.path.join(self.videos_info[video_name]['path'], f"{frame_number:05d}.jpg")
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None and frame.size > 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            return frame
        except Exception as e:
            pass

        return np.zeros((480, 640, 3), dtype=np.uint8)

    def load_bbox_data(self):
        bbox_data = {}
        bbox_file = os.path.join(self.data_root, f"gt_bbox_{self.split}.json")

        if os.path.exists(bbox_file):
            with open(bbox_file, 'r') as f:
                data = json.load(f)
                for frame_id, bboxes in data.items():
                    processed_bboxes = []
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        processed_bboxes.append([x1, y1, x2, y2])
                    bbox_data[frame_id] = processed_bboxes
        return bbox_data

    def load_skeleton_data(self):
        skeleton_data = {}
        skeleton_file = os.path.join(self.data_root, f"gt_skeleton_{self.split}.json")

        if os.path.exists(skeleton_file):
            with open(skeleton_file, 'r') as f:
                data = json.load(f)
                for frame_id, skeletons in data.items():
                    skeleton_data[frame_id] = skeletons
        return skeleton_data

    def load_event_data(self):
        event_data = {}
        events_file = os.path.join(self.data_root, f"gt_events_{self.split}.json")

        if os.path.exists(events_file):
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Loading events from: {events_file}")
            try:
                with open(events_file, 'r') as f:
                    data = json.load(f)

                    for video_name, events in data.items():
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            print(f"Processing events for video: {video_name}, found {len(events)} events")
                        for event_idx, event in enumerate(events):
                            event_id = f"{video_name}_{event_idx}"

                            if 'start_frame' not in event or 'end_frame' not in event or 'event_type' not in event:
                                continue

                            if 'player_id' in event:
                                if isinstance(event['player_id'], str):
                                    player_ids = [int(pid.strip()) for pid in event['player_id'].split(',')]
                                elif isinstance(event['player_id'], list):
                                    player_ids = [int(pid) for pid in event['player_id']]
                                else:
                                    player_ids = [int(event['player_id'])]
                            else:
                                player_ids = [0]

                            original_event_type = event['event_type']
                            if original_event_type in EVENT_TYPE_TO_LABEL:
                                mapped_event_type = EVENT_TYPE_TO_LABEL[original_event_type]
                            else:
                                mapped_event_type = 8

                            event_data[event_id] = {
                                'video_name': video_name,
                                'start_frame': event['start_frame'],
                                'end_frame': event['end_frame'],
                                'event_type': mapped_event_type,
                                'original_event_type': original_event_type,
                                'player_ids': player_ids
                            }
            except Exception as e:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Error loading event data: {e}")
        else:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Warning: Event file not found - {events_file}")

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Loaded {len(event_data)} events total")
        return event_data

    def generate_proposals(self):
        positive_proposals = []

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Generating proposals from {len(self.event_data)} events with neg_pos_ratio={self.neg_pos_ratio}")

        for event_id, event_info in self.event_data.items():
            video_name = event_info['video_name']
            start_frame = event_info['start_frame']
            end_frame = event_info['end_frame']
            player_ids = event_info['player_ids']
            event_type = event_info['event_type']
            original_event_type = event_info['original_event_type']

            frame_id_formats = [
                lambda vn, fn: f"{vn}_{fn:05d}",
                lambda vn, fn: f"{vn}_{fn}",
                lambda vn, fn: str(fn),
                lambda vn, fn: f"{vn}_{fn:05d}" if "_" in vn else f"{vn}{fn:05d}",
            ]

            for frame_number in range(start_frame, end_frame + 1):
                frame_id = None
                valid_frame = False

                for format_func in frame_id_formats:
                    candidate_id = format_func(video_name, frame_number)
                    if candidate_id in self.bbox_data:
                        frame_id = candidate_id
                        valid_frame = True
                        break

                if not valid_frame:
                    matching_keys = [k for k in self.bbox_data.keys() if
                                     video_name in str(k) and str(frame_number) in str(k)]
                    if matching_keys:
                        frame_id = matching_keys[0]
                        valid_frame = True

                if not valid_frame:
                    continue

                if frame_id in self.bbox_data and frame_id in self.skeleton_data:
                    valid_players = True
                    for player_id in player_ids:
                        if player_id >= len(self.bbox_data[frame_id]):
                            valid_players = False
                            break

                    if valid_players:
                        if original_event_type in SINGLE_PLAYER_EVENTS:
                            for augment_idx in range(3):
                                proposal = {
                                    'frame_id': frame_id,
                                    'player_ids': player_ids,
                                    'event_type': event_type,
                                    'is_positive': True,
                                    'event_id': event_id,
                                    'is_single_player': True,
                                    'augment_idx': augment_idx
                                }
                                positive_proposals.append(proposal)
                        else:
                            proposal = {
                                'frame_id': frame_id,
                                'player_ids': player_ids,
                                'event_type': event_type,
                                'is_positive': True,
                                'event_id': event_id,
                                'is_single_player': False,
                                'augment_idx': 0
                            }
                            positive_proposals.append(proposal)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Total positive proposals: {len(positive_proposals)}")

        proposals = positive_proposals.copy()
        if self.split == 'train' and len(positive_proposals) > 0 and self.neg_pos_ratio > 0:
            positive_frame_ids = {p['frame_id'] for p in positive_proposals}
            num_neg_proposals = max(1, int(len(positive_proposals) * self.neg_pos_ratio))
            neg_proposals = self.generate_negative_proposals(num_neg_proposals, positive_frame_ids)
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Generated {len(neg_proposals)} negative proposals (target: {num_neg_proposals})")
                print(f"Actual pos:neg ratio = 1:{len(neg_proposals) / len(positive_proposals):.1f}")
            proposals.extend(neg_proposals)
            random.shuffle(proposals)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Final proposals count: {len(proposals)}")
        return proposals

    def generate_negative_proposals(self, num_neg_proposals, positive_frame_ids):
        neg_proposals = []
        max_attempts = num_neg_proposals * 2

        for _ in range(max_attempts):
            if len(neg_proposals) >= num_neg_proposals:
                break

            frame_ids = list(self.bbox_data.keys())
            if not frame_ids:
                continue
            frame_id = random.choice(frame_ids)

            if frame_id in positive_frame_ids:
                continue

            num_players = random.randint(1, 2)
            if len(self.bbox_data[frame_id]) >= num_players:
                player_ids = random.sample(range(len(self.bbox_data[frame_id])), num_players)

                proposal = {
                    'frame_id': frame_id,
                    'player_ids': player_ids,
                    'event_type': 8,
                    'is_positive': False,
                    'event_id': -1,
                    'is_single_player': num_players == 1,
                    'augment_idx': 0
                }
                neg_proposals.append(proposal)

        return neg_proposals

    def project_to_court(self, bbox):
        try:
            x1, y1, x2, y2 = bbox

            if max(x1, y1, x2, y2) <= 1.0:
                x1 = x1 * self.frame_width
                y1 = y1 * self.frame_height
                x2 = x2 * self.frame_width
                y2 = y2 * self.frame_height

            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2

            point = np.array([[bottom_center_x, bottom_center_y]], dtype=np.float32)
            point = np.array([point])

            projected_point = cv2.perspectiveTransform(point, self.homography_matrix)[0][0]
            return projected_point
        except Exception as e:
            return np.array([0.0, 0.0])

    def normalize_court_position(self, position):
        try:
            normalized_x = position[0] / self.court_width
            normalized_y = position[1] / self.court_height
            normalized_x = max(0, min(1, normalized_x))
            normalized_y = max(0, min(1, normalized_y))
            return np.array([normalized_x, normalized_y])
        except:
            return np.array([0.0, 0.0])

    def extract_multi_frame_bbox_feature(self, frame_id, player_id, augment_idx=0):
        cache_key = f"slowfast_bbox_{frame_id}_{player_id}_{augment_idx}"
        if self.cache_features and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        try:
            video_name = self.get_video_name_from_frame_id(frame_id)
            frame_number = self.get_frame_number_from_frame_id(frame_id)
            half_window = self.bbox_temporal_size // 2

            frames = []
            bboxes = []

            for offset in range(-half_window, half_window):
                target_frame_number = frame_number + offset

                if target_frame_number < 1:
                    target_frame_number = 1

                target_frame_id = f"{video_name},{target_frame_number:05d}"

                frame = self.load_frame_by_id(target_frame_id)
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                frames.append(frame)

                bbox = None
                if target_frame_id in self.bbox_data and player_id < len(self.bbox_data[target_frame_id]):
                    bbox = self.bbox_data[target_frame_id][player_id]

                    if augment_idx > 0 and bbox is not None:
                        try:
                            x1, y1, x2, y2 = bbox[:4]

                            if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                                pass
                            elif any(not np.isfinite(coord) for coord in [x1, y1, x2, y2]):
                                pass
                            else:
                                width = abs(x2 - x1)
                                height = abs(y2 - y1)

                                if width > 1e-6 and height > 1e-6:
                                    if augment_idx == 1:
                                        expansion = 0.05
                                        x1 -= width * expansion
                                        y1 -= height * expansion
                                        x2 += width * expansion
                                        y2 += height * expansion
                                    elif augment_idx == 2:
                                        shrinkage = 0.05
                                        x1 += width * shrinkage
                                        y1 += height * shrinkage
                                        x2 -= width * shrinkage
                                        y2 -= height * shrinkage

                                    bbox = [x1, y1, x2, y2]
                        except Exception as e:
                            pass

                bboxes.append(bbox)

            feature = extract_multi_frame_feature(frames, bboxes, device_global or 'cuda')

            if self.cache_features:
                self.feature_cache[cache_key] = feature

            return feature

        except Exception as e:
            return torch.zeros(1024)

    def extract_bbox_feature(self, frame_id, player_id, augment_idx=0):
        return self.extract_multi_frame_bbox_feature(frame_id, player_id, augment_idx)

    def extract_skeleton_feature(self, frame_id, player_id):
        cache_key = f"skeleton_{frame_id}_{player_id}"
        if self.cache_features and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        try:
            video_name = self.get_video_name_from_frame_id(frame_id)
            frame_number = self.get_frame_number_from_frame_id(frame_id)
            half_window = self.skeleton_temporal_size // 2

            skeletons = []
            for offset in range(-half_window, half_window + 1):
                target_frame_number = frame_number + offset

                if target_frame_number < 1:
                    skeletons.append(np.zeros((17, 3)))
                    continue

                target_frame_id = f"{video_name},{target_frame_number:05d}"

                if target_frame_id in self.skeleton_data and player_id < len(self.skeleton_data[target_frame_id]):
                    skeleton = self.skeleton_data[target_frame_id][player_id]
                    skeletons.append(self.normalize_skeleton(np.array(skeleton)))
                else:
                    skeletons.append(np.zeros((17, 3)))

            if skeletons:
                all_skeletons = np.concatenate([s.flatten() for s in skeletons])
                feature = torch.tensor(all_skeletons, dtype=torch.float32)
            else:
                feature = torch.zeros(17 * 3 * self.skeleton_temporal_size)

            if feature.shape[0] != 256:
                target_size = 256
                current_size = feature.shape[0]
                if current_size > target_size:
                    indices = torch.linspace(0, current_size - 1, target_size).long()
                    feature = feature[indices]
                elif current_size < target_size:
                    padding = target_size - current_size
                    feature = torch.cat([feature, feature[-1].repeat(padding)])

            if self.cache_features:
                self.feature_cache[cache_key] = feature

            return feature
        except Exception as e:
            return torch.zeros(256)

    def normalize_skeleton(self, skeleton):
        try:
            if skeleton.shape[0] == 0:
                return np.zeros((17, 3))

            if len(skeleton.shape) == 1:
                if skeleton.shape[0] >= 51:
                    skeleton = skeleton[:51].reshape(17, 3)
                else:
                    padded = np.zeros(51)
                    padded[:skeleton.shape[0]] = skeleton
                    skeleton = padded.reshape(17, 3)

            if skeleton.shape[0] > 0:
                center = np.mean(skeleton[:, :2], axis=0)
                skeleton[:, :2] = skeleton[:, :2] - center
                scale = np.max(np.abs(skeleton[:, :2]))
                if scale > 0:
                    skeleton[:, :2] = skeleton[:, :2] / scale

            return skeleton
        except:
            return np.zeros((17, 3))

    def extract_projected_position(self, frame_id, player_id1, player_id2=None, is_single_player=False):
        if is_single_player:
            cache_key = f"position_single_{frame_id}_{player_id1}"
        else:
            cache_key = f"position_double_{frame_id}_{player_id1}_{player_id2}"

        if self.cache_features and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        try:
            video_name = self.get_video_name_from_frame_id(frame_id)
            frame_number = self.get_frame_number_from_frame_id(frame_id)
            half_window = self.skeleton_temporal_size // 2

            positions = []
            for offset in range(-half_window, half_window + 1):
                target_frame_number = frame_number + offset

                if target_frame_number < 1:
                    positions.append(np.zeros(2))
                    continue

                target_frame_id = f"{video_name},{target_frame_number:05d}"

                player1_court_pos = None
                if target_frame_id in self.bbox_data and player_id1 < len(self.bbox_data[target_frame_id]):
                    bbox1 = self.bbox_data[target_frame_id][player_id1]
                    player1_court_pos = self.project_to_court(bbox1)

                if player1_court_pos is not None:
                    if is_single_player:
                        rel_pos = player1_court_pos - self.basket_position
                        normalized_pos = self.normalize_court_position(rel_pos)
                        positions.append(normalized_pos)
                    else:
                        if (target_frame_id in self.bbox_data and
                                player_id2 is not None and
                                player_id2 < len(self.bbox_data[target_frame_id])):

                            bbox2 = self.bbox_data[target_frame_id][player_id2]
                            player2_court_pos = self.project_to_court(bbox2)
                            rel_pos = player2_court_pos - player1_court_pos
                            normalized_pos = self.normalize_court_position(rel_pos)
                            positions.append(normalized_pos)
                        else:
                            positions.append(np.zeros(2))
                else:
                    positions.append(np.zeros(2))

            if positions:
                all_positions = np.concatenate(positions)
                feature = torch.tensor(all_positions, dtype=torch.float32)
            else:
                feature = torch.zeros(2 * self.skeleton_temporal_size)

            if feature.shape[0] != 256:
                target_size = 256
                current_size = feature.shape[0]
                if current_size > target_size:
                    indices = torch.linspace(0, current_size - 1, target_size).long()
                    feature = feature[indices]
                elif current_size < target_size:
                    padding = target_size - current_size
                    feature = torch.cat([feature, feature[-1].repeat(padding)])

            if self.cache_features:
                self.feature_cache[cache_key] = feature

            return feature
        except Exception as e:
            return torch.zeros(256)

    def extract_trajectory_feature(self, frame_id, player_id):
        cache_key = f"trajectory_{frame_id}_{player_id}"
        if self.cache_features and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        try:
            video_name = self.get_video_name_from_frame_id(frame_id)
            frame_number = self.get_frame_number_from_frame_id(frame_id)
            half_window = self.trajectory_temporal_size // 2

            trajectory = []
            for offset in range(-half_window, half_window + 1):
                target_frame_number = frame_number + offset

                if target_frame_number < 1:
                    trajectory.append(np.zeros(2))
                    continue

                target_frame_id = f"{video_name},{target_frame_number:05d}"

                if target_frame_id in self.bbox_data and player_id < len(self.bbox_data[target_frame_id]):
                    bbox = self.bbox_data[target_frame_id][player_id]
                    court_pos = self.project_to_court(bbox)
                    normalized_pos = self.normalize_court_position(court_pos)
                    trajectory.append(normalized_pos)
                else:
                    trajectory.append(np.zeros(2))

            if trajectory:
                feature = self.improved_positional_encoding(np.array(trajectory), d_model=256)
            else:
                feature = torch.zeros(256)

            if feature.shape[0] != 256:
                target_size = 256
                current_size = feature.shape[0]
                if current_size > target_size:
                    indices = torch.linspace(0, current_size - 1, target_size).long()
                    feature = feature[indices]
                elif current_size < target_size:
                    padding = target_size - current_size
                    feature = torch.cat([feature, feature[-1].repeat(padding)])

            if self.cache_features:
                self.feature_cache[cache_key] = feature

            return feature
        except Exception as e:
            return torch.zeros(256)

    def improved_positional_encoding(self, positions, d_model=256):
        seq_len = positions.shape[0]

        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()
        elif not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32)

        pe = torch.zeros(seq_len, d_model)
        position_indices = torch.arange(0, seq_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position_indices * div_term)
        pe[:, 1::2] = torch.cos(position_indices * div_term)

        spatial_features = torch.zeros(seq_len, d_model)
        for i in range(seq_len):
            spatial_features[i, :d_model // 4] = positions[i, 0]
            spatial_features[i, d_model // 4:d_model // 2] = positions[i, 1]

        combined_features = pe + self.alpha * spatial_features
        return combined_features.flatten()

    def __len__(self):
        return len(self.proposals)

    def __getitem__(self, idx):
        try:
            proposal = self.proposals[idx]

            frame_id = proposal['frame_id']
            player_ids = proposal['player_ids']
            event_type = proposal['event_type']
            is_single_player = proposal['is_single_player']
            augment_idx = proposal.get('augment_idx', 0)

            if is_single_player:
                player_id = player_ids[0]

                bbox_feature1 = self.extract_bbox_feature(frame_id, player_id, augment_idx)
                bbox_feature2 = bbox_feature1.clone()
                bbox_feature = torch.stack([bbox_feature1, bbox_feature2], dim=0)

                skeleton_feature1 = self.extract_skeleton_feature(frame_id, player_id)
                skeleton_feature2 = skeleton_feature1.clone()
                skeleton_feature = torch.stack([skeleton_feature1, skeleton_feature2], dim=0)

                project_position_encoding = self.extract_projected_position(
                    frame_id, player_id, None, is_single_player=True
                )

                trajectory_feature1 = self.extract_trajectory_feature(frame_id, player_id)
                trajectory_feature2 = trajectory_feature1.clone()
                trajectory_feature = torch.stack([trajectory_feature1, trajectory_feature2], dim=0)

                is_single_player_flag = torch.tensor(1.0)

            else:
                if len(player_ids) >= 2:
                    player_id1, player_id2 = player_ids[0], player_ids[1]
                else:
                    player_id1, player_id2 = player_ids[0], player_ids[0]

                bbox_feature1 = self.extract_bbox_feature(frame_id, player_id1, augment_idx)
                bbox_feature2 = self.extract_bbox_feature(frame_id, player_id2, augment_idx)
                bbox_feature = torch.stack([bbox_feature1, bbox_feature2], dim=0)

                skeleton_feature1 = self.extract_skeleton_feature(frame_id, player_id1)
                skeleton_feature2 = self.extract_skeleton_feature(frame_id, player_id2)
                skeleton_feature = torch.stack([skeleton_feature1, skeleton_feature2], dim=0)

                project_position_encoding = self.extract_projected_position(
                    frame_id, player_id1, player_id2, is_single_player=False
                )

                trajectory_feature1 = self.extract_trajectory_feature(frame_id, player_id1)
                trajectory_feature2 = self.extract_trajectory_feature(frame_id, player_id2)
                trajectory_feature = torch.stack([trajectory_feature1, trajectory_feature2], dim=0)

                is_single_player_flag = torch.tensor(0.0)

            if bbox_feature.shape != (2, 1024):
                bbox_feature = torch.zeros(2, 1024)
            if skeleton_feature.shape != (2, 256):
                skeleton_feature = torch.zeros(2, 256)
            if project_position_encoding.shape != (256,):
                project_position_encoding = torch.zeros(256)
            if trajectory_feature.shape != (2, 256):
                trajectory_feature = torch.zeros(2, 256)

            return {
                'bbox_feature': bbox_feature,
                'skeleton_feature': skeleton_feature,
                'project_position_encoding': project_position_encoding,
                'trajectory_feature': trajectory_feature,
                'event_type': torch.tensor(event_type, dtype=torch.long),
                'is_single_player': is_single_player_flag,
                'is_positive': proposal['is_positive'],
                'frame_id': frame_id,
                'player_ids': player_ids
            }
        except Exception as e:
            return {
                'bbox_feature': torch.zeros(2, 1024),
                'skeleton_feature': torch.zeros(2, 256),
                'project_position_encoding': torch.zeros(256),
                'trajectory_feature': torch.zeros(2, 256),
                'event_type': torch.tensor(8, dtype=torch.long),
                'is_single_player': torch.tensor(0.0),
                'is_positive': False,
                'frame_id': 'dummy',
                'player_ids': [0, 0]
            }

def improved_collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    try:
        batch_dict = {}

        for key in ['bbox_feature', 'skeleton_feature', 'project_position_encoding',
                    'trajectory_feature', 'event_type', 'is_single_player']:
            batch_dict[key] = torch.stack([item[key] for item in batch])

        batch_dict['frame_id'] = [item['frame_id'] for item in batch]
        batch_dict['is_positive'] = [item['is_positive'] for item in batch]
        batch_dict['player_ids'] = [item['player_ids'] for item in batch]

        return batch_dict
    except Exception as e:
        return None

def train_improved_model(model, train_loader, criterion, optimizer, device, epoch, rank, performance_monitor=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    batch_times = []
    data_loading_times = []
    forward_times = []
    backward_times = []

    if rank == 0:
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        print_memory_usage(f"Training start (Epoch {epoch})", rank)
    else:
        progress_bar = train_loader

    epoch_start_time = time.time()
    data_iter = iter(progress_bar)

    for batch_idx in range(len(train_loader)):
        batch_start_time = time.time()

        data_load_start = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        if batch is None:
            continue

        data_load_time = time.time() - data_load_start
        data_loading_times.append(data_load_time)

        try:
            gpu_transfer_start = time.time()
            bbox_feature = batch['bbox_feature'].to(device, non_blocking=True)
            skeleton_feature = batch['skeleton_feature'].to(device, non_blocking=True)
            project_position_encoding = batch['project_position_encoding'].to(device, non_blocking=True)
            trajectory_feature = batch['trajectory_feature'].to(device, non_blocking=True)
            event_type = batch['event_type'].to(device, non_blocking=True)
            is_single_player = batch['is_single_player'].to(device, non_blocking=True)
            gpu_transfer_time = time.time() - gpu_transfer_start

            forward_start = time.time()
            outputs = model(bbox_feature, skeleton_feature, project_position_encoding,
                            trajectory_feature, is_single_player)
            loss = criterion(outputs, event_type)
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)

            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            batch_size = event_type.size(0)
            total += batch_size
            correct += predicted.eq(event_type).sum().item()

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            if performance_monitor:
                performance_monitor.record_batch_time(batch_time)
                performance_monitor.record_data_loading_time(data_load_time)
                performance_monitor.record_forward_time(forward_time)
                performance_monitor.record_backward_time(backward_time)

                if batch_idx % 10 == 0:
                    performance_monitor.record_memory_usage()

            if rank == 0 and batch_idx % 50 == 0:
                avg_batch_time = np.mean(batch_times[-50:]) if len(batch_times) >= 50 else np.mean(batch_times)
                avg_data_time = np.mean(data_loading_times[-50:]) if len(data_loading_times) >= 50 else np.mean(
                    data_loading_times)
                avg_forward_time = np.mean(forward_times[-50:]) if len(forward_times) >= 50 else np.mean(forward_times)
                avg_backward_time = np.mean(backward_times[-50:]) if len(backward_times) >= 50 else np.mean(
                    backward_times)

                samples_per_sec = batch_size / avg_batch_time if avg_batch_time > 0 else 0

                print(f"\n[Epoch {epoch}, Batch {batch_idx}] Performance stats:")
                print(f"  Average batch time: {avg_batch_time:.4f}s")
                print(f"  Data loading time: {avg_data_time:.4f}s ({avg_data_time / avg_batch_time * 100:.1f}%)")
                print(f"  GPU transfer time: {gpu_transfer_time:.4f}s")
                print(f"  Forward time: {avg_forward_time:.4f}s ({avg_forward_time / avg_batch_time * 100:.1f}%)")
                print(f"  Backward time: {avg_backward_time:.4f}s ({avg_backward_time / avg_batch_time * 100:.1f}%)")
                print(f"  Processing speed: {samples_per_sec:.2f} samples/sec")
                print_memory_usage(f"  Memory usage", rank)

            if rank == 0:
                if hasattr(progress_bar, 'set_postfix'):
                    current_batch_time = batch_times[-1] if batch_times else 0
                    current_fps = batch_size / current_batch_time if current_batch_time > 0 else 0
                    progress_bar.set_postfix({
                        'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                        'Acc': f'{100. * correct / total:.2f}%',
                        'Time': f'{current_batch_time:.3f}s',
                        'FPS': f'{current_fps:.1f}'
                    })

        except Exception as e:
            if rank == 0:
                print(f"Training batch {batch_idx} error: {e}")
            continue

    epoch_time = time.time() - epoch_start_time
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    avg_data_time = np.mean(data_loading_times) if data_loading_times else 0
    avg_forward_time = np.mean(forward_times) if forward_times else 0
    avg_backward_time = np.mean(backward_times) if backward_times else 0

    if rank == 0:
        print(f"\n=== Epoch {epoch} Performance Summary ===")
        print(f"Total epoch time: {epoch_time:.2f} seconds")
        print(f"Average batch time: {avg_batch_time:.4f} seconds")
        print(f"Average data loading time: {avg_data_time:.4f} seconds ({avg_data_time / avg_batch_time * 100:.1f}%)")
        print(f"Average forward time: {avg_forward_time:.4f} seconds ({avg_forward_time / avg_batch_time * 100:.1f}%)")
        print(f"Average backward time: {avg_backward_time:.4f} seconds ({avg_backward_time / avg_batch_time * 100:.1f}%)")
        print(f"Total processed samples: {total}")
        print(f"Average processing speed: {total / epoch_time:.2f} samples/sec")
        print_memory_usage("End of epoch memory", rank)

    return running_loss / len(train_loader), 100. * correct / total

def compute_ap_for_class(y_true, y_scores):
    if len(y_true) == 0 or len(y_scores) == 0:
        return 0.0

    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    precision = tp / (tp + fp)
    recall = tp / max(np.sum(y_true), 1)

    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])

    return ap

def validate_improved_model(model, val_loader, criterion, device, rank, performance_monitor=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    batch_times = []
    inference_times = []

    single_correct = 0
    single_total = 0
    double_correct = 0
    double_total = 0

    class_correct = [0] * 9
    class_total = [0] * 9

    class_predictions = [[] for _ in range(9)]
    class_labels = [[] for _ in range(9)]

    validation_start_time = time.time()

    with torch.no_grad():
        if rank == 0:
            progress_bar = tqdm(val_loader, desc='Validating')
            print_memory_usage("Validation start", rank)
        else:
            progress_bar = val_loader

        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue

            batch_start_time = time.time()

            try:
                gpu_transfer_start = time.time()
                bbox_feature = batch['bbox_feature'].to(device, non_blocking=True)
                skeleton_feature = batch['skeleton_feature'].to(device, non_blocking=True)
                project_position_encoding = batch['project_position_encoding'].to(device, non_blocking=True)
                trajectory_feature = batch['trajectory_feature'].to(device, non_blocking=True)
                event_type = batch['event_type'].to(device, non_blocking=True)
                is_single_player = batch['is_single_player'].to(device, non_blocking=True)
                gpu_transfer_time = time.time() - gpu_transfer_start

                inference_start = time.time()
                outputs = model(bbox_feature, skeleton_feature, project_position_encoding,
                                trajectory_feature, is_single_player)
                loss = criterion(outputs, event_type)
                inference_time = time.time() - inference_start

                batch_size = event_type.size(0)
                inference_times.append(inference_time)

                if performance_monitor:
                    performance_monitor.record_inference_time(inference_time, batch_size)

                probs = F.softmax(outputs, dim=1)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += event_type.size(0)
                correct += predicted.eq(event_type).sum().item()

                for i in range(event_type.size(0)):
                    true_label = event_type[i].item()

                    for class_idx in range(9):
                        class_score = probs[i, class_idx].item()
                        is_positive = 1 if true_label == class_idx else 0

                        class_predictions[class_idx].append(class_score)
                        class_labels[class_idx].append(is_positive)

                for i in range(event_type.size(0)):
                    pred = predicted[i].item()
                    true = event_type[i].item()

                    class_total[true] += 1
                    if pred == true:
                        class_correct[true] += 1

                    if is_single_player[i].item() == 1.0:
                        single_total += 1
                        if predicted[i].eq(event_type[i]).item():
                            single_correct += 1
                    else:
                        double_total += 1
                        if predicted[i].eq(event_type[i]).item():
                            double_correct += 1

                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                current_fps = batch_size / inference_time if inference_time > 0 else 0

                if rank == 0:
                    if hasattr(progress_bar, 'set_postfix'):
                        progress_bar.set_postfix({
                            'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                            'Acc': f'{100. * correct / total:.2f}%',
                            'FPS': f'{current_fps:.1f}',
                            'Time': f'{inference_time:.3f}s'
                        })

                if rank == 0 and batch_idx % 30 == 0 and batch_idx > 0:
                    avg_inference_time = np.mean(inference_times[-30:])
                    avg_batch_time = np.mean(batch_times[-30:])
                    avg_fps = batch_size / avg_inference_time if avg_inference_time > 0 else 0

                    print(f"\n[Validation Batch {batch_idx}] Performance stats:")
                    print(f"  Average inference time: {avg_inference_time:.4f}s")
                    print(f"  Average batch time: {avg_batch_time:.4f}s")
                    print(f"  Average inference FPS: {avg_fps:.2f}")
                    print(f"  GPU transfer time: {gpu_transfer_time:.4f}s")
                    print_memory_usage(f"  Memory usage", rank)

            except Exception as e:
                if rank == 0:
                    print(f"Validation batch {batch_idx} error: {e}")
                continue

    validation_time = time.time() - validation_start_time
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    avg_fps = total / sum(inference_times) if sum(inference_times) > 0 else 0

    if rank == 0:
        print(f"\n=== Validation Performance Summary ===")
        print(f"Total validation time: {validation_time:.2f} seconds")
        print(f"Average inference time: {avg_inference_time:.4f} seconds")
        print(f"Average batch time: {avg_batch_time:.4f} seconds")
        print(f"Total processed samples: {total}")
        print(f"Overall inference FPS: {avg_fps:.2f}")
        print(f"Validation throughput: {total / validation_time:.2f} samples/sec")
        print_memory_usage("End of validation memory", rank)

    if dist.is_initialized():
        stats_tensor = torch.tensor([
                                        running_loss, correct, total, single_correct, single_total,
                                        double_correct, double_total
                                    ] + class_correct + class_total).float().to(device)

        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        running_loss, correct, total, single_correct, single_total, double_correct, double_total = stats_tensor[
                                                                                                   :7].tolist()
        class_correct = stats_tensor[7:16].tolist()
        class_total = stats_tensor[16:25].tolist()

    class_aps = []
    single_class_aps = []
    double_class_aps = []

    if rank == 0:
        for class_idx in range(9):
            if len(class_predictions[class_idx]) > 0:
                ap = compute_ap_for_class(class_labels[class_idx], class_predictions[class_idx])
                class_aps.append(ap)

                original_type = LABEL_TO_EVENT_TYPE.get(class_idx, 9)
                if original_type in SINGLE_PLAYER_EVENTS:
                    single_class_aps.append(ap)
                elif original_type in DOUBLE_PLAYER_EVENTS:
                    double_class_aps.append(ap)
            else:
                class_aps.append(0.0)

        overall_map = np.mean(class_aps) if class_aps else 0.0
        single_map = np.mean(single_class_aps) if single_class_aps else 0.0
        double_map = np.mean(double_class_aps) if double_class_aps else 0.0

        avg_loss = running_loss / len(val_loader)
        accuracy = 100. * correct / total
        single_acc = 100. * single_correct / max(single_total, 1)
        double_acc = 100. * double_correct / max(double_total, 1)

        print(f"\n=== Detailed Validation Results ===")
        print(f"Overall accuracy: {accuracy:.2f}%")
        print(f"Overall mAP: {overall_map:.4f}")
        print(f"Single player event accuracy: {single_acc:.2f}% ({single_correct}/{single_total})")
        print(f"Single player event mAP: {single_map:.4f}")
        print(f"Double player event accuracy: {double_acc:.2f}% ({double_correct}/{double_total})")
        print(f"Double player event mAP: {double_map:.4f}")

        print(f"\n=== Class-wise Detailed Metrics ===")
        for i in range(9):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                class_ap = class_aps[i]
                original_type = LABEL_TO_EVENT_TYPE.get(i, 9)
                event_category = "Single" if original_type in SINGLE_PLAYER_EVENTS else (
                    "Double" if original_type in DOUBLE_PLAYER_EVENTS else "Background")
                print(
                    f"{CLASS_NAMES[i]} ({event_category}): Acc {class_acc:.1f}% ({class_correct[i]}/{class_total[i]}), AP {class_ap:.4f}")

        return avg_loss, accuracy, single_acc, double_acc, overall_map, single_map, double_map
    else:
        return 0, 0, 0, 0, 0, 0, 0

def inference_improved_model(model, test_loader, device, performance_monitor=None):
    model.eval()
    predictions = []

    inference_times = []
    batch_times = []
    total_samples = 0

    inference_start_time = time.time()

    print_memory_usage("Inference start", rank=0)

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing')
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue

            batch_start_time = time.time()

            try:
                gpu_transfer_start = time.time()
                bbox_feature = batch['bbox_feature'].to(device, non_blocking=True)
                skeleton_feature = batch['skeleton_feature'].to(device, non_blocking=True)
                project_position_encoding = batch['project_position_encoding'].to(device, non_blocking=True)
                trajectory_feature = batch['trajectory_feature'].to(device, non_blocking=True)
                is_single_player = batch['is_single_player'].to(device, non_blocking=True)
                frame_ids = batch['frame_id']
                player_ids_list = batch['player_ids']
                gpu_transfer_time = time.time() - gpu_transfer_start

                inference_time_start = time.time()
                outputs = model(bbox_feature, skeleton_feature, project_position_encoding,
                                trajectory_feature, is_single_player)
                _, predicted = outputs.max(1)
                inference_time = time.time() - inference_time_start

                batch_size = predicted.size(0)
                total_samples += batch_size
                inference_times.append(inference_time)

                if performance_monitor:
                    performance_monitor.record_inference_time(inference_time, batch_size)

                for i in range(batch_size):
                    current_player_ids = player_ids_list[i]
                    valid_player_ids = [pid for pid in current_player_ids if pid != -1]

                    model_prediction = predicted[i].item()
                    original_event_type = LABEL_TO_EVENT_TYPE.get(model_prediction, 9)

                    prediction = {
                        'frame_id': frame_ids[i],
                        'player_ids': valid_player_ids,
                        'event_type': original_event_type,
                        'model_label': model_prediction,
                        'confidence': F.softmax(outputs[i], dim=0)[predicted[i]].item(),
                        'is_single_player': is_single_player[i].item() == 1.0
                    }
                    predictions.append(prediction)

                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                current_fps = batch_size / inference_time if inference_time > 0 else 0

                progress_bar.set_postfix({
                    'FPS': f'{current_fps:.1f}',
                    'InferTime': f'{inference_time:.3f}s',
                    'BatchTime': f'{batch_time:.3f}s'
                })

                if batch_idx % 20 == 0 and batch_idx > 0:
                    recent_inference_times = inference_times[-20:]
                    recent_batch_times = batch_times[-20:]

                    avg_inference_time = np.mean(recent_inference_times)
                    avg_batch_time = np.mean(recent_batch_times)
                    avg_fps = batch_size / avg_inference_time if avg_inference_time > 0 else 0

                    print(f"\n[Inference Batch {batch_idx}] Performance stats:")
                    print(f"  Average inference time: {avg_inference_time:.4f}s")
                    print(f"  Average batch time: {avg_batch_time:.4f}s")
                    print(f"  Average FPS: {avg_fps:.2f}")
                    print(f"  GPU transfer time: {gpu_transfer_time:.4f}s")
                    print_memory_usage(f"  Memory usage", rank=0)

            except Exception as e:
                print(f"Inference batch {batch_idx} error: {e}")
                continue

    total_inference_time = time.time() - inference_start_time
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    overall_fps = total_samples / sum(inference_times) if sum(inference_times) > 0 else 0

    print(f"\n=== Inference Performance Summary ===")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"Average inference time: {avg_inference_time:.4f} seconds")
    print(f"Average batch time: {avg_batch_time:.4f} seconds")
    print(f"Total processed samples: {total_samples}")
    print(f"Overall inference FPS: {overall_fps:.2f}")
    print(f"Inference throughput: {total_samples / total_inference_time:.2f} samples/sec")
    print_memory_usage("End of inference memory", rank=0)

    return predictions

def filter_predictions(predictions, confidence_threshold=0.3):
    filtered_predictions = []

    for pred in predictions:
        if pred['confidence'] >= confidence_threshold:
            player_ids = pred['player_ids']

            if not player_ids or all(pid == -1 for pid in player_ids):
                continue

            unique_player_ids = list(set(player_ids))

            filtered_pred = {
                'frame_id': pred['frame_id'],
                'player_ids': unique_player_ids,
                'event_type': pred['event_type'],
                'confidence': pred['confidence'],
                'is_single_player': pred['is_single_player']
            }

            filtered_predictions.append(filtered_pred)

    return filtered_predictions

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main_worker(rank, world_size, args):
    print(f"Running on rank {rank}, world_size {world_size}")

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    performance_monitor = PerformanceMonitor(rank)

    timestamp = get_timestamp()
    if rank == 0:
        print(f"Starting improved model training (Fixed SlowFast + 4-GPU parallel + pos:neg ratio experiments + performance monitoring), timestamp: {timestamp}")
        print(f"Using device: {device}")
        print(f"Pos:neg sample ratio: 1:{args.neg_pos_ratio}")

        print_system_info(rank)
        print_memory_usage("Training start", rank)

        print(f"\n=== Ablation Study Settings ===")
        print(f"use_bbox_feat: {args.use_bbox_feat}")
        print(f"use_skeleton_feat: {args.use_skeleton_feat}")
        print(f"use_position_feat: {args.use_position_feat}")
        print(f"use_trajectory_feat: {args.use_trajectory_feat}")

    try:
        if rank == 0:
            print("\nInitializing SlowFast feature extractor...")
        slowfast_init_start = time.time()
        initialize_slowfast(device)
        slowfast_init_time = time.time() - slowfast_init_start
        if rank == 0:
            print(f"SlowFast feature extractor initialized successfully, time: {slowfast_init_time:.2f}s")
            print_memory_usage("After SlowFast init", rank)
    except Exception as e:
        if rank == 0:
            print(f"SlowFast feature extractor initialization failed: {e}")
        cleanup_distributed()
        return

    try:
        if rank == 0:
            print("\nCreating datasets...")
        dataset_start = time.time()

        train_dataset = ImprovedBasketballEventDataset(
            data_root=args.data_root,
            split='train',
            neg_pos_ratio=args.neg_pos_ratio,
            cache_features=True,
            bbox_temporal_size=8,
            skeleton_temporal_size=5,
            trajectory_temporal_size=30,
            alpha=args.alpha
        )

        val_dataset = ImprovedBasketballEventDataset(
            data_root=args.data_root,
            split='val',
            neg_pos_ratio=0.5,
            cache_features=True,
            bbox_temporal_size=8,
            skeleton_temporal_size=5,
            trajectory_temporal_size=30,
            alpha=args.alpha
        )

        dataset_time = time.time() - dataset_start
        if rank == 0:
            print(f"Dataset creation completed, time: {dataset_time:.2f}s")
            print(f"Train dataset: {len(train_dataset)} samples, alpha={args.alpha}, pos:neg=1:{args.neg_pos_ratio}")
            print(f"Val dataset: {len(val_dataset)} samples")
            print_memory_usage("After dataset creation", rank)
    except Exception as e:
        if rank == 0:
            print(f"Dataset creation failed: {e}")
        cleanup_distributed()
        return

    if rank == 0:
        print("\nCreating data loaders...")
    dataloader_start = time.time()

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=improved_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=improved_collate_fn
    )

    dataloader_time = time.time() - dataloader_start
    if rank == 0:
        print(f"Data loader creation completed, time: {dataloader_time:.2f}s")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if rank == 0:
        print("\nCreating model...")
    model_start = time.time()

    model = ImprovedBasketballEventModel(
        num_classes=9,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        dropout=0.2,
        use_bbox_feat=args.use_bbox_feat,
        use_skeleton_feat=args.use_skeleton_feat,
        use_position_feat=args.use_position_feat,
        use_trajectory_feat=args.use_trajectory_feat
    )
    model = model.to(device)

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    model_time = time.time() - model_start
    if rank == 0:
        print(f"Model creation completed, time: {model_time:.2f}s")
        print("Model wrapped as DDP")
        print_memory_usage("After model creation", rank)

    class_weights = torch.ones(9)

    for i in range(9):
        original_type = LABEL_TO_EVENT_TYPE.get(i, 9)
        if original_type in SINGLE_PLAYER_EVENTS:
            class_weights[i] = 2.0
        elif original_type in DOUBLE_PLAYER_EVENTS:
            class_weights[i] = 1.5
        else:
            class_weights[i] = 0.5

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=(rank == 0)
    )

    performance_monitor.start_training()

    num_epochs = args.epochs
    best_single_acc = 0.0
    best_overall_map = 0.0
    best_single_map = 0.0

    if rank == 0:
        ablation_suffix = f"_b{int(args.use_bbox_feat)}_s{int(args.use_skeleton_feat)}_p{int(args.use_position_feat)}_t{int(args.use_trajectory_feat)}"
        ratio_suffix = f"_ratio1_{args.neg_pos_ratio}"
        best_model_path = f'models/improved_basketball_ablation{ablation_suffix}{ratio_suffix}_{timestamp}.pth'
        os.makedirs('models', exist_ok=True)
        print("Starting improved model training (Fixed SlowFast + 4-GPU parallel + Ablation Study + pos:neg ratio experiments + performance monitoring)...")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        performance_monitor.start_epoch()

        try:
            train_loss, train_acc = train_improved_model(
                model, train_loader, criterion, optimizer, device, epoch, rank, performance_monitor
            )

            val_loss, val_acc, single_acc, double_acc, overall_map, single_map, double_map = validate_improved_model(
                model, val_loader, criterion, device, rank, performance_monitor
            )

            if rank == 0:
                scheduler.step(single_map)

                print(f'\nEpoch: {epoch} (pos:neg = 1:{args.neg_pos_ratio})')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'Overall mAP: {overall_map:.4f}')
                print(f'Single player events - Acc: {single_acc:.2f}%, mAP: {single_map:.4f}')
                print(f'Double player events - Acc: {double_acc:.2f}%, mAP: {double_map:.4f}')

                save_model = False
                save_reason = ""

                if single_map > best_single_map:
                    best_single_map = single_map
                    save_model = True
                    save_reason = f"Single player event mAP improved to {single_map:.4f}"
                elif single_map == best_single_map and overall_map > best_overall_map:
                    best_overall_map = overall_map
                    save_model = True
                    save_reason = f"Overall mAP improved to {overall_map:.4f}"
                elif single_map == best_single_map and overall_map == best_overall_map and single_acc > best_single_acc:
                    best_single_acc = single_acc
                    save_model = True
                    save_reason = f"Single player event accuracy improved to {single_acc:.2f}%"

                if save_model:
                    performance_stats = {
                        'avg_batch_time': performance_monitor.get_avg_batch_time(),
                        'avg_fps': performance_monitor.get_avg_fps(),
                        'peak_gpu_memory': performance_monitor.get_peak_memory()[0],
                        'peak_cpu_memory': performance_monitor.get_peak_memory()[1],
                        'training_time_so_far': time.time() - performance_monitor.training_start_time,
                        'avg_data_loading_time': performance_monitor.get_avg_data_loading_time(),
                        'avg_forward_time': performance_monitor.get_avg_forward_time(),
                        'avg_backward_time': performance_monitor.get_avg_backward_time(),
                        'throughput': performance_monitor.get_throughput()
                    }

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'single_acc': single_acc,
                        'double_acc': double_acc,
                        'overall_map': overall_map,
                        'single_map': single_map,
                        'double_map': double_map,
                        'val_loss': val_loss,
                        'timestamp': timestamp,
                        'class_mapping': {'event_to_label': EVENT_TYPE_TO_LABEL, 'label_to_event': LABEL_TO_EVENT_TYPE},
                        'alpha': args.alpha,
                        'neg_pos_ratio': args.neg_pos_ratio,
                        'performance_stats': performance_stats,
                        'ablation_settings': {
                            'use_bbox_feat': args.use_bbox_feat,
                            'use_skeleton_feat': args.use_skeleton_feat,
                            'use_position_feat': args.use_position_feat,
                            'use_trajectory_feat': args.use_trajectory_feat,
                            'alpha': args.alpha,
                            'neg_pos_ratio': args.neg_pos_ratio
                        }
                    }, best_model_path)
                    print(f'Best model saved to {best_model_path}, {save_reason}')

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            if rank == 0:
                print(f"Training error (Epoch {epoch}): {e}")
            continue

    performance_monitor.end_training()

    if rank == 0:
        print(f'\nTraining completed (pos:neg = 1:{args.neg_pos_ratio}).')
        print(f'Best single player event mAP: {best_single_map:.4f}')
        print(f'Best overall mAP: {best_overall_map:.4f}')
        print(f'Best single player event accuracy: {best_single_acc:.2f}%')

        performance_monitor.print_summary()
        print_memory_usage("After training", rank)

    cleanup_distributed()

def run_testing(args):
    print("Running test mode...")

    device = torch.device('cuda:0')

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    timestamp = get_timestamp()

    performance_monitor = PerformanceMonitor(rank=0)

    print_system_info(rank=0)
    print_memory_usage("Test start", rank=0)

    try:
        print("\nInitializing SlowFast feature extractor...")
        slowfast_init_start = time.time()
        initialize_slowfast(device)
        slowfast_init_time = time.time() - slowfast_init_start
        print(f"SlowFast feature extractor initialized successfully, time: {slowfast_init_time:.2f}s")
        print_memory_usage("After SlowFast init", rank=0)
    except Exception as e:
        print(f"SlowFast feature extractor initialization failed: {e}")
        return

    try:
        print("\nCreating test dataset...")
        dataset_start = time.time()
        test_dataset = ImprovedBasketballEventDataset(
            data_root=args.data_root,
            split='val',
            neg_pos_ratio=0,
            cache_features=True,
            bbox_temporal_size=8,
            skeleton_temporal_size=5,
            trajectory_temporal_size=30,
            alpha=args.alpha
        )
        dataset_time = time.time() - dataset_start
        print(f"Test dataset created successfully, {len(test_dataset)} samples, alpha={args.alpha}, time: {dataset_time:.2f}s")
        print_memory_usage("After dataset creation", rank=0)
    except Exception as e:
        print(f"Test dataset creation failed: {e}")
        return

    print("\nCreating data loader...")
    dataloader_start = time.time()
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=improved_collate_fn
    )
    dataloader_time = time.time() - dataloader_start
    print(f"Data loader creation completed, time: {dataloader_time:.2f}s")
    print(f"Test batches: {len(test_loader)}")

    if os.path.exists(args.model_path):
        print(f"\nLoading model weights: {args.model_path}")
        model_load_start = time.time()
        checkpoint = torch.load(args.model_path, map_location=device)

        ablation_settings = checkpoint.get('ablation_settings', {
            'use_bbox_feat': True,
            'use_skeleton_feat': True,
            'use_position_feat': True,
            'use_trajectory_feat': True,
            'neg_pos_ratio': 5
        })

        training_neg_pos_ratio = ablation_settings.get('neg_pos_ratio', checkpoint.get('neg_pos_ratio', 5))

        training_performance = checkpoint.get('performance_stats', {})

        print(f"Ablation settings loaded from model: {ablation_settings}")
        print(f"Training pos:neg sample ratio: 1:{training_neg_pos_ratio}")
        if training_performance:
            print(f"Training performance stats:")
            print(f"  Average batch time: {training_performance.get('avg_batch_time', 0):.4f}s")
            print(f"  Average FPS: {training_performance.get('avg_fps', 0):.2f}")
            print(f"  Peak GPU memory: {training_performance.get('peak_gpu_memory', 0):.2f}GB")

        model = ImprovedBasketballEventModel(
            num_classes=9,
            hidden_dim=512,
            num_heads=8,
            num_layers=4,
            dropout=0.2,
            use_bbox_feat=ablation_settings['use_bbox_feat'],
            use_skeleton_feat=ablation_settings['use_skeleton_feat'],
            use_position_feat=ablation_settings['use_position_feat'],
            use_trajectory_feat=ablation_settings['use_trajectory_feat']
        )
        model = model.to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model_load_time = time.time() - model_load_start
        print(f"Model weights loaded successfully, time: {model_load_time:.2f}s")
        print_memory_usage("After model loading", rank=0)
    else:
        print(f"Model file does not exist: {args.model_path}")
        return

    print("\nStarting inference...")
    performance_monitor.start_training()

    raw_predictions = inference_improved_model(model, test_loader, device, performance_monitor)

    performance_monitor.end_training()

    print("\nProcessing prediction results...")
    filtered_predictions = filter_predictions(raw_predictions, confidence_threshold=0.3)

    os.makedirs('results', exist_ok=True)

    ablation_suffix = f"_b{int(ablation_settings['use_bbox_feat'])}_s{int(ablation_settings['use_skeleton_feat'])}_p{int(ablation_settings['use_position_feat'])}_t{int(ablation_settings['use_trajectory_feat'])}"
    ratio_suffix = f"_ratio1_{training_neg_pos_ratio}"
    predictions_path = f'results/test_predictions_ablation{ablation_suffix}{ratio_suffix}_{timestamp}.json'

    results_with_performance = {
        'predictions': filtered_predictions,
        'performance_stats': {
            'total_inference_time': performance_monitor.total_training_time,
            'avg_inference_fps': performance_monitor.get_avg_fps(),
            'avg_batch_time': performance_monitor.get_avg_batch_time(),
            'peak_gpu_memory': performance_monitor.get_peak_memory()[0],
            'peak_cpu_memory': performance_monitor.get_peak_memory()[1],
            'total_samples': len(raw_predictions),
            'filtered_samples': len(filtered_predictions),
            'throughput': performance_monitor.get_throughput()
        },
        'model_info': {
            'training_neg_pos_ratio': training_neg_pos_ratio,
            'ablation_settings': ablation_settings,
            'model_path': args.model_path,
            'test_timestamp': timestamp,
            'training_performance': training_performance
        }
    }

    with open(predictions_path, 'w') as f:
        json.dump(results_with_performance, f, indent=2)

    print(f'\n=== Test Results Summary ===')
    print(f'Prediction results saved to: {predictions_path}')
    print(f'Raw predictions count: {len(raw_predictions)}')
    print(f'Filtered predictions count: {len(filtered_predictions)}')
    print(f'Model training pos:neg sample ratio: 1:{training_neg_pos_ratio}')

    single_pred_count = sum(1 for pred in filtered_predictions if pred['is_single_player'])
    double_pred_count = len(filtered_predictions) - single_pred_count

    event_type_counts = {}
    single_event_counts = {}
    double_event_counts = {}

    for pred in filtered_predictions:
        event_type = pred['event_type']
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

        if pred['is_single_player']:
            single_event_counts[event_type] = single_event_counts.get(event_type, 0) + 1
        else:
            double_event_counts[event_type] = double_event_counts.get(event_type, 0) + 1

    print(f"\n=== Prediction Results Statistics (training ratio: 1:{training_neg_pos_ratio}) ===")
    print(f"Single player event predictions: {single_pred_count}")
    print(f"Double player event predictions: {double_pred_count}")

    print("\n=== Event Type Prediction Distribution ===")
    for event_type in sorted(set(list(single_event_counts.keys()) + list(double_event_counts.keys()))):
        if 1 <= event_type <= 8:
            class_name = CLASS_NAMES[event_type - 1]
            event_category = "Single" if event_type in SINGLE_PLAYER_EVENTS else "Double"
        else:
            class_name = "Background"
            event_category = "Background"

        single_count = single_event_counts.get(event_type, 0)
        double_count = double_event_counts.get(event_type, 0)
        total_count = single_count + double_count

        print(f"  {class_name} ({event_category}): {total_count} (single:{single_count}, double:{double_count})")

    performance_monitor.print_summary()
    print_memory_usage("After test", rank=0)

    print(f"\n=== Test Completed ===")

def main():
    parser = argparse.ArgumentParser(
        description='Basketball Event Recognition with Multi-GPU Training, Ablation Study, Positive-Negative Ratio Experiments and Performance Monitoring')
    parser.add_argument('--data_root', type=str,
                        default='/work6/q_hu/AG/SportsHHI/cnn_method/RKU_dataset/annotations',
                        help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU (default: 4)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--world_size', type=int, default=4,
                        help='Number of GPUs to use (default: 4)')
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12355',
                        help='Master port for distributed training')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run testing, no training')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to model checkpoint for testing')

    parser.add_argument('--use_bbox_feat', action='store_true', default=True,
                        help='Whether to use bbox features (default: True)')
    parser.add_argument('--no_bbox_feat', action='store_true',
                        help='Disable bbox features')
    parser.add_argument('--use_skeleton_feat', action='store_true', default=True,
                        help='Whether to use skeleton features (default: True)')
    parser.add_argument('--no_skeleton_feat', action='store_true',
                        help='Disable skeleton features')
    parser.add_argument('--use_position_feat', action='store_true', default=True,
                        help='Whether to use position features (default: True)')
    parser.add_argument('--no_position_feat', action='store_true',
                        help='Disable position features')
    parser.add_argument('--use_trajectory_feat', action='store_true', default=True,
                        help='Whether to use trajectory features (default: True)')
    parser.add_argument('--no_trajectory_feat', action='store_true',
                        help='Disable trajectory features')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Alpha parameter for spatial feature weighting in positional encoding (default: 0.1)')
    parser.add_argument('--neg_pos_ratio', type=int, default=5,
                        help='Negative to positive sample ratio for training (default: 5, meaning 1:5)')

    args = parser.parse_args()

    if args.no_bbox_feat:
        args.use_bbox_feat = False
    if args.no_skeleton_feat:
        args.use_skeleton_feat = False
    if args.no_position_feat:
        args.use_position_feat = False
    if args.no_trajectory_feat:
        args.use_trajectory_feat = False

    print(f"Using {args.world_size} GPUs for distributed training")
    print(f"Data path: {args.data_root}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Training epochs: {args.epochs}")
    print(f"Pos:neg sample ratio: 1:{args.neg_pos_ratio}")

    print(f"\n=== Ablation Study Settings ===")
    print(f"Use bbox features: {args.use_bbox_feat}")
    print(f"Use skeleton features: {args.use_skeleton_feat}")
    print(f"Use position features: {args.use_position_feat}")
    print(f"Use trajectory features: {args.use_trajectory_feat}")

    print(f"\n=== Pos:Neg Sample Ratio Experiment Description ===")
    print(f"Current setting: 1:{args.neg_pos_ratio}")
    print(f"Recommended experiment settings:")
    print(f"  - 1:1  (balanced dataset)")
    print(f"  - 1:10 (high negative sample ratio)")
    print(f"  - 1:15 (higher negative sample ratio)")
    print(f"  - 1:20 (very high negative sample ratio)")

    if args.test_only and args.model_path:
        run_testing(args)
    else:
        mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size)

if __name__ == '__main__':
    main()