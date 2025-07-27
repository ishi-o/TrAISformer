#!/usr/bin/env python
# coding: utf-8
# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch implementation of TrAISformer---A generative transformer for
AIS trajectory prediction

https://arxiv.org/abs/2109.03958

"""
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import pdb
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import models, trainers, datasets, utils
from config_trAISformer import Config

# cf: 配置项
cf = Config()

# 若配置了tb_log, 则将部分信息输出到日志中, 详见trainers.py
TB_LOG = cf.tb_log
tb = {}
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter()

# 确定性训练, 固定种子以及设置cudnn的确定性属性
utils.set_seed(42)
# pi 常数在 pytorch 2.x 版本已有原生定义, 故注释此行
# torch.pi = torch.acos(torch.zeros(1)).item() * 2


def DrawTrajectory(batch_idx, sample_idx, input_coords, pred_coords):
    '''
    画出给定一系列真实轨迹集合和预测轨迹集合, 第batch_idx个batch的第一条轨迹
    '''
    real_traj = input_coords[batch_idx].detach().cpu().numpy()
    pred_traj = pred_coords[batch_idx].detach().cpu().numpy()
    real_lat = real_traj[:, 0] * 180 / np.pi
    real_lon = real_traj[:, 1] * 180 / np.pi
    pred_lat = pred_traj[:, 0] * 180 / np.pi
    pred_lon = pred_traj[:, 1] * 180 / np.pi
    plt.figure()
    # plt.plot(real_lon[0], real_lat[0], markersize=2, label='start_r')
    # plt.plot(pred_lon[0], pred_lat[0], markersize=2, label='start_p')

    plt.plot(real_lon, real_lat, label=r'$real/\degree$')
    plt.plot(pred_lon, pred_lat, label=r'$pred/\degree$')

    plt.legend()
    plt.xlabel('LON(jing du)')
    plt.ylabel('LAT(wei du)')

    plt.savefig(
        cf.savedir
        + 'traj_'
        + str(int(batch_idx / 5))
        + '_'
        + str(int(sample_idx / 10))
        + '.png'
    )


if __name__ == "__main__":

    # 配置项, device: 设备(本项目默认为cuda驱动的GPU), init_seqlen: 初始序列长度
    device = cf.device
    init_seqlen = cf.init_seqlen

    ## Logging
    # ===============================
    # 若不存在该目录则创建
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    # 定义日志信息的格式
    utils.new_log(cf.savedir, "log")

    ## Data
    # ===============================
    # 速度阈值, 只取速度超过moving_threshold的数据
    moving_threshold = 0.05
    # 数据集所在路径
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    # 加载训练集、验证集、测试集
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:  # 只读打开二进制文件.pkl
            l_pred_errors = pickle.load(f)
        # 对轨迹数据预处理
        for V in l_pred_errors:
            try:
                # V["traj"]是轨迹的点集
                # 第三列为速度, 取出第一个SOG超过阈值的点的编号
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                # 所有点均不满足, 则取最后一点的编号
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            # 覆盖掉原来的V["traj"], 相当于去除了前方的一系列停留点
            V["traj"] = V["traj"][moving_idx:, :]
        # 取出预处理后的轨迹数据中, 所有点的特征值均不为nan, 且点数量大于min_seqlen的轨迹
        Data[phase] = [
            x
            for x in l_pred_errors
            if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen
        ]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        # 将Data数组根据要求封装为不同的Dataset子类对象
        # 因为 输入是轨迹(假设长度为L)的前L-1个点, 标签是后L-1个点
        # 轨迹数据最多是cf.max_seqlen个点, 所以需要cf.max_seqlen+1保证序列长度足够
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(
                Data[phase], max_seqlen=cf.max_seqlen + 1, device=cf.device
            )
        else:
            aisdatasets[phase] = datasets.AISDataset(
                Data[phase], max_seqlen=cf.max_seqlen + 1, device=cf.device
            )
        # 测试时不打乱数据保证确定性
        if phase == "test":
            shuffle = False
        else:
            shuffle = True
        # 一切Dataset子类均封装为DataLoader对象
        aisdls[phase] = DataLoader(
            aisdatasets[phase], batch_size=cf.batch_size, shuffle=shuffle
        )
    # 2 * 样本数 * 序列长度, 乘2不知道是何意
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

    ## Model
    # ===============================
    # 模型
    model = models.TrAISformer(cf, partition_model=None)

    ## Trainer
    # ===============================
    # 训练器
    trainer = trainers.Trainer(
        model,
        aisdatasets["train"],
        aisdatasets["valid"],
        cf,
        savedir=cf.savedir,
        device=cf.device,
        aisdls=aisdls,
        INIT_SEQLEN=init_seqlen,
    )

    ## Training
    # ===============================
    # cf.retrain决定此次运行是否训练
    if cf.retrain:
        trainer.train()

    ## Evaluation
    # ===============================
    # 加载训练好的模型
    model.load_state_dict(torch.load(cf.ckpt_path))

    v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
    max_seqlen = init_seqlen + 6 * 4

    model.eval()
    # 最小误差, 均值误差, 掩码
    l_min_errors, l_mean_errors, l_masks = [], [], []
    # 进度条
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    # 不带梯度传播地测试
    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros(
                (batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)
            ).to(cf.device)
            for i_sample in range(cf.n_samples):
                preds = trainers.sample(
                    model,
                    seqs_init,
                    max_seqlen - init_seqlen,
                    temperature=1.0,
                    sample=True,
                    sample_mode=cf.sample_mode,
                    r_vicinity=cf.r_vicinity,
                    top_k=cf.top_k,
                )
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen :]

                if it % 5 == 0 and i_sample % 10 == 0:
                    DrawTrajectory(
                        batch_idx=it,
                        sample_idx=i_sample,
                        input_coords=input_coords,
                        pred_coords=pred_coords,
                    )

            # Accumulation through batches
            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, cf.init_seqlen :])

    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()

    ## Plot
    # ===============================
    # 画性能图
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors)

    timestep = 6
    plt.plot(1, pred_errors[timestep], "o")
    plt.plot([1, 1], [0, pred_errors[timestep]], "r")
    plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(
        1.12,
        pred_errors[timestep] - 0.5,
        "{:.4f}".format(pred_errors[timestep]),
        fontsize=10,
    )

    timestep = 12
    plt.plot(2, pred_errors[timestep], "o")
    plt.plot([2, 2], [0, pred_errors[timestep]], "r")
    plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(
        2.12,
        pred_errors[timestep] - 0.5,
        "{:.4f}".format(pred_errors[timestep]),
        fontsize=10,
    )

    timestep = 18
    plt.plot(3, pred_errors[timestep], "o")
    plt.plot([3, 3], [0, pred_errors[timestep]], "r")
    plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(
        3.12,
        pred_errors[timestep] - 0.5,
        "{:.4f}".format(pred_errors[timestep]),
        fontsize=10,
    )
    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.xlim([0, 12])
    plt.ylim([0, 20])
    # plt.ylim([0,pred_errors.max()+0.5])
    plt.savefig(cf.savedir + "prediction_error.png")

    # Yeah, done!!!
