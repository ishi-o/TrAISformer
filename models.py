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

"""Models for TrAISformer.
    https://arxiv.org/abs/2109.03958

The code is built upon:
    https://github.com/karpathy/minGPT
"""

import math
import logging
import pdb


import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.

    因果自注意力层
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seqlen, config.max_seqlen)).view(
                1, 1, config.max_seqlen, config.max_seqlen
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """
    an unassuming Transformer block
    自定义的Transformer模块
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TrAISformer(nn.Module):
    """
    Transformer for AIS trajectories.
    """

    def __init__(self, config, partition_model=None):
        super().__init__()
        # 根据fourhot算法, 需要N[lat, lon, sog, cog](区间数)
        # 作者的配置: lat、lon每区间0.01d, sog每区间1 knot, cog每区间5d
        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        # 四个特征的N之和
        self.full_size = config.full_size
        # embd向量大小: [lat, lon, sog, cog]: [256, 256, 128, 128]
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        # 注册缓冲向量att_sizes: 论文中fourhot算法中的N
        self.register_buffer(
            "att_sizes",
            torch.tensor(
                [config.lat_size, config.lon_size, config.sog_size, config.cog_size]
            ),
        )
        self.register_buffer(
            "emb_sizes",
            torch.tensor(
                [
                    config.n_lat_embd,
                    config.n_lon_embd,
                    config.n_sog_embd,
                    config.n_cog_embd,
                ]
            ),
        )
        # 默认使用均值划分(uniform)
        if hasattr(config, "partition_mode"):
            self.partition_mode = config.partition_mode
        else:
            self.partition_mode = "uniform"
        self.partition_model = partition_model
        # 对应论文中的模糊损失: 取不同的N进行两次预测, 加权求最终损失值
        if hasattr(config, "blur"):
            self.blur = config.blur
            self.blur_learnable = config.blur_learnable
            self.blur_loss_w = config.blur_loss_w
            self.blur_n = config.blur_n
            if self.blur:
                # 模糊层是一个一维卷积层, 采用复制填充
                self.blur_module = nn.Conv1d(
                    1, 1, 3, padding=1, padding_mode='replicate', groups=1, bias=False
                )
                if not self.blur_learnable:
                    # 模糊参数不自学, 则均设为不参与梯度计算, 值为1/3
                    for params in self.blur_module.parameters():
                        params.requires_grad = False
                        params.fill_(1 / 3)
            else:
                self.blur_module = None

        if hasattr(config, "label_smoothing"):
            self.label_smoothing = config.label_smoothing
        else:
            self.label_smoothing = 0.0

        # 根据论文的fourhot算法, 需要给定lat, lon的最小最大值
        if hasattr(config, "lat_min"):  # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max - config.lat_min
            self.lon_range = config.lon_max - config.lon_min
            self.sog_range = 30.0

        if hasattr(config, "mode"):  # mode: "pos" or "velo".
            # "pos": predict directly the next positions.   直接预测下一个点的位置
            # "velo": predict the velocities, use them to   预测速度, 然后通过速度预测位置
            # calculate the next positions.
            self.mode = config.mode
        else:
            self.mode = "pos"

        # Passing from the 4-D space to a high-dimentional space
        # 将四维的特征空间编码通过Embedding(嵌入层)成为高维特征空间
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)

        #
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        # Dropout层
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        # nn.Sequential: 按顺序地连接神经网络层, 自动处理前向传播
        # 创建config.n_layer个Block, 每个Block由config配置, 在论文中设置的是n_layer=8
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # decoder head 解码头
        # nn.LayerNorm: 层归一化
        self.ln_f = nn.LayerNorm(config.n_embd)
        # mlp模式: 全连接层作为任务头
        if self.mode in ("mlp_pos", "mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            self.head = nn.Linear(
                config.n_embd, self.full_size, bias=False
            )  # Classification head

        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        '''
        初始化权重
        '''
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.

        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        # 将参数分为两种: 需要/不需要经过正则化权重衰减的
        decay = set()
        no_decay = set()
        # 白名单(全连接层和卷积层)模块的参数需要权重衰减
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        # 黑名单(归一化层和嵌入层)不需要
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():  # module名和module对象
            for pn, p in m.named_parameters():  # parameter名和parameter对象
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    # 偏置量不需要权重衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        # 验证所有参数均进行了分类
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        # 封装为列表传递给优化器
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        # 使用AdamW作为优化器并返回
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def to_indexes(self, x, mode="uniform"):
        """Convert tokens to indexes.

        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated
                to [0,1).
            model: currenly only supports "uniform".

        Returns:
            idxs: a Tensor (dtype: Long) of indexes.

        对应论文中的fourhot编码算法, 其中x经过了归一化
        """
        bs, seqlen, data_dim = x.shape
        if mode == "uniform":
            idxs = (x * self.att_sizes).long()
            return idxs, idxs
        elif mode in ("freq", "freq_uniform"):

            idxs = (x * self.att_sizes).long()
            idxs_uniform = idxs.clone()
            discrete_lats, discrete_lons, lat_ids, lon_ids = self.partition_model(
                x[:, :, :2]
            )
            #             pdb.set_trace()
            idxs[:, :, 0] = torch.round(lat_ids.reshape((bs, seqlen))).long()
            idxs[:, :, 1] = torch.round(lon_ids.reshape((bs, seqlen))).long()
            return idxs, idxs_uniform

    def forward(self, x, masks=None, with_targets=False, return_loss_tuple=False):
        """
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated
                to [0,1).
            masks: a Tensor of the same size of x. masks[idx] = 0. if
                x[idx] is a padding.
            with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:],
                otherwise inputs = x.
        Returns:
            logits, loss
        """

        if self.mode in (
            "mlp_pos",
            "mlp",
        ):
            idxs, idxs_uniform = x, x  # use the real-values of x.
        else:
            # Convert to indexes
            idxs, idxs_uniform = self.to_indexes(x, mode=self.partition_mode)

        # 若x是带标签的, 则视前L-1点为输入, 后L-1点为标签
        # 实际上为了方便嵌入, 这里的idxs最低维不是onehot向量, 而是离散的整数值
        if with_targets:
            inputs = idxs[:, :-1, :].contiguous()
            targets = idxs[:, 1:, :].contiguous()
            targets_uniform = idxs_uniform[:, 1:, :].contiguous()
            inputs_real = x[:, :-1, :].contiguous()
            targets_real = x[:, 1:, :].contiguous()
        # 否则标签为空
        else:
            inputs_real = x
            inputs = idxs
            targets = None

        batchsize, seqlen, _ = inputs.size()
        assert (
            seqlen <= self.max_seqlen
        ), "Cannot forward, model block size is exhausted."

        # forward the GPT model
        # 嵌入
        lat_embeddings = self.lat_emb(inputs[:, :, 0])  # (batchsize, seqlen, lat_size)
        lon_embeddings = self.lon_emb(inputs[:, :, 1])
        sog_embeddings = self.sog_emb(inputs[:, :, 2])
        cog_embeddings = self.cog_emb(inputs[:, :, 3])
        token_embeddings = torch.cat(
            (lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings), dim=-1
        )

        position_embeddings = self.pos_emb[
            :, :seqlen, :
        ]  # each position maps to a (learnable) vector (1, seqlen, n_embd)
        # 经过dropout, 自定义Block, 归一化层, 全连接任务头, 得到logit输出
        fea = self.drop(token_embeddings + position_embeddings)
        fea = self.blocks(fea)
        fea = self.ln_f(fea)  # (bs, seqlen, n_embd)
        logits = self.head(fea)  # (bs, seqlen, full_size) or (bs, seqlen, n_embd)

        lat_logits, lon_logits, sog_logits, cog_logits = torch.split(
            logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1
        )

        # Calculate the loss
        # 计算损失
        loss = None
        loss_tuple = None
        if targets is not None:
            # 经过ce获得四者损失再相加
            # print(sog_logits.view(-1, self.sog_size).shape)
            # print(targets[:, :, 2].view(-1).shape)

            # cross_entropy内置label_smoothing
            sog_loss = F.cross_entropy(
                sog_logits.view(-1, self.sog_size),
                targets[:, :, 2].view(-1),
                reduction="none",
                label_smoothing=self.label_smoothing,
            ).view(batchsize, seqlen)
            cog_loss = F.cross_entropy(
                cog_logits.view(-1, self.cog_size),
                targets[:, :, 3].view(-1),
                reduction="none",
                label_smoothing=self.label_smoothing,
            ).view(batchsize, seqlen)
            lat_loss = F.cross_entropy(
                lat_logits.view(-1, self.lat_size),
                targets[:, :, 0].view(-1),
                reduction="none",
                label_smoothing=self.label_smoothing,
            ).view(batchsize, seqlen)
            lon_loss = F.cross_entropy(
                lon_logits.view(-1, self.lon_size),
                targets[:, :, 1].view(-1),
                reduction="none",
                label_smoothing=self.label_smoothing,
            ).view(batchsize, seqlen)

            # blur版本数据先softmax再一维卷积再nll_loss(和ce区别是softmax和nll中间加了个卷积)
            if self.blur:
                lat_probs = F.softmax(lat_logits, dim=-1)
                lon_probs = F.softmax(lon_logits, dim=-1)
                sog_probs = F.softmax(sog_logits, dim=-1)
                cog_probs = F.softmax(cog_logits, dim=-1)

                for _ in range(self.blur_n):
                    blurred_lat_probs = self.blur_module(
                        lat_probs.reshape(-1, 1, self.lat_size)
                    ).reshape(lat_probs.shape)
                    blurred_lon_probs = self.blur_module(
                        lon_probs.reshape(-1, 1, self.lon_size)
                    ).reshape(lon_probs.shape)
                    blurred_sog_probs = self.blur_module(
                        sog_probs.reshape(-1, 1, self.sog_size)
                    ).reshape(sog_probs.shape)
                    blurred_cog_probs = self.blur_module(
                        cog_probs.reshape(-1, 1, self.cog_size)
                    ).reshape(cog_probs.shape)

                    blurred_lat_loss = F.nll_loss(
                        blurred_lat_probs.view(-1, self.lat_size),
                        targets[:, :, 0].view(-1),
                        reduction="none",
                    ).view(batchsize, seqlen)
                    blurred_lon_loss = F.nll_loss(
                        blurred_lon_probs.view(-1, self.lon_size),
                        targets[:, :, 1].view(-1),
                        reduction="none",
                    ).view(batchsize, seqlen)
                    blurred_sog_loss = F.nll_loss(
                        blurred_sog_probs.view(-1, self.sog_size),
                        targets[:, :, 2].view(-1),
                        reduction="none",
                    ).view(batchsize, seqlen)
                    blurred_cog_loss = F.nll_loss(
                        blurred_cog_probs.view(-1, self.cog_size),
                        targets[:, :, 3].view(-1),
                        reduction="none",
                    ).view(batchsize, seqlen)

                    lat_loss += self.blur_loss_w * blurred_lat_loss
                    lon_loss += self.blur_loss_w * blurred_lon_loss
                    sog_loss += self.blur_loss_w * blurred_sog_loss
                    cog_loss += self.blur_loss_w * blurred_cog_loss

                    lat_probs = blurred_lat_probs
                    lon_probs = blurred_lon_probs
                    sog_probs = blurred_sog_probs
                    cog_probs = blurred_cog_probs

            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)

            if masks is not None:
                loss = (loss * masks).sum(dim=1) / masks.sum(dim=1)

            loss = loss.mean()

        if return_loss_tuple:
            return logits, loss, loss_tuple
        else:
            return logits, loss
