import logging
import os
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

import utils
from config_trAISformer import Config

logger = logging.getLogger(__name__)

# 若配置了tb_log, 则将部分信息输出到日志中, 详见trainers.py
TB_LOG = Config().tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter()


class Trainer:

    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        config,
        savedir=None,
        device=torch.device("cpu"),
        aisdls={},
        INIT_SEQLEN=0,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir

        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN

    def save_checkpoint(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(
            f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}"
        )
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        (
            model,
            config,
            aisdls,
            INIT_SEQLEN,
        ) = (
            self.model,
            self.config,
            self.aisdls,
            self.INIT_SEQLEN,
        )
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        if model.mode in (
            "gridcont_gridsin",
            "gridcont_gridsigmoid",
            "gridcont2_gridsigmoid",
        ):
            return_loss_tuple = True
        else:
            return_loss_tuple = False

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            losses = []
            n_batches = len(loader)
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            d_loss, d_reg_loss, d_n = 0, 0, 0
            for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:

                # place data on the correct device
                seqs = seqs.to(self.device)
                masks = masks[:, :-1].to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    if return_loss_tuple:
                        logits, loss, loss_tuple = model(
                            seqs,
                            masks=masks,
                            with_targets=True,
                            return_loss_tuple=return_loss_tuple,
                        )
                    else:
                        logits, loss = model(seqs, masks=masks, with_targets=True)
                    loss = (
                        loss.mean()
                    )  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                d_loss += loss.item() * seqs.shape[0]
                if return_loss_tuple:
                    reg_loss = loss_tuple[-1]
                    reg_loss = reg_loss.mean()
                    d_reg_loss += reg_loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                            seqs >= 0
                        ).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(
                                max(1, config.warmup_tokens)
                            )
                        else:
                            # cosine learning rate decay
                            progress = float(
                                self.tokens - config.warmup_tokens
                            ) / float(
                                max(1, config.final_tokens - config.warmup_tokens)
                            )
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}"
                    )

                    # tb logging
                    if TB_LOG:
                        tb.add_scalar("loss", loss.item(), epoch * n_batches + it)
                        tb.add_scalar("lr", lr, epoch * n_batches + it)

                        for name, params in model.head.named_parameters():
                            tb.add_histogram(
                                f"head.{name}", params, epoch * n_batches + it
                            )
                            tb.add_histogram(
                                f"head.{name}.grad", params.grad, epoch * n_batches + it
                            )
                        if model.mode in ("gridcont_real",):
                            for name, params in model.res_pred.named_parameters():
                                tb.add_histogram(
                                    f"res_pred.{name}", params, epoch * n_batches + it
                                )
                                tb.add_histogram(
                                    f"res_pred.{name}.grad",
                                    params.grad,
                                    epoch * n_batches + it,
                                )

            if is_train:
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, {d_reg_loss / d_n:.5f}, lr {lr:e}."
                    )
                else:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, lr {lr:e}."
                    )
            else:
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}."
                    )
                else:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}."
                    )

            if not is_train:
                test_loss = float(np.mean(losses))
                #                 logging.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0

        for epoch in range(config.max_epochs):

            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                best_epoch = epoch
                self.save_checkpoint(best_epoch + 1)

            ## SAMPLE AND PLOT
            # ==========================================================================================
            # ==========================================================================================
            raw_model = model.module if hasattr(self.model, "module") else model
            seqs, masks, seqlens, mmsis, time_starts = next(iter(aisdls["test"]))
            n_plots = 7
            init_seqlen = INIT_SEQLEN
            seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
            preds = utils.sample(
                raw_model,
                seqs_init,
                96 - init_seqlen,
                temperature=1.0,
                sample=True,
                sample_mode=self.config.sample_mode,
                r_vicinity=self.config.r_vicinity,
                top_k=self.config.top_k,
            )

            img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')
            plt.figure(figsize=(9, 6), dpi=150)
            cmap = plt.cm.get_cmap("jet")
            preds_np = preds.detach().cpu().numpy()
            inputs_np = seqs.detach().cpu().numpy()
            for idx in range(n_plots):
                c = cmap(float(idx) / (n_plots))
                try:
                    seqlen = seqlens[idx].item()
                except:
                    continue
                plt.plot(
                    inputs_np[idx][:init_seqlen, 1],
                    inputs_np[idx][:init_seqlen, 0],
                    color=c,
                )
                plt.plot(
                    inputs_np[idx][:init_seqlen, 1],
                    inputs_np[idx][:init_seqlen, 0],
                    "o",
                    markersize=3,
                    color=c,
                )
                plt.plot(
                    inputs_np[idx][:seqlen, 1],
                    inputs_np[idx][:seqlen, 0],
                    linestyle="-.",
                    color=c,
                )
                plt.plot(
                    preds_np[idx][init_seqlen:, 1],
                    preds_np[idx][init_seqlen:, 0],
                    "x",
                    markersize=4,
                    color=c,
                )
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.savefig(img_path, dpi=150)
            plt.close()

        # Final state
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(
            f"Last epoch: {epoch:03d}, saving model to {self.config.ckpt_path}"
        )
        save_path = self.config.ckpt_path.replace(
            "model.pt", f"model_{epoch + 1:03d}.pt"
        )
        torch.save(raw_model.state_dict(), save_path)
