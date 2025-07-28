import os

from .train import tr_cf


class PathConfig:

    datadir = f"./data/{tr_cf.dataset_name}/"
    trainset_name = f"{tr_cf.dataset_name}_train.pkl"
    validset_name = f"{tr_cf.dataset_name}_valid.pkl"
    testset_name = f"{tr_cf.dataset_name}_test.pkl"

    filename = (
        f"{tr_cf.dataset_name}"
        + f"-{tr_cf.mode}-{tr_cf.sample_mode}-{tr_cf.top_k}-{tr_cf.r_vicinity}"
        + f"-blur-{tr_cf.blur}-{tr_cf.blur_learnable}-{tr_cf.blur_n}-{tr_cf.blur_loss_w}"
        + f"-data_size-{tr_cf.lat_size}-{tr_cf.lon_size}-{tr_cf.sog_size}-{tr_cf.cog_size}"
        + f"-embd_size-{tr_cf.n_lat_embd}-{tr_cf.n_lon_embd}-{tr_cf.n_sog_embd}-{tr_cf.n_cog_embd}"
        + f"-head-{tr_cf.n_head}-{tr_cf.n_layer}"
        + f"-bs-{tr_cf.batch_size}"
        + f"-lr-{tr_cf.learning_rate}"
        + f"-seqlen-{tr_cf.init_seqlen}-{tr_cf.max_seqlen}"
        + f"-lbsmooth-{tr_cf.label_smoothing}"
    )
    savedir = "./results/" + filename + "/"

    ckpt_path = os.path.join(savedir, "model.pt")


path_cf = PathConfig()
