"""
Standalone runner for PaD-TS on the MHEALTH dataset.
Mirrors run.py but imports the mhealth config directly,
so no modifications to the original run.py are needed.

Usage:
    python run_mhealth.py
    python run_mhealth.py -w 48        # custom window size
    python run_mhealth.py -s 50000     # custom training steps
"""

import torch
import argparse
import numpy as np
from resample import UniformSampler, Batch_Same_Sampler
from Model import PaD_TS
from diffmodel_init import create_gaussian_diffusion
from training import Trainer
from data_preprocessing.real_dataloader import CustomDataset
from torchsummary import summary
from data_preprocessing.sampling import sampling
from eval_run import (
    discriminative_score,
    predictive_score,
    BMMD_score_naive,
    VDS_score,
)
from configs.mhealth_config import (
    Training_args,
    Model_args,
    Diffusion_args,
    DataLoader_args,
    Data_args,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-window", "-w", default=24, type=int, help="Window Size", required=False
    )
    parser.add_argument(
        "-steps", "-s", default=0, type=int, help="Training Step", required=False
    )
    args = parser.parse_args()

    train_arg = Training_args()
    model_arg = Model_args()
    diff_arg = Diffusion_args()
    dl_arg = DataLoader_args()
    d_arg = Data_args()

    if args.window != 24:
        d_arg.window = int(args.window)
        train_arg.save_dir = f"./OUTPUT/{d_arg.name}_{d_arg.window}_MMD/"
        model_arg.input_shape = (d_arg.window, d_arg.dim)

    if args.steps != 0:
        train_arg.lr_anneal_steps = args.steps

    print("======Load Data======")
    dataset = CustomDataset(
        name=d_arg.name,
        proportion=d_arg.proportion,
        data_root=d_arg.data_root,
        window=d_arg.window,
        save2npy=d_arg.save2npy,
        neg_one_to_one=d_arg.neg_one_to_one,
        seed=d_arg.seed,
        period=d_arg.period,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dl_arg.batch_size,
        shuffle=dl_arg.shuffle,
        num_workers=dl_arg.num_workers,
        drop_last=dl_arg.drop_last,
        pin_memory=dl_arg.pin_memory,
    )

    model = PaD_TS(
        hidden_size=model_arg.hidden_size,
        num_heads=model_arg.num_heads,
        n_encoder=model_arg.n_encoder,
        n_decoder=model_arg.n_decoder,
        feature_last=model_arg.feature_last,
        mlp_ratio=model_arg.mlp_ratio,
        input_shape=model_arg.input_shape,
    )
    diffusion = create_gaussian_diffusion(
        predict_xstart=diff_arg.predict_xstart,
        diffusion_steps=diff_arg.diffusion_steps,
        noise_schedule=diff_arg.noise_schedule,
        loss=diff_arg.loss,
        rescale_timesteps=diff_arg.rescale_timesteps,
    )
    if train_arg.schedule_sampler == "batch":
        schedule_sampler = Batch_Same_Sampler(diffusion)
    elif train_arg.schedule_sampler == "uniform":
        schedule_sampler = UniformSampler(diffusion)
    else:
        raise NotImplementedError(f"Unknown sampler: {train_arg.schedule_sampler}")

    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        data=dataloader,
        batch_size=dl_arg.batch_size,
        lr=train_arg.lr,
        weight_decay=train_arg.weight_decay,
        lr_anneal_steps=train_arg.lr_anneal_steps,
        log_interval=train_arg.log_interval,
        save_interval=train_arg.save_interval,
        save_dir=train_arg.save_dir,
        schedule_sampler=schedule_sampler,
        mmd_alpha=train_arg.mmd_alpha,
    )
    summary(model)
    print("Loss Function: ", diff_arg.loss)
    print("Save Directory: ", train_arg.save_dir)
    print("Schedule Sampler: ", train_arg.schedule_sampler)
    print("Batch Size: ", dl_arg.batch_size)
    print("Diffusion Steps: ", diff_arg.diffusion_steps)
    print("Epochs: ", train_arg.lr_anneal_steps)
    print("Alpha: ", train_arg.mmd_alpha)
    print("Window Size: ", d_arg.window)
    print("Data shape: ", model_arg.input_shape)
    print("Hidden: ", model_arg.hidden_size)

    print("======Training======")
    trainer.train()
    print("======Done======")

    print("======Generate Samples======")
    concatenated_tensor = sampling(
        model,
        diffusion,
        dataset.sample_num,
        dataset.window,
        dataset.var_num,
        dl_arg.batch_size,
    )
    np.save(
        f"{train_arg.save_dir}ddpm_fake_{d_arg.name}_{dataset.window}.npy",
        concatenated_tensor.cpu(),
    )
    print(f"{train_arg.save_dir}ddpm_fake_{d_arg.name}_{dataset.window}.npy")

    print("======Diff Eval======")
    np_fake = np.array(concatenated_tensor.detach().cpu())

    # Note: eval_run.py hardcodes dataset names. For MHEALTH evaluation,
    # we load the ground truth directly instead of relying on eval_run's
    # if/elif chains (which don't include mhealth).
    ori_path = f"./OUTPUT/samples/{d_arg.name}_norm_truth_{d_arg.window}_train.npy"
    ori_data = np.load(ori_path)
    fake_unnorm = (np_fake + 1) * 0.5  # unnormalize from [-1,1] to [0,1]
    fake_unnorm = fake_unnorm[: ori_data.shape[0]]

    print("======Discriminative Score======")
    from eval_utils.discriminative_metric import discriminative_score_metrics
    from eval_utils.metric_utils import display_scores

    disc_scores = []
    for i in range(5):
        temp_disc, fake_acc, real_acc, _ = discriminative_score_metrics(
            ori_data, fake_unnorm
        )
        disc_scores.append(temp_disc)
        print(f"Iter {i}: {temp_disc}, {fake_acc}, {real_acc}")
    print(f"{d_arg.name}:")
    display_scores(disc_scores)

    print("======Predictive Score======")
    from eval_utils.predictive_metric import predictive_score_metrics

    pred_scores = []
    for i in range(5):
        temp_pred = predictive_score_metrics(ori_data, fake_unnorm)
        pred_scores.append(temp_pred)
        print(f"{i} epoch: {temp_pred}")
    print(f"{d_arg.name}:")
    display_scores(pred_scores)

    print("======VDS Score======")
    import torch as th
    from eval_utils.MMD import BMMD_Naive, cross_correlation_distribution, VDS_Naive

    ori_t = th.tensor(ori_data).float()
    fake_t = th.tensor(fake_unnorm).float()
    vds = VDS_Naive(ori_t, fake_t, "rbf").mean()
    print(f"{d_arg.name} VDS Score: {vds}")

    print("======FDDS Score======")
    ori_ccd = cross_correlation_distribution(ori_t).unsqueeze(-1)
    fake_ccd = cross_correlation_distribution(fake_t).unsqueeze(-1)
    fdds = BMMD_Naive(ori_ccd, fake_ccd, "rbf").mean()
    print(f"{d_arg.name} FDDS Score: {fdds}")

    print("======Finished======")
