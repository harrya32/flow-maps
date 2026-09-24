import os
import time
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE
from torchvision import transforms

import lpips
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.metrics.FID import FID


from mfm.networks.utils import flow_model_torch_wrapper
from mfm.utils import wasserstein_distance, plot_arch, plot_lidar, plot_sphere
from mfm.flow_matchers.ema import EMA
from mfm.flow_matchers.eval_utils import FIDImageDataset
from mfm.flow_matchers.maizels_eval import rbf_mmd2


class FlowNetTrainBase(pl.LightningModule):
    def __init__(
        self,
        flow_matcher,
        flow_net,
        skipped_time_points=None,
        ot_sampler=None,
        args=None,
    ):
        super().__init__()
        self.flow_matcher = flow_matcher
        self.flow_net = flow_net
        self.ot_sampler = ot_sampler
        self.skipped_time_points = skipped_time_points

        self.optimizer_name = args.flow_optimizer
        self.lr = args.flow_lr
        self.weight_decay = args.flow_weight_decay
        self.whiten = args.whiten
        self.working_dir = args.working_dir
        self.is_maizels = args.data_type == "maizels"
        self.seed_current = int(getattr(args, "seed_current", 0))
        self._step_start_time = None

    def forward(self, t, xt):
        return self.flow_net(t, xt)

    def _configured_timesteps(self, batch_length):
        configured = getattr(self.trainer.datamodule, "times", None)
        if configured is not None and len(configured) == batch_length:
            return torch.as_tensor(configured, dtype=torch.float32).tolist()
        return torch.linspace(0.0, 1.0, batch_length).tolist()

    def _compute_loss(self, main_batch):
        main_batch_filtered = [
            x for i, x in enumerate(main_batch) if i not in self.skipped_time_points
        ]

        x0s, x1s = main_batch_filtered[:-1], main_batch_filtered[1:]
        ts, xts, uts = self._process_flow(x0s, x1s)

        # The learned geopath defines fixed conditional-flow targets. Detaching
        # avoids optimizing it again and avoids higher-order JVP backward during
        # velocity-field training (unsupported by Apple MPS).
        t = torch.cat(ts).detach()
        xt = torch.cat(xts).detach()
        ut = torch.cat(uts).detach()
        vt = self(t[:, None], xt)

        loss = mean_squared_error(vt, ut)

        return loss

    def _process_flow(self, x0s, x1s):
        ts, xts, uts = [], [], []
        t_start = self.timesteps[0]

        for i, (x0, x1) in enumerate(zip(x0s, x1s)):
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)

            if self.ot_sampler is not None:
                x0, x1 = self.ot_sampler.sample_plan(
                    x0,
                    x1,
                    replace=True,
                )
            if self.skipped_time_points and i + 1 >= self.skipped_time_points[0]:
                t_start_next = self.timesteps[i + 2]
            else:
                t_start_next = self.timesteps[i + 1]

            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, t_start, t_start_next
            )

            ts.append(t)

            xts.append(xt)
            uts.append(ut)
            t_start = t_start_next
        return ts, xts, uts

    def training_step(self, batch, batch_idx):
        main_batch = batch["train_samples"][0]
        self.timesteps = self._configured_timesteps(len(main_batch))
        loss = self._compute_loss(main_batch)
        if self.flow_matcher.alpha != 0:
            self.log(
                "FlowNet/mean_geopath_cfm",
                (self.flow_matcher.geopath_net_output.detach().abs().mean()),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.log(
            "FlowNet/train_loss_cfm",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if self.is_maizels:
            self.log("loss", loss, on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        main_batch = batch["val_samples"][0]

        self.timesteps = self._configured_timesteps(len(main_batch))
        val_loss = self._compute_loss(main_batch)
        self.log(
            "FlowNet/val_loss_cfm",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if self.is_maizels:
            self.log(
                "validation_loss",
                val_loss,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
        return val_loss

    def on_train_batch_start(self, batch, batch_idx):
        if self.is_maizels:
            self._step_start_time = time.perf_counter()

    def on_fit_start(self):
        if self.flow_matcher.alpha == 0:
            return
        geopath_net = self.flow_matcher.geopath_net.to(self.device)
        geopath_net.eval()
        geopath_net.requires_grad_(False)
        self.flow_matcher.geopath_net = geopath_net

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.is_maizels:
            return
        optimizer = self.optimizers(use_pl_optimizer=False)
        self.log(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        if self._step_start_time is not None:
            self.log(
                "step_time",
                time.perf_counter() - self._step_start_time,
                on_step=True,
                on_epoch=False,
                logger=True,
            )

    def on_before_optimizer_step(self, optimizer):
        if not self.is_maizels:
            return
        grad_norm_sq = torch.zeros((), device=self.device)
        for parameter in self.flow_net.parameters():
            if parameter.grad is not None:
                grad_norm_sq = grad_norm_sq + parameter.grad.detach().pow(2).sum()
        self.log(
            "grad",
            torch.sqrt(grad_norm_sq),
            on_step=True,
            on_epoch=False,
            logger=True,
        )

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.flow_net, EMA):
            self.flow_net.update_ema()

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
            )

        return optimizer


class FlowNetTrainTrajectory(FlowNetTrainBase):
    evaluation_sampler = "euler"

    def evaluation_trajectory(
        self,
        x0,
        *,
        start_time: float,
        end_time: float,
        n_steps: int,
        seed: int | None = None,
    ):
        del seed
        node = NeuralODE(
            flow_model_torch_wrapper(self.flow_net),
            solver="euler",
            sensitivity="adjoint",
            atol=1e-5,
            rtol=1e-5,
        )
        return node.trajectory(
            x0,
            t_span=torch.linspace(
                float(start_time),
                float(end_time),
                max(1, int(n_steps)) + 1,
                device=x0.device,
                dtype=x0.dtype,
            ),
        )

    def test_step(self, batch, batch_idx):
        self.timesteps = self._configured_timesteps(len(batch))
        data_type = self.trainer.datamodule.data_type

        t_exclude = self.skipped_time_points[0] if self.skipped_time_points else None
        if t_exclude is not None:
            traj = self.evaluation_trajectory(
                batch[t_exclude - 1],
                start_time=self.timesteps[t_exclude - 1],
                end_time=self.timesteps[t_exclude],
                n_steps=100,
                seed=self.seed_current + 2901,
            )
            X_mid_pred = traj[-1]
            traj = self.evaluation_trajectory(
                batch[t_exclude - 1],
                start_time=self.timesteps[t_exclude - 1],
                end_time=self.timesteps[t_exclude + 1],
                n_steps=100,
                seed=self.seed_current + 2902,
            )
            if data_type == "arch":
                plot_arch(
                    batch,
                    traj,
                    time_steps=[t_exclude - 1, t_exclude + 1],
                    n_samples=400,
                    fname=os.path.join(os.getcwd(), f"arch_trajs.png"),
                )
            elif data_type == "sphere":
                mean_distance_from_sphere = torch.abs(
                    torch.sqrt((X_mid_pred**2).sum(dim=1)) - 1
                ).mean()
                self.log(
                    "mean_distance_from_sphere",
                    mean_distance_from_sphere,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
                plot_sphere(
                    batch,
                    traj,
                    time_steps=[t_exclude - 1, t_exclude + 1],
                    n_samples=100,
                    fname=os.path.join(os.getcwd(), f"sphere_trajs.png"),
                )

            EMD = wasserstein_distance(X_mid_pred, batch[t_exclude], power=1)
            self.final_EMD = EMD

            self.log("test_EMD", EMD, on_step=False, on_epoch=True, prog_bar=True)
            if data_type == "scrna" and self.trainer.datamodule.data_name in (
                "cite",
                "multi",
            ):
                mmd2 = rbf_mmd2(
                    X_mid_pred.detach().cpu().numpy(),
                    batch[t_exclude].detach().cpu().numpy(),
                    np.random.default_rng(self.seed_current + 2901),
                )
                self.final_MMD2 = mmd2
                timepoints = list(self.trainer.datamodule.timepoint_splits)
                heldout_tag = str(timepoints[t_exclude]).replace(".", "p")
                # Keep the original MFM name while also exposing the shared
                # cross-method names used by this repository's CITE/Multi runs.
                self.log(
                    "mfm/test_EMD", EMD, on_step=False, on_epoch=True, prog_bar=False
                )
                self.log(
                    "final_eval/euler_mean_emd",
                    EMD,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "mfm/test_rbf_MMD2",
                    mmd2,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"distribution_eval/{heldout_tag}_euler_rbf_mmd2",
                    mmd2,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "distribution_eval/euler_rbf_mmd2_mean",
                    mmd2,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "final_eval/euler_mean_rbf_mmd2",
                    mmd2,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

                if self.evaluation_sampler != "euler":
                    sampler = self.evaluation_sampler
                    self.log(
                        f"distribution_eval/{heldout_tag}_{sampler}_emd",
                        EMD,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                    self.log(
                        f"distribution_eval/{heldout_tag}_{sampler}_rbf_mmd2",
                        mmd2,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                    self.log(
                        f"sf2m/test_EMD_{sampler}",
                        EMD,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                    self.log(
                        f"sf2m/test_rbf_MMD2_{sampler}",
                        mmd2,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                    self.log(
                        f"final_eval/{sampler}_mean_emd",
                        EMD,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                    self.log(
                        f"final_eval/{sampler}_mean_rbf_mmd2",
                        mmd2,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )


class SF2MFlowNetTrainTrajectory(FlowNetTrainTrajectory):
    """Train SF2M velocity and score fields and evaluate its stochastic SDE."""

    evaluation_sampler = "euler_maruyama"

    def __init__(self, *args, score_net, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_net = score_net
        config = kwargs["args"]
        self.sf2m_sigma = float(config.sf2m_sigma)
        self.sf2m_score_weight = float(config.sf2m_score_weight)

    def _compute_sf2m_loss(self, main_batch):
        retained_indices = [
            index
            for index in range(len(main_batch))
            if index not in self.skipped_time_points
        ]
        retained = [main_batch[index] for index in retained_indices]
        ts, xts, uts, score_scales, noises = [], [], [], [], []
        for pair_index, (x0, x1) in enumerate(zip(retained[:-1], retained[1:])):
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)
            t_start = self.timesteps[retained_indices[pair_index]]
            t_end = self.timesteps[retained_indices[pair_index + 1]]
            t, xt, ut, score_scale, noise = (
                self.flow_matcher.sample_location_flow_and_score(
                    x0,
                    x1,
                    t_start,
                    t_end,
                )
            )
            ts.append(t)
            xts.append(xt)
            uts.append(ut)
            score_scales.append(score_scale)
            noises.append(noise)

        t = torch.cat(ts).detach()
        xt = torch.cat(xts).detach()
        ut = torch.cat(uts).detach()
        score_scale = torch.cat(score_scales).detach()
        noise = torch.cat(noises).detach()
        velocity = self(t[:, None], xt)
        score = self.score_net(t[:, None], xt)
        velocity_loss = mean_squared_error(velocity, ut)
        # Denoising score matching written in its endpoint-stable form:
        # sigma_t * score(xt, t) = -epsilon.
        score_loss = torch.mean((score_scale * score + noise) ** 2)
        total = velocity_loss + self.sf2m_score_weight * score_loss
        return total, velocity_loss, score_loss

    def _shared_sf2m_step(self, batch, *, stage: str):
        main_batch = batch[f"{stage}_samples"][0]
        self.timesteps = self._configured_timesteps(len(main_batch))
        total, velocity_loss, score_loss = self._compute_sf2m_loss(main_batch)
        log_stage = "train" if stage == "train" else "val"
        self.log(
            f"FlowNet/{log_stage}_loss_cfm",
            total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"SF2M/{log_stage}_velocity_loss",
            velocity_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            f"SF2M/{log_stage}_score_loss",
            score_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            f"SF2M/{log_stage}_loss",
            total,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        if self.is_maizels:
            name = "loss" if stage == "train" else "validation_loss"
            self.log(
                name,
                total,
                on_step=stage == "train",
                on_epoch=stage != "train",
                logger=True,
            )
        return total

    def training_step(self, batch, batch_idx):
        del batch_idx
        return self._shared_sf2m_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        del batch_idx
        return self._shared_sf2m_step(batch, stage="val")

    @torch.no_grad()
    def evaluation_trajectory(
        self,
        x0,
        *,
        start_time: float,
        end_time: float,
        n_steps: int,
        seed: int | None = None,
    ):
        n_steps = max(1, int(n_steps))
        start_time = float(start_time)
        end_time = float(end_time)
        if end_time < start_time:
            raise ValueError("SF2M rollout end time precedes its start time.")
        dt = (end_time - start_time) / n_steps
        state = x0
        trajectory = [state]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed_current if seed is None else int(seed))
        for step in range(n_steps):
            t = torch.full(
                (state.shape[0], 1),
                start_time + step * dt,
                dtype=state.dtype,
                device=state.device,
            )
            velocity = self.flow_net(t, state)
            score = self.score_net(t, state)
            drift = velocity + 0.5 * self.sf2m_sigma**2 * score
            noise = torch.randn(
                state.shape,
                generator=generator,
                dtype=state.dtype,
                device="cpu",
            ).to(state.device)
            state = state + dt * drift + self.sf2m_sigma * np.sqrt(dt) * noise
            trajectory.append(state)
        return torch.stack(trajectory, dim=0)

    def on_before_optimizer_step(self, optimizer):
        if not self.is_maizels:
            return
        grad_norm_sq = torch.zeros((), device=self.device)
        for parameter in self.parameters():
            if parameter.grad is not None:
                grad_norm_sq = grad_norm_sq + parameter.grad.detach().pow(2).sum()
        self.log(
            "grad",
            torch.sqrt(grad_norm_sq),
            on_step=True,
            on_epoch=False,
            logger=True,
        )

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.score_net, EMA):
            self.score_net.update_ema()


class FlowNetTrainLidar(FlowNetTrainBase):
    def test_step(self, batch, batch_idx):
        x0, cloud_points = batch
        node = NeuralODE(
            flow_model_torch_wrapper(self.flow_net),
            solver="euler",
            sensitivity="adjoint",
        )
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, 101),
            ).cpu()

        if self.whiten:
            traj_shape = traj.shape
            traj = traj.reshape(-1, 3)
            traj = self.trainer.datamodule.scaler.inverse_transform(
                traj.detach().numpy()
            ).reshape(traj_shape)
            cloud_points = torch.tensor(
                self.trainer.datamodule.scaler.inverse_transform(
                    cloud_points.detach().numpy()
                )
            )
        traj = torch.transpose(torch.tensor(traj), 0, 1)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.view_init(elev=30, azim=-115, roll=0)

        plot_lidar(ax, cloud_points, xs=traj)
        plt.savefig(os.path.join(os.getcwd(), f"lidar.png"), dpi=300)
        wandb.log({"lidar": wandb.Image(plt)})


class FlowNetTrainImage(FlowNetTrainBase):
    def on_test_start(self):
        self.node = NeuralODE(
            flow_model_torch_wrapper(self.flow_net).to(self.device),
            solver="tsit5",
            sensitivity="adjoint",
            atol=1e-5,
            rtol=1e-5,
        )
        self.all_outputs = []
        self.vae = self.trainer.datamodule.vae.to(self.device)
        self.postprocess = self.trainer.datamodule.process.postprocess
        image_size = self.trainer.datamodule.image_size
        self.ambient_x0 = transforms.Resize((image_size, image_size))(
            self.trainer.datamodule.ambient_x0
        )
        self.ambient_x1 = self.trainer.datamodule.ambient_x1

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            traj = self.node.trajectory(
                batch.to(self.device),
                t_span=torch.linspace(0, 1, 100).to(self.device),
            )
        traj = traj.transpose(0, 1)
        traj = traj.reshape(*traj.shape[0:2], *self.trainer.datamodule.dim)
        output = traj[:, -1]
        output = self.vae.decode(output).sample.cpu().detach()
        self.all_outputs.append(output)

    def on_test_epoch_end(self):
        all_outputs = torch.cat(self.all_outputs, dim=0).to(self.device)
        fid = self.compute_fid(
            FIDImageDataset(self.ambient_x1, self.postprocess),
            FIDImageDataset(all_outputs, self.postprocess),
        )
        lpips = self.compute_lpips(self.ambient_x0, all_outputs)
        self.log("FID", fid, on_step=False, on_epoch=True, prog_bar=True)
        self.log("LPIPS", lpips, on_step=False, on_epoch=True, prog_bar=True)

    def compute_lpips(self, x0_data, gen_data):
        loss_fn = lpips.LPIPS(net="vgg").to(x0_data.device)
        return loss_fn(x0_data, gen_data).mean().item()

    def compute_fid(self, val_data, gen_data):
        feature_extractor = InceptionFeatureExtractor()
        val_feat = feature_extractor.get_features(val_data)
        gen_feat = feature_extractor.get_features(gen_data)
        return FID().compute_metric(val_feat, None, gen_feat)
