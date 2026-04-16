# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

import time
import numpy as np
import omegaconf
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

from grasp_gen.dataset.dataset import collate
from grasp_gen.models.grasp_gen import GraspGen
from grasp_gen.models.m2t2 import M2T2
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


DEFAULT_CHECKPOINT_REPO_ID = "adithyamurali/GraspGenModels"
DEFAULT_CHECKPOINT_REPO_SUBFOLDER = "checkpoints"
DEFAULT_CHECKPOINT_REPO_REVISION = "main"


def _checkpoint_path(base_dir: Path, checkpoint: str) -> str:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_absolute():
        return str(checkpoint_path)
    return str(base_dir / checkpoint_path)


def _checkpoint_repo_settings(cfg: omegaconf.DictConfig) -> tuple[str, str, str]:
    repo_id = DEFAULT_CHECKPOINT_REPO_ID
    subfolder = DEFAULT_CHECKPOINT_REPO_SUBFOLDER
    revision = DEFAULT_CHECKPOINT_REPO_REVISION

    if "checkpoint_repository" in cfg:
        if "repo_id" in cfg.checkpoint_repository:
            repo_id = cfg.checkpoint_repository.repo_id
        if "subfolder" in cfg.checkpoint_repository:
            subfolder = cfg.checkpoint_repository.subfolder
        if "revision" in cfg.checkpoint_repository:
            revision = cfg.checkpoint_repository.revision

    if "GRASPGEN_CHECKPOINT_REPO_ID" in os.environ:
        repo_id = os.environ["GRASPGEN_CHECKPOINT_REPO_ID"]
    if "GRASPGEN_CHECKPOINT_REPO_SUBFOLDER" in os.environ:
        subfolder = os.environ["GRASPGEN_CHECKPOINT_REPO_SUBFOLDER"]
    if "GRASPGEN_CHECKPOINT_REPO_REVISION" in os.environ:
        revision = os.environ["GRASPGEN_CHECKPOINT_REPO_REVISION"]

    subfolder = subfolder.strip("/")

    return repo_id, subfolder, revision


def _download_checkpoint_if_missing(
    checkpoint: str,
    cfg: omegaconf.DictConfig,
) -> str:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return str(checkpoint_path)

    repo_id, subfolder, revision = _checkpoint_repo_settings(cfg)
    checkpoint_name = checkpoint_path.name
    remote_path = checkpoint_name
    if subfolder != "":
        remote_path = f"{subfolder}/{checkpoint_name}"

    repo_files = set(list_repo_files(repo_id=repo_id, revision=revision))

    if remote_path in repo_files:
        if subfolder == "":
            remote_subfolder = None
        else:
            remote_subfolder = subfolder
    elif checkpoint_name in repo_files:
        remote_subfolder = None
        remote_path = checkpoint_name
    else:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint} does not exist locally and was not found in "
            f"HuggingFace repo {repo_id} (looked for {remote_path})."
        )

    logger.info(
        "Downloading missing checkpoint %s from %s (%s)",
        checkpoint_name,
        repo_id,
        remote_path,
    )

    return hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_name,
        subfolder=remote_subfolder,
        revision=revision,
    )


def _normalize_eval_section(cfg: omegaconf.DictConfig) -> None:
    if "eval" not in cfg:
        cfg.eval = omegaconf.OmegaConf.create({})

    if "model_name" not in cfg.eval:
        if "model_name" in cfg:
            cfg.eval.model_name = cfg.model_name
        elif "diffusion" in cfg and "discriminator" in cfg:
            cfg.eval.model_name = "diffusion-discriminator"
        elif "m2t2" in cfg:
            cfg.eval.model_name = "m2t2"

    if "checkpoint" not in cfg.eval:
        if "checkpoint" in cfg:
            cfg.eval.checkpoint = cfg.checkpoint
        elif "generator_checkpoint" in cfg:
            cfg.eval.checkpoint = cfg.generator_checkpoint

    if "discriminator" not in cfg and "discriminator_checkpoint" in cfg:
        cfg.discriminator = omegaconf.OmegaConf.create({})

    if (
        "discriminator" in cfg
        and "checkpoint" not in cfg.discriminator
        and "discriminator_checkpoint" in cfg
    ):
        cfg.discriminator.checkpoint = cfg.discriminator_checkpoint


def _validate_eval_section(cfg: omegaconf.DictConfig, gripper_config: str) -> None:
    if "eval" not in cfg:
        raise KeyError(f"Missing key eval in config: {gripper_config}")

    if "model_name" not in cfg.eval:
        raise KeyError(
            "Missing key eval.model_name in config. "
            "Set eval.model_name or top-level model_name."
        )

    if "checkpoint" not in cfg.eval:
        raise KeyError(
            "Missing key eval.checkpoint in config. "
            "Set eval.checkpoint, top-level checkpoint, or generator_checkpoint."
        )

    if cfg.eval.model_name == "diffusion-discriminator":
        if "discriminator" not in cfg or "checkpoint" not in cfg.discriminator:
            raise KeyError(
                "Missing discriminator.checkpoint in config for "
                "model_name=diffusion-discriminator. "
                "Set discriminator.checkpoint or top-level discriminator_checkpoint."
            )


def load_grasp_cfg(gripper_config: str) -> omegaconf.DictConfig:
    """
    Loads the grasp configuration file and updates the checkpoint paths to be relative to the gripper config file.

    Assumes that the checkpoint paths are in the same directory as the gripper config file.

    Args:
        gripper_config: Path to the gripper configuration file
    Returns:
        grasp_cfg: Hydra config object with updated checkpoint paths
    """
    cfg = omegaconf.OmegaConf.load(gripper_config)
    _normalize_eval_section(cfg)
    _validate_eval_section(cfg, gripper_config)

    ckpt_root_dir = Path(gripper_config).parent
    cfg.eval.checkpoint = _checkpoint_path(ckpt_root_dir, cfg.eval.checkpoint)
    cfg.eval.checkpoint = _download_checkpoint_if_missing(cfg.eval.checkpoint, cfg)

    if cfg.eval.model_name == "diffusion-discriminator":
        cfg.discriminator.checkpoint = _checkpoint_path(
            ckpt_root_dir, cfg.discriminator.checkpoint
        )
        cfg.discriminator.checkpoint = _download_checkpoint_if_missing(
            cfg.discriminator.checkpoint, cfg
        )

    if (
        "data" in cfg
        and "diffusion" in cfg
        and "discriminator" in cfg
        and "gripper_name" in cfg.data
        and "gripper_name" in cfg.diffusion
        and "gripper_name" in cfg.discriminator
    ):
        assert (
            cfg.data.gripper_name
            == cfg.diffusion.gripper_name
            == cfg.discriminator.gripper_name
        )

    return cfg


class GraspGenSampler:
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
    ):
        """
        Args:
            cfg: Hydra config object
        """

        self.cfg = cfg
        # Initialize model and load checkpoint
        if cfg.eval.model_name == "m2t2":
            model = M2T2.from_config(cfg.m2t2)
            ckpt = torch.load(cfg.eval.checkpoint)
            model.load_state_dict(ckpt["model"])
        elif cfg.eval.model_name == "diffusion-discriminator":
            model = GraspGen.from_config(
                cfg.diffusion,
                cfg.discriminator,
            )
            if not os.path.exists(cfg.eval.checkpoint):
                raise FileNotFoundError(
                    f"Checkpoint {cfg.eval.checkpoint} does not exist"
                )
            if not os.path.exists(cfg.discriminator.checkpoint):
                raise FileNotFoundError(
                    f"Checkpoint {cfg.discriminator.checkpoint} does not exist"
                )

            model.load_state_dict(cfg.eval.checkpoint, cfg.discriminator.checkpoint)
            model.eval()
        else:
            raise NotImplementedError(
                f"Model name not implemented {cfg.eval.model_name}"
            )

        self.model = model.cuda().eval()

    @staticmethod
    def run_inference(
        object_pc: np.ndarray | torch.Tensor,
        grasp_sampler: "GraspGenSampler",
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = -1,
        min_grasps: int = 40,
        max_tries: int = 6,
        remove_outliers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run grasp generation inference on a point cloud.

        Args:
            object_pc: Point cloud to generate grasps for
            grasp_sampler: Initialized GraspGenSampler instance
            grasp_threshold: Threshold for valid grasps. If -1.0, then the top topk_grasps grasps will be ranked and returned
            num_grasps: Number of grasps to generate
            topk_grasps: Maximum number of grasps to return
            min_grasps: Minimum number of grasps required. If fewer grasps are found, inference will be retried
            max_tries: Maximum number of inference attempts to make before returning results

        Returns:
            grasps: Generated grasp poses
            grasp_conf: Confidence scores for the grasps
        """
        if type(object_pc) == np.ndarray:
            object_pc = torch.from_numpy(object_pc).cuda().float()

        if grasp_threshold == -1.0 and topk_num_grasps == -1:
            topk_num_grasps = 100

        all_grasps = []
        all_conf = []
        num_tries = 0

        while sum(len(g) for g in all_grasps) < min_grasps and num_tries < max_tries:
            num_tries += 1
            t0 = time.time()
            output = grasp_sampler.sample(
                object_pc,
                threshold=grasp_threshold,
                num_grasps=num_grasps,
                remove_outliers=remove_outliers,
            )
            grasp_conf = output[1]
            grasps = output[0]

            # Sort and prune grasps within this iteration
            if topk_num_grasps != -1 and len(grasps) > 0:
                grasp_conf, grasps = zip(
                    *sorted(zip(grasp_conf, grasps), key=lambda x: x[0], reverse=True)
                )
                grasps = torch.stack(grasps)
                grasp_conf = torch.stack(grasp_conf)
                grasps = grasps[:topk_num_grasps]
                grasp_conf = grasp_conf[:topk_num_grasps]

            all_grasps.append(grasps)
            all_conf.append(grasp_conf)

            logger.info(
                f"Found {len(grasps)} grasps in iteration {len(all_grasps)}, total grasps: {sum(len(g) for g in all_grasps)}"
            )
            t1 = time.time()
            logger.info(f"Time taken for inference: {t1 - t0} seconds")

        if len(all_grasps) == 0:
            return torch.tensor([]), torch.tensor([])

        # Concatenate all grasps and confidences
        grasps = torch.cat(all_grasps, dim=0)
        grasp_conf = torch.cat(all_conf, dim=0)
        grasps[:, 3, 3] = 1  # TODO: Fix this in grasp_gen.py later.

        return grasps, grasp_conf

    @torch.inference_mode()
    def sample(
        self,
        obj_pcd: np.ndarray,
        threshold: float = -1.0,
        num_grasps: int = 200,
        remove_outliers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obj_pcd: np.array of shape (N, 3)
            obj_pts_color (Optional): np.array of shape (N, 4)

        Returns:
            grasps: torch.tensor of shape (M, 6)
            grasp_conf: torch.tensor of shape (M,)
            grasp_contacts: torch.tensor of shape (M, 3)
        """

        if remove_outliers:
            obj_pcd, _ = point_cloud_outlier_removal(obj_pcd)

        obj_pcd_center = obj_pcd.mean(axis=0)
        obj_pts_color = torch.zeros_like(obj_pcd)
        obj_mean_points = obj_pcd - obj_pcd_center[None]

        data = {}
        data["task"] = "pick"
        data["inputs"] = torch.cat(
            [obj_mean_points, obj_pts_color[:, :3].squeeze(1)], dim=-1
        ).float()
        data["points"] = obj_mean_points

        data_batch = collate([data])
        grasp_key = "grasps"
        with torch.inference_mode():
            if self.cfg.eval.model_name == "m2t2":
                model_outputs = self.model.infer(data_batch, self.cfg.eval)
            elif self.cfg.eval.model_name == "diffusion-discriminator":
                grasp_key = "grasps_pred"
                self.model.grasp_generator.num_grasps_per_object = num_grasps

                model_outputs, _, _ = self.model.infer(data_batch)

            else:
                raise NotImplementedError(f"Invalid model {self.cfg.eval.model_name}!")

        if len(model_outputs[grasp_key][0]) == 0:
            return [], [], []

        grasps = model_outputs[grasp_key][0]

        if self.cfg.eval.model_name == "diffusion-discriminator":
            grasp_conf = model_outputs["grasp_confidence"][0][:, 0]
            logger.info(
                f"Confidences min: {grasp_conf.min():.5f}, max: {grasp_conf.max():.5f}"
            )
            mask_best_grasps = grasp_conf >= threshold
            logger.info(
                f"Thresholding grasps @ {threshold}. Only {mask_best_grasps.sum()}/{mask_best_grasps.shape[0]} grasps remaining"
            )

            grasps = grasps[mask_best_grasps]
            grasp_conf = grasp_conf[mask_best_grasps]

        elif self.cfg.eval.model_name == "m2t2":
            grasps = grasps[0]
            grasp_conf = model_outputs["grasp_confidence"][0][0]
        else:
            raise NotImplementedError(f"Invalid model {self.cfg.eval.model_name}!")
        grasps[:, :3, 3] += obj_pcd_center
        return grasps, grasp_conf, None
