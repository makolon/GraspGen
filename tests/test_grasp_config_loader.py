from pathlib import Path

import pytest

import grasp_gen.grasp_server as grasp_server
from grasp_gen.grasp_server import load_grasp_cfg


def _write_yaml(tmp_path: Path, file_name: str, contents: str) -> Path:
    config_path = tmp_path / file_name
    config_path.write_text(contents)
    return config_path


def test_load_grasp_cfg_builds_eval_from_flat_keys(tmp_path):
    (tmp_path / "gen.pth").write_text("gen")
    (tmp_path / "dis.pth").write_text("dis")

    config_path = _write_yaml(
        tmp_path,
        "flat_config.yaml",
        """
data:
  gripper_name: franka_panda
diffusion:
  gripper_name: franka_panda
discriminator:
  gripper_name: franka_panda
model_name: diffusion-discriminator
checkpoint: gen.pth
discriminator_checkpoint: dis.pth
""",
    )

    cfg = load_grasp_cfg(str(config_path))

    assert cfg.eval.model_name == "diffusion-discriminator"
    assert cfg.eval.checkpoint == str(tmp_path / "gen.pth")
    assert cfg.discriminator.checkpoint == str(tmp_path / "dis.pth")


def test_load_grasp_cfg_infers_model_name_from_sections(tmp_path):
    (tmp_path / "gen.pth").write_text("gen")
    (tmp_path / "dis.pth").write_text("dis")

    config_path = _write_yaml(
        tmp_path,
        "inferred_model.yaml",
        """
data:
  gripper_name: franka_panda
diffusion:
  gripper_name: franka_panda
discriminator:
  gripper_name: franka_panda
  checkpoint: dis.pth
checkpoint: gen.pth
""",
    )

    cfg = load_grasp_cfg(str(config_path))

    assert cfg.eval.model_name == "diffusion-discriminator"
    assert cfg.eval.checkpoint == str(tmp_path / "gen.pth")
    assert cfg.discriminator.checkpoint == str(tmp_path / "dis.pth")


def test_load_grasp_cfg_errors_when_checkpoint_is_missing(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        "missing_checkpoint.yaml",
        """
data:
  gripper_name: franka_panda
diffusion:
  gripper_name: franka_panda
discriminator:
  gripper_name: franka_panda
  checkpoint: dis.pth
model_name: diffusion-discriminator
""",
    )

    with pytest.raises(KeyError, match="eval.checkpoint"):
        load_grasp_cfg(str(config_path))


def test_load_grasp_cfg_downloads_missing_checkpoint(tmp_path, monkeypatch):
    config_path = _write_yaml(
        tmp_path,
        "download_checkpoint.yaml",
        """
model_name: m2t2
checkpoint: gen.pth
m2t2: {}
""",
    )

    downloaded_checkpoint = tmp_path / "cache_gen.pth"
    downloaded_checkpoint.write_text("gen")

    def fake_list_repo_files(repo_id, revision):
        assert repo_id == "adithyamurali/GraspGenModels"
        assert revision == "main"
        return ["checkpoints/gen.pth"]

    def fake_hf_hub_download(repo_id, filename, subfolder, revision):
        assert repo_id == "adithyamurali/GraspGenModels"
        assert filename == "gen.pth"
        assert subfolder == "checkpoints"
        assert revision == "main"
        return str(downloaded_checkpoint)

    monkeypatch.setattr(grasp_server, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(grasp_server, "hf_hub_download", fake_hf_hub_download)

    cfg = load_grasp_cfg(str(config_path))
    assert cfg.eval.checkpoint == str(downloaded_checkpoint)


def test_load_grasp_cfg_raises_when_remote_checkpoint_is_missing(
    tmp_path, monkeypatch
):
    config_path = _write_yaml(
        tmp_path,
        "missing_remote_checkpoint.yaml",
        """
model_name: m2t2
checkpoint: gen.pth
m2t2: {}
""",
    )

    def fake_list_repo_files(repo_id, revision):
        assert repo_id == "adithyamurali/GraspGenModels"
        assert revision == "main"
        return []

    monkeypatch.setattr(grasp_server, "list_repo_files", fake_list_repo_files)

    with pytest.raises(FileNotFoundError, match="HuggingFace repo"):
        load_grasp_cfg(str(config_path))