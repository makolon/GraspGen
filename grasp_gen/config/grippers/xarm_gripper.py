from pathlib import Path

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra

from grasp_gen.robot import load_control_points_core, load_default_gripper_config


OPEN_CONFIGURATION = 0.0
BASE_TO_TCP = 0.172
WIDTH = 0.0889236


def translation_matrix(offset):
    transform = np.eye(4)
    transform[:3, 3] = offset
    return transform


def rotation_x_matrix(angle):
    return tra.rotation_matrix(angle, [1, 0, 0])


class GripperModel(object):
    def __init__(self, data_root_dir=None, joint_angle=OPEN_CONFIGURATION):
        if data_root_dir is None:
            data_root_dir = "{}/assets/xarm/meshes".format(
                Path(__file__).parent.parent.parent
            )

        self.joint_angle = joint_angle
        self.mesh = self._load_mesh(data_root_dir)

    def _load_mesh(self, data_root_dir):
        meshes = {}
        for name in [
            "base_link",
            "left_outer_knuckle",
            "left_finger",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_finger",
            "right_inner_knuckle",
        ]:
            meshes[name] = trimesh.load(
                "{}/{}.stl".format(data_root_dir, name), force="mesh"
            )

        joint_angle = self.joint_angle
        parts = [meshes["base_link"]]

        left_outer = meshes["left_outer_knuckle"].copy()
        left_outer.apply_transform(
            translation_matrix([0.0, 0.035, 0.059098]).dot(
                rotation_x_matrix(joint_angle)
            )
        )
        parts.append(left_outer)

        left_finger = meshes["left_finger"].copy()
        left_finger.apply_transform(
            translation_matrix([0.0, 0.035, 0.059098])
            .dot(rotation_x_matrix(joint_angle))
            .dot(translation_matrix([0.0, 0.035465, 0.042039]))
            .dot(rotation_x_matrix(-joint_angle))
        )
        parts.append(left_finger)

        left_inner = meshes["left_inner_knuckle"].copy()
        left_inner.apply_transform(
            translation_matrix([0.0, 0.02, 0.074098]).dot(
                rotation_x_matrix(joint_angle)
            )
        )
        parts.append(left_inner)

        right_outer = meshes["right_outer_knuckle"].copy()
        right_outer.apply_transform(
            translation_matrix([0.0, -0.035, 0.059098]).dot(
                rotation_x_matrix(-joint_angle)
            )
        )
        parts.append(right_outer)

        right_finger = meshes["right_finger"].copy()
        right_finger.apply_transform(
            translation_matrix([0.0, -0.035, 0.059098])
            .dot(rotation_x_matrix(-joint_angle))
            .dot(translation_matrix([0.0, -0.035465, 0.042039]))
            .dot(rotation_x_matrix(joint_angle))
        )
        parts.append(right_finger)

        right_inner = meshes["right_inner_knuckle"].copy()
        right_inner.apply_transform(
            translation_matrix([0.0, -0.02, 0.074098]).dot(
                rotation_x_matrix(-joint_angle)
            )
        )
        parts.append(right_inner)

        return trimesh.util.concatenate(parts)

    def get_gripper_collision_mesh(self):
        return self.mesh

    def get_gripper_visual_mesh(self):
        return self.mesh


def get_gripper_offset_bins():
    offset_bins = np.linspace(0.0, WIDTH, 11).tolist()
    offset_bin_weights = [
        0.16652107,
        0.21488856,
        0.37031708,
        0.55618503,
        0.75124664,
        0.93943357,
        1.07824539,
        1.19423112,
        1.55731375,
        3.17161779,
    ]
    return offset_bins, offset_bin_weights


def load_control_points() -> torch.Tensor:
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = load_control_points_core(gripper_config)
    control_points = np.vstack([control_points, np.zeros(3)])
    control_points = np.hstack([control_points, np.ones([len(control_points), 1])])
    control_points = torch.from_numpy(control_points).float()
    return control_points.T


def load_control_points_for_visualization():
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = load_control_points_core(gripper_config)

    mid_point = (control_points[0] + control_points[1]) / 2

    control_points = [
        control_points[-2],
        control_points[0],
        mid_point,
        [0, 0, 0],
        mid_point,
        control_points[1],
        control_points[-1],
    ]
    return [control_points]


def get_transform_from_base_link_to_tool_tcp():
    return tra.translation_matrix([0.0, 0.0, BASE_TO_TCP])
