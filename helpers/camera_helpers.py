import json
from typing import TypedDict

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from omni.isaac.sensor import Camera
from pytorch3d import transforms

from helpers.helper import Helper
from utils.utils_3d import matrix_to_wxyz, pR_to_transf, pr_to_transf


class CameraFrameData(TypedDict):
    rgb: np.ndarray
    depth: np.ndarray
    bb: dict
    rendering_time: float


def process_frame_data_rgb(data) -> CameraFrameData:
    res: CameraFrameData = {"rendering_time": data["rendering_time"]}
    if "rgba" not in data:
        logger.warning(f"No RGBA data received from camera...")
        return res
    rgba = data["rgba"]
    res["rgb"] = rgba[:, :, :3]
    return res


def process_frame_data(data) -> CameraFrameData:
    res: CameraFrameData = {"rendering_time": data["rendering_time"]}

    if data["bounding_box_2d_tight"] is not None:
        rgba = data["rgba"]
        res["rgb"] = rgba[:, :, :3]
        res["depth"] = data["distance_to_image_plane"]
        bb_2d = data["bounding_box_2d_tight"]["data"]
        bb_obj = data["bounding_box_2d_tight"]["info"]["primPaths"]

        res["bb"] = {path: list(bb)[1:-1] for path, bb in zip(bb_obj, bb_2d)}
    return res


def expand_and_clip_bb(bb, img_width=1920, img_height=1080, expand_ratio=0.10):
    x0, y0, x1, y1 = bb

    # Compute center, width, and height of the box
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    width = x1 - x0
    height = y1 - y0

    # Increase width and height by the expand_ratio
    width *= 1 + expand_ratio
    height *= 1 + expand_ratio

    # Compute new box corners
    new_x0 = center_x - width / 2
    new_y0 = center_y - height / 2
    new_x1 = center_x + width / 2
    new_y1 = center_y + height / 2

    # Clip the box to be within the image bounds
    new_x0 = max(0, min(img_width, new_x0))
    new_y0 = max(0, min(img_height, new_y0))
    new_x1 = max(0, min(img_width, new_x1))
    new_y1 = max(0, min(img_height, new_y1))

    return [new_x0, new_y0, new_x1, new_y1]


def move_gripper_absolute(transforms, distances):
    """
    Moves the gripper an absolute distance along its local axes.

    :param transforms: An array of shape (n, 4, 4) representing the transformation matrices of the grippers.
    :param distances: A list or array with three elements [x, y, z] representing the absolute distances to move along the local axes.
    :return: An array of shape (n, 4, 4) representing the new transformation matrices after the move.
    """
    # Ensure distances are a numpy array
    distances = np.array(distances)

    # Create new array for transformed matrices
    new_transforms = np.copy(transforms)

    # Iterate through each transformation matrix
    for i in range(len(transforms)):
        # Extract the rotation matrix and translation vector
        rotation_matrix = transforms[i, :3, :3]
        translation_vector = transforms[i, :3, 3]

        # Calculate new translation vector along each local axis
        # Note: Assumes the columns of rotation_matrix are unit vectors for local axes
        new_translation_vector = translation_vector + (rotation_matrix @ distances)

        # Update the translation component of the transformation matrix
        new_transforms[i, :3, 3] = new_translation_vector

    return new_transforms


def generate_hemisphere_points(n_points):
    """Generate n_points uniformly distributed on the upper hemisphere."""
    # Randomly sample points on the unit sphere
    theta = np.random.uniform(0, 2 * np.pi, n_points)  # azimuthal angle
    phi = np.random.uniform(0, np.pi / 2, n_points)  # polar angle (0 to pi/2 for upper hemisphere)

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.vstack((x, y, z)).T


class CameraHelper(Helper):
    def __init__(self, cam_cfg: DictConfig, logdir: str):
        super().__init__(cam_cfg, f"{logdir}/camera")

    def init_robot_cam(self, robot_cam_prim_path, **kwargs):
        f_stop = self.cfg.f_stop
        res = [self.cfg.resolution_h, self.cfg.resolution_w]
        pixel_size = self.cfg.pixel_size
        focal_length = self.cfg.focal_length
        focus_distance = self.cfg.focus_distance

        # Camera intrinsics parameters
        horizontal_aperture = pixel_size * res[0]
        vertical_aperture = pixel_size * res[1]

        fx, fy = focal_length / pixel_size, focal_length / pixel_size
        cx, cy = res[0] / 2, res[1] / 2

        # Initialize camera
        self.robot_camera = Camera(prim_path=robot_cam_prim_path, resolution=res, **kwargs)
        self.robot_camera.initialize()
        self.robot_camera.set_focal_length(focal_length / 100.0)
        self.robot_camera.set_focus_distance(focus_distance)
        self.robot_camera.set_lens_aperture(f_stop * 100.0)
        self.robot_camera.set_horizontal_aperture(horizontal_aperture / 10.0)
        self.robot_camera.set_vertical_aperture(vertical_aperture / 10.0)
        self.robot_camera.set_clipping_range(0.01, 1.0e5)

        self.robot_camera.add_motion_vectors_to_frame()
        self.robot_camera.add_distance_to_image_plane_to_frame()
        self.robot_camera.add_bounding_box_2d_tight_to_frame()

        intr_save_path = self.get_export_path("_intr", "json")
        json.dump(self.intrinsics_matrix.tolist(), open(intr_save_path, "w"))

        logger.info(f"Initialized camera with resolution {res}. Intrinsics saved to {intr_save_path}.")

        self.recording_buffer = []

    @property
    def frame_data(self) -> CameraFrameData:
        frame_data = self.robot_camera.get_current_frame()
        res = process_frame_data(frame_data)

        if "rgb" not in res:
            logger.warning(f"No RGB data received from camera...")

        if "depth" in res:
            res["depth"] = np.where(res["depth"] < 10.0, res["depth"], np.ones_like(res["depth"]) * 10.0)

        return res

    @property
    def extrinsics_matrix(self) -> np.array:
        return np.linalg.inv(self.robot_camera.get_view_matrix_ros())

    @property
    def intrinsics_matrix(self):
        return self.robot_camera.get_intrinsics_matrix()

    def rgbd_to_world_pc(self) -> np.array:
        extr, intr = self.extrinsics_matrix, self.intrinsics_matrix
        # TODO

    def rgbd_to_cam_pc(self) -> np.array:
        cam_transf = np.linalg.inv(np.asarray(self.extrinsics_matrix))

    def start_recording(self):
        logger.info(f"Started recording.")
        pass

    def save_recording(self):
        logger.info(f"Terminated recording.")
        pass

    def compute_inspect_pose(self, depth, target_bb):
        """Compute the camera position and orientation to inspect an object in an RGBD area."""
        bb_pixels_uv = np.stack(
            np.meshgrid(
                np.arange(target_bb[0], target_bb[2]),
                np.arange(target_bb[1], target_bb[3]),
            ),
            axis=-1,
        ).reshape([-1, 2])
        bb_pixels_depth = np.clip(depth[bb_pixels_uv[:, 1], bb_pixels_uv[:, 0]], 0.0, 10.0)
        bb_pixels_depth = np.ones_like(bb_pixels_depth) * np.min(bb_pixels_depth)

        cam_pos, cam_quat = self.robot_camera.get_world_pose()

        bb_points = self.robot_camera.get_world_points_from_image_coords(bb_pixels_uv, bb_pixels_depth)
        bb_points_mean = bb_points.mean(axis=0)
        # Horizontal inspection
        hand_to_bb_vec = bb_points_mean - cam_pos
        bb_hand_dist = np.linalg.norm(hand_to_bb_vec)
        # hand_to_bb_vec[0:2] = hand_to_bb_vec[0:2] / np.linalg.norm(hand_to_bb_vec[0:2])
        hand_to_bb_vec[2] = -0.75
        hand_to_bb_vec = hand_to_bb_vec / bb_hand_dist
        plan_pos = bb_points_mean - hand_to_bb_vec * self.cfg.inspection_dist

        # TODO: This should be camera transformation targets
        hand_z = hand_to_bb_vec / np.linalg.norm(hand_to_bb_vec)  # np.array([0.0, 0.0, 1.0])
        hand_y = -np.cross(hand_z, np.array([0.0, 0.0, 1.0]))  # np.array([0.0, 1.0, 0.0])
        hand_x = np.cross(hand_y, hand_z)  # np.array([1.0, 0.0, 0.0])
        plan_rot = np.stack([-hand_x, -hand_y, hand_z], axis=1)  # np.eye(3)

        plan_transf = pR_to_transf(plan_pos, plan_rot)
        plan_pos, plan_rot = plan_transf[:3, 3].tolist(), plan_transf[:3, :3]

        return plan_pos, matrix_to_wxyz(plan_rot)

    def compute_inspect_pose_group(self, depth, target_bb, n_grasps=64, n_dist_range=[0.2, 0.4]):
        """Compute a group of camera positions and orientations to inspect an object in an RGBD area."""
        bb_pixels_uv = np.stack(
            np.meshgrid(
                np.arange(target_bb[0], target_bb[2]),
                np.arange(target_bb[1], target_bb[3]),
            ),
            axis=-1,
        ).reshape([-1, 2])
        bb_pixels_depth = np.clip(depth[bb_pixels_uv[:, 1], bb_pixels_uv[:, 0]], 0.0, 10.0)
        bb_pixels_depth = np.ones_like(bb_pixels_depth) * np.min(bb_pixels_depth)

        cam_pos, cam_quat = self.robot_camera.get_world_pose()

        bb_points = self.robot_camera.get_world_points_from_image_coords(bb_pixels_uv, bb_pixels_depth)
        bb_points_mean = bb_points.mean(axis=0)

        hand_sphere_xyz = generate_hemisphere_points(n_grasps)
        hand_sphere_xyz = hand_sphere_xyz * np.random.uniform(n_dist_range[0], n_dist_range[1], size=(n_grasps, 1))

        # DEBUG
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(hand_sphere_xyz)
        # o3d.visualization.draw_geometries([pcd])

        hand_pos = hand_sphere_xyz + bb_points_mean[None]
        hand_to_bb_vec = bb_points_mean[None] - hand_pos
        bb_hand_dist = np.linalg.norm(hand_to_bb_vec, axis=1)
        hand_to_bb_vec = hand_to_bb_vec / bb_hand_dist[:, None]

        hand_z = hand_to_bb_vec
        hand_y = -np.cross(hand_z, np.array([[0.0, 0.0, 1.0]]))
        hand_x = np.cross(hand_y, hand_z)
        plan_rot = np.stack([-hand_x, -hand_y, hand_z], axis=-1)  # B x 3 x 3

        plan_pos, plan_rot = hand_pos, plan_rot
        # plan_transf = pR_to_transf(plan_pos, plan_rot)
        # plan_transf = np.matmul(plan_transf, self.tool_to_gripper_transform)
        # plan_pos, plan_rot = plan_transf[:3, 3].tolist(), plan_transf[:3, :3]

        # Rotate the camera about the axis at [bb_points_mean], pointing towards z, about

        # inspect_rot = np.matmul(plan_rot, transforms.euler_angles_to_matrix(torch.tensor([0, 0, -np.pi/2]), "XYZ").numpy())
        plan_rot = matrix_to_wxyz(plan_rot)
        return plan_pos, plan_rot
