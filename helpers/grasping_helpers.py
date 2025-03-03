import argparse
import json
import os
import pickle as pkl
import sys

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh as tm
from graspnetAPI import GraspGroup
from scipy.spatial.transform import Rotation as R

sys.path.append("..")

import argparse
import json
from functools import cache

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from omegaconf import DictConfig

from helpers.helper import Helper
from models.graspnet.models.graspnets import GraspNet, pred_decode
from models.graspnet.utils.collision_detector import ModelFreeCollisionDetector
from models.graspnet.utils.data_utils import (
    CameraInfo,
    create_point_cloud_from_depth_image,
)
from utils.utils_plotly import plot_point_cloud


class GraspNetHelper(Helper):
    def __init__(self, grasp_cfg: DictConfig, logdir):
        super().__init__(grasp_cfg, f"{logdir}/graspnet")

        self.net = self.get_net()

    def get_net(self):
        # Init the model
        net = GraspNet(
            input_feature_dim=0,
            num_view=self.cfg.num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        # Load checkpoint
        checkpoint = torch.load(self.cfg.checkpoint_path)
        net.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        print("-> loaded checkpoint %s (epoch: %d)" % (self.cfg.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()

        return net

    def get_grasps(self, net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfg.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfg.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self, gg, cloud, extra_geom=[]):
        grippers = gg.to_open3d_geometry_list()
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([cloud, frame, *grippers] + extra_geom)

    def intr_to_cam_info(self, res, intrinsics, factor_depth=1.0):
        return CameraInfo(
            float(res[0]),
            float(res[1]),
            intrinsics[0][0],
            intrinsics[1][1],
            intrinsics[0][2],
            intrinsics[1][2],
            factor_depth,
        )

    def process_data(self, color, depth, seg_mask, intrinsic: np.array, extrinsics: np.array):
        # generate cloud
        cam_info = self.intr_to_cam_info((color.shape[1], color.shape[0]), intrinsic)
        cloud = create_point_cloud_from_depth_image(depth, cam_info, extrinsics=extrinsics)

        # get valid points
        seg_mask = ((depth > 0) & (depth < 5.0) & (seg_mask > 0)).reshape(-1)

        cloud_seg = cloud[seg_mask]
        color_seg = color.reshape(-1, 3)[seg_mask]

        # Remove outliers
        cloud_seg_o3d = o3d.geometry.PointCloud()
        cloud_seg_o3d.points = o3d.utility.Vector3dVector(cloud_seg)
        ind = np.array(cloud_seg_o3d.remove_statistical_outlier(nb_neighbors=32, std_ratio=2.0)[1])

        # sample points
        if len(ind) >= self.cfg.num_point:
            idxs = np.random.choice(len(ind), self.cfg.num_point, replace=False)
        else:
            ind_ind = np.random.choice(len(ind), self.cfg.num_point - len(ind), replace=True)
            idxs = np.concatenate([ind, ind[ind_ind]], axis=0)

        end_points = {
            "point_clouds": torch.tensor(cloud_seg[idxs], dtype=torch.float32, device=torch.device("cuda:0")).unsqueeze(0),
            "cloud_colors": torch.tensor(color_seg[idxs], dtype=torch.float32, device=torch.device("cuda:0")).unsqueeze(0),
        }

        return cloud, end_points

    def query_grasp_multiview(
        self,
        rgbs,
        depths,
        ws_masks,
        seg_masks,
        intrinsics,
        extrinsicses,
        viz=False,
        quiet=False,
    ):
        pkl.dump(
            [rgbs, depths, seg_masks, intrinsics, extrinsicses],
            open(self.get_export_path("grasp_query_data", "pkl"), "wb"),
        )

        all_points, end_points = [], []
        base_extrinsics = extrinsicses[0]
        for i, (rgb, depth, ws_mask, seg_mask, extrinsics) in enumerate(zip(rgbs, depths, ws_masks, seg_masks, extrinsicses)):
            depth = np.clip(depth, 0, 5.0)
            # Turn to base frame
            extrinsics = np.linalg.inv(base_extrinsics) @ extrinsics
            points, end_point = self.process_data(rgb, depth, seg_mask, intrinsics, extrinsics)
            all_points.append(points)
            end_points.append(end_point)

            if not quiet:
                cv2.imwrite(
                    self.get_export_path(f"grasp_query_rgb-{i}", "png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.uint8),
                )
                cv2.imwrite(
                    self.get_export_path(f"grasp_query_depth_normalized-{i}", "png"),
                    cv2.normalize(depth, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16),
                )

        all_points = np.concatenate(all_points, axis=0)
        end_points = {
            "point_clouds": torch.cat([ep["point_clouds"][0] for ep in end_points], axis=0).unsqueeze(0),
            "cloud_colors": torch.cat([ep["cloud_colors"][0] for ep in end_points], axis=0),
        }

        gg = self.get_grasps(self.net, end_points)
        if self.cfg.collision_thresh > 0:
            gg = self.collision_detection(gg, all_points)

        gg.nms()
        gg.sort_by_score()

        transl = gg.translations
        orient = gg.rotation_matrices
        widths = gg.widths
        scores = gg.scores

        if not quiet:
            json_save_path = self.get_export_path("grasps", "json")
            json.dump(
                {
                    "grasp_pos": gg.translations.tolist(),
                    "orient": gg.rotation_matrices.tolist(),
                    "widths": gg.widths.tolist(),
                    "scores": gg.scores.tolist(),
                    "extrinsicses": np.asarray(extrinsicses).tolist(),
                },
                open(json_save_path, "w"),
            )

            tm.Trimesh(
                vertices=np.matmul(
                    end_points["point_clouds"][0].cpu().numpy(),
                    base_extrinsics[:3, :3].T,
                )
                + base_extrinsics[:3, 3],
                vertex_colors=end_points["cloud_colors"].cpu().numpy(),
            ).export(self.get_export_path("end_point_cloud", "ply"))
            tm.Trimesh(vertices=np.matmul(all_points, base_extrinsics[:3, :3].T) + base_extrinsics[:3, 3]).export(
                self.get_export_path("full_point_cloud", "ply")
            )

        transf = np.eye(4)[None].repeat(len(transl), axis=0)
        transf[:, :3, 3] = np.array(transl)
        transf[:, :3, :3] = np.array(orient)

        # Robot hand transformation
        robot_tf = np.eye(4)
        robot_tf[:3, :3] = R.from_euler("y", [np.pi / 2]).as_matrix()
        robot_tf[0, 3] = -0.019

        transf = np.matmul(transf, robot_tf[None])
        transf_global = np.matmul(extrinsicses[0], transf)

        full_cloud_o3d = None
        if viz:
            full_cloud_o3d = o3d.geometry.PointCloud()
            full_cloud_o3d.points = o3d.utility.Vector3dVector(all_points)
            self.vis_grasps(gg, full_cloud_o3d)

        return (
            np.array(transf_global),
            np.array(transf),
            np.array(widths),
            np.array(scores),
            [gg, full_cloud_o3d],
        )

    def query_grasp(
        self,
        rgb,
        depth,
        ws_mask,
        seg_mask,
        intrinsics,
        extrinsics,
        viz=False,
        quiet=False,
    ):
        pkl.dump(
            [rgb, depth, seg_mask, intrinsics, extrinsics],
            open(self.get_export_path("grasp_query_data", "pkl"), "wb"),
        )

        depth = np.clip(depth, 0, 5.0)
        all_points, end_points = self.process_data(rgb, depth, seg_mask, intrinsics, np.eye(4))

        gg = self.get_grasps(self.net, end_points)
        if self.cfg.collision_thresh > 0:
            gg = self.collision_detection(gg, all_points)

        gg.nms()
        gg.sort_by_score()

        gg = gg[:64]

        transl = gg.translations
        orient = gg.rotation_matrices
        widths = gg.widths
        scores = gg.scores

        if not quiet:
            cv2.imwrite(
                self.get_export_path("grasp_query_rgb", "png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.uint8),
            )
            cv2.imwrite(
                self.get_export_path("grasp_query_depth_normalized", "png"),
                cv2.normalize(depth, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16),
            )

            json_save_path = self.get_export_path("grasps", ".json")
            json.dump(
                {
                    "grasp_pos": gg.translations.tolist(),
                    "orient": gg.rotation_matrices.tolist(),
                    "widths": gg.widths.tolist(),
                    "scores": gg.scores.tolist(),
                },
                open(json_save_path, "w"),
            )

            full_cloud_o3d = o3d.geometry.PointCloud()
            full_cloud_o3d.points = o3d.utility.Vector3dVector(all_points)
            o3d.io.write_point_cloud(self.get_export_path("full_point_cloud", ".pcd"), full_cloud_o3d)

        transf = np.eye(4)[None].repeat(len(transl), axis=0)
        transf[:, :3, 3] = np.array(transl)
        transf[:, :3, :3] = np.array(orient)

        robot_tf = np.eye(4)
        robot_tf[:3, :3] = R.from_euler("y", [np.pi / 2]).as_matrix()
        robot_tf[0, 3] = -0.085

        transf = np.matmul(transf, robot_tf[None])
        transf_global = np.matmul(extrinsics, transf)

        # if viz:
        #     full_cloud_o3d = o3d.geometry.PointCloud()
        #     full_cloud_o3d.points = o3d.utility.Vector3dVector(all_points)
        #     self.vis_grasps(gg, full_cloud_o3d)

        return (
            np.array(transf_global),
            np.array(transf),
            np.array(widths),
            np.array(scores),
            [gg, full_cloud_o3d],
        )
