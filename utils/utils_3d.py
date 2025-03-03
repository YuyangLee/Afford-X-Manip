import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import trimesh as tm
import open3d as o3d

wxyz_to_matrix = lambda q: R.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()
matrix_to_wxyz = lambda m: R.from_matrix(m).as_quat()[..., [3, 0, 1, 2]]
rpy_to_matrix = lambda rpy: R.from_euler("xyz", rpy).as_matrix()


def pr_to_transf(p, quat):
    transf = np.eye(4)
    transf[:3, 3] = p
    transf[:3, :3] = wxyz_to_matrix(quat)
    return transf


def pR_to_transf(p, R):
    transf = np.eye(4)
    transf[:3, 3] = p
    transf[:3, :3] = R
    return transf


def get_cam_params(
    cam_pos: torch.Tensor,
    cam_look_at: torch.Tensor,
    width=640,
    height=480,
    focal_length=24,
    horizontal_aperture=20.954999923706055,
    vertical_aperture=15.290800094604492,
):
    device = cam_pos.device
    num_envs = len(cam_pos)
    cam_front = cam_look_at - cam_pos
    cam_right = torch.cross(cam_front, torch.tensor([[0.0, 0.0, 1.0]], device=device))
    cam_up = torch.cross(cam_right, cam_front)

    cam_right = cam_right / (torch.norm(cam_right, dim=-1, keepdim=True) + 1e-12)
    cam_front = cam_front / (torch.norm(cam_front, dim=-1, keepdim=True) + 1e-12)
    cam_up = cam_up / (torch.norm(cam_up, dim=-1, keepdim=True) + 1e-12)

    # Camera convention difference between ROS and Isaac Sim
    R = torch.stack([cam_right, -cam_up, cam_front], dim=1)  # .transpose(-1, -2)
    t = -torch.bmm(R, cam_pos.unsqueeze(-1)).squeeze()
    extrinsics = torch.eye(4, device=device).unsqueeze(0).tile([num_envs, 1, 1])
    extrinsics[:, :3, :3] = R
    extrinsics[:, :3, 3] = t

    fx = width * focal_length / horizontal_aperture
    fy = height * focal_length / vertical_aperture
    cx = width * 0.5
    cy = height * 0.5

    intrinsics = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device).unsqueeze(0).tile([num_envs, 1, 1])

    return extrinsics, intrinsics


def tm_to_o3d(mesh_tm: tm.Trimesh):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_tm.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_tm.faces)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d
