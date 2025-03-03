from dataclasses import dataclass
from enum import Enum
import json
from typing import Optional
import numpy as np
import trimesh as tm
import torch

from helpers.camera_helpers import move_gripper_absolute
from utils.consts import *
from helpers.helper import Helper
from loguru import logger


from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import (
    delete_prim,
    get_prim_at_path,
    is_prim_path_valid,
    set_prim_property,
    set_targets,
)
from omni.physx import get_physx_interface, get_physx_simulation_interface
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf, PhysxSchema, Usd, UsdPhysics, UsdShade, UsdGeom
from pytorch3d import transforms
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, Mesh, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.util.logger import setup_logger
import omni.isaac.core.utils.prims as prims_utils
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
    PoseCostMetric,
)

setup_logger(level="warning")


def plot_iters_traj_3d(trajectory, d_id=1, dof=7, seed=0):
    # Third Party
    import matplotlib.pyplot as plt

    ax = plt.axes(projection="3d")
    c = 0
    h = trajectory[0][0].shape[1] - 1
    x = [x for x in range(h)]

    for k in range(len(trajectory)):
        q = trajectory[k]

        for i in range(len(q)):
            # ax.plot3D(x,[c for _ in range(h)],  q[i][seed, :, d_id].cpu())#, 'r')
            ax.scatter3D(
                x,
                [c for _ in range(h)],
                q[i][seed, :h, d_id].cpu(),
                c=q[i][seed, :, d_id].cpu(),
            )
            # @plt.show()
            c += 1
    # plt.legend()
    plt.show()


class GraspAction(Enum):
    CLOSE = 1
    OPEN = -1
    NONE = 0


def find_mesh_prims_in_prim(prim):
    meshes = []

    # Traverse only the children of the specified prim
    for child in prim.GetChildren():
        # Check if the child is a mesh
        if child.IsA(UsdGeom.Mesh):
            meshes.append(child)
        # Recursively check if the child has its own children
        meshes.extend(find_mesh_prims_in_prim(child))

    return meshes


class Franka(Robot):
    def __init__(
        self,
        cfg,
        prim_path: str,
        name: Optional[str] = "franka",
        usd_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.cfg = cfg
        self._usd_path = usd_path
        self._name = name
        self._prim_path = prim_path

        self._usd_path = "assets/franka/franka_panda.usd"
        add_reference_to_stage(self._usd_path, self._prim_path)

        super().__init__(prim_path=self._prim_path, name=name, articulation_controller=None, **kwargs)

        self._end_effector_prim_path = f"{self._prim_path}/panda_hand"
        self._gripper = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.0, 0.0]),
            action_deltas=np.array([0.05, 0.05]),
        )

        self.magic_grasp_prim_path = f"{self._end_effector_prim_path}/magic_grasp_joint"

        self._r_finger = RigidPrim(
            prim_path=f"{self._prim_path}/panda_rightfinger",
            name=self.name + "_right_finger",
        )
        self._l_finger = RigidPrim(
            prim_path=f"{self._prim_path}/panda_leftfinger",
            name=self.name + "_left_finger",
        )

        self.attach_obj_flag = False
        self.inspection_q: list[float] = [
            -2.0238492e-08,
            -1.0471565e00,
            -6.9897020e-08,
            -2.0944028e00,
            -6.1968379e-08,
            2.3561783e00,
            7.6636827e-01,
        ]

    @property
    def l_finger(self) -> RigidPrim:
        return self._l_finger

    @property
    def r_finger(self) -> RigidPrim:
        return self._r_finger

    @property
    def tcp_pos(self) -> np.array:
        return (self._l_finger.get_world_pose()[0] + self._r_finger.get_world_pose()[0]) / 2

    @property
    def end_effector(self) -> RigidPrim:
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )

    def post_reset(self) -> None:
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(dof_index=self.gripper.joint_dof_indicies[0], mode="position")
        self._articulation_controller.switch_dof_control_mode(dof_index=self.gripper.joint_dof_indicies[1], mode="position")
        return

    @property
    def ee_pose(self) -> np.array:
        ee_pos, ee_quat = self.gripper.get_world_pose()
        ee_rotmat = transforms.quaternion_to_matrix(torch.tensor(ee_quat)).numpy()
        ee_transf = np.eye(4)
        ee_transf[:3, :3] = ee_rotmat
        ee_transf[:3, 3] = ee_pos
        return ee_transf

    @property
    def tcp_pos(self) -> np.array:
        ee_pos, ee_quat = self.gripper.get_world_pose()
        ee_rotmat = transforms.quaternion_to_matrix(torch.tensor(ee_quat)).numpy()
        ee_pos = ee_pos + np.matmul(ee_rotmat, np.array([0.0, 0.0, 0.085]))
        return ee_pos

    def setup_magic_grasp(self, stage):
        self.holder_joint = UsdPhysics.FixedJoint.Define(stage, self.magic_grasp_prim_path)
        self.holder_joint.CreateBody0Rel().SetTargets([self.cfg.hold_obj_link_path])
        self.holder_joint.GetJointEnabledAttr().Set(False)
        self.attach_obj_flag = False

    def attach_nearest_object(self, stage, candidate_objects, mode="mesh"):
        # Find nearest object
        dst = 0.2  # Maximum tolerable distance: 20 cm
        self.hold_object_target_prim = None
        gripper_pos = self.tcp_pos
        for prim in candidate_objects:
            obj_usdgeom = UsdGeom.Xformable(get_prim_at_path(prim + "/base_link"))
            obj_pose = np.asarray(obj_usdgeom.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).transpose()
            if mode == "mesh":
                obj_vmesh = UsdGeom.Mesh(get_prim_at_path(prim + "/base_link/visuals"))
                obj_verts = np.array(obj_vmesh.GetPointsAttr().Get())
                obj_verts = np.matmul(obj_pose, np.pad(obj_verts, ((0, 0), (0, 1)), constant_values=1.0).T).T[:, :3]
                gripper_to_obj_dist = np.min(np.linalg.norm(gripper_pos[None] - obj_verts, axis=1)).min()
            elif mode == "pose":
                obj_pos = obj_pose[:3, 3]
                gripper_to_obj_dist = np.linalg.norm(gripper_pos - obj_pos)
            if gripper_to_obj_dist < dst:
                dst = gripper_to_obj_dist
                self.hold_object_target_prim = prim + "/base_link"

            from_prim_tf = self.ee_pose
            to_prim_tf = obj_pose
            translation_matrix = np.linalg.inv(from_prim_tf) @ to_prim_tf
            translation, orientation = pose_from_tf_matrix(translation_matrix)

        if self.hold_object_target_prim is None:
            logger.error(f"No attachable object! Make sure that the object (origin) is near enough.")
            self.attach_obj_flag = False

        else:
            self.holder_joint.CreateBody1Rel().SetTargets([self.hold_object_target_prim])
            self.holder_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*translation.astype(float)))
            self.holder_joint.GetLocalRot0Attr().Set(Gf.Quatf(*orientation.astype(float)))
            self.holder_joint.GetLocalPos1Attr().Set(tuple(Gf.Vec3f(0.0, 0.0, 0.0)))
            self.holder_joint.GetLocalRot1Attr().Set(Gf.Quatf(*np.array([1, 0, 0, 0]).astype(float)))
            self.holder_joint.GetJointEnabledAttr().Set(True)

            logger.info(f"Attaching object: {self.hold_object_target_prim} (prim distance {dst} m)")

        self.attach_obj_flag = True
        return self.hold_object_target_prim

    def release_grasped_object(self):
        if not self.attach_obj_flag:
            logger.warning("No object is currently attached.")
            return

        logger.info(f"Released object.")
        # self.holder_joint.GetJointEnabledAttr().Set(False)
        # self.holder_joint.GetBody0Rel().SetTargets([])
        # self.holder_joint.GetBody1Rel().SetTargets([])
        prims_utils.delete_prim(self.magic_grasp_prim_path)

        obj_pose = np.asarray(
            UsdGeom.Xformable(get_prim_at_path(self.hold_object_target_prim)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ).transpose()[:3, 3]
        get_physx_interface().apply_force_at_pos(self.hold_object_target_prim, Gf.Vec3d(0.0, 0.0, -1.0), Gf.Vec3d(*obj_pose))
        self.hold_object_target_prim = None


class RMPFlowHelper(Helper):
    def __init__(
        self,
        motion_gen_cfg,
        scene_cfg,
        logdir,
        robot_file="ridgeback_franka.yaml",
        **kwargs,
    ):
        super().__init__(motion_gen_cfg, f"{logdir}/motion_gen")
        self.scene_cfg = scene_cfg

    def setup_collider(self, world_config=None, robot_file="ridgeback_franka.yaml"):
        tensor_args = TensorDeviceType()

        if world_config is None:
            # world_config = "collision_base.yml"
            scene_collision_mesh_path = (
                self.scene_cfg.slam_collision_mesh_path if self.cfg.use_slam_collision else self.scene_cfg.gt_collision_mesh_path
            )
            scene_mesh = Mesh(
                name="scene",
                file_path=scene_collision_mesh_path,
                pose=[0.0, 0.0, -0.01, 1.0, 0.0, 0.0, 0.0],
                tensor_args=tensor_args,
            )
            scene_mesh.file_path = os.path.abspath(scene_collision_mesh_path)
            # world_config = WorldConfig(mesh=[scene_mesh])
            # # world_config = WorldConfig.create_collision_support_world(world_config)

            spheres = scene_mesh.get_bounding_spheres(n_spheres=256, surface_sphere_radius=0.025)
            world_config = WorldConfig(sphere=spheres)

        self.world_config = world_config
        if isinstance(self.world_config, WorldConfig):
            self.world_config.save_world_as_mesh(self.get_export_path("collision", "stl"))

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            self.world_config,
            tensor_args,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            interpolation_steps=500,
            # interpolation_steps=5000,
            interpolation_dt=0.02,
            # interpolation_dt=0.02,
            position_threshold=0.5,
            rotation_threshold=np.pi / 2,
            self_collision_check=False,
            use_cuda_graph=True,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        self.robot_cfg = RobotConfig.from_dict(self.robot_cfg, tensor_args)

        logger.info(f"Warming up cuRobo motion generation helper for robot {robot_file}")
        self.motion_gen.warmup(parallel_finetune=True)

        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_config,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(self.ik_config)

    def plan_to_goal_q(self, start_q: torch.Tensor, goal_q: torch.Tensor, quiet=False, callback_fn=None):
        start_state = JointState.from_position(start_q.unsqueeze(0))
        goal_state = JointState.from_position(goal_q.unsqueeze(0))

        result = self.motion_gen.plan_single_js(start_state, goal_state, MotionGenPlanConfig(max_attempts=100))

        success = result.success[0].item()
        q_traj = result.get_interpolated_plan() if success else None

        if not quiet:
            motion_plan_path = self.get_export_path("plan_traj", "json")
            logger.info(f"Motion planning to target q, success: {success}. Logged to {motion_plan_path}.")
            json.dump(
                {
                    "start_q": start_q.tolist(),
                    "goal_state": goal_q.tolist(),
                    "result": {
                        "success": success,
                        "solve_time": result.solve_time,
                        "traj_len": len(q_traj) if q_traj is not None else 0,
                        "q_traj": (q_traj.position.tolist() if q_traj is not None else []),
                        "q_traj_velo": (q_traj.velocity.tolist() if q_traj is not None else []),
                        "q_traj_acc": (q_traj.acceleration.tolist() if q_traj is not None else []),
                    },
                },
                open(motion_plan_path, "w"),
            )

        if callback_fn is not None and success:
            callback_fn(success, q_traj)

        return success, q_traj

    def compute_ik(self, goal_pos, goal_quat, callback_fn=None):
        goal = Pose(goal_pos, goal_quat)
        result = self.ik_solver.solve_single(goal.unsqueeze(0))
        if result.success:
            if callback_fn is not None:
                callback_fn(np.array(result.js_solution.position))
            return True, np.array(goal.js_solution.position)
        return False, None

    def plan_to_goal_pose(
        self,
        start_q: np.array,
        goal_pos=torch.tensor([0.0, 0.0, 1.0]),
        goal_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        grasp_appr=False,
        quiet=False,
        callback_fn=None,
    ):
        start_state = JointState.from_position(start_q[None])
        pose_metric = None
        success = False
        goal_pose = Pose(goal_pos, goal_quat)
        # TODO: The approach vector in relative pose.
        if grasp_appr:
            pose_metric = PoseCostMetric.create_grasp_approach_metric(offset_position=self.cfg.grasp_approach_offset, tstep_fraction=0.8)
            result = self.motion_gen.plan_single(
                start_state,
                goal_pose,
                MotionGenPlanConfig(
                    max_attempts=100,
                    parallel_finetune=True,
                    pose_cost_metric=pose_metric,
                ),
            )
            success = result.success[0].item()
            q_traj = result.get_interpolated_plan() if success else None
            # if result_appr.success[0].item():
            #     q_traj_appr = result_appr.get_interpolated_plan()
            #     appr_q = q_traj_appr[-1]
            #     result_reach = self.motion_gen.plan_single(appr_q, goal_pose, MotionGenPlanConfig(max_attempts=100, parallel_finetune=True, pose_cost_metric=pose_metric))
            #     if result_reach.success[0].item():
            #         q_traj_reach = result_reach.get_interpolated_plan()
            #         q_traj = np.concatenate([q_traj_appr, q_traj_reach, q_traj_reach[::-1]], axis=0)
            #         success = True

        else:
            result = self.motion_gen.plan_single(
                start_state,
                goal_pose,
                MotionGenPlanConfig(
                    max_attempts=100,
                    parallel_finetune=True,
                    pose_cost_metric=pose_metric,
                ),
            )
            success = result.success[0].item()
            q_traj = result.get_interpolated_plan()  # if success else None

        if not quiet:
            motion_plan_path = self.get_export_path("plan_traj", "json")
            logger.info(f"Motion planning to target pose, success: {success}. Logged to {motion_plan_path}.")
            json.dump(
                {
                    "start_q": start_q.tolist(),
                    "goal_state": {
                        "pos": goal_pos.tolist(),
                        "quat": goal_quat.tolist(),
                    },
                    "grasp_appr": grasp_appr,
                    "result": {
                        "success": success,
                        "solve_time": result.solve_time,
                        "traj_len": len(q_traj) if q_traj is not None else 0,
                        "q_traj": (q_traj.position.tolist() if q_traj is not None else []),
                        "q_traj_velo": (q_traj.velocity.tolist() if q_traj is not None else []),
                        "q_traj_acc": (q_traj.acceleration.tolist() if q_traj is not None else []),
                    },
                },
                open(motion_plan_path, "w"),
            )

        if callback_fn is not None:  # and success:
            callback_fn(success, q_traj)

        return success, q_traj
