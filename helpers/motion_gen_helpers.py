import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import omni.isaac.core.utils.prims as prims_utils
import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_logger
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from loguru import logger
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.transformations import pose_from_tf_matrix
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.physx import get_physx_interface
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from pytorch3d import transforms
from tqdm import trange

from helpers.helper import Helper

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


def get_motion_plan_callback_fn(world, robot, ctrl_interval=2, record_callback=None):
    def motion_plan_callback_fn(succ, traj):
        if not succ:
            return

        # # DEBUG
        # robot.set_joint_positions(traj.position[-1].cpu().numpy())
        # return

        for i in trange(len(traj) * ctrl_interval, desc="Executing motion plan"):
            if i % ctrl_interval == 0:
                art_action = ArticulationAction(
                    traj.position[i // ctrl_interval].cpu().numpy(),
                    traj.velocity[i // ctrl_interval].cpu().numpy(),
                    joint_indices=np.arange(len(robot.dof_names) - 2),
                )
            # DEBUG
            # robot.set_joint_positions(np.array(traj.position[i // ctrl_interval].tolist() + [0.04, 0.04]))
            robot.apply_action(art_action)
            world.step(render=True)
            if record_callback is not None:
                record_callback()

        for _ in range(50):
            world.step(render=True)
            if record_callback is not None:
                record_callback()

    return motion_plan_callback_fn


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


class RidgebackFranka(Robot):
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

        self._usd_path = "assets/ridgeback_franka/ridgeback_franka.usd"
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
        self.inspection_q: list[float] = [0.0, -1.047, 0.0, -2.094, 0.0, 2.356, 0.766]

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
        dst = 0.3
        self.hold_object_target_prim = None
        gripper_pos = self.tcp_pos
        from_prim_tf = self.ee_pose
        for prim in candidate_objects.values():
            print()
            obj_usdgeom = UsdGeom.Xformable(get_prim_at_path(prim))
            obj_pose = np.asarray(obj_usdgeom.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).transpose()
            if mode == "mesh":
                obj_vmesh = UsdGeom.Mesh(get_prim_at_path(prim))
                obj_verts = np.array(obj_vmesh.GetPointsAttr().Get())
                obj_verts = np.matmul(obj_pose, np.pad(obj_verts, ((0, 0), (0, 1)), constant_values=1.0).T).T[:, :3]
                gripper_to_obj_dist = np.min(np.linalg.norm(gripper_pos[None] - obj_verts, axis=1)).min()
            elif mode == "pose":
                obj_pos = obj_pose[:3, 3]
                gripper_to_obj_dist = np.linalg.norm(gripper_pos - obj_pos)
            if gripper_to_obj_dist < dst:
                dst = gripper_to_obj_dist
                self.hold_object_target_prim = prim
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
        self.gripper.apply_action(ArticulationAction([0.04, 0.04]))


class IIWAPanda(Robot):
    def __init__(
        self,
        cfg,
        prim_path: str,
        name: Optional[str] = "iiwa_panda",
        usd_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.cfg = cfg
        self._usd_path = usd_path
        self._name = name
        self._prim_path = prim_path

        self._usd_path = "assets/iiwa/iiwa.usd"
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

    @property
    def inspection_q(self) -> list[float]:
        return [0.785375, -0.785375, -0.5235833, -1.57, 2.61791, -1.57, 0]
        # return [ 0.        , -1.30899694,  0.        , -2.0943951 ,  0.        , 1.30899694, 3.14159265]

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

    def set_base_pose(self, xyz, rpy):
        quat = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(torch.from_numpy(rpy), "XYZ")).numpy()
        self.set_world_pose(xyz, quat)

    # -------- revised by ZXM -----------
    def set_robot_scale(self, scale_factor):
        robot_prim = get_prim_at_path(self._prim_path)
        if robot_prim:
            robot_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(scale_factor, scale_factor, scale_factor))
            logger.info(f"Set robot scale to {scale_factor}.")
        else:
            logger.warning(f"Could not find robot prim at {self._prim_path} to set scale.")

    # -----------------------------------

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
    def ee_pos_quat(self) -> tuple:
        ee_pos, ee_quat = self.gripper.get_world_pose()
        return ee_pos, ee_quat

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
        dst = 0.1
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
                # self.hold_object_target_prim = prim
                self.hold_object_target_prim = prim + "/base_link"
                from_prim_tf = self.ee_pose
                to_prim_tf = obj_pose
                translation_matrix = np.linalg.inv(from_prim_tf) @ to_prim_tf
                translation, orientation = pose_from_tf_matrix(translation_matrix)

        if self.hold_object_target_prim is None:
            logger.error(f"No attachable object! Make sure that the object (origin) is near enough.")
            self.attach_obj_flag = False
            return self.hold_object_target_prim

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


class MotionGenHelper(Helper):
    def __init__(self, motion_gen_cfg, scene_cfg, logdir, device="cuda:0", **kwargs):
        super().__init__(motion_gen_cfg, f"{logdir}/motion_gen")
        self.scene_cfg = scene_cfg

        self.robot_pos, self.robot_quat = (
            torch.zeros(3, dtype=torch.float32, device=device),
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device),
        )

        self.device = device

    def update_robot_pose(self, robot_pos, robot_quat):
        self.robot_pos, self.robot_quat = (
            robot_pos.to(self.robot_pos.device).float(),
            robot_quat.to(self.robot_quat.device).float(),
        )

    def pose_to_local(self, pos, quat):
        rotmat = transforms.quaternion_to_matrix(quat)
        robot_rotmat = transforms.quaternion_to_matrix(self.robot_quat)
        pos = torch.matmul(robot_rotmat.T, pos - self.robot_pos)
        rotmat = torch.matmul(robot_rotmat.T, rotmat)
        quat = transforms.matrix_to_quaternion(rotmat)
        return pos, quat

    def pose_to_local_batch(self, pos, quat):
        rotmat = transforms.quaternion_to_matrix(quat)
        robot_rotmat = transforms.quaternion_to_matrix(self.robot_quat)
        pos = torch.matmul(robot_rotmat.T, (pos - self.robot_pos[None]).transpose(-1, -2)).transpose(-1, -2)
        rotmat = torch.matmul(robot_rotmat.T.unsqueeze(0), rotmat)
        quat = transforms.matrix_to_quaternion(rotmat)
        return pos, quat

    def setup_collider(self, world_config=None, robot_file="ridgeback_franka.yml"):
        tensor_args = TensorDeviceType()

        self.world_config = world_config
        if isinstance(self.world_config, WorldConfig):
            self.world_config.save_world_as_mesh(self.get_export_path("collision", "ply"))

        collision_checker_type = CollisionCheckerType.PRIMITIVE if len(world_config.mesh) == 0 else CollisionCheckerType.MESH
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            self.world_config,
            tensor_args,
            collision_checker_type=collision_checker_type,
            interpolation_steps=5000,
            interpolation_dt=0.02,
            # position_threshold=0.01,
            # rotation_threshold=np.pi/18,
            self_collision_check=False,
            evaluate_interpolated_trajectory=True,
            use_cuda_graph=True,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        self.robot_cfg = RobotConfig.from_dict(self.robot_cfg, tensor_args)

        logger.info(f"Warming up cuRobo motion generation helper for robot {robot_file}")
        self.motion_gen.warmup(parallel_finetune=True)

        # self.ik_config = IKSolverConfig.load_from_robot_config(
        #     self.robot_cfg,
        #     self.world_config,
        #     rotation_threshold=0.05,
        #     position_threshold=0.005,
        #     num_seeds=20,
        #     self_collision_check=False,
        #     self_collision_opt=False,
        #     tensor_args=tensor_args,
        #     use_cuda_graph=True,
        # )
        # self.ik_solver = IKSolver(self.ik_config)

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
        goal_pos, goal_quat = self.pose_to_local(goal_pos, goal_quat)
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

            if not success:
                return False, None
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

    def plan_to_goal_pose_batch(
        self,
        start_q: np.array,
        goal_pos=torch.tensor([[0.0, 0.0, 1.0]]),
        goal_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        grasp_appr=False,
        quiet=False,
        callback_fn=None,
    ):
        batch_size = goal_pos.shape[0]
        start_state = JointState.from_position(start_q[None].tile([batch_size, 1]))
        pose_metric = None
        success = False
        goal_pos, goal_quat = self.pose_to_local_batch(goal_pos, goal_quat)
        goal_pose = Pose(goal_pos.contiguous(), goal_quat.contiguous())

        result = self.motion_gen.plan_batch(
            start_state,
            goal_pose,
            MotionGenPlanConfig(max_attempts=100, parallel_finetune=True, pose_cost_metric=pose_metric),
        )
        success = result.success
        # show the reason of failure
        for i in range(batch_size):
            if not success[i]:
                logger.warning(f"Target {i} planning failed.")

                target_pos = goal_pos[i]

                if result.position_error is not None and result.position_error[i].item() > 0.05:
                    logger.debug(f"Target {i} has a large position error: {result.position_error[i].item()}.")
                    continue

                if result.rotation_error is not None and result.rotation_error[i].item() > 0.1:
                    logger.debug(f"Target {i} has a large rotation error: {result.rotation_error[i].item()}.")
                    continue

                logger.warning(f"Target {i} failed due to unknown reasons. Status: {result.status}")

        if result.success.sum() == 0:
            return success, None

        q_traj = result.optimized_plan[torch.where(success)[0].tolist()]

        if not quiet:
            motion_plan_path = self.get_export_path("plan_traj", "json")
            logger.info(f"Motion planning to target pose, success: {sum(success)} / {batch_size}. Logged to {motion_plan_path}.")
            json.dump(
                {
                    "start_q": start_q.tolist(),
                    "goal_state": {
                        "pos": goal_pos.tolist(),
                        "quat": goal_quat.tolist(),
                    },
                    "grasp_appr": grasp_appr,
                    "result": {
                        "success": success.tolist(),
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
            callback_fn(success.sum() > 0, q_traj[0])

        return success, q_traj
