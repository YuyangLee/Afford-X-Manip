import json
import os
from multiprocessing import Process, Queue

import cv2
import hydra
import numpy as np
import torch
import yaml
from loguru import logger
from omegaconf import DictConfig, open_dict
from tqdm import trange

from utils.consts import *
from utils.utils_assets import og_decrypt_file
from utils.utils_record import export_to_cam

os.environ["CUROBO_TORCH_CUDA_GRAPH_RESET"] = "1"
os.environ["CUROBO_USE_LRU_CACHE"] = "1"
# Isaac APIs
from omni.isaac.kit import SimulationApp
from pytorch3d import transforms

from helpers.affordance_helpers import AffordanceHelper
from helpers.grasping_helpers import GraspNetHelper
from utils.utils_3d import matrix_to_wxyz, rpy_to_matrix
from utils.utils_misc import prep_logdir
from utils.utils_query import select_bb

app = SimulationApp({"headless": False})

import sys

import omni.kit.viewport as vp
from curobo.util.usd_helper import UsdHelper
from omni.isaac.core import World
from omni.isaac.core.utils.prims import add_reference_to_stage, get_prim_at_path
from omni.isaac.core.utils.semantics import add_update_semantics, remove_all_semantics
from omni.isaac.sensor import Camera
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from trimesh.caching import tracked_array

from helpers.camera_helpers import CameraHelper, process_frame_data_rgb
from helpers.motion_gen_helpers import (
    IIWAPanda,
    MotionGenHelper,
    get_motion_plan_callback_fn,
)


def load_meta():
    with open("data/tasks.json", "r") as f:
        data = json.load(f)
    return data


def verify_task(task_name, object, data):
    # check if object is in the task
    if task_name not in data:
        raise ValueError(f"Task {task_name} not found in the data.")
    if object not in data[task_name]:
        return False
    return True


def load_scene(cfg: DictConfig):
    # Handle and merge config
    scene_id, scene_stage_id = cfg.scene_id.split("/")
    scene_usd_path = os.path.join("data", "scenes", scene_id, "Collected_scene/scene.usd")  # Portalble USD for release
    scene_cfg_path = os.path.join("data", "scenes", scene_id, "meta.yaml")
    layout_cfg_path = os.path.join(
        "data",
        "scenes",
        scene_id,
        f"cfg_{scene_stage_id}",
        f"{cfg.task_name}-{cfg.i_cfg}.yaml",
    )
    scene_cfg = yaml.load(open(scene_cfg_path, "r"), Loader=yaml.FullLoader)
    layout_cfg = yaml.load(open(layout_cfg_path, "r"), Loader=yaml.FullLoader)

    with open_dict(cfg):
        cfg.scene = scene_cfg
        cfg.layout = layout_cfg

    cfg.task_name = cfg.task_name.replace("_", " ")

    # Load assets
    object_pos = []
    candidate_objects = []
    add_reference_to_stage(usd_path=scene_usd_path, prim_path=cfg.scene_prim_path)
    logger.info(f"Loaded scene from config: {scene_cfg_path}. Executing task(s): {cfg.task_name}.")

    target_table_prim = get_prim_at_path(os.path.join(cfg.scene_prim_path, scene_stage_id))
    target_transf = UsdGeom.Xformable(target_table_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    target_table_geom_prim = get_prim_at_path(os.path.join(cfg.scene_prim_path, scene_stage_id, "base_link", "visuals"))
    table_verts = np.asarray(UsdGeom.Mesh(target_table_geom_prim).GetPointsAttr().Get())
    table_height_world = table_verts[:, 2].max()

    # Load OmniGibson key
    og_dir = os.environ.get("OG_DIR", "./OmniGibson")
    os.environ["KEY_PATH"] = f"{og_dir}/omnigibson/data/omnigibson.key"

    for i_obj, obj_cfg in enumerate(layout_cfg):
        obj_cfg_lvis_id = obj_cfg["object_lvis_cat"]
        obj_cfg_og_id = obj_cfg["object_og_obj_id"]
        obj_position = np.array(obj_cfg["object_position"])

        obj_position = np.matmul(np.array(target_transf.ExtractRotationMatrix()), obj_position) + np.array(target_transf.ExtractTranslation())
        obj_rot = target_transf.ExtractRotation().GetQuaternion()
        object_pos.append(obj_position)

        obj_name, obj_id = obj_cfg_og_id.split("-")
        encr_obj_usd_path = os.path.join(og_dir, "omnigibson/data/og_dataset/objects", obj_name, obj_id, "usd", f"{obj_id}.encrypted.usd")
        # obj_usd_path = os.path.join("/tmp", f"{obj_name}-{obj_id}.usd")
        obj_usd_path = encr_obj_usd_path.replace(".encrypted.usd", ".usd")
        og_decrypt_file(encr_obj_usd_path, obj_usd_path)

        obj_prim_path = f"/Object/{obj_name}"
        obj_prim = add_reference_to_stage(usd_path=obj_usd_path, prim_path=obj_prim_path)
        obj_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*obj_position))
        obj_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(obj_rot.GetReal(), obj_rot.GetImaginary()))
        obj_baselink_prim = get_prim_at_path(f"{obj_prim_path}/base_link")
        obj_baselink_prim.GetAttribute("physxArticulation:articulationEnabled").Set(False)

        for child in obj_baselink_prim.GetChildren():
            if child.GetName().endswith("joint"):
                joint = UsdPhysics.FixedJoint(child)
                joint.GetJointEnabledAttr().Set(False)
                print(f"Disabled joint {child.GetName()}")

        remove_all_semantics(obj_prim)
        add_update_semantics(obj_prim, obj_cfg_lvis_id)
        candidate_objects.append(obj_prim_path)

        logger.info(f"Added object {obj_cfg_lvis_id} to the scene at position {obj_position}.")

    return cfg, candidate_objects, np.stack(object_pos, axis=0), table_height_world


def process_target_name(target: str) -> str:
    parts = target.split("_")

    if len(parts) < 2:
        return target
    try:
        last_num = int(parts[-1])
        second_last_num = int(parts[-2])
        return "_".join(parts[:-1])
    except ValueError:
        try:
            last_num = int(parts[-1])
            return target
        except ValueError:
            return target


def delete_objects_from_collision_mesh(world_config, obj_list):
    mesh_world_config = world_config.get_mesh_world()
    mesh_dict = {}
    for mesh in mesh_world_config.mesh:
        for obj in obj_list:
            if obj in mesh.name:
                mesh_world_config.mesh.remove(mesh)
    return mesh_world_config


@hydra.main(version_base=None, config_name="config", config_path="cfg")
def main(cfg: DictConfig):
    logdir = prep_logdir()
    meta_data = load_meta()

    seed = cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Setup stage
    world = World()

    ## Setup scene
    scene_id, scene_stage_id = cfg.scene_id.split("/")
    camera_idx = scene_stage_id
    scene_stage_id = process_target_name(scene_stage_id)
    cfg, candidate_objects, obj_pos, table_height = load_scene(cfg)

    ## Setup robot
    robot = IIWAPanda(
        cfg,
        prim_path=cfg.robot_prim_path,
        name="iiwa_panda",
        usd_path=cfg.robot_usd_path,
    )
    world.scene.add(robot)
    camera = CameraHelper(cfg.camera, logdir)
    camera.init_robot_cam(cfg.robot_cam_prim_path)

    afford_rsn = AffordanceHelper(cfg.affordance, logdir, seed=seed)
    graspnet = GraspNetHelper(cfg.graspnet, logdir)
    motion_gen = MotionGenHelper(cfg.motion_gen, cfg.scene, logdir)

    gt_obj_pos = obj_pos[np.random.choice([0, 1])]

    # Simulation
    world.reset()
    robot.set_joint_positions(np.array(robot.inspection_q + [0.04, 0.04]))
    robot_base_xyz, robot_base_rpy = (
        np.array([*np.array(np.array(cfg.scene.surfaces[camera_idx].spawn_base))[:3]]),
        np.array([0, 0, np.array(cfg.scene.surfaces[camera_idx].spawn_base)[3] * np.pi / 180]),
    )
    base_placement_axis = cfg.scene.surfaces[camera_idx].placement_axis
    robot_base_xyz[base_placement_axis] = gt_obj_pos[base_placement_axis]  # + np.random.uniform(-0.1, 0.1)

    robot.set_base_pose(robot_base_xyz, robot_base_rpy)
    robot_base_rotmat = rpy_to_matrix(robot_base_rpy)
    motion_gen.update_robot_pose(
        torch.from_numpy(robot_base_xyz),
        transforms.matrix_to_quaternion(torch.from_numpy(robot_base_rotmat)),
    )

    world.step(render=True)
    robot.setup_magic_grasp(world.stage)

    camera_path = f"/World/viewer_camera_{camera_idx}"
    vp.utility.get_active_viewport().camera_path = camera_path

    record_frame = None
    if cfg.capture:
        rec_queue = Queue()
        rec_path = os.path.join(logdir, "record.mp4")
        rec_process = Process(target=export_to_cam, args=(rec_queue, 60, [1920, 1080], rec_path))
        rec_cam = Camera(prim_path=camera_path, resolution=(1920, 1080))
        rec_cam.initialize()
        rec_cam.add_motion_vectors_to_frame()
        rec_process.start()

        def record_frame():
            frame = process_frame_data_rgb(rec_cam.get_current_frame())
            if "rgb" in frame:
                rec_queue.put(frame["rgb"])

    def hold_sim(steps, reason="Having a break."):
        logger.info(reason)
        for _ in trange(steps, desc=reason):
            world.step(render=True)
            if cfg.capture:
                record_frame()

    def termination(success=False):
        if success:
            logger.info(f"Task completed successfully.")
        else:
            logger.error(f"Terminated")
        if cfg.capture:
            rec_queue.put(None)
            rec_process.join()

        app.close()
        exit()

    usd_helper = UsdHelper()
    usd_helper.load_stage(world.stage)

    world_config = usd_helper.get_obstacles_from_stage(
        only_paths=[os.path.join(cfg.scene_prim_path, scene_stage_id), "/Object"],
        ignore_substring=["visuals"],
        reference_prim_path="/Robot/iiwa7_link_0",
    )
    world_config = delete_objects_from_collision_mesh(world_config, ["clipboard"])
    world_config = world_config.get_obb_world()

    scene_dir = os.path.join(logdir, "scene")
    os.makedirs(scene_dir, exist_ok=True)
    motion_gen.setup_collider(world_config, "iiwa_panda.yml")

    # Save the scene
    world_config.save_world_as_mesh(os.path.join(logdir, "scene/extracted_collision_mesh.obj"))

    # Enumerating through each tasks.
    logger.info(f"Task: [{cfg.task_name}]")
    hold_sim(50)

    ## Stage 2: Query affordance reasoning model for object selection
    cam_frame_data = camera.frame_data
    img, depth = cam_frame_data["rgb"], cam_frame_data["depth"]
    bb_list, masks, scores = afford_rsn.affordance_query(cam_frame_data["rgb"], cfg.task_name, cam_frame_data["depth"])
    target_bb = [int(b) for b in bb_list]
    target_score = scores[0][0]
    target_mask = masks

    selected_obj = list(cam_frame_data["bb"].keys())[select_bb(list(cam_frame_data["bb"].values()), target_bb)[1]]
    selected_obj = selected_obj.split("/")[2]
    check_object = verify_task(cfg.task_name, selected_obj, meta_data)

    if target_score < cfg.aff_threshold_soft:
        logger.warning(
            f"Suboptimal affordance grounding for task: [{cfg.task_name}]. The best candidate has the score: {target_score}, which is lower than the soft threshold {cfg.aff_threshold_soft}."
        )
    torch.cuda.empty_cache()

    ## Stage 3: Approach the object for grasp planning
    n_graps_plan_captures = cfg.scene.surfaces[camera_idx].get("n_graps_plan_captures", 3)
    n_max_grasps_plan_failure = 100

    imgs, depths, target_bbs, target_masks, extrinsicses = [], [], [], [], []
    while n_graps_plan_captures > 0:
        inspect_pos, inspect_quat = camera.compute_inspect_pose_group(depth, target_bb, 256)
        inspect_pos, inspect_quat = (
            torch.tensor(inspect_pos, dtype=torch.float32, device=cfg.device),
            torch.tensor(inspect_quat, dtype=torch.float32, device=cfg.device),
        )

        for i_grasp in range(len(inspect_pos)):
            curr_q = torch.tensor(robot.get_joint_positions()[:-2], dtype=torch.float32, device=cfg.device)

            succ, q_traj = motion_gen.plan_to_goal_pose(
                curr_q,
                inspect_pos[i_grasp],
                inspect_quat[i_grasp],
                callback_fn=get_motion_plan_callback_fn(world, robot, record_callback=record_frame),
            )

            if not succ:
                logger.warning(f"Inspect pose plan failed; {n_graps_plan_captures} captures remaining")
                n_max_grasps_plan_failure -= 1
                if n_max_grasps_plan_failure <= 0:
                    termination()
                continue

            n_graps_plan_captures -= 1

            cam_frame_data = camera.frame_data
            imgs.append(cam_frame_data["rgb"])
            depths.append(cam_frame_data["depth"])
            extrinsicses.append(camera.extrinsics_matrix)

            target_score = 0.0
            while target_score < cfg.aff_threshold_soft:
                img_proc, bb_list, masks, scores = afford_rsn.affordance_query(
                    cam_frame_data["rgb"],
                    cfg.task_name,
                    cam_frame_data["depth"],
                    require_img=True,
                )
                target_score = scores[0][0]
            img, depth = cam_frame_data["rgb"], cam_frame_data["depth"]
            target_bb = [int(b) for b in bb_list]
            target_mask = masks

            target_bbs.append(target_bb)
            target_masks.append(target_mask)

            if n_graps_plan_captures == 0:
                break

    torch.cuda.empty_cache()

    bb_mask = np.zeros_like(depth, dtype=bool)
    bb_mask[target_bb[1] : target_bb[3], target_bb[0] : target_bb[2]] = 1.0

    grasp_transfs, _, _, scores, _ = graspnet.query_grasp_multiview(
        imgs,
        depths,
        target_bbs,
        target_masks,
        camera.intrinsics_matrix,
        extrinsicses,
        cfg.debug,
    )

    if len(grasp_transfs) == 0:
        termination()

    curr_q = torch.tensor(robot.get_joint_positions()[:-2], dtype=torch.float32, device=cfg.device)
    grasp_pos, grasp_quat = (
        torch.tensor(grasp_transfs[:, :3, 3], dtype=torch.float32, device=cfg.device),
        torch.tensor(matrix_to_wxyz(grasp_transfs[:, :3, :3]), dtype=torch.float32, device=cfg.device),
    )

    torch.cuda.empty_cache()
    succ, _ = motion_gen.plan_to_goal_pose_batch(
        curr_q,
        grasp_pos,
        grasp_quat,
        grasp_appr=False,
        callback_fn=get_motion_plan_callback_fn(world, robot, record_callback=record_frame),
    )

    if succ.float().sum() == 0:
        logger.error(f"Grasp pose plan failed.")
        termination()

    i_grasp = torch.where(succ)[0].tolist()[0]

    hold_sim(50)
    attached_obj = robot.attach_nearest_object(world.stage, candidate_objects)
    if attached_obj is None:
        logger.error(f"Failed to attach object.")
        termination()
    attached_obj = attached_obj.split("/")[2]  # = lvis label, cfg.task_name is the task name

    # Check object correctness
    check_object = verify_task(cfg.task_name, attached_obj, meta_data)
    wrong_object = (attached_obj is None) or not check_object
    if wrong_object:
        termination()

    robot.gripper.close()
    hold_sim(50)

    ## Stage 5: Transport object to the target location
    curr_q = torch.tensor(robot.get_joint_positions()[:-2], dtype=torch.float32, device=cfg.device)
    goal_q = torch.tensor(robot.inspection_q, dtype=torch.float32, device=cfg.device)
    succ, _ = motion_gen.plan_to_goal_q(
        curr_q,
        goal_q,
        callback_fn=get_motion_plan_callback_fn(world, robot, record_callback=record_frame),
    )

    hold_sim(50)

    termination(succ)
    app.close()


if __name__ == "__main__":
    main()
