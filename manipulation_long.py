import json
import os
from multiprocessing import Process, Queue

import hydra
import numpy as np
import torch
from curobo.geom.types import WorldConfig
from loguru import logger
from omegaconf import DictConfig
from tqdm import trange

from utils.utils_record import export_to_cam


from omni.isaac.kit import SimulationApp

# Seed everything
from scipy.spatial.transform import Rotation as R

from helpers.affordance_helpers import AffordanceHelper
from helpers.grasping_helpers import GraspNetHelper
from utils.utils_3d import (
    matrix_to_wxyz,
)
from utils.utils_misc import prep_logdir

os.environ["CUROBO_TORCH_CUDA_GRAPH_RESET"] = "1"
os.environ["CUROBO_USE_LRU_CACHE"] = "1"

app = SimulationApp({"headless": False})

import omni.kit.viewport as vp
from curobo.util.usd_helper import UsdHelper
from omni.isaac.core import World
from omni.isaac.core.utils.prims import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

from helpers.camera_helpers import CameraHelper, process_frame_data_rgb
from helpers.motion_gen_helpers import (
    MotionGenHelper,
    RidgebackFranka,
    get_motion_plan_callback_fn,
)


def load_scene(cfg: DictConfig):
    scene_usd_path = "data/scenes/long_horizon/final.usd"
    scene_setup = json.load(open("data/scenes/long_horizon/config.json", "r"))
    scene_setup = DictConfig(scene_setup)

    add_reference_to_stage(usd_path=scene_usd_path, prim_path=cfg.scene_prim_path)
    logger.info(f"Loaded scene from USD: {scene_usd_path}. Executing task(s).")

    candidate_objects = scene_setup.candidate_object_prims

    return cfg, scene_setup, candidate_objects


@hydra.main(version_base=None, config_name="config", config_path="cfg")
def main(cfg: DictConfig):
    logdir = prep_logdir()

    seed = cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Setup stage
    world = World()

    ## Setup scene
    cfg, scene_cfg, candidate_objects = load_scene(cfg)

    ## Setup robot
    robot = RidgebackFranka(
        cfg,
        prim_path=cfg.robot_prim_path,
        name="ridgeback_panda",
        usd_path="assets/ridgeback_franka/ridgeback_franka.usd",
    )
    world.scene.add(robot)
    camera = CameraHelper(cfg.camera, logdir)
    camera.init_robot_cam(cfg.robot_cam_prim_path)

    afford_rsn = AffordanceHelper(cfg.affordance, logdir, seed, "cuda:1")
    graspnet = GraspNetHelper(cfg.graspnet, logdir)
    motion_gen = MotionGenHelper(cfg.motion_gen, None, logdir)

    world.reset()
    robot.set_joint_positions(np.array(scene_cfg.bot_root + robot.inspection_q + [0.04, 0.04]))

    world.step(render=True)

    camera_paths = [
        "/World/viewer_camera_0",
        "/World/viewer_camera_1",
        cfg.robot_cam_prim_path,
    ]
    rec_cams, rec_procs, rec_queues = [], [], []
    vp.utility.get_active_viewport().camera_path = camera_paths[-1]

    record_frame = None
    if cfg.capture:
        for i_cam, camera_path in enumerate(camera_paths):
            rec_queue = Queue()
            rec_queues.append(rec_queue)
            rec_path = os.path.join(logdir, f"record-{i_cam}.avi")
            rec_proc = Process(
                target=export_to_cam,
                args=(
                    rec_queue,
                    60,
                    [1920, 1080],
                    rec_path,
                ),
            )
            rec_procs.append(rec_proc)
            rec_cam = Camera(prim_path=camera_path, resolution=(1920, 1080))
            rec_cam.initialize()
            rec_cam.add_motion_vectors_to_frame()
            rec_proc.start()

            rec_cams.append(rec_cam)

        def record_frame():
            for rec_cam, rec_queue in zip(rec_cams, rec_queues):
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
        if cfg.capture:
            for rec_queue in rec_queues:
                rec_queue.put(None)
            for rec_proc in rec_procs:
                rec_proc.join()

        app.close()
        exit()

    usd_helper = UsdHelper()
    usd_helper.load_stage(world.stage)

    world_config = WorldConfig.from_dict({
        "mesh": {
            "scene": {
                "pose": [0, 0, 0, 1, 0, 0, 0],
                "file_path": "data/scenes/long_horizon/scene_collision.stl",
            }
        },
    })
    # cuRobo loading problem
    world_config.mesh[0].file_path = "data/scenes/long_horizon/scene_collision.stl"

    world_config = world_config.get_mesh_world()

    scene_dir = os.path.join(logdir, "scene")
    os.makedirs(scene_dir, exist_ok=True)
    motion_gen.setup_collider(world_config, "ridgeback_franka.yaml")

    hold_sim(50)
    for task_dict in scene_cfg.scripts:
        ## Stage 1: Initialize the robot
        robot.setup_magic_grasp(world.stage)

        task = task_dict["prompt"]
        logger.info(f"Task: [{task}]")
        hold_sim(50)

        ## Stage 2: Query affordance reasoning model for object selection
        cam_frame_data = camera.frame_data
        img, depth = cam_frame_data["rgb"], cam_frame_data["depth"]

        bb_list, masks, scores = afford_rsn.affordance_query(cam_frame_data["rgb"], task, cam_frame_data["depth"])
        target_bb = [int(b) for b in bb_list]
        target_score = scores[0][0]
        target_mask = masks

        if target_score < cfg.aff_threshold_soft:
            logger.warning(
                f"Suboptimal affordance grounding for task: [{task}]. The best candidate has the score: {target_score}, which is lower than the soft threshold {cfg.aff_threshold_soft}."
            )
        torch.cuda.empty_cache()

        ## Stage 3: Approach the object for grasp planning
        start_q = torch.tensor(robot.get_joint_positions()[:10], dtype=torch.float32, device=cfg.device)
        inspect_pos, inspect_quat = camera.compute_inspect_pose(depth, target_bb)
        inspect_pos, inspect_quat = (
            torch.tensor(inspect_pos, dtype=torch.float32, device=cfg.device),
            torch.tensor(inspect_quat, dtype=torch.float32, device=cfg.device),
        )
        succ, _ = motion_gen.plan_to_goal_pose(
            start_q,
            inspect_pos,
            inspect_quat,
            callback_fn=get_motion_plan_callback_fn(world, robot, record_callback=record_frame),
        )

        if not succ:
            termination(False)
        torch.cuda.empty_cache()

        ## Stage 4: Plan and execute the grasp
        cam_frame_data = camera.frame_data
        bb_list, masks, scores = afford_rsn.affordance_query(cam_frame_data["rgb"], task, cam_frame_data["depth"])
        img, depth = cam_frame_data["rgb"], cam_frame_data["depth"]
        target_score = scores[0][0]
        target_bb = [int(b) for b in bb_list]
        target_mask = masks

        bb_mask = np.zeros_like(depth, dtype=bool)
        bb_mask[target_bb[1] : target_bb[3], target_bb[0] : target_bb[2]] = 1.0

        extrinsics = camera.extrinsics_matrix
        grasp_transfs, grasp_transfs_local, widths, scores, [gg, full_cloud_o3d] = graspnet.query_grasp(
            img,
            depth,
            target_bb,
            target_mask,
            camera.intrinsics_matrix,
            extrinsics,
            cfg.debug,
        )

        if len(grasp_transfs) == 0:
            termination(False)

        curr_q = torch.tensor(robot.get_joint_positions()[:-2], dtype=torch.float32, device=cfg.device)
        grasp_pos, grasp_quat = (
            torch.tensor(grasp_transfs[:, :3, 3], dtype=torch.float32, device=cfg.device),
            torch.tensor(
                matrix_to_wxyz(grasp_transfs[:, :3, :3]),
                dtype=torch.float32,
                device=cfg.device,
            ),
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
            termination(False)
        i_grasp = torch.where(succ)[0].tolist()[0]

        # hold_process(30)
        attached_obj = robot.attach_nearest_object(world.stage, candidate_objects, mode="pose")
        if attached_obj is None:
            termination(False)

        hold_sim(30)
        grasp_width = np.clip(widths[i_grasp].item() * 0.75, 0.0, 0.08)
        for _ in range(30):
            robot.gripper.apply_action(ArticulationAction([grasp_width / 2, grasp_width / 2], None))
            world.step(render=True)

        ## Stage 5: Transport object to the target location
        curr_q = torch.tensor(robot.get_joint_positions()[:-2], dtype=torch.float32, device=cfg.device)
        goal_q = torch.tensor(
            [
                -0.7,
                -2.9,
                np.pi / 2,
                0.0,
                np.pi / 6,
                0.0,
                -np.pi / 4,
                0.0,
                np.pi / 2,
                np.pi / 4,
            ],
            dtype=torch.float32,
            device=cfg.device,
        )
        succ, _ = motion_gen.plan_to_goal_q(
            curr_q,
            goal_q,
            callback_fn=get_motion_plan_callback_fn(world, robot, record_callback=record_frame),
        )

        if not succ:
            termination(False)

        hold_sim(30)
        robot.release_grasped_object()
        hold_sim(30)

        # Stage 6: Going back
        curr_q = torch.tensor(robot.get_joint_positions()[:-2], dtype=torch.float32, device=cfg.device)
        goal_q = torch.tensor(
            scene_cfg.bot_root + robot.inspection_q,
            dtype=torch.float32,
            device=cfg.device,
        )
        succ, _ = motion_gen.plan_to_goal_q(
            curr_q,
            goal_q,
            callback_fn=get_motion_plan_callback_fn(world, robot, record_callback=record_frame),
        )

        if not succ:
            termination(False)
        hold_sim(30)

        torch.cuda.empty_cache()

    termination(True)


if __name__ == "__main__":
    main()
