# Afford-X: Task-oriented Manipulation

This repo is an official implementation of the simulation studies in Afford-X. The Afford-X model is released in [GitHub: ZhuXMMM/Afford-X](https://github.com/ZhuXMMM/Afford-X).

## Setup Environment

### Install Isaac Sim

Install NVIDIA Isaac Sim following [this page](https://developer.nvidia.com/isaac/sim). Due to its active development, please install the version 2023.1.1 for compatibility.

The following environment should be set up within the Isaac Sim Python environment.

### Clone this Repo

Clone this repo with dependencies:

```shell
git clone https://github.com/YuyangLee/Afford-X
```

### Install Dependencies from PyPI

First, install dependencies from PyPI:

```shell
pip install -r requirements.txt
```

### Install Afford-X

The affordance reasoning capabilities comes from the Afford-X model. Install it according to [this section](https://github.com/ZhuXMMM/Afford-X?tab=readme-ov-file#installation).

We recommend cloning the repo into `models/affordance_reasoning`. In this way, the model should be successfully wrapped by the `AffordanceHelper` in [`helpers/affordance_helpers.py`](./helpers/affordance_helpers.py).

### Install GraspNet Baseline

Install the GraspNet module following [this section](https://github.com/graspnet/graspnet-baseline?tab=readme-ov-file#installation).

We recommand cloning the repo into `models/graspnet`, after which it should be successfully wrapped by the `GraspNetHelper` in [`helpers/grasping_helpers.py`](./helpers/grasping_helpers.py).

### Install cuRobo

Install NVIDIA cuRobo following [this docs](https://curobo.org/get_started/1_install_instructions.html).

We recommand cloning the repo into `models/curobo`, which will be wrapped by the `MotionGenHelper` in [`helpers/motion_gen_helpers.py`](helpers/motion_gen_helpers.py).

### Install OmniGibson

We perform the task-oriented manipulation experiments in Isaac Sim with assets from OmniGibson and Objaverse. According to their license, you have to acquire the data and the license from OmniGibson to run the manipulation experiments.

We will use a history version of OmniGibson for reproducability:

```shell
git clone https://github.com/StanfordVL/OmniGibson
cd OmniGibson
git checkout 7840446
python scripts/download_datasets.py
```

This will download the OG dataset into `OmniGibson/omnigibson/data`.

### Download Additional Data

In the last step, we will download some additional data that includes the task-related configs and part of the assets. Download the zip files from [this Google Drive folder](https://drive.google.com/drive/folders/1L0EEIxxMmV80_gYhSjmDPH8dpjcYOLCE?usp=sharing) and unzip the contents into `data/`:

- `scenes.zip` -> `data/scenes/`


## cuRobo Config

You first need to link the files to your cuRobo content directory. Assume its in `models/curobo`, then create the soft links:

```shell
ln -s $PWD/assets/curobo_cfg/*.yml models/curobo/src/curobo/content/configs/robot/
ln -s $PWD/assets/curobo_cfg/iiwa_panda models/curobo/src/curobo/content/assets/robot/
ln -s $PWD/assets/curobo_cfg/iiwa_panda.urdf models/curobo/src/curobo/content/assets/robot/
ln -s $PWD/assets/ridgeback_franka/ models/curobo/src/curobo/content/assets/robot/
```

## Simulation Studies

### Table-top Grasping

To simulate table-top task-oriented manipulation:

```shell
export OG_DIR=OmniGibson  # Change the path to your omnigibson directory

python manipulation.py scene_id=hotel_suite_large/desk_ohjotf_0 task_name=drink_water_with i_cfg=0 seed=42
```

Scenes current available:

- `beechwood/cfg_coffee_table_qlmqyy_1`
- `beechwood/cfg_countertop_tpuwys_6`
- `hotel_suite_large/cfg_desk_ohjotf_0`
- `merom/cfg_countertop_tpuwys_0`
- `office_vendor_machine/cfg_conference_table_jxixdw_0`
- `office_vendor_machine/cfg_shelf_zdsmhe_2`

Tasks current available:

- `Clean_electronic_screens_with`
- `drink_water_with`
- `protect_items_from_rain_with`
- `stream_video_with`
- `spread_butter_with`
- `stir_the_chocolate_with`

For each `scene_id` and `task_name`, you can specify `i_cfg` in `[0, 1, 2, 3, 4]` to apply various and random table layout, and specify different `seed` for randomization.

### Long-horizon Manipulation

To simulate long-horizon manipulation:

```shell
export OG_DIR=OmniGibson  # Change the path to your omnigibson directory

python manipulation_long.py
```

## Known Issues

- Some textures are missing in the released scene data.


## FAQ

In case of `no space left on device` error:

```shell
echo 524280 | sudo tee /proc/sys/fs/inotify/max_user_watches
```
