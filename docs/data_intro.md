## Data Introduction

---

V2XSet data is structured as following:

```sh
V2XSet
├── train # data for training
│   ├── 2021_08_22_21_41_24  # scenario folder
│     ├── data_protocol.yaml # the simulation parameters used to collect the data in Carla
│     └──  -1 # The infra's id 
│       └── 00000.pcd - 00700.pcd # the point clouds data from timestamp 0 to 700
│       ├── 00000.yaml - 00700.yaml # corresponding metadata for each timestamp
│       ├── 00000_camera0.png - 00700_camera0.png # frontal camera images
│       ├── 00000_camera1.png - 00700_camera1.png # right rear camera images
│       ├── 00000_camera2.png - 00700_camera2.png # left rear camera images
│       └── 00000_camera3.png - 00700_camera3.png # back camera images
|      └──  112 # The connected vehicle id
├── validate  
├── test
```

### 1. Data Split
OPV2V dataset can be divided into 4 different folders: `train`, `validation`, `test`
- `train`: contains all training data
- `validate`: used for validation during training
- `test`: test set 

### 2. Scenario Database
V2XSet has 58 scenarios in total, where each of them contains data stream from different agents across different timestamps.
Each scenario is named by the time it was gathered, e.g., `2021_08_22_21_41_24`.

### 3. Agent Contents
Under each scenario folder,  the data of every intelligent agent~(i.e. infrastructure or connected automated vehicle) appearing in the current scenario is saved in different folders. Each folder is named by the agent's unique id, e.g., 1732. <strong>A negative id means infrastructure.</strong>

In each agent folder, data across different timestamps will be saved. Those timestamps are represented by five digits integers
as the prefix of the filenames (e.g., 00700.pcd). There are three types of files inside the agent folders: LiDAR point clouds (`.pcd` files), camera images (`.png` files), and metadata (`.yaml` files).

#### 3.1 Lidar point cloud
The LiDAR data is saved with Open3d package and has a postfix ".pcd" in the name. 

#### 3.2 Camera images
Each CAV and Infra is equipped with 4 RGB cameras to capture the 360 degree of view of the surrounding scene.`camera0`, `camera1`, `camera2`, and `camera3` represent the front, right rear, left rear, and back cameras respectively.

#### 3.3  Data Annotation
All the metadata is saved in yaml files. It records the following important information at the current timestamp:
- **ego information**:  Current ego pose with and without GPS noise under Carla world coordinates, ego speed in km/h, the LiDAR pose, and future planning trajectories. 
- **calibration**: The intrinsic matrix and extrinsic matrix from each camera to the LiDAR sensor.
- **objects annotation**: The pose and velocity of each surrounding human driving vehicle that has at least one point hit by the agent's LiDAR sensor. See [data annotation section](data_annotation_tutorial.md) for more details. 

### 4. Data Collection Protocol
Besides agent contents, every scenario database also has a yaml file named `data_protocol.yaml`. 
This yaml file records the simulation configuration to collect the current scenario.

