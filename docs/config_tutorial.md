## Tutorial: Config System in V2X-ViT

---
We incorporate modular and inheritance design into the config system to enable users conveniently
modify the model/training/inference parameters. Specifically, we use **yaml** files to configure all the 
important parameters.

### Config File Location
To train a model from scratch, all the yaml files should be saved in `v2xvit/hypes_yaml`, and users should use the `load_yaml()` function in [`v2xvit/hypes_yaml/yaml_utils.py`](https://github.com/DerrickXuNu/OpenCOOD/blob/main/opencood/hypes_yaml/yaml_utils.py#L8) to load the parameters into a dictionary.

To train from a saved checkpoint or test, the yaml files should be located with the checkpoint together in the `v2xvit/logs/model_name`.

### Config Name Style
We follow the below style to name config yaml files.
```python
{backbone}_{fusion_strategy}.yaml
```

### Noise Simulation
Communication noise simulation is defined in the `wild_setting` group in the yaml file.
```
wild_setting: # setting related to noise
  async: true
  async_mode: 'sim'
  async_overhead: 100
  backbone_delay: 10
  data_size: 1.06
  loc_err: true
  ryp_std: 0.2
  seed: 25
  transmission_speed: 27
  xyz_std: 0.2
```
`async`: whether add communication delay. <br>
`aysnc_mode`:  sim or real mode. In sim mode, the delay is a constant while in real mode, the delay has a uniform distribution.
The major experiment in the paper used sim mode whereas the 'Effects of transmission size' study used real
 mode. <br>
`async_overhead`: the communication delay in ms. In sim mode, it represents a constant number. In real mode,
the systematic async will be a random number from 0 to `aysnc_overhead`. <br>
`backbone_delay`: an estimate of backbone computation time. Only useful in real mode. <br>
`data_size`: transmission data size in Mb. Only used in real mode. <br>
`transmission_speed`: data transmitting speed during communication. By default 27 Mb/s. Only used in real mode. <br>
`loc_err`: whether to add localization error. <br>
`xyz_std`: the standard deviation of positional GPS error. <br>
`ryp_std`: the standard deviation of angular GPS error. <br>
`seed`: random seed for noise simulation. <strong>please keep it as 25 during testing </strong>.




### A concrete example
Now let's go through the `point_pillar_opv2v_fusion.yaml` as an example.

```yaml
name: point_pillar_intermediate_fusion # this parameter together with the current timestamp will  define the name of the saved folder for the model. 
root_dir: "v2xset/train" # this is where the training data locate
validate_dir: "v2xset/validate" # during training, it defines the validation folder. during testing, it defines the testing folder path.

yaml_parser: "load_point_pillar_params" # we need specific loading functions for different backbones.
train_params: # the common training parameters
  batch_size: &batch_size 2
  epoches: 60
  eval_freq: 1
  save_freq: 1

wild_setting: # setting related to noise
  async: true
  async_mode: 'sim'
  async_overhead: 100
  backbone_delay: 10
  data_size: 1.06
  loc_err: true
  ryp_std: 0.2
  seed: 25
  transmission_speed: 27
  xyz_std: 0.2

fusion:
  core_method: 'IntermediateFusionDataset' # LateFusionDataset, EarlyFusionDataset, and IntermediateFusionDataset are supported
  args:
    cur_ego_pose_flag: True
    # when the cur_ego_pose_flag is set to True, there is no time gap
    # between  the time when the LiDAR data is captured by connected
    # agents and when the extracted features are received by
    # the ego vehicle, which is equal to implement STCM. When set to False,
    # STCM has to be used. To validate STCM, V2X-ViT will set this as False.

# preprocess-related
preprocess:
  # options: BasePreprocessor, SpVoxelPreprocessor, BevPreprocessor
  core_method: 'SpVoxelPreprocessor'
  args:
    voxel_size: &voxel_size [0.4, 0.4, 4] # the voxel resolution for PointPillar
    max_points_per_voxel: 32 # maximum points allowed in each voxel
    max_voxel_train: 32000 # the maximum voxel number during training
    max_voxel_test: 70000 # the maximum voxel number during testing
  # LiDAR point cloud cropping range
  cav_lidar_range: &cav_lidar [-140.8, -40, -3, 140.8, 40, 1]

# data augmentation options.
data_augment:
  - NAME: random_world_flip
    ALONG_AXIS_LIST: [ 'x' ]

  - NAME: random_world_rotation
    WORLD_ROT_ANGLE: [ -0.78539816, 0.78539816 ]

  - NAME: random_world_scaling
    WORLD_SCALE_RANGE: [ 0.95, 1.05 ]

# post processing related.
postprocess:
  core_method: 'VoxelPostprocessor' # VoxelPostprocessor and BevPostprocessor are supported
  anchor_args: # anchor generator parameters
    cav_lidar_range: *cav_lidar # the range is consistent with the lidar cropping range to generate the correct ancrhors
    l: 3.9 # the default length of the anchor
    w: 1.6 # the default width
    h: 1.56 # the default height
    r: [0, 90] # the yaw angles. 0, 90 meaning for each voxel, two anchors will be generated with 0 and 90 degree yaw angle
    feature_stride: 2 # the feature map is shrank twice compared the input voxel tensor
    num: &achor_num 2 # for each location in the feature map, 2 anchors will be generated
  target_args: # used to generate positive and negative samples for object detection
    pos_threshold: 0.6 
    neg_threshold: 0.45
    score_threshold: 0.20
  order: 'hwl' # hwl or lwh
  max_num: 100 # maximum number of objects in a single frame. use this number to make sure different frames have the same dimension in the same batch
  nms_thresh: 0.15

# model related
model:
  core_method: point_pillar_opv2v # trainer will load the corresponding model python file with the same name
  args: # detailed parameters of the point pillar model
    voxel_size: *voxel_size 
    lidar_range: *cav_lidar
    anchor_number: *achor_num

    pillar_vfe:
      use_norm: true
      with_distance: false
      use_absolute_xyz: true
      num_filters: [64]
    point_pillar_scatter:
      num_features: 64

    base_bev_backbone:
      layer_nums: [3, 5, 8]
      layer_strides: [2, 2, 2]
      num_filters: [64, 128, 256]
      upsample_strides: [1, 2, 4]
      num_upsample_filter: [128, 128, 128]
      compression: 0 # whether to compress the features before fusion to reduce the bandwidth
      backbone_fix: false # whether fix the pointpillar backbone weights during training.
    anchor_num: *achor_num

loss: # loss function
  core_method: point_pillar_loss # trainer will load the loss function with the same name
  args:
    cls_weight: 1.0 # classification weights
    reg: 2.0 # regression weights

optimizer: # optimzer setup
  core_method: Adam # the name has to exist in Pytorch optimizer library
  lr: 0.002
  args:
    eps: 1e-10
    weight_decay: 1e-4

lr_scheduler: # learning rate schedular
  core_method: multistep #step, multistep and Exponential are supported
  gamma: 0.1
  step_size: [15, 30]

```