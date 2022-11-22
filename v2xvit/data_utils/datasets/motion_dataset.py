"""
Dataset class for motion
"""
import math
import random
from collections import OrderedDict

import numpy as np
import torch

import v2xvit
import v2xvit.data_utils.post_processor as post_processor
from v2xvit.hypes_yaml.yaml_utils import load_yaml
from v2xvit.utils import box_utils, pcd_utils
from v2xvit.data_utils.datasets import basedataset, IntermediateFusionDataset
from v2xvit.data_utils.pre_processor import build_preprocessor
from v2xvit.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum, downsample_lidar


from v2xvit.utils.transformation_utils import x1_to_x2


class MotionDataset(basedataset.BaseDataset):
    # 32 scenario, 96 cars in total, 21086 frames in total
    # 11037 frames in total in need, 10 epoch in need
    def __init__(self, params, visualize, train=True):
        super(MotionDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)
        self.N = params['motion_method']['args']['N']  # use N history frams, default 4
        self.k_range = params['motion_method']['args']['k']  # random k in possible k k_range
        self.frame_database = []
        for i, scenario in self.scenario_database.items():
            for j, car in scenario.items():
                car_frames = list(car.values())
                for f_id in range(0, len(car_frames) - self.N - max(self.k_range), 2):
                    tem = OrderedDict()
                    k = random.choice(self.k_range)  # also for k in k_range
                    tem.update({'history': car_frames[f_id:f_id + self.N],
                                'target': car_frames[f_id + self.N + k - 1],
                                'deltaT': k})
                    self.frame_database.append(tem)

    def __len__(self):
        return len(self.frame_database)

    def __getitem__(self, idx):
        # get history frames from each car, and target is current frame
        base_data_dict = self.frame_database[idx]
        history = base_data_dict['history']
        target = base_data_dict['target']
        deltaT = base_data_dict['deltaT']
        history_feature = []
        history_speed = []

        params_tar = load_yaml(target['yaml'])
        tar_pose = params_tar['lidar_pose']
        target_pcd = pcd_utils.pcd_to_np(target['lidar'])
        target_pcd = shuffle_points(target_pcd)
        target_pcd = mask_ego_points(target_pcd)
        target_pcd = mask_points_by_range(target_pcd, self.params['preprocess']['cav_lidar_range'])
        target_feature = [self.pre_processor.preprocess(target_pcd)]

        for frame in history:
            params = load_yaml(frame['yaml'])
            velocity = params['ego_speed']
            velocity /= 30.0
            history_speed.append(velocity)
            cav_pose = params['lidar_pose']
            transformation_matrix = x1_to_x2(cav_pose, tar_pose)

            pcd = pcd_utils.pcd_to_np(frame['lidar'])
            pcd = shuffle_points(pcd)
            pcd = mask_ego_points(pcd)
            pcd[:, :3] = box_utils.project_points_by_matrix_torch(pcd[:, :3], transformation_matrix)
            pcd = mask_points_by_range(pcd, self.params['preprocess']['cav_lidar_range'])
            precessed_pcd = self.pre_processor.preprocess(pcd)
            history_feature.append(precessed_pcd)

        merge_history_feature = IntermediateFusionDataset.merge_features_to_dict(history_feature)
        merge_target_feature = IntermediateFusionDataset.merge_features_to_dict(target_feature)

        data = OrderedDict()
        data.update({'sources': merge_history_feature})
        data.update({'target': merge_target_feature})
        data.update({'deltaT': deltaT})
        data.update({'sources_speed': history_speed})
        data.update({'target_speed': [load_yaml(target['yaml'])['ego_speed'] / 30.0]})
        return data

    def collate_batch_train(self, batch):
        """
        Collate the batch data for training.

        Parameters
        ----------
        batch : list
            A list of data dictionary.

        Returns
        -------
        batch_data : dict
            The batch data dictionary.
        """
        batch_data = OrderedDict()
        sources = []
        targets = []
        deltaTs = []
        sources_speed = []
        targets_speed = []
        for i in range(len(batch)):
            sources.append(batch[i]['sources'])
            targets.append(batch[i]['target'])
            deltaTs.append(batch[i]['deltaT'])
            sources_speed.append(batch[i]['sources_speed'])
            targets_speed.append(batch[i]['target_speed'])
        processed_sources = IntermediateFusionDataset.merge_features_to_dict(sources)
        processed_targets = IntermediateFusionDataset.merge_features_to_dict(targets)
        processed_sources = self.pre_processor.collate_batch(processed_sources)
        processed_targets = self.pre_processor.collate_batch(processed_targets)
        # dict:3
        #  'voxel_features': Tensor:(B*N*size, 32, 4)
        #  'voxel_coords': Tensor:(B*N*size, 4)
        #  'voxel_num_points': Tensor:(B*N*size)
        batch_data.update({'sources': processed_sources})  # warning: need to reshape to BL...
        # dict:3
        #  'voxel_features': Tensor:(B*size, 32, 4)
        #  'voxel_coords': Tensor:(B*size, 4)
        #  'voxel_num_points': Tensor:(B*size)
        batch_data.update({'target': processed_targets})
        batch_data.update({'deltaT': torch.Tensor(deltaTs)})  # list:batch_size
        batch_data.update({'sources_speed': torch.Tensor(sources_speed)})  # list:batch_size
        batch_data.update({'target_speed': torch.Tensor(targets_speed)})  # list:batch_size
        return batch_data
