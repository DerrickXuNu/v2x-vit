import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from v2xvit.models.sub_modules.pillar_vfe import PillarVFE
from v2xvit.models.sub_modules.point_pillar_scatter import PointPillarScatter
from v2xvit.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from v2xvit.models.sub_modules.fuse_utils import regroup
from v2xvit.models.sub_modules.downsample_conv import DownsampleConv
from v2xvit.models.sub_modules.naive_compress import NaiveCompressor
from v2xvit.models.sub_modules.v2xvit_basic import V2XTransformer


class PointPillarTransformer(nn.Module):
    def __init__(self, args):
        super(PointPillarTransformer, self).__init__()

        self.max_cav = args['max_cav']
        try:
            self.learnable_backbone = args['learnable_backbone']
        except KeyError:
            self.learnable_backbone = False
        try:
            self.learnable_motion = args['learnable_motion']
        except KeyError:
            self.learnable_motion = False
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = V2XTransformer(args['transformer'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

        self.his_len = args.get('his_len', 0)
        self.use_motion = args.get('use_motion', False)
        self.motion_predict = None
        print('learnable backbone: ', self.learnable_backbone)
        print('learnable motion: ', self.learnable_motion)


    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def motion_fix(self):
        for p in self.motion_predict.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # B, max_cav, 3(dt dv infra), 1, 1 # B, max_cav, 3(v delay infra), 1, 1
        prior_encoding = \
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        if not self.use_motion:
            batch_dict = self.pillar_vfe(batch_dict)
            # n, c -> N, C, H, W
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)

            spatial_features_2d = batch_dict['spatial_features_2d']
            # downsample feature to reduce memory
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            # compressor
            if self.compression:
                spatial_features_2d = self.naive_compressor(spatial_features_2d)
            # N, C, H, W -> B,  L, C, H, W
            regroup_feature, mask = regroup(spatial_features_2d,
                                            record_len,
                                            self.max_cav)
        # motion predict added here
        if self.use_motion:
            if self.learnable_backbone:
                motion = self.vox2spatial(data_dict['history'], need_nchw=True)
                # motion is NCHW, need to -> B,4,CHW
                motion = rearrange(motion, '(b l) c h w -> b l c h w', l=self.his_len)
                original_time_delay = data_dict['original_time_delay'].view(motion.shape[0], -1)
            else:
                with torch.no_grad():
                    motion = self.vox2spatial(data_dict['history'], need_nchw=True)
                    # motion is NCHW, need to -> B,4,CHW
                    motion = rearrange(motion, '(b l) c h w -> b l c h w', l=self.his_len)
                    original_time_delay = data_dict['original_time_delay'].view(motion.shape[0], -1)
            if self.learnable_motion:
                target = self.motion_predict(motion, original_time_delay).squeeze(1)  # BCHW
                target = torch.split(target, record_len.view(-1).tolist(), dim=0)

            else:
                with torch.no_grad():
                    target = self.motion_predict(motion, original_time_delay).squeeze(1)  # BCHW
                    target = torch.split(target, record_len.view(-1).tolist(), dim=0)
            # target is list [N, C, H, W]
            mask = []
            features = []
            for split_feature in target:
                feature_shape = split_feature.shape
                padding_len = self.max_cav - feature_shape[0]
                mask.append([1] * feature_shape[0] + [0] * padding_len)
                padding_tensor = torch.zeros(padding_len, feature_shape[1],
                                             feature_shape[2], feature_shape[3])
                padding_tensor = padding_tensor.to(split_feature.device)
                split_feature = torch.cat([split_feature, padding_tensor], dim=0)
                split_feature = split_feature.unsqueeze(0)
                features.append(split_feature)
            mask = torch.from_numpy(np.array(mask)).to(target[0].device)
            regroup_feature = torch.cat(features, dim=0)
            # batch = regroup_feature.shape[0]
            # for i in range(batch):
            #     regroup_feature[i, :target[i].shape[0], :, :, :] = target[i]
        # prior encoding added
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2).contiguous()

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict

    def vox2spatial(self, batch_dict, bs=2, need_nchw = False):  # n, c -> N, C, H, W
        """
        Convert voxel features to spatial features.
        """
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        voxel_num_points = batch_dict['voxel_num_points']
        batch_size = bs

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points
                      }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # N, C, H, W -> B,  L, C, H, W
        # regroup_feature, mask = regroup(spatial_features_2d,
        #                                 torch.Tensor([self.his_len] * batch_size),
        #                                 self.his_len)
        regroup_features = rearrange(spatial_features_2d, '(b l) c h w -> b l c h w', b=batch_size)
        return regroup_features if not need_nchw else spatial_features_2d

    def load_motion(self, motion_predict):
        self.motion_predict = motion_predict
