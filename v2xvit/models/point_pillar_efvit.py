"""
Vanilla pointpillar for early and late fusion.
"""
import torch
import torch.nn as nn

from v2xvit.models.sub_modules.fuse_utils import regroup
from v2xvit.models.sub_modules.pillar_vfe import PillarVFE
from v2xvit.models.sub_modules.point_pillar_scatter import PointPillarScatter
from v2xvit.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from v2xvit.models.sub_modules.downsample_conv import DownsampleConv
from v2xvit.models.sub_modules.v2xvit_basic import V2XTransformer


class PointPillarEFVit(nn.Module):
    def __init__(self, args):
        super(PointPillarEFVit, self).__init__()

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

        self.fusion_net = V2XTransformer(args['transformer'])
        self.max_cav = 1

        self.cls_head = nn.Conv2d(args['cls_head_dim'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['cls_head_dim'],
                                  7 * args['anchor_number'],
                                  kernel_size=1)

    def forward(self, data_dict, bs=4):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']
        prior_encoding = \
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)   # 4 64 200 704
        batch_dict = self.backbone(batch_dict)  # 4 384 100 352

        spatial_features_2d = batch_dict['spatial_features_2d']
        '''
        DownsampleConv(
          (layers): ModuleList(
            (0): DoubleConv(
              (double_conv): Sequential(
                (0): Conv2d(384, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (1): ReLU(inplace=True)
                (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (3): ReLU(inplace=True)
              )
            )
          )
        )
        '''
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        regroup_feature, mask = regroup(spatial_features_2d,    #
                                        record_len,
                                        self.max_cav)
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