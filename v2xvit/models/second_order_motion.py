import torch
import torch.nn as nn
import torch.nn.functional as F
from .sub_modules.optical_flow import OpticalFlow


class SecondOrderMotion(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.optical_flow = OpticalFlow(256, model_cfg['optical_flow']['out_channels'])
        try:
            self.only_least_squares = model_cfg['only_least_squares']
        except KeyError:
            self.only_least_squares = False
        print('only least squares: {}'.format(self.only_least_squares))
        # self.register_buffer('A', torch.Tensor([[-0.5909,  0.2273,  0.0909],
        #                                         [ 0.6364, -0.0909,  0.3636]], requires_grad=False))
        # self.register_buffer('B', torch.Tensor([[-0.8158, -0.6842,  0.3947],
        #                                         [-0.5789, -0.4211,  0.4737]], requires_grad=False) )

    def forward(self, history_frames, delta_t):
        # history_frames: (B, T, C, H, W)
        frames = torch.split(history_frames, 1, dim=1)
        assert len(frames) in [4, 3]
        if len(frames) == 4:
            # EQVI, diff: v0 is f4's velocity
            f1, f2, f3, f4 = frames
            f1 = f1.squeeze(1)
            f2 = f2.squeeze(1)
            f3 = f3.squeeze(1)
            f4 = f4.squeeze(1)
            # f2f1 = self.optical_flow(f2, f1)
            # f2f3 = self.optical_flow(f2, f3)
            # f2f4 = self.optical_flow(f2, f4)
            # v = -0.5909* f2f1 + 0.2273 * f2f3 + 0.0909 * f2f4
            # a = 0.6364 * f2f1 - 0.0909 * f2f3 + 0.3636 * f2f4
            f4f3 = self.optical_flow(f4, f3)
            f4f2 = self.optical_flow(f4, f2)
            f4f1 = self.optical_flow(f4, f1)
            v_e = -0.8158 * f4f3 - 0.6842 * f4f2 + 0.3947 * f4f1
            a_e = -0.5789 * f4f3 - 0.4211 * f4f2 + 0.4737 * f4f1
            if self.only_least_squares:
                a = a_e
                v = v_e
            else:
                a_qvi = -2 * f4f3 + f4f2
                v_qvi = -2 * f4f3 + 0.5 * f4f2
                a_2 = 0.5 * f4f1 - 1.5 * f4f3
                a_3 = 2 * f4f1 / 3 - f4f2
                mask1 = a_qvi * a_2 > 0
                mask2 = a_qvi * a_3 > 0
                mask3 = a_2 * a_3 > 0
                mask = mask1 * mask2 * mask3
                alpha = self.get_alpha(a_qvi, a_2)
                v = v_e * alpha + v_qvi * (1 - alpha)
                a = a_e * alpha + a_qvi * (1 - alpha)
                v = v * mask + v_qvi * (~mask)
                a = a * mask + a_qvi * (~mask)
        else:
            # QVI, diff: v0 is f3's velocity
            f1, f2, f3 = frames
            f1 = f1.squeeze(1)
            f2 = f2.squeeze(1)
            f3 = f3.squeeze(1)
            f2f1 = self.optical_flow(f2, f1)
            f2f3 = self.optical_flow(f2, f3)
            a = f2f3 + f2f1
            v = 0.5* (f2f3 - f2f1) + a
        delta_t = delta_t.view(-1)
        delta_t = delta_t[:, None, None, None].to(frames[-1].dtype)
        pred_frame = frames[-1].squeeze(1) + delta_t * v + 0.5 * a * delta_t ** 2
        return pred_frame.unsqueeze(1)   # B 1 C H W

    def get_alpha(self, a, b, omega = 5, gamma = 1):
        # a: (B, 2, H, W)
        # b: (B, 2, H, W)
        # omega: float
        # gamma: float
        # alpha: (B, 1, H, W)
        z = (torch.abs(a - b) - gamma) * omega
        e1 = torch.exp(z)
        e2 = torch.exp(-z)
        alpha = -0.5 * (e1 - e2) / (e1 + e2) + 0.5
        return alpha
