import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./model')
from base_backbone import BaseBackbone
from vit import VisionTransformerMul3DMB
from mlp import MLP
from mmcv.runner.base_module import ModuleList


class FusionModel(BaseBackbone):
    def __init__(self, vit_cfg={"kernel_size": (1, 1, 1),
                                "in_channels": 1,
                                "strides": ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                                "out_indices": -1,
                                "qkv_bias": True,
                                "drop_rate": 0.0,
                                "drop_path_rate": 0.3,
                                "patch_cfg": dict(input_size=(64, 64, 64)),
                                "init_cfg": dict(
                                    type='Pretrained',
                                    checkpoint='')
                                },
                 mlp_cfg={"input_dim": 20,
                          "hidden_dims": [64],
                          "output_dim": 2,
                          "dropout_rate": 0.2,
                          "init_cfg": None},
                 init_cfg=None):
        super(FusionModel, self).__init__(init_cfg)

        self.vit = VisionTransformerMul3DMB(kernel_size=vit_cfg['kernel_size'],
                                            in_channels=vit_cfg['in_channels'],
                                            strides=vit_cfg['strides'],
                                            out_indices=vit_cfg['out_indices'],
                                            qkv_bias=vit_cfg['qkv_bias'],
                                            drop_rate=vit_cfg['drop_rate'],
                                            drop_path_rate=vit_cfg['drop_path_rate'],
                                            patch_cfg=vit_cfg['patch_cfg'],
                                            init_cfg=vit_cfg['init_cfg'],
                                            single_model=False)

        self.mlp = MLP(input_dim=mlp_cfg['input_dim'],
                       hidden_dims=mlp_cfg['hidden_dims'],
                       output_dim=mlp_cfg['output_dim'],
                       dropout_rate=mlp_cfg['dropout_rate'])

        self.fc_all = nn.Linear(2 * 64, 2)

    def init_weights(self):
        super(GcnMain, self).init_weights()
        if self.init_cfg is not None:
            pretrained = self.init_cfg.get('checkpoint', None)
        else:
            pretrained = None
        if pretrained is not None:
            pretrained = pretrained[0]

        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, x, t):
        batch_size = x.size(0)
        vit_cls_mb, vit_feature = self.vit(x)
        mlp_cls, mlp_feature = self.mlp(t)
        tol_feature = torch.cat((vit_feature, mlp_feature[0]), dim=1)
        all_cls_mb = self.fc_all(tol_feature.view(batch_size, -1))
        return vit_cls_mb, mlp_cls, all_cls_mb


# model = FusionModel()
# x = torch.rand((2, 1, 64, 64, 64))
# t = torch.rand((2, 20))
# vit_cls_mb, mlp_cls, all_cls_mb = model(x, t)
# print(vit_cls_mb.size(), mlp_cls.size(), all_cls_mb.size())
