import sys
sys.path.append('/mnt/LungLocalNFS/tanweixiong/zjzl/code/ScreenLungNet')
from base_backbone import BaseBackbone
import torch
import torch.nn as nn



class MLP(BaseBackbone):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='ReLU', dropout_rate=0.0):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        current_dim = input_dim

        # 构建隐藏层结构
        for hidden_dim in hidden_dims:
            layer_seq = nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                getattr(nn, activation)(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            )
            self.hidden_layers.append(layer_seq)
            current_dim = hidden_dim

        # 输出层
        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        hidden_features = []

        # 逐层计算并记录特征
        for layer in self.hidden_layers:
            x = layer(x)
            hidden_features.append(x)

        # 最终输出
        output = self.output_layer(x)
        return output, hidden_features


# 示例用法
if __name__ == "__main__":
    # 创建模型
    model = MLP(
        input_dim=20,
        hidden_dims=[64],
        output_dim=2,
        dropout_rate=0.2
    )

    # 测试数据
    dummy_input = torch.randn(32, 20)  # 批量大小32
    output, features = model(dummy_input)

    print("输出形状:", output.shape)
    print("隐含层特征数量:", len(features))
    print("第一个隐含层特征形状:", features[0].shape)