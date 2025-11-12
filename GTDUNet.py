import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.fftpack as fft
import numpy as np
from .s2block import S2Block



def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


def tsvd_denoise(tensor, threshold_ratio=0.9):
    """
    T-SVD based tensor denoising with energy thresholding
    :param tensor: input tensor of shape [m, n, h, w]
    :param threshold_ratio: energy ratio to keep (0-1)
    :return: denoised tensor
    """
    # 将 PyTorch 张量转换为 NumPy 数组进行处理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.cpu().detach().numpy()

    m, n, h, w = tensor.shape
    tensor_f = fft.dct(tensor, axis=2, norm='ortho')  # DCT along height
    tensor_f = fft.dct(tensor_f, axis=3, norm='ortho')  # DCT along width

    # Compute energy threshold
    singular_values = []
    for i in range(m):
        for j in range(n):
            U, S, V = torch.linalg.svd(torch.from_numpy(tensor_f[i, j]), full_matrices=False)
            S = S.numpy()
            singular_values.extend(S)

    singular_values_sorted = np.sort(singular_values)[::-1]
    cum_energy = np.cumsum(singular_values_sorted ** 2)
    total_energy = cum_energy[-1]
    keep_index = np.where(cum_energy <= threshold_ratio * total_energy)[0]
    threshold = singular_values_sorted[keep_index[-1]] if len(keep_index) > 0 else 0

    # Apply thresholded T-SVD
    denoised_f = np.zeros_like(tensor_f)
    for i in range(m):
        for j in range(n):
            U, S, V = torch.linalg.svd(torch.from_numpy(tensor_f[i, j]), full_matrices=False)
            S = S.numpy()
            S_thresh = S * (S >= threshold)
            denoised_f[i, j] = U.numpy() @ np.diag(S_thresh) @ V.numpy()

    # Inverse DCT
    denoised = fft.idct(denoised_f, axis=3, norm='ortho')
    denoised = fft.idct(denoised, axis=2, norm='ortho')
    # 将 NumPy 数组转换回 PyTorch 张量
    denoised = torch.from_numpy(denoised)
    return denoised.to(device)


class GraphGenerator:
    def __init__(self, num_nodes, k, p):
        """
        初始化图生成器
        :param num_nodes: 图的节点数量
        :param k: 每个节点的初始邻居数
        :param p: 重新连接的概率
        """
        self.num_nodes = num_nodes
        self.k = k
        self.p = p

    def generate_graph(self):
        """
        生成 WS 小世界图并转换为 PyTorch Geometric 格式
        :return: 图的边索引
        """
        nx_graph = nx.watts_strogatz_graph(n=self.num_nodes, k=self.k, p=self.p)
        return from_networkx(nx_graph).edge_index


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        初始化图卷积层
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        """
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化权重参数
        """
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        """
        图卷积层前向传播
        :param x: 节点特征，形状为 [num_nodes, in_channels]
        :param edge_index: 图的边索引，形状为 [2, num_edges]
        :return: 经过卷积处理后的节点特征，形状为 [num_nodes, out_channels]
        """
        num_nodes = x.size(0)

        # 构建稀疏邻接矩阵
        adj_values = torch.ones(edge_index.size(1), device=x.device)  # 边的权重
        adj_matrix = torch.sparse_coo_tensor(
            edge_index,
            adj_values,
            (num_nodes, num_nodes),
            device=x.device)

        # 计算度矩阵
        degree = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        degree[degree == 0] = 1  # 避免除零错误
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)

        # 归一化邻接矩阵
        normalized_adj = torch.sparse.mm(degree_inv_sqrt, adj_matrix)
        normalized_adj = torch.sparse.mm(normalized_adj, degree_inv_sqrt)

        # 图卷积操作
        output = torch.sparse.mm(normalized_adj, x)  # L_sym @ X
        output = torch.mm(output, self.weight)  # (L_sym @ X) @ W
        return output


class GCNImageFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        初始化图卷积图像特征提取器
        :param in_channels: 输入通道数
        :param hidden_channels: 隐藏层通道数
        :param out_channels: 输出通道数
        """
        super(GCNImageFeatureExtractor, self).__init__()
        self.conv1 = GraphConvLayer(in_channels, hidden_channels)
        self.conv2 = GraphConvLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        特征提取器前向传播
        :param x: 节点特征
        :param edge_index: 图的边索引
        :return: 提取的特征
        """
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def getGCN(images, edge_index):
    """
    对输入的图像张量进行特征提取
    :param images: 输入的图像张量，形状为 [n, m, h, w]
    :return: 提取的特征张量，形状与输入一致 [n, m, h, w]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    n, m, h, w = images.shape

    feature_extractor = GCNImageFeatureExtractor(in_channels=h * w, hidden_channels=60, out_channels=h * w).to(device)
    all_features = []
    for i in range(n):
        image = images[i]
        # 将每个波段的所有像素特征拼接作为节点特征
        node_features = image.view(m, -1)
        output = feature_extractor(node_features, edge_index)
        # 将输出恢复为原始的 [m, h, w] 形状
        output = output.view(m, h, w)
        all_features.append(output)
    all_features = torch.stack(all_features)
    return all_features


class RMBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv0_1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.conv0_2 = nn.Conv2d(in_channels, hidden_channels, 5, 1, 2)
        self.conv0_3 = nn.Conv2d(in_channels, hidden_channels, 7, 1, 3)
        self.conv1 = nn.Conv2d(hidden_channels * 3, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        rs1 = self.relu(self.conv0_1(x))
        rs2 = self.relu(self.conv0_2(x))
        rs3 = self.relu(self.conv0_3(x))

        # 一阶段
        rs1_1 = rs1 + rs2 + rs3
        rs2_1 = rs2 + rs3

        # 二阶段
        rs3_1 = rs3 + rs1_1 + rs2_1
        rs2_2 = rs2_1 + rs1_1 + rs3_1
        rs1_2 = rs1_1 + rs2_2 + rs3_1

        rs = torch.cat([rs1_2, rs2_2, rs3_1], dim=1)

        rs = self.conv1(rs)
        rs = torch.add(x, rs)
        return rs

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 16, 3, 1, 1, bias=False),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        return self.upsamle(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        else:
            self.up0 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.LeakyReLU()
            )
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0),
                nn.LeakyReLU()
            )
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        x1 = self.up1(x1)
        x = x1 + x2
        return self.relu(self.conv(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 2, 2, 0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.conv(x)


class GTDUNet(nn.Module):
    def __init__(self, dim, edge_index, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super().__init__()
        ms_dim = 8
        pan_dim = 1

        self.edge_index = edge_index

        self.relu = nn.LeakyReLU()
        self.upsample = Upsample(ms_dim)

        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(ms_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(pan_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.to_hrms = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, ms_dim, 3, 1, 1)
        )
        self.gout = nn.Conv2d(in_channels=9, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True)

        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)
        dim3 = dim1
        dim4 = dim0

        # layer 0
        self.s2block0 = S2Block(dim0, dim0 // dim_head, dim_head, int(dim0 * se_ratio_mlp))
        self.down0 = Down(dim0, dim1)
        # self.resblock0 = ResBlock(dim0, int(se_ratio_rb * dim0), dim0)
        self.rmblock0 = RMBlock(dim0, int(se_ratio_rb * dim0), dim0)

        # layer 1
        self.s2block1 = S2Block(dim1, dim1 // dim_head, dim_head, int(dim1 * se_ratio_mlp))
        self.down1 = Down(dim1, dim2)
        self.rmblock1 = RMBlock(dim1, int(se_ratio_rb * dim1), dim1)

        # layer 2
        self.s2block2 = S2Block(dim2, dim2 // dim_head, dim_head, int(dim2 * se_ratio_mlp))
        self.up0 = Up(dim2, dim3)
        self.rmblock2 = RMBlock(dim2, int(se_ratio_rb * dim2), dim2)

        # layer 3
        self.s2block3 = S2Block(dim3, dim3 // dim_head, dim_head, int(dim3 * se_ratio_mlp))
        self.up1 = Up(dim3, dim4)
        self.rmblock3 = RMBlock(dim3, int(se_ratio_rb * dim3), dim3)

        # layer 4
        self.s2block4 = S2Block(dim4, dim4 // dim_head, dim_head, int(dim4 * se_ratio_mlp))

    def forward(self, x, y, y_denoised):  # x = ms, y = pan
        x = self.upsample(x)
        skip_c0 = x
        # y_pan = y

        x = self.raise_ms_dim(x)
        y = self.raise_pan_dim(y)
        y = torch.cat([y[:, :31, :, :], y_denoised], dim=1)

        # layer 0
        x = self.s2block0(x, y)  # 32 64 64
        skip_c10 = x  # 32 64 64
        x = self.down0(x)  # 64 32 32
        y = self.rmblock0(y)  # 32 64 64
        skip_c11 = y  # 32 64 64
        y = self.down0(y)  # 64 32 32

        # layer 1
        x = self.s2block1(x, y)  # 64 32 32
        skip_c20 = x
        x = self.down1(x)  # 128 16 16
        y = self.rmblock1(y)  # 64 32 32
        skip_c21 = y  # 64 32 32
        y = self.down1(y)  # 128 16 16

        # layer 2
        x = self.s2block2(x, y)  # 128 16 16
        x = self.up0(x, skip_c20)  # 64 32 32
        y = self.rmblock2(y)  # 128 16 16
        y = self.up0(y, skip_c21)  # 64 32 32

        # layer 3
        x = self.s2block3(x, y)  # 64 32 32
        x = self.up1(x, skip_c10)  # 32 64 64
        y = self.rmblock3(y)  # 64 32 32
        y = self.up1(y, skip_c11)  # 32 64 64
        y = torch.cat([y[:, :31, :, :], y_denoised], dim=1)

        # layer 4
        x = self.s2block4(x, y)  # 32 64 64

        x = self.to_hrms(x)

        x_pm = torch.cat([x,y_denoised], dim=1)
        #x_t = get_t_svd(x_pm)
        x_g = getGCN(x_pm, self.edge_index)
        #x_g = x_g[:, :8, :, :]
        x_g = self.gout(x_g)
        x = self.relu(x + x_g)

        output = x + skip_c0  # 8 64 64

        return output


def summaries(model, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

