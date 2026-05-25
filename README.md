# MC-PIKAN：基于蒙特卡洛采样的物理信息 Kolmogorov–Arnold 网络求解 Volterra 积分方程

## 概述

MC-PIKAN（Monte Carlo Physics-Informed Kolmogorov–Arnold Network）是一种结合蒙特卡洛积分与物理信息 KAN 网络的数值方法，用于求解各类 Volterra 积分方程，包括：

- **一维有界核 Volterra 积分方程**（1D bounded kernel）
- **二维非矩形域上的非线性 Volterra 积分方程**（2D nonlinear, non-rectangular domain）
- **高维有界核 Volterra 积分方程**（10D bounded kernel with boundary conditions）

本方法的核心思想是将 Volterra 积分方程中的积分项通过蒙特卡洛随机采样近似，并以基于 Chebyshev 多项式的 KAN 网络作为解函数的逼近器，通过最小化物理残差损失进行无监督训练。

---

## 方法原理

### 1. Volterra 积分方程

**第二类 Volterra 积分方程**的一般形式为：

$$u(\mathbf{x}) = f(\mathbf{x}) + \int_{\Omega(\mathbf{x})} K(\mathbf{x}, \mathbf{s}) \, u(\mathbf{s}) \, d\mathbf{s}$$

其中 $u(\mathbf{x})$ 为未知函数，$f(\mathbf{x})$ 为已知源项，$K(\mathbf{x}, \mathbf{s})$ 为积分核，$\Omega(\mathbf{x})$ 为依赖于 $\mathbf{x}$ 的积分域。

### 2. 蒙特卡洛积分近似

对积分项采用蒙特卡洛估计：

$$\int_{\Omega(\mathbf{x})} K(\mathbf{x}, \mathbf{s}) \, u(\mathbf{s}) \, d\mathbf{s} \approx \frac{|\Omega(\mathbf{x})|}{N_s} \sum_{i=1}^{N_s} K(\mathbf{x}, \mathbf{s}_i) \, u(\mathbf{s}_i)$$

其中 $\{\mathbf{s}_i\}_{i=1}^{N_s}$ 为从积分域 $\Omega(\mathbf{x})$ 中均匀采样的随机点。

### 3. Chebyshev KAN 层（ChebyKANLayer）

本项目使用基于 Chebyshev 多项式的 KAN 层替代传统 MLP 层。其前向传播过程为：

**显式表达式形式（优化版）：**

$$T_n(x) = \cos(n \cdot \arccos(x)), \quad x \in [-1, 1]$$

输入首先经过 $\tanh$ 归一化映射到 $[-1, 1]$，然后计算各阶 Chebyshev 多项式值：

$$y_j = \sum_{i=1}^{d_{in}} \sum_{k=0}^{p} w_{ijk} \cdot T_k(\tanh(x_i))$$

其中 $p$ 为多项式阶数（degree），$w_{ijk}$ 为可学习权重参数。

**递推关系形式（未优化版）：**

$$T_0(x) = 1, \quad T_1(x) = x, \quad T_{n+1}(x) = 2x \cdot T_n(x) - T_{n-1}(x)$$

### 4. 物理残差损失函数

以一维问题为例，损失函数定义为：

$$\mathcal{L} = \frac{1}{N_x} \sum_{j=1}^{N_x} \left[ u_\theta(x_j) - f(x_j) - \frac{x_j}{N_s} \sum_{i=1}^{N_s} K(x_j, x_j s_i) \cdot u_\theta(x_j s_i) \right]^2$$

对于高维问题（10D），损失函数包含物理残差项和边界条件项的自适应加权：

$$\mathcal{L} = (1 - w_{ph}) \cdot \mathcal{L}_{ph} + (1 - w_{bc}) \cdot \mathcal{L}_{bc}$$

其中权重通过归一化方式自适应调整。

---

## 项目结构

```
MC-PIKAN_Official/
├── layers.py              # KAN 层实现（ChebyKANLayer、JacobiKANLayer）
├── models.py              # 网络模型定义（CPIKANModel、PINNModel）
├── train.py               # 通用训练器（支持 Adam/LBFGS、动态学习率）
├── utils.py               # 高维问题工具函数（边界采样、网格生成）
├── random_seed.py         # 随机种子设置
├── problems/              # 问题定义
│   ├── functions.py       # 解析解与源项函数
│   └── Volterra.py        # Volterra 积分方程类（1D、2DNR、10D）
├── experiments/           # 实验脚本
│   ├── Volterra1D_MCPI.py     # 一维实验
│   ├── Volterra2DNR_MCPI.py   # 二维非矩形域实验
│   ├── Volterra10D_MCPI.py    # 十维实验
│   └── sympy_for_solution/    # 符号计算验证
├── draw/                  # 绘图脚本
│   ├── draw_loss.py       # 损失曲线绘制
│   ├── draw_10D.py        # 十维结果可视化
│   ├── draw_2DNR.py       # 二维结果可视化
│   ├── draw_curf.py       # 曲线绘制
│   └── Chebyshev Polynomials.py  # Chebyshev 多项式可视化
└── results/               # 输出结果
    ├── best_models/       # 最优模型权重
    ├── data/              # 数值数据
    ├── figure/            # 图片输出
    └── reports/           # 实验报告
```

---

## 环境依赖

- Python >= 3.8
- PyTorch >= 1.12（推荐支持 CUDA）
- NumPy
- Matplotlib
- tqdm
- torchinfo
- seaborn（绘图需要）

安装依赖：

```bash
pip install torch numpy matplotlib tqdm torchinfo seaborn
```

---

## 快速开始

### 1. 一维 Volterra 积分方程

```bash
python experiments/Volterra1D_MCPI.py
```

此脚本将：
1. 构造一维有界核 Volterra 积分方程问题实例
2. 使用 PIKAN 或 PINNs 模型进行训练
3. 输出预测解与精确解的对比图和绝对误差

### 2. 二维非矩形域问题

```bash
python experiments/Volterra2DNR_MCPI.py
```

### 3. 十维高维问题

```bash
python experiments/Volterra10D_MCPI.py
```

> **注意**：十维实验需要 CUDA GPU 支持，且默认使用两阶段训练策略（Adam + L-BFGS）。

---

## 模型配置说明

### CPIKANModel（Chebyshev PIKAN）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_dim` | 输入维度 | 1 |
| `output_dim` | 输出维度 | 1 |
| `num_layers` | 网络层数 | 3 |
| `hidden_dim` | 隐藏层维度 | 10 |
| `degree` | Chebyshev 多项式阶数 | 3 |
| `dtype` | 数据类型 | `torch.float32` |

### PINNModel（对比基线）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_dim` | 输入维度 | 1 |
| `output_dim` | 输出维度 | 1 |
| `num_layers` | 网络层数 | 3 |
| `hidden_dim` | 隐藏层维度 | 20 |
| `dtype` | 数据类型 | `torch.float32` |

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_epochs` | 最大训练轮数 | 1000 |
| `lr` | 初始学习率 | 1e-3 |
| `epsilon` | 收敛阈值 | 1e-20 |
| `opt_method` | 优化器（`Adam`/`LBFGS`） | `Adam` |
| `dynamic_lr` | 是否启用动态学习率 | `False` |
| `lr_decay` | 学习率衰减因子 | 0.5 |

---

## 自定义问题

若需求解新的 Volterra 积分方程，可通过继承 `Volterratype` 基类实现：

```python
from problems.Volterra import Volterratype
import torch

class MyVolterra(Volterratype):
    def __init__(self, X_grid, s):
        super().__init__(X_grid, s)
        # 初始化问题特定的张量

    def loss_fn(self, u):
        """
        定义物理残差损失函数
        u: 神经网络模型（可调用）
        返回: 标量损失值
        """
        # 实现蒙特卡洛近似的积分残差
        ...
        return loss
```

使用方式：

```python
from models import CPIKANModel
from train import train

model = CPIKANModel(input_dim=..., degree=5)
problem = MyVolterra(X_grid=..., s=...)
train(model, problem, max_epochs=5000, lr=1e-3)
```

---

## 数值实验结果

### 评估指标

本项目使用**相对 $L^2$ 误差**作为主要评价指标：

$$\text{Relative } L^2 \text{ error} = \frac{\| u_{exact} - u_{pred} \|_2}{\| u_{exact} \|_2}$$

### 实验设置

| 问题 | 配点数 $N_x$ | 采样数 $N_s$ | 模型结构 | 训练策略 |
|------|------|------|------|------|
| 1D | 50 | 40 | [3, 10, 3] | Adam, 2000 epochs |
| 2D-NR | 100 | 40 | [3, 10, 3] | Adam, 5000 epochs |
| 10D | 10000 | 10 | [3, 10, 7] | Adam 10k + LBFGS 1k |

---

## 核心设计选择

1. **显式 Chebyshev 计算**：使用 $T_n(x) = \cos(n \cdot \arccos(x))$ 代替递推关系，避免高阶递推中的数值误差累积，同时可利用 GPU 并行化。

2. **蒙特卡洛积分**：相比传统数值积分（如 Gauss 求积），蒙特卡洛方法在高维问题中不受维度灾难影响，计算复杂度与维度无关。

3. **两阶段训练**：高维问题采用 Adam 快速收敛至损失较低区域后，切换至 L-BFGS 进行精细优化。

4. **自适应损失权重**：十维问题中物理残差和边界条件的损失权重通过归一化自适应调整，平衡多目标优化。

---

## 许可证

本项目仅供学术研究使用。
