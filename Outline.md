# MC-PIKAN Official 项目大纲

- 概览
  - 目标：使用 MC-PIKAN 框架对高维 Volterra 积分方程进行神经网络近似求解。
  - 语言/依赖：Python + PyTorch；依赖点包括 torch、tqdm、matplotlib、torchinfo，以及随机种子工具等。
  - 涉及的问题：Volterra1D、Volterra2DNR、Volterra10D 等高维问题及其解函数。
 
- 代码结构
  - 顶层文件与目录
  - 关键模块
  - 实验脚本
  - 数据与问题定义
  - 辅助资料与对话
  - 产物与缓存
 
- 关键模块详情
  - train.py
    - 功能：训练循环、动态学习率、进度显示、模型保存
  - models.py
    - 模型族：CPIKANModel、Unoptd_cPIKANModel、PINNModel
    - 辅助：print_modelsize
  - KANlayers.py
    - ChebyKANLayer、unoptd_ChebyKANLayer、JacobiKANLayer
  - problems/Volterra.py
    - Volterra1D、Volterra2DNR、Volterra10D：损失函数 loss_fn，接口统一
  - problems/functions.py
    - Volterra1D_solution、Volterra1D_f、Volterra1D_K
    - Volterra2DNR_f、Volterra2DNR_solution
    - Volterra10D_f、Volterra10D_solution、Volterra10D_integral
  - utils.py
    - boundary sampling、全局网格生成、维度灵活网格辅助
  - experiments/
    - Volterra1D_MCPI.py、Volterra2DNR_MCPI.py、Volterra10D_MCPI.py
  - antigravity_conversations
    - Refining MC-PIKAN Codebase.md
    - walkthrough.md.resolved
    - implementation_plan.md.resolved
    - task.md.resolved
 
- 运行与依赖
  - 依赖：torch、torchvision、tqdm、matplotlib、torchinfo、random_seed
  - 运行示例
    - python experiments/Volterra1D_MCPI.py
    - python experiments/Volterra2DNR_MCPI.py
    - python experiments/Volterra10D_MCPI.py
  - 产出位置：results/、Volterra*.png、对应模型权重
 
- 架构与实现要点
  - 自定义层：ChebyKANLayer、JacobiKANLayer、Unoptd 版本
  - 物理信息损失：1D/2D/10D 的 loss_fn 及自适应权重策略
  - 数据/网格工具：boundary points、grid generation等
  - 设备适配：对 CPU/GPU 自动切换
 
 - 使用与扩展建议
   - 增加单元测试
   - 增加更多问题/核/边界条件
   - 优化可视化和指标
 
 ### LaTeX Version (1D Volterra)
 
 1) 方程
 
 ```latex
 u(x) = f(x) + \int_{0}^{x} K(x,\xi)\,u(\xi)\,d\xi,\quad 0 \le x \le 1
 ```
 
 2) 核与源项
 
 ```latex
 K(x,\xi) = -\sin\left(\pi (x-\xi)\right)
 ```
 
 ```latex
 f(x) = \left(1 + \frac{1}{2\pi}\right) \sin(\pi x) - \frac{x}{2}\cos(\pi x)
 ```
 
 3) 已知解
 
 ```latex
 u(x) = \sin(\pi x)
 ```
 
 4) 蒙特卡洛离散化近似
 
 Let $ s \sim \text{Uniform}(0,1) $，$ \xi = x s $，则
 
 ```latex
 \int_{0}^{x} K(x,\xi) u(\xi) d\xi = x \int_{0}^{1} K(x, x s) u(x s) ds
 \approx x \cdot \frac{1}{N_s} \sum_{i=1}^{N_s} K\big(x, x s_i\big) u(x s_i)
 ```
 
 5) 离散残差与损失
 
 ```latex
 R(x_j) = u(x_j) - f(x_j) - \frac{ x_j }{N_s} \sum_{i=1}^{N_s} K\big(x_j, x_j s_i\big) u(x_j s_i)
 ```
 
 ```latex
 L = \frac{1}{N_x} \sum_{j=1}^{N_x} R(x_j)^2
 ```
 
 6) 代码映射要点（简要对照）
 
 - `xs = x_e * s_e` 对应上式中的 \{x_j\} 与 \{s_i\} 的网格扩展
 - `k_vals = K(x_e, xs)` 对应 $K(x_j,\xi_i)$
 - `u_vals = u(xs)` 对应 $u(\xi_i)$
 - `mean(product, dim=1)` 对应对 \( \frac{1}{N_s} \sum K(x_j,\xi_i) u(\xi_i) \)
 - `inner_mean` 对应残差 $R(x_j)$
 - `L` 对应损失 $L$
