# 第7章：拓扑优化基础

拓扑优化是一种数学方法，用于在给定设计空间内寻找材料的最优分布，以满足特定的性能目标和约束条件。本章将深入探讨拓扑优化的核心数学原理，从经典的SIMP方法开始，逐步介绍敏感度分析、数值稳定性技术和实用优化算法。我们将重点关注连续体结构的拓扑优化，为后续的高级主题（水平集方法、多材料优化等）奠定基础。

## 7.1 SIMP方法与密度插值

### 7.1.1 拓扑优化的数学框架

拓扑优化问题的一般形式可以表述为：

$$
\begin{aligned}
\min_{\boldsymbol{\rho}} \quad & c(\boldsymbol{\rho}) = \mathbf{U}^T\mathbf{K}\mathbf{U} \\
\text{s.t.} \quad & \mathbf{K}\mathbf{U} = \mathbf{F} \\
& V(\boldsymbol{\rho}) = \int_\Omega \rho \, d\Omega \leq V_0 \\
& 0 < \rho_{\min} \leq \rho_e \leq 1, \quad e = 1, ..., N
\end{aligned}
$$

其中：
- $\boldsymbol{\rho} = [\rho_1, \rho_2, ..., \rho_N]^T$ 是设计变量（单元密度）
- $c(\boldsymbol{\rho})$ 是目标函数（通常为柔度）
- $\mathbf{K}$ 是全局刚度矩阵
- $\mathbf{U}$ 是位移向量
- $\mathbf{F}$ 是外力向量
- $V_0$ 是体积约束

### 7.1.2 SIMP插值模型

SIMP (Solid Isotropic Material with Penalization) 方法通过幂律插值建立密度与材料属性的关系：

$$
E_e(\rho_e) = \rho_e^p E_0
$$

其中 $p$ 是惩罚因子（通常 $p = 3$），$E_0$ 是实体材料的弹性模量。

单元刚度矩阵的插值：
$$
\mathbf{k}_e(\rho_e) = \rho_e^p \mathbf{k}_e^0
$$

这种插值策略的关键在于惩罚中间密度值，推动优化结果趋向0-1分布。

### 7.1.3 插值函数的数学性质

为了保证优化问题的良定性，插值函数应满足：

1. **单调性**：$\frac{\partial E(\rho)}{\partial \rho} > 0$
2. **凸性条件**：对于柔度最小化，需要 $p \geq \max(2, 2/\nu)$，其中 $\nu$ 是泊松比
3. **Hashin-Shtrikman界限**：插值函数应在理论界限内

修正的SIMP模型（RAMP）：
$$
E(\rho) = E_0 \frac{\rho(1 + q)}{1 + q\rho}
$$

其中 $q$ 是惩罚参数（通常 $q = 8$）。

### 7.1.4 离散化与有限元公式

考虑2D平面应力问题，单元刚度矩阵：

$$
\mathbf{k}_e^0 = t \int_{\Omega_e} \mathbf{B}^T \mathbf{D}_0 \mathbf{B} \, d\Omega
$$

其中：
- $t$ 是厚度
- $\mathbf{B}$ 是应变-位移矩阵
- $\mathbf{D}_0$ 是材料本构矩阵

对于双线性四边形单元，使用2×2高斯积分：
$$
\mathbf{k}_e^0 = t \sum_{i=1}^{4} w_i \mathbf{B}^T(\xi_i, \eta_i) \mathbf{D}_0 \mathbf{B}(\xi_i, \eta_i) |\mathbf{J}(\xi_i, \eta_i)|
$$

#### 应变-位移矩阵的构造

对于四节点四边形单元，位移场插值：
$$
\mathbf{u}(x,y) = \sum_{i=1}^4 N_i(x,y) \mathbf{u}_i
$$

其中形函数在参数坐标系中：
$$
\begin{aligned}
N_1(\xi, \eta) &= \frac{1}{4}(1-\xi)(1-\eta) \\
N_2(\xi, \eta) &= \frac{1}{4}(1+\xi)(1-\eta) \\
N_3(\xi, \eta) &= \frac{1}{4}(1+\xi)(1+\eta) \\
N_4(\xi, \eta) &= \frac{1}{4}(1-\xi)(1+\eta)
\end{aligned}
$$

应变-位移关系：
$$
\boldsymbol{\varepsilon} = \begin{bmatrix} \varepsilon_{xx} \\ \varepsilon_{yy} \\ \gamma_{xy} \end{bmatrix} = \begin{bmatrix} \frac{\partial u}{\partial x} \\ \frac{\partial v}{\partial y} \\ \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \end{bmatrix}
$$

通过链式法则，需要雅可比矩阵：
$$
\mathbf{J} = \begin{bmatrix} 
\frac{\partial x}{\partial \xi} & \frac{\partial y}{\partial \xi} \\
\frac{\partial x}{\partial \eta} & \frac{\partial y}{\partial \eta}
\end{bmatrix} = \sum_{i=1}^4 \begin{bmatrix}
\frac{\partial N_i}{\partial \xi} x_i & \frac{\partial N_i}{\partial \xi} y_i \\
\frac{\partial N_i}{\partial \eta} x_i & \frac{\partial N_i}{\partial \eta} y_i
\end{bmatrix}
$$

形函数对物理坐标的导数：
$$
\begin{bmatrix} 
\frac{\partial N_i}{\partial x} \\ 
\frac{\partial N_i}{\partial y} 
\end{bmatrix} = \mathbf{J}^{-1} \begin{bmatrix} 
\frac{\partial N_i}{\partial \xi} \\ 
\frac{\partial N_i}{\partial \eta} 
\end{bmatrix}
$$

最终应变-位移矩阵：
$$
\mathbf{B} = \begin{bmatrix}
\frac{\partial N_1}{\partial x} & 0 & \frac{\partial N_2}{\partial x} & 0 & \cdots \\
0 & \frac{\partial N_1}{\partial y} & 0 & \frac{\partial N_2}{\partial y} & \cdots \\
\frac{\partial N_1}{\partial y} & \frac{\partial N_1}{\partial x} & \frac{\partial N_2}{\partial y} & \frac{\partial N_2}{\partial x} & \cdots
\end{bmatrix}
$$

#### 本构矩阵

平面应力状态下的本构矩阵：
$$
\mathbf{D}_0 = \frac{E}{1-\nu^2} \begin{bmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \frac{1-\nu}{2}
\end{bmatrix}
$$

平面应变状态：
$$
\mathbf{D}_0 = \frac{E}{(1+\nu)(1-2\nu)} \begin{bmatrix}
1-\nu & \nu & 0 \\
\nu & 1-\nu & 0 \\
0 & 0 & \frac{1-2\nu}{2}
\end{bmatrix}
$$

#### 数值积分点选择

高斯积分点位置和权重（2×2方案）：
$$
\xi_{1,2} = \pm \frac{1}{\sqrt{3}}, \quad \eta_{1,2} = \pm \frac{1}{\sqrt{3}}, \quad w_i = 1
$$

完全积分vs减缩积分：
- **完全积分**（2×2）：精确积分双线性项，避免零能模式
- **减缩积分**（1×1）：降低计算成本，但可能引入沙漏模式
- **选择性减缩积分**：剪切项减缩，避免剪切锁定

### 7.1.5 多材料SIMP扩展

对于 $m$ 种材料的拓扑优化：
$$
E_e = \sum_{i=1}^m \rho_{e,i}^p E_i
$$

约束条件：
$$
\sum_{i=1}^m \rho_{e,i} = 1, \quad \rho_{e,i} \geq 0
$$

Zuo-Kang插值方案：
$$
E_e = \sum_{i=1}^m \rho_{e,i} E_i \left( \sum_{j=1}^m \rho_{e,j}^p \right)^{-1}
$$

### 7.1.6 各向异性材料的SIMP

对于正交各向异性材料，弹性张量插值：
$$
\mathbf{C}_e(\rho_e, \theta_e) = \rho_e^p \mathbf{T}(\theta_e)^T \mathbf{C}_0 \mathbf{T}(\theta_e)
$$

其中 $\mathbf{T}(\theta_e)$ 是旋转变换矩阵，$\theta_e$ 是纤维方向角。

离散材料方向优化（DMO）：
$$
\mathbf{C}_e = \sum_{k=1}^{N_\theta} w_{e,k} \rho_e^p \mathbf{C}_0^{(k)}
$$

其中 $w_{e,k}$ 是方向权重，满足 $\sum_k w_{e,k} = 1$。

## 7.2 敏感度分析与伴随方法

### 7.2.1 直接微分法

目标函数对设计变量的敏感度：

$$
\frac{\partial c}{\partial \rho_e} = \frac{\partial c}{\partial \mathbf{U}} \frac{\partial \mathbf{U}}{\partial \rho_e} + \frac{\partial c}{\partial \mathbf{K}} \frac{\partial \mathbf{K}}{\partial \rho_e}
$$

从平衡方程 $\mathbf{K}\mathbf{U} = \mathbf{F}$ 微分：
$$
\frac{\partial \mathbf{K}}{\partial \rho_e} \mathbf{U} + \mathbf{K} \frac{\partial \mathbf{U}}{\partial \rho_e} = 0
$$

因此：
$$
\frac{\partial \mathbf{U}}{\partial \rho_e} = -\mathbf{K}^{-1} \frac{\partial \mathbf{K}}{\partial \rho_e} \mathbf{U}
$$

### 7.2.2 伴随方法

引入拉格朗日函数：
$$
\mathcal{L} = c(\mathbf{U}, \boldsymbol{\rho}) + \boldsymbol{\lambda}^T(\mathbf{K}\mathbf{U} - \mathbf{F})
$$

伴随方程：
$$
\mathbf{K}^T \boldsymbol{\lambda} = -\frac{\partial c}{\partial \mathbf{U}}
$$

对于柔度目标函数 $c = \mathbf{U}^T\mathbf{K}\mathbf{U}$：
$$
\boldsymbol{\lambda} = -2\mathbf{U}
$$

最终敏感度：
$$
\frac{\partial c}{\partial \rho_e} = -p\rho_e^{p-1} \mathbf{u}_e^T \mathbf{k}_e^0 \mathbf{u}_e
$$

### 7.2.3 体积约束的敏感度

体积约束函数：
$$
g(\boldsymbol{\rho}) = \sum_{e=1}^N \rho_e v_e - V_0
$$

敏感度：
$$
\frac{\partial g}{\partial \rho_e} = v_e
$$

其中 $v_e$ 是单元体积。

### 7.2.4 应力约束的敏感度

von Mises应力：
$$
\sigma_{vm} = \sqrt{\sigma^T \mathbf{M} \sigma}
$$

其中 $\mathbf{M}$ 是von Mises矩阵。

应力对密度的敏感度（考虑qp松弛）：
$$
\frac{\partial \sigma_{vm}}{\partial \rho_e} = \frac{1}{2\sigma_{vm}} \left( 2\sigma^T \mathbf{M} \frac{\partial \sigma}{\partial \rho_e} \right)
$$

其中：
$$
\frac{\partial \sigma}{\partial \rho_e} = p\rho_e^{p-1} \mathbf{D}_0 \mathbf{B} \mathbf{u}_e + \rho_e^p \mathbf{D}_0 \mathbf{B} \frac{\partial \mathbf{u}_e}{\partial \rho_e}
$$

#### von Mises矩阵的具体形式

对于2D平面应力：
$$
\mathbf{M} = \begin{bmatrix}
1 & -0.5 & 0 \\
-0.5 & 1 & 0 \\
0 & 0 & 3
\end{bmatrix}
$$

对于3D情况：
$$
\mathbf{M} = \begin{bmatrix}
1 & -0.5 & -0.5 & 0 & 0 & 0 \\
-0.5 & 1 & -0.5 & 0 & 0 & 0 \\
-0.5 & -0.5 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 3 & 0 & 0 \\
0 & 0 & 0 & 0 & 3 & 0 \\
0 & 0 & 0 & 0 & 0 & 3
\end{bmatrix}
$$

#### 主应力约束

最大主应力：
$$
\sigma_1 = \frac{\sigma_{xx} + \sigma_{yy}}{2} + \sqrt{\left(\frac{\sigma_{xx} - \sigma_{yy}}{2}\right)^2 + \tau_{xy}^2}
$$

主应力敏感度：
$$
\frac{\partial \sigma_1}{\partial \rho_e} = \frac{\partial \sigma_1}{\partial \boldsymbol{\sigma}} \cdot \frac{\partial \boldsymbol{\sigma}}{\partial \rho_e}
$$

其中：
$$
\frac{\partial \sigma_1}{\partial \boldsymbol{\sigma}} = \begin{bmatrix}
\frac{1}{2} + \frac{\sigma_{xx} - \sigma_{yy}}{4\sqrt{(\sigma_{xx} - \sigma_{yy})^2/4 + \tau_{xy}^2}} \\
\frac{1}{2} - \frac{\sigma_{xx} - \sigma_{yy}}{4\sqrt{(\sigma_{xx} - \sigma_{yy})^2/4 + \tau_{xy}^2}} \\
\frac{\tau_{xy}}{\sqrt{(\sigma_{xx} - \sigma_{yy})^2/4 + \tau_{xy}^2}}
\end{bmatrix}
$$

### 7.2.5 多载荷敏感度

对于 $L$ 个载荷工况的加权目标：
$$
c = \sum_{l=1}^L w_l c_l = \sum_{l=1}^L w_l \mathbf{U}_l^T \mathbf{K} \mathbf{U}_l
$$

敏感度：
$$
\frac{\partial c}{\partial \rho_e} = -\sum_{l=1}^L w_l p\rho_e^{p-1} \mathbf{u}_{e,l}^T \mathbf{k}_e^0 \mathbf{u}_{e,l}
$$

### 7.2.6 特征值问题的敏感度

对于频率约束的拓扑优化，考虑广义特征值问题：
$$
\mathbf{K}\boldsymbol{\phi}_i = \lambda_i \mathbf{M}\boldsymbol{\phi}_i
$$

特征值敏感度（Nelson方法）：
$$
\frac{\partial \lambda_i}{\partial \rho_e} = \boldsymbol{\phi}_i^T \left( \frac{\partial \mathbf{K}}{\partial \rho_e} - \lambda_i \frac{\partial \mathbf{M}}{\partial \rho_e} \right) \boldsymbol{\phi}_i
$$

考虑质量归一化 $\boldsymbol{\phi}_i^T \mathbf{M} \boldsymbol{\phi}_i = 1$。

对于重复特征值的情况，需要考虑特征向量的导数：
$$
\frac{\partial \boldsymbol{\phi}_i}{\partial \rho_e} = \sum_{j \neq i} \frac{\boldsymbol{\phi}_j^T \left( \frac{\partial \mathbf{K}}{\partial \rho_e} - \lambda_i \frac{\partial \mathbf{M}}{\partial \rho_e} \right) \boldsymbol{\phi}_i}{\lambda_i - \lambda_j} \boldsymbol{\phi}_j
$$

### 7.2.7 瞬态问题的敏感度

对于瞬态响应优化，考虑Newmark时间积分：
$$
\mathbf{M}\ddot{\mathbf{U}}^{(n+1)} + \mathbf{C}\dot{\mathbf{U}}^{(n+1)} + \mathbf{K}\mathbf{U}^{(n+1)} = \mathbf{F}^{(n+1)}
$$

使用伴随方法，引入伴随变量 $\boldsymbol{\lambda}(t)$，满足终端条件：
$$
\mathbf{M}\ddot{\boldsymbol{\lambda}} - \mathbf{C}\dot{\boldsymbol{\lambda}} + \mathbf{K}\boldsymbol{\lambda} = -\frac{\partial g}{\partial \mathbf{U}}
$$

其中 $g$ 是时间积分目标函数。

### 7.2.8 热弹性耦合的敏感度

考虑热应力问题：
$$
\boldsymbol{\sigma} = \mathbf{D}(\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_T)
$$

其中热应变：
$$
\boldsymbol{\varepsilon}_T = \alpha \Delta T \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}
$$

温度场满足：
$$
\mathbf{K}_T \mathbf{T} = \mathbf{Q}
$$

耦合敏感度需要考虑温度场对结构响应的影响：
$$
\frac{\partial c}{\partial \rho_e} = \frac{\partial c}{\partial \mathbf{U}} \frac{\partial \mathbf{U}}{\partial \rho_e} + \frac{\partial c}{\partial \mathbf{T}} \frac{\partial \mathbf{T}}{\partial \rho_e}
$$

## 7.3 过滤技术与棋盘格现象

### 7.3.1 棋盘格现象的数学解释

棋盘格模式源于有限元离散化的数值病态。对于双线性四边形单元，棋盘格模式具有人工高刚度。

考虑特征值分析：
$$
\mathbf{K}\boldsymbol{\phi}_i = \lambda_i \mathbf{M}\boldsymbol{\phi}_i
$$

棋盘格模式对应于零能量模式或伪刚体模式。

### 7.3.2 密度过滤

密度过滤通过加权平均修改密度场：
$$
\tilde{\rho}_e = \frac{\sum_{i \in N_e} w_{ei} v_i \rho_i}{\sum_{i \in N_e} w_{ei} v_i}
$$

权重函数（线性衰减）：
$$
w_{ei} = \max(0, r_{min} - d_{ei})
$$

其中 $r_{min}$ 是过滤半径，$d_{ei}$ 是单元中心距离。

过滤后的敏感度链式法则：
$$
\frac{\partial c}{\partial \rho_i} = \sum_{e \in N_i} \frac{\partial c}{\partial \tilde{\rho}_e} \frac{\partial \tilde{\rho}_e}{\partial \rho_i}
$$

### 7.3.3 敏感度过滤

直接对敏感度进行过滤：
$$
\frac{\widehat{\partial c}}{\partial \rho_e} = \frac{1}{\max(\gamma, \rho_e) \sum_{i \in N_e} w_{ei}} \sum_{i \in N_e} w_{ei} \rho_i \frac{\partial c}{\partial \rho_i}
$$

其中 $\gamma$ 是小正数，防止除零。

### 7.3.4 Heaviside投影

为获得清晰的0-1设计，使用Heaviside投影：
$$
\bar{\rho}_e = \frac{\tanh(\beta \eta) + \tanh(\beta(\tilde{\rho}_e - \eta))}{\tanh(\beta \eta) + \tanh(\beta(1 - \eta))}
$$

其中 $\beta$ 控制陡度，$\eta$ 是阈值（通常0.5）。

投影的导数：
$$
\frac{\partial \bar{\rho}_e}{\partial \tilde{\rho}_e} = \frac{\beta \text{sech}^2(\beta(\tilde{\rho}_e - \eta))}{\tanh(\beta \eta) + \tanh(\beta(1 - \eta))}
$$

### 7.3.5 鲁棒设计公式

考虑制造不确定性的三场公式：
$$
\min_{\boldsymbol{\rho}} \quad c(\bar{\boldsymbol{\rho}}^d) + c(\bar{\boldsymbol{\rho}}^i) + c(\bar{\boldsymbol{\rho}}^e)
$$

其中：
- $\bar{\boldsymbol{\rho}}^d$ = 稀释设计（dilated）
- $\bar{\boldsymbol{\rho}}^i$ = 中间设计（intermediate）  
- $\bar{\boldsymbol{\rho}}^e$ = 腐蚀设计（eroded）

通过不同的投影阈值实现：
$$
\eta^d = 0.5 - \Delta\eta, \quad \eta^i = 0.5, \quad \eta^e = 0.5 + \Delta\eta
$$

#### 统一投影框架

广义投影函数：
$$
\bar{\rho} = \frac{1}{1 + \exp(-\beta(\tilde{\rho} - \eta))}
$$

三场投影的梯度链：
$$
\frac{\partial c}{\partial \rho_i} = \sum_{j \in \{d,i,e\}} \sum_{k} \frac{\partial c^j}{\partial \bar{\rho}_k^j} \frac{\partial \bar{\rho}_k^j}{\partial \tilde{\rho}_k} \frac{\partial \tilde{\rho}_k}{\partial \rho_i}
$$

#### 制造约束的数学表达

最小特征尺寸约束（使用几何约束）：
$$
\int_{\Omega} H(\bar{\rho} - \eta) \cdot H(\eta - \bar{\rho}_{\text{dilated}}) \, d\Omega = 0
$$

其中 $H$ 是Heaviside函数，$\bar{\rho}_{\text{dilated}}$ 是膨胀操作后的密度场。

### 7.3.6 非局部过滤算子

考虑更一般的非局部过滤：
$$
\tilde{\rho}(\mathbf{x}) = \int_{\Omega} w(\mathbf{x}, \mathbf{y}) \rho(\mathbf{y}) \, d\mathbf{y}
$$

高斯核函数：
$$
w(\mathbf{x}, \mathbf{y}) = \frac{1}{(2\pi r^2)^{d/2}} \exp\left(-\frac{\|\mathbf{x} - \mathbf{y}\|^2}{2r^2}\right)
$$

其中 $d$ 是空间维度，$r$ 是过滤半径。

### 7.3.7 PDE过滤器

使用Helmholtz方程作为过滤器：
$$
-r^2 \nabla^2 \tilde{\rho} + \tilde{\rho} = \rho
$$

边界条件：
$$
\nabla \tilde{\rho} \cdot \mathbf{n} = 0 \quad \text{on } \partial\Omega
$$

有限元离散化：
$$
(\mathbf{K}_f + \mathbf{M}_f) \tilde{\boldsymbol{\rho}} = \mathbf{M}_f \boldsymbol{\rho}
$$

其中：
- $\mathbf{K}_f = r^2 \int_{\Omega} \nabla N_i \cdot \nabla N_j \, d\Omega$
- $\mathbf{M}_f = \int_{\Omega} N_i N_j \, d\Omega$

### 7.3.8 形态学过滤

数学形态学操作：

**腐蚀操作**：
$$
(\rho \ominus B)(\mathbf{x}) = \inf_{\mathbf{b} \in B} \rho(\mathbf{x} + \mathbf{b})
$$

**膨胀操作**：
$$
(\rho \oplus B)(\mathbf{x}) = \sup_{\mathbf{b} \in B} \rho(\mathbf{x} - \mathbf{b})
$$

**开操作**（先腐蚀后膨胀）：
$$
\rho \circ B = (\rho \ominus B) \oplus B
$$

**闭操作**（先膨胀后腐蚀）：
$$
\rho \bullet B = (\rho \oplus B) \ominus B
$$

其中 $B$ 是结构元素。

### 7.3.9 各向异性过滤

方向依赖的过滤核：
$$
w(\mathbf{x}, \mathbf{y}) = \frac{1}{Z} \exp\left(-(\mathbf{x} - \mathbf{y})^T \mathbf{A}^{-1} (\mathbf{x} - \mathbf{y})\right)
$$

其中 $\mathbf{A}$ 是各向异性张量：
$$
\mathbf{A} = \mathbf{R}(\theta) \begin{bmatrix} r_1^2 & 0 \\ 0 & r_2^2 \end{bmatrix} \mathbf{R}(\theta)^T
$$

$r_1, r_2$ 是主方向的过滤半径，$\theta$ 是旋转角度。

## 7.4 MMA优化器与收敛准则

### 7.4.1 移动渐近线方法（MMA）

MMA (Method of Moving Asymptotes) 通过一系列凸子问题逼近原始非凸问题。

原始问题：
$$
\begin{aligned}
\min_{\mathbf{x}} \quad & f_0(\mathbf{x}) \\
\text{s.t.} \quad & f_i(\mathbf{x}) \leq 0, \quad i = 1, ..., m \\
& \alpha_j \leq x_j \leq \beta_j, \quad j = 1, ..., n
\end{aligned}
$$

MMA子问题的近似函数：
$$
f_i^{(k)}(\mathbf{x}) = r_i^{(k)} + \sum_{j=1}^n \left( \frac{p_{ij}^{(k)}}{U_j^{(k)} - x_j} + \frac{q_{ij}^{(k)}}{x_j - L_j^{(k)}} \right)
$$

其中 $L_j^{(k)}$ 和 $U_j^{(k)}$ 是移动渐近线。

### 7.4.2 渐近线更新策略

渐近线位置的自适应更新：
$$
L_j^{(k)} = x_j^{(k)} - s_j^{(k)}(x_j^{\max} - x_j^{\min})
$$
$$
U_j^{(k)} = x_j^{(k)} + s_j^{(k)}(x_j^{\max} - x_j^{\min})
$$

其中 $s_j^{(k)}$ 是移动限制参数：

$$
s_j^{(k)} = \begin{cases}
0.7 s_j^{(k-1)} & \text{if } (x_j^{(k)} - x_j^{(k-1)})(x_j^{(k-1)} - x_j^{(k-2)}) < 0 \\
1.2 s_j^{(k-1)} & \text{if } (x_j^{(k)} - x_j^{(k-1)})(x_j^{(k-1)} - x_j^{(k-2)}) > 0 \\
s_j^{(k-1)} & \text{otherwise}
\end{cases}
$$

### 7.4.3 系数计算

根据函数值和梯度计算MMA系数：

当 $\frac{\partial f_i}{\partial x_j} > 0$：
$$
p_{ij}^{(k)} = (U_j^{(k)} - x_j^{(k)})^2 \frac{\partial f_i}{\partial x_j}\bigg|_{x^{(k)}}, \quad q_{ij}^{(k)} = 0
$$

当 $\frac{\partial f_i}{\partial x_j} < 0$：
$$
p_{ij}^{(k)} = 0, \quad q_{ij}^{(k)} = -(x_j^{(k)} - L_j^{(k)})^2 \frac{\partial f_i}{\partial x_j}\bigg|_{x^{(k)}}
$$

常数项：
$$
r_i^{(k)} = f_i(x^{(k)}) - \sum_{j=1}^n \left( \frac{p_{ij}^{(k)}}{U_j^{(k)} - x_j^{(k)}} + \frac{q_{ij}^{(k)}}{x_j^{(k)} - L_j^{(k)}} \right)
$$

### 7.4.4 对偶问题与KKT条件

MMA子问题的拉格朗日函数：
$$
\mathcal{L} = f_0^{(k)}(\mathbf{x}) + \sum_{i=1}^m \lambda_i f_i^{(k)}(\mathbf{x}) + \sum_{j=1}^n (\mu_j^- (\alpha_j - x_j) + \mu_j^+ (x_j - \beta_j))
$$

KKT条件：
$$
\frac{\partial \mathcal{L}}{\partial x_j} = 0, \quad j = 1, ..., n
$$

显式解：
$$
x_j = \frac{\sqrt{p_j^*(\lambda)} - \sqrt{q_j^*(\lambda)} + L_j \sqrt{q_j^*(\lambda)/p_j^*(\lambda)} + U_j}{\sqrt{q_j^*(\lambda)/p_j^*(\lambda)} + 1}
$$

其中：
$$
p_j^*(\lambda) = p_{0j} + \sum_{i=1}^m \lambda_i p_{ij}, \quad q_j^*(\lambda) = q_{0j} + \sum_{i=1}^m \lambda_i q_{ij}
$$

### 7.4.5 内点法求解对偶问题

对偶问题：
$$
\max_{\lambda \geq 0} \quad \psi(\lambda)
$$

使用内点牛顿法，引入障碍函数：
$$
\psi_\mu(\lambda) = \psi(\lambda) - \mu \sum_{i=1}^m \ln(\lambda_i)
$$

牛顿方向：
$$
\nabla^2 \psi_\mu \Delta\lambda = -\nabla \psi_\mu
$$

### 7.4.6 收敛准则

多准则收敛判定：

1. **KKT残差**：
$$
r_{KKT} = \max\left\{ \left\|\nabla_x \mathcal{L}\right\|_\infty, \max_i |\min(0, -f_i)|, \max_i |\lambda_i f_i| \right\} < \epsilon_{KKT}
$$

2. **设计变量变化**：
$$
\Delta x = \frac{\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\|_2}{\|\mathbf{x}^{(k)}\|_2} < \epsilon_x
$$

3. **目标函数变化**：
$$
\Delta f = \frac{|f_0^{(k+1)} - f_0^{(k)}|}{|f_0^{(k)}|} < \epsilon_f
$$

4. **灰度指标**（0-1性）：
$$
M_{nd} = \frac{4 \sum_{e=1}^N \rho_e(1-\rho_e)}{N} < \epsilon_{gray}
$$

典型阈值：$\epsilon_{KKT} = 10^{-6}$, $\epsilon_x = 10^{-3}$, $\epsilon_f = 10^{-4}$, $\epsilon_{gray} = 0.01$

#### 自适应收敛准则

动态调整收敛阈值：
$$
\epsilon^{(k+1)} = \begin{cases}
\epsilon^{(k)} / \gamma & \text{if } r^{(k)} < \alpha \epsilon^{(k)} \\
\epsilon^{(k)} \cdot \gamma & \text{if } r^{(k)} > \beta \epsilon^{(k)} \\
\epsilon^{(k)} & \text{otherwise}
\end{cases}
$$

其中 $\gamma = 1.5$，$\alpha = 0.1$，$\beta = 0.9$。

### 7.4.7 GCMMA扩展

广义MMA（GCMMA）增加了二阶项：
$$
f_i^{(k)}(\mathbf{x}) = r_i^{(k)} + \sum_{j=1}^n \left( \frac{p_{ij}^{(k)}}{U_j^{(k)} - x_j} + \frac{q_{ij}^{(k)}}{x_j - L_j^{(k)}} \right) + \sum_{j=1}^n \sum_{l=1}^n x_j A_{ijl}^{(k)} x_l
$$

二阶项系数通过拟牛顿方法更新：
$$
\mathbf{A}^{(k+1)} = \mathbf{A}^{(k)} + \frac{\mathbf{y}\mathbf{y}^T}{\mathbf{y}^T\mathbf{s}} - \frac{\mathbf{A}^{(k)}\mathbf{s}\mathbf{s}^T\mathbf{A}^{(k)}}{\mathbf{s}^T\mathbf{A}^{(k)}\mathbf{s}}
$$

其中 $\mathbf{s} = \mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}$，$\mathbf{y} = \nabla f^{(k+1)} - \nabla f^{(k)}$。

### 7.4.8 全局收敛性保证

引入信赖域约束：
$$
\|\mathbf{x} - \mathbf{x}^{(k)}\|_\infty \leq \Delta^{(k)}
$$

信赖域半径更新：
$$
\Delta^{(k+1)} = \begin{cases}
\min(2\Delta^{(k)}, \Delta_{\max}) & \text{if } \rho^{(k)} > 0.75 \\
\Delta^{(k)} & \text{if } 0.25 \leq \rho^{(k)} \leq 0.75 \\
0.5\Delta^{(k)} & \text{if } \rho^{(k)} < 0.25
\end{cases}
$$

其中 $\rho^{(k)} = \frac{f(\mathbf{x}^{(k)}) - f(\mathbf{x}^{(k+1)})}{q(\mathbf{x}^{(k)}) - q(\mathbf{x}^{(k+1)})}$ 是实际下降与预测下降的比值。

### 7.4.9 并行MMA实现

对于大规模问题，MMA子问题可以分解：

**分离变量**：
$$
\min_{\mathbf{x}, \mathbf{y}, \mathbf{z}} \sum_{j=1}^n \left( \frac{p_{0j}^+ + \sum_{i=1}^m \lambda_i p_{ij}^+}{U_j - x_j} + \frac{q_{0j}^+ + \sum_{i=1}^m \lambda_i q_{ij}^+}{x_j - L_j} \right) + \sum_{i=1}^m \lambda_i z_i
$$

约束：
$$
\sum_{j=1}^n \left( \frac{p_{ij}}{U_j - x_j} + \frac{q_{ij}}{x_j - L_j} \right) - z_i + y_i^2 = b_i
$$

**并行策略**：
1. 变量分组：$\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_P]$
2. 处理器 $p$ 负责 $\mathbf{x}_p$ 的更新
3. 使用全局归约操作计算约束值
4. 主处理器更新拉格朗日乘子

### 7.4.10 稳定化技术

**移动限制**：
$$
\max(\alpha_j, x_j^{(k)} - 0.1(x_j^{\max} - x_j^{\min})) \leq x_j^{(k+1)} \leq \min(\beta_j, x_j^{(k)} + 0.1(x_j^{\max} - x_j^{\min}))
$$

**阻尼更新**：
$$
x_j^{(k+1)} = (1 - \omega) x_j^{(k)} + \omega x_j^{*}
$$

其中 $\omega \in (0, 1]$ 是阻尼因子，$x_j^{*}$ 是MMA子问题的解。

## 7.5 应力约束与疲劳优化

### 7.5.1 应力约束的挑战

应力约束拓扑优化面临的主要挑战：

1. **奇异性**：$\rho \to 0$ 时应力趋于无穷
2. **局部性**：大量局部约束（每个单元）
3. **非线性**：应力-密度关系高度非线性

### 7.5.2 qp松弛方法

松弛的应力约束：
$$
\sigma_{vm,e} \leq \frac{\sigma_{allow}}{\rho_e^q}
$$

其中 $q \in (0, 1)$ 是松弛参数（通常 $q = 0.5$）。

等效形式：
$$
\rho_e^q \sigma_{vm,e} \leq \sigma_{allow}
$$

### 7.5.3 应力聚合函数

P-norm聚合：
$$
\sigma_{PN} = \left( \sum_{e=1}^N \left(\frac{\sigma_{vm,e}}{\sigma_{allow}}\right)^P \right)^{1/P}
$$

KS (Kreisselmeier-Steinhauser) 函数：
$$
\sigma_{KS} = \frac{1}{\xi} \ln\left( \sum_{e=1}^N \exp(\xi \sigma_{vm,e}/\sigma_{allow}) \right)
$$

其中 $P$ 或 $\xi$ 是聚合参数。

### 7.5.4 局部应力约束的处理

块聚合策略：
$$
\max_{e \in B_k} \sigma_{vm,e} \leq \sigma_{allow}, \quad k = 1, ..., N_B
$$

其中 $B_k$ 是第 $k$ 个块。

自适应约束删除：
$$
\mathcal{A} = \{e : \sigma_{vm,e} > \alpha \sigma_{allow}\}
$$

只保留活跃集 $\mathcal{A}$ 中的约束。

### 7.5.5 疲劳优化

高周疲劳的Basquin定律：
$$
N_f = \left(\frac{\sigma_a}{\sigma'_f}\right)^{-1/b}
$$

其中：
- $N_f$ 是疲劳寿命（循环次数）
- $\sigma_a$ 是应力幅值
- $\sigma'_f$ 是疲劳强度系数
- $b$ 是疲劳强度指数

修正的Goodman准则：
$$
\frac{\sigma_a}{S_e} + \frac{\sigma_m}{S_{ut}} = \frac{1}{n}
$$

其中：
- $S_e$ 是耐久极限
- $S_{ut}$ 是极限强度
- $n$ 是安全系数

### 7.5.6 多载荷工况

考虑 $M$ 个载荷工况：
$$
\min_{\boldsymbol{\rho}} \quad \sum_{i=1}^M w_i c_i(\boldsymbol{\rho})
$$

其中 $w_i$ 是权重系数。

最坏情况设计：
$$
\min_{\boldsymbol{\rho}} \quad \max_{i=1,...,M} c_i(\boldsymbol{\rho})
$$

使用辅助变量重构：
$$
\begin{aligned}
\min_{\boldsymbol{\rho}, z} \quad & z \\
\text{s.t.} \quad & c_i(\boldsymbol{\rho}) \leq z, \quad i = 1, ..., M
\end{aligned}
$$

### 7.5.7 应力集中的网格自适应

误差估计器：
$$
\eta_e = h_e \|\nabla \sigma_{vm}\|_{L^2(\Omega_e)}
$$

网格细化准则：
$$
\text{refine if } \eta_e > \theta \max_i \eta_i
$$

其中 $\theta \in (0, 1)$ 是细化阈值。

## 本章小结

本章系统介绍了拓扑优化的基础理论和数值方法：

### 核心概念
- **SIMP方法**：通过幂律插值 $E(\rho) = \rho^p E_0$ 实现材料分布的连续化
- **敏感度分析**：伴随方法高效计算 $\frac{\partial c}{\partial \rho_e} = -p\rho_e^{p-1} \mathbf{u}_e^T \mathbf{k}_e^0 \mathbf{u}_e$
- **过滤技术**：密度过滤、敏感度过滤和Heaviside投影消除数值不稳定性
- **MMA优化器**：通过移动渐近线构造凸子问题序列
- **应力约束**：qp松弛和聚合函数处理局部应力约束

### 关键公式汇总

1. **拓扑优化标准形式**：
   $$\min_{\boldsymbol{\rho}} c(\boldsymbol{\rho}) \text{ s.t. } \mathbf{K}\mathbf{U} = \mathbf{F}, V(\boldsymbol{\rho}) \leq V_0$$

2. **密度过滤**：
   $$\tilde{\rho}_e = \frac{\sum_{i \in N_e} w_{ei} v_i \rho_i}{\sum_{i \in N_e} w_{ei} v_i}$$

3. **MMA近似函数**：
   $$f_i^{(k)}(\mathbf{x}) = r_i^{(k)} + \sum_{j=1}^n \left( \frac{p_{ij}^{(k)}}{U_j^{(k)} - x_j} + \frac{q_{ij}^{(k)}}{x_j - L_j^{(k)}} \right)$$

4. **应力聚合**：
   $$\sigma_{KS} = \frac{1}{\xi} \ln\left( \sum_{e=1}^N \exp(\xi \sigma_{vm,e}/\sigma_{allow}) \right)$$

### 实践要点
- 惩罚因子 $p$ 通常取3，过滤半径 $r_{min}$ 取1.5倍单元尺寸
- MMA需要合理设置移动限制和渐近线更新策略
- 应力约束问题建议使用块聚合减少约束数量
- 多载荷工况优化需要权衡不同工况的重要性

## 练习题

### 基础题

**习题7.1** 证明SIMP插值下，当 $p > 1$ 时，中间密度（$0 < \rho < 1$）的材料效率低于线性插值。

<details>
<summary>提示</summary>
比较 $\rho^p$ 和 $\rho$ 在 $0 < \rho < 1$ 区间的大小关系。
</details>

<details>
<summary>答案</summary>

当 $p > 1$ 且 $0 < \rho < 1$ 时，有 $\rho^p < \rho$。

证明：设 $f(\rho) = \rho - \rho^p$，则：
$$f'(\rho) = 1 - p\rho^{p-1}$$

当 $\rho < 1$ 时，$\rho^{p-1} < 1$，因此当 $\rho < 1/p^{1/(p-1)}$ 时，$f'(\rho) > 0$。

由于 $f(0) = 0$ 且 $f(1) = 0$，函数在 $(0, 1)$ 区间先增后减，且 $f(\rho) > 0$。

因此 $\rho^p < \rho$，说明SIMP惩罚了中间密度。
</details>

**习题7.2** 推导二维四节点四边形单元的敏感度计算公式，假设单元厚度为 $t$，材料弹性模量为 $E$，泊松比为 $\nu$。

<details>
<summary>提示</summary>
从单元应变能出发，利用链式法则求导。
</details>

<details>
<summary>答案</summary>

单元应变能：
$$U_e = \frac{1}{2}\mathbf{u}_e^T \mathbf{k}_e \mathbf{u}_e = \frac{1}{2}\rho_e^p \mathbf{u}_e^T \mathbf{k}_e^0 \mathbf{u}_e$$

柔度对密度的导数：
$$\frac{\partial c}{\partial \rho_e} = -\frac{\partial U_e}{\partial \rho_e} = -\frac{p}{2}\rho_e^{p-1} \mathbf{u}_e^T \mathbf{k}_e^0 \mathbf{u}_e$$

考虑对称性，最终：
$$\frac{\partial c}{\partial \rho_e} = -p\rho_e^{p-1} \mathbf{u}_e^T \mathbf{k}_e^0 \mathbf{u}_e$$
</details>

**习题7.3** 给定过滤半径 $r_{min} = 1.5h$（$h$ 为单元尺寸），计算2D规则网格中心单元的过滤权重矩阵。

<details>
<summary>提示</summary>
使用线性衰减权重函数 $w = \max(0, r_{min} - d)$。
</details>

<details>
<summary>答案</summary>

对于单元间距为 $h$ 的规则网格，中心单元到邻居的距离：
- 直接相邻：$d = h$
- 对角相邻：$d = \sqrt{2}h \approx 1.414h$

权重计算：
- 中心单元：$w_0 = 1.5h - 0 = 1.5h$
- 直接相邻（4个）：$w_1 = 1.5h - h = 0.5h$
- 对角相邻（4个）：$w_2 = 1.5h - 1.414h = 0.086h$

归一化权重矩阵：
$$\mathbf{W} = \frac{1}{3.844h} \begin{bmatrix}
0.086h & 0.5h & 0.086h \\
0.5h & 1.5h & 0.5h \\
0.086h & 0.5h & 0.086h
\end{bmatrix}$$
</details>

### 挑战题

**习题7.4** 考虑带有频率约束的拓扑优化问题，推导第一阶固有频率对设计变量的敏感度。

<details>
<summary>提示</summary>
从广义特征值问题出发，使用伴随方法。
</details>

<details>
<summary>答案</summary>

广义特征值问题：
$$(\mathbf{K} - \omega^2 \mathbf{M})\boldsymbol{\phi} = \mathbf{0}$$

对 $\rho_e$ 求导：
$$\left(\frac{\partial \mathbf{K}}{\partial \rho_e} - \omega^2 \frac{\partial \mathbf{M}}{\partial \rho_e} - 2\omega\frac{\partial \omega}{\partial \rho_e}\mathbf{M}\right)\boldsymbol{\phi} + (\mathbf{K} - \omega^2 \mathbf{M})\frac{\partial \boldsymbol{\phi}}{\partial \rho_e} = \mathbf{0}$$

左乘 $\boldsymbol{\phi}^T$，利用质量归一化 $\boldsymbol{\phi}^T\mathbf{M}\boldsymbol{\phi} = 1$：

$$\frac{\partial \omega^2}{\partial \rho_e} = \boldsymbol{\phi}^T \left(\frac{\partial \mathbf{K}}{\partial \rho_e} - \omega^2 \frac{\partial \mathbf{M}}{\partial \rho_e}\right) \boldsymbol{\phi}$$

对于SIMP插值：
$$\frac{\partial \omega^2}{\partial \rho_e} = p\rho_e^{p-1}\boldsymbol{\phi}_e^T\mathbf{k}_e^0\boldsymbol{\phi}_e - \omega^2\boldsymbol{\phi}_e^T\mathbf{m}_e^0\boldsymbol{\phi}_e$$
</details>

**习题7.5** 分析MMA算法中渐近线位置对收敛性的影响，给出最优渐近线选择策略。

<details>
<summary>提示</summary>
考虑Hessian矩阵的正定性和曲率匹配。
</details>

<details>
<summary>答案</summary>

MMA近似的Hessian矩阵：
$$H_{jj} = \frac{2p_j}{(U_j - x_j)^3} + \frac{2q_j}{(x_j - L_j)^3}$$

为保证正定性和良好收敛，渐近线应满足：

1. **曲率匹配条件**：
   $$\frac{1}{U_j - x_j} + \frac{1}{x_j - L_j} \approx \frac{|\nabla^2 f|}{|\nabla f|}$$

2. **自适应策略**：
   - 振荡时（方向改变）：收缩渐近线，$s \leftarrow 0.7s$
   - 单调时（方向不变）：扩张渐近线，$s \leftarrow 1.2s$

3. **最优初始设置**：
   $$L_j^{(0)} = x_j^{(0)} - 0.5(x_j^{\max} - x_j^{\min})$$
   $$U_j^{(0)} = x_j^{(0)} + 0.5(x_j^{\max} - x_j^{\min})$$

这种策略平衡了局部收敛速度和全局稳定性。
</details>

**习题7.6** 设计一个自适应Heaviside投影策略，使得优化过程从连续松弛逐渐过渡到离散设计。

<details>
<summary>提示</summary>
考虑continuation方法，逐步增加投影陡度参数 $\beta$。
</details>

<details>
<summary>答案</summary>

自适应投影策略：

1. **初始阶段**（迭代1-50）：
   $$\beta = 1, \quad \text{无投影或弱投影}$$

2. **过渡阶段**（迭代51-150）：
   $$\beta^{(k)} = \min(1 + 0.1(k-50), 8)$$

3. **强化阶段**（迭代151-250）：
   $$\beta^{(k)} = \min(8 \times 2^{(k-150)/50}, 64)$$

4. **收敛准则调整**：
   ```
   if 灰度指标 < 0.05 and k > 100:
       β = min(2β, 64)
   ```

5. **稳定性保护**：
   ```
   if 目标函数振荡 > 5%:
       β = max(β/2, 1)
   ```

这种策略确保：
- 早期探索设计空间
- 中期逐渐清晰化
- 后期强制0-1设计
- 避免过早收敛到局部最优
</details>

**习题7.7** 推导考虑几何非线性的拓扑优化敏感度公式，假设采用Total Lagrangian格式。

<details>
<summary>提示</summary>
考虑切线刚度矩阵和几何刚度矩阵的贡献。
</details>

<details>
<summary>答案</summary>

几何非线性平衡方程：
$$\mathbf{F}_{int}(\mathbf{U}, \boldsymbol{\rho}) = \mathbf{F}_{ext}$$

切线刚度矩阵：
$$\mathbf{K}_T = \mathbf{K}_L + \mathbf{K}_G + \mathbf{K}_{u}$$

其中：
- $\mathbf{K}_L$：线性刚度矩阵
- $\mathbf{K}_G$：几何刚度矩阵
- $\mathbf{K}_{u}$：初始位移刚度矩阵

敏感度（使用伴随方法）：

1. 伴随方程：
   $$\mathbf{K}_T^T \boldsymbol{\lambda} = -\frac{\partial c}{\partial \mathbf{U}}$$

2. 目标函数敏感度：
   $$\frac{\partial c}{\partial \rho_e} = \frac{\partial c}{\partial \mathbf{U}}\frac{\partial \mathbf{U}}{\partial \rho_e} + \boldsymbol{\lambda}^T \frac{\partial \mathbf{F}_{int}}{\partial \rho_e}$$

3. 内力对密度的导数：
   $$\frac{\partial \mathbf{F}_{int}}{\partial \rho_e} = p\rho_e^{p-1} \int_{\Omega_e} \mathbf{B}_L^T \mathbf{S} \, d\Omega$$

其中 $\mathbf{S}$ 是第二Piola-Kirchhoff应力，$\mathbf{B}_L$ 是大变形应变-位移矩阵。

最终敏感度需要通过Newton-Raphson迭代中的增量求解获得。
</details>

## 常见陷阱与错误

### 1. 数值不稳定性
- **问题**：未加过滤导致棋盘格模式
- **解决**：始终使用密度或敏感度过滤，过滤半径至少1.2倍单元尺寸

### 2. 初始设计影响
- **问题**：均匀初始密度可能导致局部最优
- **解决**：尝试不同初始设计，如随机扰动或基于经验的非均匀分布

### 3. 惩罚参数选择
- **问题**：$p$ 值过大导致收敛困难
- **解决**：使用continuation方法，从 $p=1$ 开始逐步增加到 $p=3$

### 4. 体积约束过严
- **问题**：体积分数过小导致无可行解
- **解决**：从较大体积分数开始，逐步减小

### 5. MMA参数调试
- **问题**：移动限制过大导致振荡
- **解决**：自适应调整移动限制，监控目标函数历史

### 6. 应力约束处理
- **问题**：直接使用局部应力约束导致计算量爆炸
- **解决**：使用聚合函数或自适应约束删除策略

### 7. 网格依赖性
- **问题**：不同网格密度得到不同拓扑
- **解决**：固定过滤半径的物理尺寸，而非单元数量

### 8. 灰度单元处理
- **问题**：优化结果含大量中间密度
- **解决**：增加惩罚因子或使用Heaviside投影

## 最佳实践检查清单

### 问题定义阶段
- [ ] 明确定义设计域、载荷和边界条件
- [ ] 选择合适的目标函数（柔度、应力、频率等）
- [ ] 确定合理的体积约束或其他约束条件
- [ ] 评估是否需要考虑制造约束

### 数值实现阶段
- [ ] 选择合适的单元类型和网格密度
- [ ] 实现敏感度分析并验证（有限差分对比）
- [ ] 配置过滤方案（类型、半径）
- [ ] 选择优化算法（MMA、OC、GCMMA）
- [ ] 设置合理的收敛准则

### 优化过程监控
- [ ] 记录目标函数和约束历史
- [ ] 监控灰度指标变化
- [ ] 检查KKT条件满足程度
- [ ] 观察设计演化过程
- [ ] 识别和处理数值问题

### 后处理阶段
- [ ] 提取清晰的0-1设计
- [ ] 验证结构性能（FEA重分析）
- [ ] 检查制造可行性
- [ ] 进行网格无关性研究
- [ ] 评估对参数变化的敏感性

### 验证与确认
- [ ] 基准问题测试（悬臂梁、MBB梁）
- [ ] 与解析解或文献结果对比
- [ ] 参数研究（惩罚因子、过滤半径）
- [ ] 鲁棒性分析（初始设计、网格密度）
- [ ] 计算效率评估