# 第6章：有限元方法与结构分析

有限元方法（FEM）是3D打印结构分析的核心数学工具。本章从变分原理出发，系统介绍Galerkin方法、单元构造、系统组装和特征值分析，重点探讨3D打印特有的多物理场耦合问题。我们将深入推导数学原理，分析数值稳定性，并提供大量练习帮助掌握FEM在增材制造中的应用。

## 6.1 Galerkin方法与弱形式

### 6.1.1 强形式到弱形式的转换

考虑线弹性力学的平衡方程（强形式）：
$$-\nabla \cdot \boldsymbol{\sigma} = \mathbf{f} \quad \text{in } \Omega$$

其中 $\boldsymbol{\sigma}$ 是应力张量，$\mathbf{f}$ 是体力。本构关系为：
$$\boldsymbol{\sigma} = \mathbb{C} : \boldsymbol{\varepsilon}$$

应变-位移关系：
$$\boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)$$

边界条件：
- Dirichlet边界：$\mathbf{u} = \overline{\mathbf{u}}$ on $\Gamma_D$
- Neumann边界：$\boldsymbol{\sigma} \cdot \mathbf{n} = \overline{\mathbf{t}}$ on $\Gamma_N$

### 6.1.2 变分原理与虚功原理

引入试函数空间和测试函数空间：
$$\mathcal{V} = \{\mathbf{v} \in [H^1(\Omega)]^d : \mathbf{v}|_{\Gamma_D} = 0\}$$
$$\mathcal{S} = \{\mathbf{u} \in [H^1(\Omega)]^d : \mathbf{u}|_{\Gamma_D} = \overline{\mathbf{u}}\}$$

弱形式：找到 $\mathbf{u} \in \mathcal{S}$，使得对所有 $\mathbf{v} \in \mathcal{V}$：
$$\int_\Omega \boldsymbol{\varepsilon}(\mathbf{v}) : \mathbb{C} : \boldsymbol{\varepsilon}(\mathbf{u}) \, d\Omega = \int_\Omega \mathbf{v} \cdot \mathbf{f} \, d\Omega + \int_{\Gamma_N} \mathbf{v} \cdot \overline{\mathbf{t}} \, d\Gamma$$

双线性形式和线性泛函：
$$a(\mathbf{u}, \mathbf{v}) = \int_\Omega \boldsymbol{\varepsilon}(\mathbf{v}) : \mathbb{C} : \boldsymbol{\varepsilon}(\mathbf{u}) \, d\Omega$$
$$l(\mathbf{v}) = \int_\Omega \mathbf{v} \cdot \mathbf{f} \, d\Omega + \int_{\Gamma_N} \mathbf{v} \cdot \overline{\mathbf{t}} \, d\Gamma$$

### 6.1.3 Galerkin离散化

有限维逼近空间：
$$\mathcal{V}_h = \text{span}\{\phi_1, \phi_2, ..., \phi_n\} \subset \mathcal{V}$$

位移场的离散表示：
$$\mathbf{u}_h = \sum_{i=1}^n u_i \phi_i$$

离散弱形式：
$$\sum_{j=1}^n a(\phi_j, \phi_i) u_j = l(\phi_i), \quad i = 1, 2, ..., n$$

矩阵形式：
$$\mathbf{K}\mathbf{u} = \mathbf{f}$$

其中刚度矩阵 $K_{ij} = a(\phi_j, \phi_i)$，载荷向量 $f_i = l(\phi_i)$。

### 6.1.4 Lax-Milgram定理与适定性

问题适定性的充分条件：
1. **连续性**：$|a(\mathbf{u}, \mathbf{v})| \leq M \|\mathbf{u}\|_V \|\mathbf{v}\|_V$
2. **强制性**：$a(\mathbf{v}, \mathbf{v}) \geq \alpha \|\mathbf{v}\|_V^2$

对于线弹性问题，Korn不等式保证了强制性：
$$\|\boldsymbol{\varepsilon}(\mathbf{v})\|_{L^2} \geq C_K \|\mathbf{v}\|_{H^1}$$

条件数估计：
$$\kappa(\mathbf{K}) \leq \frac{M}{\alpha} \cdot \frac{H^2}{h^2}$$

其中 $H$ 是域尺寸，$h$ 是网格尺寸。

### 6.1.5 误差估计与收敛性

Céa引理给出最佳逼近性质：
$$\|\mathbf{u} - \mathbf{u}_h\|_V \leq \frac{M}{\alpha} \inf_{\mathbf{v}_h \in \mathcal{V}_h} \|\mathbf{u} - \mathbf{v}_h\|_V$$

对于 $p$ 阶多项式基函数，如果精确解 $\mathbf{u} \in H^{p+1}(\Omega)$：
$$\|\mathbf{u} - \mathbf{u}_h\|_{H^1} \leq C h^p |\mathbf{u}|_{H^{p+1}}$$
$$\|\mathbf{u} - \mathbf{u}_h\|_{L^2} \leq C h^{p+1} |\mathbf{u}|_{H^{p+1}}$$

#### Aubin-Nitsche技巧

对于 $L^2$ 误差的超收敛性，考虑对偶问题：
$$a(\mathbf{v}, \mathbf{z}) = (\mathbf{e}, \mathbf{v})_{L^2} \quad \forall \mathbf{v} \in \mathcal{V}$$

其中 $\mathbf{e} = \mathbf{u} - \mathbf{u}_h$。若对偶解满足正则性估计 $\|\mathbf{z}\|_{H^2} \leq C\|\mathbf{e}\|_{L^2}$，则：
$$\|\mathbf{e}\|_{L^2}^2 = a(\mathbf{e}, \mathbf{z}) = a(\mathbf{e}, \mathbf{z} - \mathbf{z}_h) \leq C\|\mathbf{e}\|_{H^1}\|\mathbf{z} - \mathbf{z}_h\|_{H^1}$$

#### 后验误差估计

残差型估计子：
$$\eta_K^2 = h_K^2 \|\mathbf{f} + \nabla \cdot \boldsymbol{\sigma}_h\|_{L^2(K)}^2 + \sum_{e \in \partial K} h_e \|[\boldsymbol{\sigma}_h \cdot \mathbf{n}]\|_{L^2(e)}^2$$

其中 $[\cdot]$ 表示跨单元边界的跳跃。全局估计：
$$\|\mathbf{u} - \mathbf{u}_h\|_{H^1} \leq C \left(\sum_{K \in \mathcal{T}_h} \eta_K^2\right)^{1/2}$$

### 6.1.6 稳定化方法

#### SUPG/PSPG稳定化

对于对流占优问题，添加稳定项：
$$a_{stab}(\mathbf{u}_h, \mathbf{v}_h) = \sum_K \tau_K \int_K (\mathbf{a} \cdot \nabla \mathbf{v}_h) \cdot \mathcal{R}(\mathbf{u}_h) \, d\Omega$$

其中残差 $\mathcal{R}(\mathbf{u}_h) = \mathbf{f} - \mathcal{L}\mathbf{u}_h$，稳定参数：
$$\tau_K = \frac{h_K}{2\|\mathbf{a}\|} \left(\coth(Pe_K) - \frac{1}{Pe_K}\right)$$

Péclet数 $Pe_K = \frac{\|\mathbf{a}\|h_K}{2\nu}$。

#### 最小二乘稳定化

对于混合问题，满足inf-sup条件的替代方案：
$$a_{LS}(\mathbf{u}_h, \mathbf{v}_h) = a(\mathbf{u}_h, \mathbf{v}_h) + \sum_K \delta_K (\mathcal{L}\mathbf{u}_h, \mathcal{L}\mathbf{v}_h)_{L^2(K)}$$

### 6.1.7 自适应网格细化

#### 误差指示子与标记策略

Dörfler标记：选择最小单元集合 $\mathcal{M} \subset \mathcal{T}_h$ 使得：
$$\sum_{K \in \mathcal{M}} \eta_K^2 \geq \theta \sum_{K \in \mathcal{T}_h} \eta_K^2$$

典型取 $\theta = 0.5$。

#### 细化与粗化

最长边二分法保证网格质量：
- 标记需细化的单元
- 递归标记邻居以保持相容性
- 执行局部细化
- 投影解到新网格

收敛性保证：存在常数 $\gamma < 1$ 使得：
$$\|\mathbf{u} - \mathbf{u}_{h_{n+1}}\|_{H^1} \leq \gamma \|\mathbf{u} - \mathbf{u}_{h_n}\|_{H^1} + \text{高阶项}$$

## 6.2 单元类型：四面体、六面体、高阶单元

### 6.2.1 四面体单元

#### 线性四面体（T4）
参考单元坐标：$(0,0,0)$, $(1,0,0)$, $(0,1,0)$, $(0,0,1)$

形函数：
$$N_1 = 1 - \xi - \eta - \zeta$$
$$N_2 = \xi$$
$$N_3 = \eta$$
$$N_4 = \zeta$$

雅可比矩阵：
$$\mathbf{J} = \begin{bmatrix}
x_2-x_1 & x_3-x_1 & x_4-x_1 \\
y_2-y_1 & y_3-y_1 & y_4-y_1 \\
z_2-z_1 & z_3-z_1 & z_4-z_1
\end{bmatrix}$$

体积：$V = \frac{1}{6}|\det(\mathbf{J})|$

#### 二次四面体（T10）
增加6个中节点，形函数包含二次项：
$$N_i = (2L_i - 1)L_i \quad \text{(角节点)}$$
$$N_{ij} = 4L_i L_j \quad \text{(边中点)}$$

其中 $L_i$ 是体积坐标。

### 6.2.2 六面体单元

#### 线性六面体（H8）
参考单元：$[-1,1]^3$

形函数：
$$N_i(\xi, \eta, \zeta) = \frac{1}{8}(1 + \xi_i\xi)(1 + \eta_i\eta)(1 + \zeta_i\zeta)$$

等参变换：
$$\mathbf{x} = \sum_{i=1}^8 N_i(\boldsymbol{\xi}) \mathbf{x}_i$$

#### 二次六面体（H20/H27）
Serendipity单元（H20）：仅边中点，无面心和体心
$$N_i = \frac{1}{8}(1+\xi_i\xi)(1+\eta_i\eta)(1+\zeta_i\zeta)(\xi_i\xi+\eta_i\eta+\zeta_i\zeta-2)$$

Lagrange单元（H27）：完全二次，包含所有节点

### 6.2.3 高阶谱单元

#### Legendre基函数
一维Legendre多项式递推：
$$P_0(x) = 1, \quad P_1(x) = x$$
$$P_{n+1}(x) = \frac{2n+1}{n+1}xP_n(x) - \frac{n}{n+1}P_{n-1}(x)$$

三维张量积基：
$$\phi_{ijk}(\xi, \eta, \zeta) = P_i(\xi)P_j(\eta)P_k(\zeta)$$

#### 分层基函数（Hierarchical）
边模式、面模式、体模式的分离：
- 顶点模式：线性形函数
- 边模式：$\phi_e^p = \frac{1-\xi^2}{4}P_{p-2}'(\xi)$
- 面模式：$\phi_f^{pq} = \frac{(1-\xi^2)(1-\eta^2)}{16}P_{p-2}'(\xi)P_{q-2}'(\eta)$

### 6.2.4 锁定现象与缓解策略

#### 体积锁定（不可压缩材料）
当泊松比 $\nu \to 0.5$，体积模量 $K \to \infty$

B-bar方法：修正应变-位移矩阵
$$\bar{\mathbf{B}} = \mathbf{B}_{dev} + \frac{1}{3}\bar{\mathbf{B}}_{vol}$$

其中 $\bar{\mathbf{B}}_{vol}$ 使用降阶积分计算。

#### 剪切锁定（薄结构）
减缩积分：使用较少的Gauss点
- 完全积分：$(p+1)^3$ 个Gauss点
- 减缩积分：$p^3$ 个Gauss点

沙漏控制：添加稳定项防止零能模式
$$\mathbf{K}_{stab} = \alpha \mathbf{K}_{hourglass}$$

### 6.2.5 单元质量评估

#### 雅可比比率
$$\rho_J = \frac{\min_{\xi \in \Omega_e} \det(\mathbf{J})}{\max_{\xi \in \Omega_e} \det(\mathbf{J})}$$

理想值：$\rho_J = 1$（均匀变形）

#### 长宽比（Aspect Ratio）
$$AR = \frac{\max(l_i)}{\min(l_i)}$$

其中 $l_i$ 是单元边长。建议 $AR < 10$。

#### 偏斜度（Skewness）
$$S = \frac{|V_{actual} - V_{ideal}|}{V_{ideal}}$$

建议 $S < 0.5$ 以保证精度。

#### 条件数估计

单元刚度矩阵条件数：
$$\kappa(K_e) \sim \left(\frac{\max(l_i)}{\min(l_i)}\right)^2 \cdot \frac{E}{\nu(1-2\nu)}$$

对于畸变单元，条件数急剧增加，导致数值不稳定。

### 6.2.6 p-自适应与hp-FEM

#### 误差指示子

基于Legendre展开的误差估计：
$$\eta_p = \|u_p - u_{p-1}\|_{H^1(K)}$$

谱收敛率：对于解析函数
$$\|u - u_p\|_{H^1} \leq C e^{-\alpha p}$$

#### hp-决策准则

光滑度指示子：
$$\sigma_K = \frac{\|u_p - u_{p-1}\|_{L^2(K)}}{\|u_p - u_{p-2}\|_{L^2(K)}}$$

- 若 $\sigma_K > \theta_p$：p-细化（解光滑）
- 若 $\sigma_K < \theta_h$：h-细化（解有奇异性）

### 6.2.7 积分规则与精度

#### Gauss-Legendre积分

$n$ 点规则精确积分 $2n-1$ 阶多项式：

| 积分点数 | 位置 $\xi_i$ | 权重 $w_i$ |
|---------|-------------|-----------|
| 1 | 0 | 2 |
| 2 | $\pm 1/\sqrt{3}$ | 1 |
| 3 | 0, $\pm\sqrt{3/5}$ | 8/9, 5/9 |

三维张量积：$(n_{Gauss})^3$ 个积分点

#### 积分精度要求

刚度矩阵积分：至少 $2p-1$ 阶精度（$p$ 为基函数阶数）
质量矩阵积分：至少 $2p$ 阶精度
非线性项：根据非线性度自适应选择

## 6.3 刚度矩阵组装与边界条件

### 6.3.1 局部到全局映射

#### 自由度编号

节点自由度映射：
$$\text{DOF}_{global} = \text{ndof} \times (\text{node\_id} - 1) + \text{dof\_component}$$

其中 $\text{ndof}$ 是每节点自由度数（如3D弹性为3）。

#### 连接矩阵

单元 $e$ 的连接矩阵 $\mathbf{L}^e$：
$$\mathbf{u}^e = \mathbf{L}^e \mathbf{u}_{global}$$

稀疏性：$\mathbf{L}^e$ 每行仅一个非零元素1。

### 6.3.2 刚度矩阵组装算法

#### 单元刚度矩阵计算

对于单元 $e$：
$$K_{ij}^e = \int_{\Omega_e} \mathbf{B}_i^T \mathbf{D} \mathbf{B}_j \, d\Omega$$

其中应变-位移矩阵：
$$\mathbf{B}_i = \begin{bmatrix}
\frac{\partial N_i}{\partial x} & 0 & 0 \\
0 & \frac{\partial N_i}{\partial y} & 0 \\
0 & 0 & \frac{\partial N_i}{\partial z} \\
\frac{\partial N_i}{\partial y} & \frac{\partial N_i}{\partial x} & 0 \\
0 & \frac{\partial N_i}{\partial z} & \frac{\partial N_i}{\partial y} \\
\frac{\partial N_i}{\partial z} & 0 & \frac{\partial N_i}{\partial x}
\end{bmatrix}$$

弹性矩阵 $\mathbf{D}$ （各向同性）：
$$\mathbf{D} = \frac{E}{(1+\nu)(1-2\nu)} \begin{bmatrix}
1-\nu & \nu & \nu & 0 & 0 & 0 \\
\nu & 1-\nu & \nu & 0 & 0 & 0 \\
\nu & \nu & 1-\nu & 0 & 0 & 0 \\
0 & 0 & 0 & \frac{1-2\nu}{2} & 0 & 0 \\
0 & 0 & 0 & 0 & \frac{1-2\nu}{2} & 0 \\
0 & 0 & 0 & 0 & 0 & \frac{1-2\nu}{2}
\end{bmatrix}$$

#### 组装过程

全局刚度矩阵：
$$\mathbf{K} = \sum_{e=1}^{n_{elem}} (\mathbf{L}^e)^T \mathbf{K}^e \mathbf{L}^e$$

稀疏存储格式：
- CSR (Compressed Sparse Row)：行压缩存储
- COO (Coordinate)：坐标格式，便于并行组装
- Skyline：利用带宽结构

### 6.3.3 边界条件处理

#### Dirichlet边界条件

**罚函数法**：
$$\mathbf{K}_{modified} = \mathbf{K} + \alpha \mathbf{P}$$

其中 $\mathbf{P}_{ii} = 1$ 若节点 $i$ 受约束，$\alpha \sim 10^{10} \max(K_{ii})$。

**Lagrange乘子法**：
$$\begin{bmatrix}
\mathbf{K} & \mathbf{G}^T \\
\mathbf{G} & \mathbf{0}
\end{bmatrix} \begin{bmatrix}
\mathbf{u} \\
\boldsymbol{\lambda}
\end{bmatrix} = \begin{bmatrix}
\mathbf{f} \\
\mathbf{g}
\end{bmatrix}$$

**消去法**（推荐）：
将自由度分为约束 $(c)$ 和自由 $(f)$：
$$\begin{bmatrix}
\mathbf{K}_{ff} & \mathbf{K}_{fc} \\
\mathbf{K}_{cf} & \mathbf{K}_{cc}
\end{bmatrix} \begin{bmatrix}
\mathbf{u}_f \\
\mathbf{u}_c
\end{bmatrix} = \begin{bmatrix}
\mathbf{f}_f \\
\mathbf{f}_c
\end{bmatrix}$$

求解缩减系统：
$$\mathbf{K}_{ff} \mathbf{u}_f = \mathbf{f}_f - \mathbf{K}_{fc} \mathbf{u}_c$$

#### Neumann边界条件

表面力积分：
$$\mathbf{f}_i = \int_{\Gamma_N} N_i \overline{\mathbf{t}} \, d\Gamma$$

对于线性单元，均布载荷：
$$\mathbf{f}_{node} = \frac{A_{face}}{n_{nodes}} \overline{\mathbf{t}}$$

#### 周期边界条件

主从节点约束：
$$\mathbf{u}_{slave} = \mathbf{u}_{master} + \boldsymbol{\Delta}$$

通过约束矩阵 $\mathbf{C}$ 实现：
$$\mathbf{C} \mathbf{u} = \mathbf{d}$$

### 6.3.4 求解器选择

#### 直接求解器

**LU分解**：
$$\mathbf{K} = \mathbf{L}\mathbf{U}$$

复杂度：$O(n^3)$ （密集），$O(n^{3/2})$ （2D稀疏），$O(n^2)$ （3D稀疏）

**Cholesky分解**（对称正定）：
$$\mathbf{K} = \mathbf{L}\mathbf{L}^T$$

存储和计算减半。

#### 迭代求解器

**共轭梯度法（CG）**：
适用于对称正定系统，收敛率：
$$\|\mathbf{e}_k\|_{\mathbf{K}} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \|\mathbf{e}_0\|_{\mathbf{K}}$$

**预条件**：
- Jacobi：$\mathbf{M} = \text{diag}(\mathbf{K})$
- ILU(0)：不完全LU分解
- AMG：代数多重网格

### 6.3.5 并行化策略

#### 域分解

非重叠域分解（Schur补）：
$$\mathbf{S} = \mathbf{K}_{\Gamma\Gamma} - \mathbf{K}_{\Gamma I} \mathbf{K}_{II}^{-1} \mathbf{K}_{I\Gamma}$$

其中 $\Gamma$ 是界面，$I$ 是内部。

#### FETI方法

Lagrange乘子强制界面连续性：
$$\sum_{s=1}^{n_s} \mathbf{B}^{(s)} \mathbf{u}^{(s)} = \mathbf{0}$$

对偶问题：
$$\mathbf{F}\boldsymbol{\lambda} = \mathbf{d}$$

其中 $\mathbf{F} = \sum_s \mathbf{B}^{(s)} \mathbf{K}^{(s)+} \mathbf{B}^{(s)T}$。