# 第24章：仿生设计与自然算法

本章探讨自然界中的数学模式及其在3D打印设计中的应用。我们将从Voronoi图和Delaunay三角化的计算几何基础出发，深入研究反应扩散系统产生的图灵斑图，探索最小曲面的变分原理，分析骨骼自适应生长的力学机制，并介绍群体智能在形状优化中的应用。这些仿生算法不仅能生成美观的结构，更重要的是它们往往具有优异的力学性能和材料效率。

## 24.1 Voronoi图与Delaunay三角化

### 24.1.1 Voronoi图的数学定义

给定欧氏空间 $\mathbb{R}^d$ 中的点集 $P = \{p_1, p_2, ..., p_n\}$，点 $p_i$ 的Voronoi胞元定义为：

$$V(p_i) = \{x \in \mathbb{R}^d : \|x - p_i\| \leq \|x - p_j\|, \forall j \neq i\}$$

Voronoi图 $\text{Vor}(P)$ 是所有Voronoi胞元的集合。对于两个相邻的生成点 $p_i, p_j$，它们的Voronoi边界是垂直平分超平面：

$$H_{ij} = \{x \in \mathbb{R}^d : \|x - p_i\| = \|x - p_j\|\}$$

### 24.1.2 Delaunay三角化与对偶关系

Delaunay三角化 $\text{Del}(P)$ 是Voronoi图的对偶结构。在二维情况下，如果两个Voronoi胞元共享一条边，则对应的生成点在Delaunay三角化中相连。

**空圆性质**：一个三角化是Delaunay三角化当且仅当每个三角形的外接圆内部不包含其他点。对于点 $p_i, p_j, p_k$ 构成的三角形，外接圆判定条件为：

$$\begin{vmatrix}
x_i - x_k & y_i - y_k & (x_i - x_k)^2 + (y_i - y_k)^2 \\
x_j - x_k & y_j - y_k & (x_j - x_k)^2 + (y_j - y_k)^2 \\
x_l - x_k & y_l - y_k & (x_l - x_k)^2 + (y_l - y_k)^2
\end{vmatrix} > 0$$

其中 $p_l$ 是待测试的第四个点。

### 24.1.3 增量构造算法

Bowyer-Watson算法通过逐点插入构造Delaunay三角化：

1. **初始化**：构造包含所有点的超三角形
2. **点插入**：对每个新点 $p$：
   - 找出所有外接圆包含 $p$ 的三角形（坏三角形）
   - 删除这些三角形，形成多边形空腔
   - 将 $p$ 与空腔边界连接，形成新三角形
3. **清理**：删除与超三角形相关的三角形

时间复杂度：最坏情况 $O(n^2)$，期望 $O(n \log n)$。

### 24.1.4 受限Delaunay三角化

在实际应用中，常需要保持特定边界。受限Delaunay三角化（CDT）在保持给定边的同时尽可能满足Delaunay性质。对于边 $e$，如果存在圆经过 $e$ 的端点且内部不包含其他可见点，则 $e$ 是局部Delaunay的。

### 24.1.5 权重Voronoi图与各向异性

加权Voronoi图（功率图）中，每个生成点 $p_i$ 具有权重 $w_i$：

$$V_w(p_i) = \{x \in \mathbb{R}^d : \|x - p_i\|^2 - w_i \leq \|x - p_j\|^2 - w_j, \forall j \neq i\}$$

各向异性Voronoi图使用非欧氏度量，如：

$$d_M(x, p) = \sqrt{(x - p)^T M (x - p)}$$

其中 $M$ 是正定矩阵，定义了局部的各向异性。

### 24.1.6 Centroidal Voronoi Tessellation (CVT)

CVT是一种特殊的Voronoi图，其中每个生成点位于其Voronoi胞元的质心：

$$p_i = \frac{\int_{V(p_i)} x \rho(x) dx}{\int_{V(p_i)} \rho(x) dx}$$

Lloyd算法迭代计算CVT：
1. 计算当前Voronoi图
2. 将每个生成点移动到其胞元质心
3. 重复直到收敛

能量函数：
$$E = \sum_{i=1}^n \int_{V(p_i)} \rho(x) \|x - p_i\|^2 dx$$

### 24.1.7 高维Voronoi图与计算复杂度

在 $d$ 维空间中，Voronoi图的组合复杂度急剧增长：

**最坏情况复杂度**：
- 2D：$O(n)$ 个顶点、边和面
- 3D：$O(n^2)$ 个顶点、边、面和胞元
- $d$维：$O(n^{\lceil d/2 \rceil})$

**Fortune算法扫描线**（2D）：
扫描线算法使用抛物线前沿（beach line）维护Voronoi图：
$$y = \frac{(x - x_i)^2 + (y_l - y_i)^2}{2(y_l - y_i)}$$

其中 $(x_i, y_i)$ 是站点，$y_l$ 是扫描线位置。

**增量算法的退化处理**：
使用符号扰动避免共球情况：
$$\tilde{p}_i = p_i + \epsilon^i \mathbf{e}_i$$

其中 $\epsilon$ 是无穷小量，$\mathbf{e}_i$ 是标准基向量。

### 24.1.8 约束Voronoi图与网格生成

**受限Voronoi图（RVT）**：
给定域 $\Omega \subset \mathbb{R}^d$ 和站点集 $P$，受限Voronoi胞元：
$$V_\Omega(p_i) = V(p_i) \cap \Omega$$

**最优传输与Voronoi图**：
Monge-Kantorovich问题的半离散形式：
$$\min_{\{V_i\}} \sum_{i=1}^n \int_{V_i} c(\mathbf{x}, p_i) \rho(\mathbf{x}) d\mathbf{x}$$

当 $c(\mathbf{x}, p_i) = \|\mathbf{x} - p_i\|^2$ 时，最优分割是加权Voronoi图。

**Lloyd松弛的收敛性**：
能量函数 $E$ 关于站点位置的梯度：
$$\nabla_{p_i} E = 2m_i(p_i - c_i)$$

其中 $m_i = \int_{V_i} \rho d\mathbf{x}$ 是质量，$c_i$ 是质心。

收敛速率：线性收敛，收敛因子 $\rho < 1$ 依赖于密度函数的Lipschitz常数。

### 24.1.9 各向异性网格与黎曼度量

**黎曼Voronoi图**：
在配备度量张量场 $\mathcal{M}(\mathbf{x})$ 的流形上：
$$d_\mathcal{M}(\mathbf{x}, \mathbf{y}) = \inf_\gamma \int_0^1 \sqrt{\dot{\gamma}(t)^T \mathcal{M}(\gamma(t)) \dot{\gamma}(t)} dt$$

其中 $\gamma$ 是连接 $\mathbf{x}$ 和 $\mathbf{y}$ 的路径。

**度量插值**：
给定顶点度量 $\mathcal{M}_i$，三角形内部的插值：
$$\mathcal{M}(\mathbf{x}) = \sum_{i=1}^3 \lambda_i(\mathbf{x}) \log \mathcal{M}_i$$

使用对数映射保证正定性。

**各向异性误差估计**：
对于有限元逼近，最优度量张量：
$$\mathcal{M}_{\text{opt}} = \det(|\mathcal{H}|)^{-1/(2p+d)} |\mathcal{H}|$$

其中 $\mathcal{H}$ 是解的Hessian矩阵，$p$ 是插值阶数。

### 24.1.10 3D打印中的应用

**多孔结构设计**：
Voronoi泡沫的力学性能建模：

相对密度与壁厚关系：
$$\bar{\rho} = C_g \left(\frac{t}{\ell}\right) + C_s \left(\frac{t}{\ell}\right)^2$$

其中 $C_g \approx 3$ 是几何因子，$\ell$ 是胞元尺寸。

有效杨氏模量（开孔泡沫）：
$$\frac{E^*}{E_s} = C_1 \bar{\rho}^2 \left[1 + C_2 \left(\frac{t}{\ell}\right)^2\right]$$

屈服强度：
$$\frac{\sigma_y^*}{\sigma_{ys}} = C_3 \bar{\rho}^{3/2} \left[1 + C_4 \bar{\rho}^{1/2}\right]$$

**梯度结构优化**：
密度场驱动的站点分布：
$$\lambda(\mathbf{x}) = \lambda_0 \left(\frac{\sigma_{\text{vm}}(\mathbf{x})}{\sigma_{\text{max}}}\right)^\alpha$$

其中 $\sigma_{\text{vm}}$ 是von Mises应力，$\alpha$ 控制梯度强度。

**Poisson盘采样算法**：
1. 初始化：随机选择第一个点
2. 活跃列表：维护候选点集合
3. 生成新点：在环形区域 $[r, 2r]$ 内采样
4. 冲突检测：使用空间哈希加速
5. 密度适应：$r(\mathbf{x}) = \frac{k}{\sqrt[d]{\lambda(\mathbf{x})}}$

**生物医学支架设计**：
互连孔隙的渗透率（Darcy定律）：
$$k = \frac{\phi^3 d_p^2}{180(1-\phi)^2}$$

其中 $d_p$ 是平均孔径，使用Voronoi面积分布估计。

表面积与体积比：
$$\frac{S}{V} = \frac{4\phi(1-\phi)}{d_p}$$

优化目标：最大化渗透率同时保持力学强度。

## 24.2 反应扩散系统：Gray-Scott模型

### 24.2.1 反应扩散方程的一般形式

反应扩散系统描述了化学物质在空间中的扩散和相互反应：

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + f(u, v)$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + g(u, v)$$

其中 $u, v$ 是物质浓度，$D_u, D_v$ 是扩散系数，$f, g$ 是反应项。

### 24.2.2 Gray-Scott模型

Gray-Scott模型描述了两种化学物质的反应：
- 物质U（基质）：持续供给，自然衰减
- 物质V（催化剂）：消耗U并自我复制

反应机制：
$$U + 2V \rightarrow 3V$$
$$V \rightarrow P$$

偏微分方程：
$$\frac{\partial u}{\partial t} = D_u \nabla^2 u - uv^2 + F(1 - u)$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + uv^2 - (F + k)v$$

参数含义：
- $F$：进料率（U的补充速率）
- $k$：移除率（V的衰减速率）
- $D_u/D_v$：扩散系数比，典型值为2

### 24.2.3 线性稳定性分析

均匀稳态解 $(u_0, v_0)$ 满足：
$$F(1 - u_0) - u_0 v_0^2 = 0$$
$$(F + k)v_0 - u_0 v_0^2 = 0$$

对小扰动 $(\delta u, \delta v) \sim e^{\lambda t + ik \cdot x}$ 进行线性化：

$$\lambda = \text{tr}(J) - D|k|^2 \pm \sqrt{(\text{tr}(J) - D|k|^2)^2 - 4(\det(J) - D_u D_v |k|^4)}$$

其中Jacobian矩阵：
$$J = \begin{pmatrix}
-F - v_0^2 & -2u_0 v_0 \\
v_0^2 & 2u_0 v_0 - F - k
\end{pmatrix}$$

图灵不稳定性条件：
1. $\text{tr}(J) < 0$（均匀态稳定）
2. $\det(J) > 0$
3. $(D_u \text{tr}(J))^2 > 4D_u D_v \det(J)$（扩散诱导不稳定）

### 24.2.4 参数空间与斑图形态

不同参数区域产生不同的斑图：

- **斑点**（Spots）：$F \approx 0.030, k \approx 0.062$
- **条纹**（Stripes）：$F \approx 0.030, k \approx 0.057$
- **迷宫**（Labyrinth）：$F \approx 0.029, k \approx 0.057$
- **螺旋**（Spirals）：$F \approx 0.014, k \approx 0.054$
- **混沌**（Chaos）：$F \approx 0.026, k \approx 0.051$

临界波数：
$$k_c = \sqrt{\frac{\det(J)}{D_u D_v}}$$

对应的特征长度尺度：
$$L = \frac{2\pi}{k_c} = 2\pi\sqrt{\frac{D_u D_v}{\det(J)}}$$

### 24.2.5 数值求解方法

使用交替方向隐式（ADI）方法求解：

空间离散化（五点差分）：
$$\nabla^2 u_{i,j} = \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2}$$

时间分裂：
1. X方向隐式：$(I - \frac{\Delta t D_u}{2h^2} \delta_x^2) u^* = u^n + \Delta t(D_u \delta_y^2 u^n + R^n)$
2. Y方向隐式：$(I - \frac{\Delta t D_u}{2h^2} \delta_y^2) u^{n+1} = u^*$

稳定性条件（显式格式）：
$$\Delta t < \frac{h^2}{4\max(D_u, D_v)}$$

### 24.2.6 三维扩展与曲面反应扩散

在三维曲面上的反应扩散：
$$\frac{\partial u}{\partial t} = D_u \Delta_S u + f(u, v)$$

其中 $\Delta_S$ 是曲面Laplace-Beltrami算子。离散化使用余切权重：

$$\Delta_S u_i = \frac{1}{2A_i} \sum_{j \in N(i)} (\cot \alpha_{ij} + \cot \beta_{ij})(u_j - u_i)$$

### 24.2.7 非线性动力学与分岔分析

**Hopf分岔**：
当参数跨越临界值时，稳定焦点转变为极限环。特征值穿越虚轴：
$$\lambda = \alpha(F, k) \pm i\omega(F, k)$$

Hopf分岔条件：
- $\alpha(F_c, k_c) = 0$
- $\omega(F_c, k_c) \neq 0$
- $\frac{d\alpha}{dF}|_{F_c} \neq 0$ （横截条件）

**Turing-Hopf分岔**：
空间模式与时间振荡的相互作用产生行波和螺旋波：
$$u(\mathbf{x}, t) = u_0 + A\cos(\mathbf{k} \cdot \mathbf{x} - \omega t + \phi)$$

色散关系：
$$\omega^2 = c^2 k^2 + \omega_0^2$$

其中 $c$ 是波速，$\omega_0$ 是本征频率。

**振幅方程**（弱非线性分析）：
近临界点的慢变振幅 $A(\mathbf{x}, t)$ 满足Ginzburg-Landau方程：
$$\frac{\partial A}{\partial t} = \sigma A + \xi \nabla^2 A - g|A|^2 A$$

其中 $\sigma = \epsilon(F - F_c)$ 是线性增长率，$g$ 是非线性饱和系数。

### 24.2.8 多物种反应扩散系统

**Brusselator模型**：
自催化振荡反应：
$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + a - (b+1)u + u^2v$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + bu - u^2v$$

图灵不稳定性条件：$b > 1 + a^2$ 且 $D_v/D_u > (b-1+a^2)^2/(4a^2b)$

**FitzHugh-Nagumo模型**：
神经元激发的简化模型：
$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + u - \frac{u^3}{3} - v$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + \epsilon(u + a - bv)$$

行波解：$u(\mathbf{x}, t) = U(\mathbf{x} - ct)$ 满足：
$$c U' = D_u U'' + U - \frac{U^3}{3} - V$$
$$c V' = D_v V'' + \epsilon(U + a - bV)$$

**Schnakenberg模型**：
三分子自催化：
$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + \gamma(a - u + u^2v)$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + \gamma(b - u^2v)$$

稳态：$(u_0, v_0) = (a + b, b/(a+b)^2)$

### 24.2.9 数值方法的稳定性与精度

**谱方法**（周期边界条件）：
傅里叶变换：
$$\hat{u}_k = \mathcal{F}[u] = \int u(\mathbf{x}) e^{-i\mathbf{k} \cdot \mathbf{x}} d\mathbf{x}$$

演化方程：
$$\frac{d\hat{u}_k}{dt} = -D_u k^2 \hat{u}_k + \mathcal{F}[f(u, v)]_k$$

使用FFT计算非线性项，复杂度 $O(N \log N)$。

**指数时间差分（ETD）方法**：
精确处理线性部分：
$$u^{n+1} = e^{\Delta t L} u^n + \int_0^{\Delta t} e^{(\Delta t - \tau)L} N(u(\tau)) d\tau$$

其中 $L$ 是线性算子，$N$ 是非线性项。

ETD2阶Runge-Kutta：
$$a_n = e^{\Delta t L/2} u^n + \frac{e^{\Delta t L/2} - I}{L} N(u^n)$$
$$u^{n+1} = e^{\Delta t L} u^n + \frac{e^{\Delta t L} - I}{L} N(u^n) + \frac{e^{\Delta t L} - I - \Delta t L}{\Delta t L^2}(N(a_n) - N(u^n))$$

**自适应网格细化（AMR）**：
误差指示器：
$$\eta_i = h^2 \|\nabla^2 u\|_{L^2(\Omega_i)}$$

细化准则：$\eta_i > \theta_{\text{refine}} \max_j \eta_j$
粗化准则：$\eta_i < \theta_{\text{coarsen}} \max_j \eta_j$

### 24.2.10 在3D打印中的应用

**仿生纹理生成的参数控制**：

初始条件设计：
- 单点扰动：$u(x, y, 0) = u_0 + A\delta(x - x_0, y - y_0)$
- 随机噪声：$u(x, y, 0) = u_0 + \xi(x, y)$，$\xi \sim \mathcal{N}(0, \sigma^2)$
- 预设模板：$u(x, y, 0) = u_0 + A\sum_i \exp(-\|\mathbf{x} - \mathbf{x}_i\|^2/2\sigma^2)$

边界条件影响：
- Neumann边界：$\nabla u \cdot \mathbf{n} = 0$ 产生垂直于边界的条纹
- Dirichlet边界：$u = u_0$ 抑制边界附近的图案
- 周期边界：实现无缝平铺纹理

**功能表面的定量设计**：

疏水性优化（Cassie-Baxter模型）：
$$\cos \theta^* = f_s \cos \theta_s + f_a \cos \theta_a$$

其中 $\theta^*$ 是表观接触角，$f_s, f_a$ 是固液和气液界面分数。

使用反应扩散生成的柱状结构：
$$f_s = \frac{\pi r^2}{L^2}, \quad f_a = 1 - f_s$$

结构色设计（光子晶体）：
布拉格条件：$2nd\sin\theta = m\lambda$

通过控制反应扩散的特征波长 $L = 2\pi/k_c$ 实现特定波长的反射：
$$L = \frac{\lambda}{2n\sin\theta}$$

**力学性能的图案化增强**：

各向异性刚度张量：
$$\mathbb{C}_{ij} = \mathbb{C}_0 + \Delta\mathbb{C} \cdot \Phi(\mathbf{x})$$

其中 $\Phi(\mathbf{x})$ 是反应扩散生成的图案函数。

有效模量的Hashin-Shtrikman界：
$$E_{\text{lower}} \leq E_{\text{eff}} \leq E_{\text{upper}}$$

$$E_{\text{upper}} = E_1 + \frac{f_2}{1/(E_2 - E_1) + f_1/(3K_1)}$$

**多尺度嵌套结构**：

分级反应扩散：
$$\frac{\partial u_i}{\partial t} = D_i \nabla^2 u_i + f_i(u_i, v_i) + \epsilon_{i,i+1} g(u_{i+1})$$

其中 $\epsilon_{i,i+1}$ 是尺度耦合强度，$g$ 是耦合函数。

特征尺度分离：$L_{i+1}/L_i > 5$ 避免模式锁定。

## 24.3 最小曲面与肥皂膜

### 24.3.1 平均曲率与最小曲面方程

曲面 $S$ 的平均曲率定义为两个主曲率的平均值：

$$H = \frac{\kappa_1 + \kappa_2}{2}$$

对于参数化曲面 $\mathbf{r}(u, v)$，平均曲率可表示为：

$$H = \frac{eg - 2fF + fG}{2(EG - F^2)^{3/2}}$$

其中第一基本形式系数：
- $E = \mathbf{r}_u \cdot \mathbf{r}_u$
- $F = \mathbf{r}_u \cdot \mathbf{r}_v$  
- $G = \mathbf{r}_v \cdot \mathbf{r}_v$

第二基本形式系数：
- $e = \mathbf{n} \cdot \mathbf{r}_{uu}$
- $f = \mathbf{n} \cdot \mathbf{r}_{uv}$
- $g = \mathbf{n} \cdot \mathbf{r}_{vv}$

最小曲面满足 $H = 0$，即最小曲面方程：

$$(1 + f_y^2)f_{xx} - 2f_x f_y f_{xy} + (1 + f_x^2)f_{yy} = 0$$

对于图形式曲面 $z = f(x, y)$。

### 24.3.2 变分原理与Plateau问题

最小曲面是面积泛函的极值点：

$$A[S] = \int_S dA = \int_D \sqrt{EG - F^2} \, du \, dv$$

Euler-Lagrange方程导出平均曲率流：

$$\frac{\partial \mathbf{r}}{\partial t} = -2H\mathbf{n}$$

**Plateau问题**：给定闭合曲线 $\Gamma$，求以 $\Gamma$ 为边界的最小曲面。

Douglas-Radó定理：对于任意Jordan曲线，存在以其为边界的最小曲面。

### 24.3.3 经典最小曲面

**Catenoid（悬链面）**：
$$x = c \cosh(z/c) \cos \theta$$
$$y = c \cosh(z/c) \sin \theta$$

第一基本形式：$ds^2 = c^2 \cosh^2(z/c)(dz^2 + d\theta^2)$

**Helicoid（螺旋面）**：
$$\mathbf{r}(u, v) = (v\cos u, v\sin u, au)$$

Helicoid和Catenoid通过参数变形相关联（关联族）。

**Scherk曲面**：
$$z = \ln\left|\frac{\cos y}{\cos x}\right|$$

满足 $e^z = \cos y / \cos x$，在 $x = \pm\pi/2, y = \pm\pi/2$ 处有奇点。

**Enneper曲面**：
$$x = u - \frac{u^3}{3} + uv^2$$
$$y = v - \frac{v^3}{3} + vu^2$$
$$z = u^2 - v^2$$

具有自交性，但局部是最小曲面。

### 24.3.4 三重周期最小曲面（TPMS）

TPMS在三个方向上周期重复，隐式方程形式：

**Schwarz P曲面**：
$$\cos x + \cos y + \cos z = 0$$

**Schwarz D曲面**：
$$\sin x \sin y \sin z + \sin x \cos y \cos z + \cos x \sin y \cos z + \cos x \cos y \sin z = 0$$

**Gyroid**：
$$\sin x \cos y + \sin y \cos z + \sin z \cos x = 0$$

体积分数计算：
$$\phi = \frac{1}{V_{\text{cell}}} \int_{f(\mathbf{x}) > t} d\mathbf{x}$$

其中 $t$ 是水平集参数。

### 24.3.5 离散最小曲面

**离散平均曲率**：
$$H_i = \frac{1}{4A_i} \sum_{j \in N(i)} (\cot \alpha_{ij} + \cot \beta_{ij}) \|\mathbf{x}_i - \mathbf{x}_j\|$$

**肥皂膜算法**（Surface Evolver）：
1. 初始化三角网格
2. 计算每个顶点的平均曲率向量
3. 更新顶点位置：$\mathbf{x}_i^{n+1} = \mathbf{x}_i^n - \Delta t \cdot 2H_i \mathbf{n}_i$
4. 保持边界固定
5. 重复直到收敛

能量下降保证：选择足够小的时间步长 $\Delta t$。

### 24.3.6 水平集方法求解

使用水平集函数 $\phi$ 表示曲面：$S = \{\mathbf{x} : \phi(\mathbf{x}) = 0\}$

平均曲率：
$$H = \nabla \cdot \left(\frac{\nabla \phi}{|\nabla \phi|}\right)$$

演化方程：
$$\frac{\partial \phi}{\partial t} + v|\nabla \phi| = 0$$

其中速度场 $v = -2H$。

重初始化保持符号距离函数：
$$\frac{\partial \phi}{\partial \tau} = \text{sign}(\phi_0)(1 - |\nabla \phi|)$$

### 24.3.7 最小曲面的几何不变量

**Weierstrass-Enneper表示**：
任何最小曲面可由全纯函数 $f$ 和亚纯函数 $g$ 参数化：
$$\mathbf{r}(z) = \Re \int_\gamma \left( f(1-g^2), if(1+g^2), 2fg \right) dz$$

其中 $z$ 是复参数，$\gamma$ 是积分路径。

高斯曲率：
$$K = -\left|\frac{4g'}{f(1+|g|^2)^2}\right|^2$$

第一基本形式：
$$ds^2 = |f|^2(1+|g|^2)^2 |dz|^2$$

**Gauss映射与全曲率**：
Gauss映射 $\mathcal{G}: S \to S^2$ 将曲面点映到单位法向量。
对于完备最小曲面，全曲率：
$$\int_S K \, dA = -4\pi \chi(S)$$

其中 $\chi(S)$ 是Euler特征数。

**共轭曲面族**：
最小曲面的单参数族：
$$\mathbf{r}_\theta = \cos\theta \cdot \mathbf{r} + \sin\theta \cdot \mathbf{r}^*$$

其中 $\mathbf{r}^*$ 是共轭最小曲面，满足Cauchy-Riemann方程。

### 24.3.8 TPMS的晶体学分类

**空间群对称性**：
TPMS按其对称性分为不同空间群：

- **立方族**（$m\bar{3}m$）：
  - P曲面：$Pm\bar{3}m$ 空间群
  - D曲面：$Pn\bar{3}m$ 空间群  
  - Gyroid：$Ia\bar{3}d$ 空间群

- **六方族**：
  - H曲面：$P6_3/mmc$ 空间群

**Bonnet变换**：
保持平均曲率的等距变形：
$$\mathbf{r}_t = \cos t \cdot \mathbf{n} + \sin t \cdot \mathbf{n} \times \nabla_S H$$

其中 $\mathbf{n}$ 是法向量，$\nabla_S H$ 是平均曲率的表面梯度。

**水平集表示的优化**：
多参数TPMS族：
$$\phi(\mathbf{x}) = \sum_{i=1}^N a_i \cos(\mathbf{k}_i \cdot \mathbf{x} + \phi_i) = t$$

通过调整系数 $a_i$ 和相位 $\phi_i$ 控制形态。

### 24.3.9 数值计算方法的高级技术

**相场方法**：
使用Allen-Cahn方程演化到最小曲面：
$$\frac{\partial \phi}{\partial t} = \Delta \phi - \frac{1}{\epsilon^2}W'(\phi)$$

其中双井势 $W(\phi) = \frac{1}{4}(\phi^2 - 1)^2$。

界面厚度：$\delta \sim \epsilon$
表面张力：$\sigma = \frac{2\sqrt{2}}{3}\epsilon$

**离散外微分（DEC）**：
在离散流形上定义微分算子：

离散外导数：
$$d\omega = \sum_{i<j} \omega_{ij} dx_i \wedge dx_j$$

离散Hodge星算子：
$$\star_k \omega = \frac{|\star\sigma_k|}{|\sigma_k|} \omega$$

离散Laplace-deRham算子：
$$\Delta = d\delta + \delta d = \star d \star d + d \star d \star$$

**多重网格方法**：
层次化求解最小曲面问题：

限制算子：$I_h^{2h}: \Omega_h \to \Omega_{2h}$
延拓算子：$I_{2h}^h: \Omega_{2h} \to \Omega_h$

V-循环：
1. 前光滑：$u_h^{(1)} = S_h^{\nu_1}(u_h^{(0)}, f_h)$
2. 残差计算：$r_h = f_h - A_h u_h^{(1)}$
3. 限制：$r_{2h} = I_h^{2h} r_h$
4. 粗网格修正：求解 $A_{2h} e_{2h} = r_{2h}$
5. 延拓修正：$u_h^{(2)} = u_h^{(1)} + I_{2h}^h e_{2h}$
6. 后光滑：$u_h^{(3)} = S_h^{\nu_2}(u_h^{(2)}, f_h)$

### 24.3.10 3D打印应用的定量分析

**轻量化结构的力学模型**：

TPMS的均匀化刚度张量（对于立方对称）：
$$\mathbb{C}^* = \begin{pmatrix}
C_{11} & C_{12} & C_{12} & 0 & 0 & 0 \\
C_{12} & C_{11} & C_{12} & 0 & 0 & 0 \\
C_{12} & C_{12} & C_{11} & 0 & 0 & 0 \\
0 & 0 & 0 & C_{44} & 0 & 0 \\
0 & 0 & 0 & 0 & C_{44} & 0 \\
0 & 0 & 0 & 0 & 0 & C_{44}
\end{pmatrix}$$

其中：
$$C_{11} = E_s \bar{\rho} \alpha_1, \quad C_{12} = E_s \bar{\rho} \alpha_2, \quad C_{44} = E_s \bar{\rho} \alpha_3$$

系数 $\alpha_i$ 依赖于具体TPMS类型：
- Gyroid: $\alpha_1 = 0.38$, $\alpha_2 = 0.11$, $\alpha_3 = 0.13$
- P曲面: $\alpha_1 = 0.42$, $\alpha_2 = 0.15$, $\alpha_3 = 0.14$
- D曲面: $\alpha_1 = 0.45$, $\alpha_2 = 0.17$, $\alpha_3 = 0.14$

**生物支架的传质分析**：

有效扩散系数（曲折度模型）：
$$D_{\text{eff}} = \frac{D_0 \phi}{\tau^2}$$

其中曲折度 $\tau$ 与几何相关：
$$\tau = 1 + p(1 - \phi)$$

对于TPMS，$p \approx 0.5-0.7$。

营养输送的Michaelis-Menten动力学：
$$\frac{\partial c}{\partial t} = D_{\text{eff}} \nabla^2 c - \frac{V_{\max} c}{K_m + c} \rho_{\text{cell}}$$

临界厚度（氧气扩散限制）：
$$L_{\text{crit}} = \sqrt{\frac{2D_{\text{eff}}(c_0 - c_{\min})}{q_{O_2} \rho_{\text{cell}}}}$$

**热交换器的性能优化**：

Nusselt数（对流换热）：
$$Nu = \frac{hL}{k} = C Re^m Pr^n$$

对于TPMS通道，$C \approx 0.1-0.3$，$m \approx 0.6-0.8$。

压降（Darcy-Forchheimer方程）：
$$-\nabla p = \frac{\mu}{K}v + \rho C_F |v| v$$

渗透率 $K$ 和Forchheimer系数 $C_F$：
$$K = \frac{\phi^3 d_h^2}{180(1-\phi)^2}, \quad C_F = \frac{1.75}{\sqrt{150K}} \frac{1-\phi}{\phi^3 d_h}$$

性能指标（热交换效率与泵功比）：
$$\eta = \frac{Q}{\Delta p \cdot V} = \frac{Nu \cdot A_s}{f \cdot Re^2}$$

## 24.4 骨小梁结构与Wolff定律

### 24.4.1 Wolff定律的数学表述

Wolff定律（1892）指出：骨骼会根据力学刺激重塑其内部结构，使主应力方向与骨小梁方向对齐。

数学表述：骨密度 $\rho$ 的演化遵循：

$$\frac{\partial \rho}{\partial t} = B(S(\sigma) - S_{\text{ref}})$$

其中：
- $S(\sigma)$：力学刺激（如应变能密度）
- $S_{\text{ref}}$：参考刺激水平（稳态值）
- $B$：重塑速率常数

**应变能密度刺激**：
$$S = \frac{1}{2} \sigma : \varepsilon = \frac{1}{2E} \sigma_{ij} \sigma_{ij}$$

**主应力刺激**：
$$S = \sum_{i=1}^3 |\sigma_i|^m$$

其中 $m \approx 2-4$ 反映疲劳损伤累积。

### 24.4.2 连续体骨重塑模型

**Cowin-Hegedus模型**：
考虑各向异性演化，使用织构张量 $\mathbf{H}$：

$$\frac{D\mathbf{H}}{Dt} = \mathbf{A} : \mathbf{H} + \mathbf{H} : \mathbf{A}^T - 2(\mathbf{H} : \mathbf{D}) \mathbf{H} + \alpha(\mathbf{I} - 3\mathbf{H})$$

其中 $\mathbf{D}$ 是变形率张量，$\alpha$ 是适应率。

**Stanford模型**（Beaupré等）：
$$\dot{\rho} = \begin{cases}
c_r(\psi - \psi_r)^q & \psi > \psi_r \\
0 & \psi_d \leq \psi \leq \psi_r \\
-c_a(\psi_d - \psi)^p & \psi < \psi_d
\end{cases}$$

其中：
- $\psi$：日平均应变能密度
- $\psi_r, \psi_d$：重塑和废用阈值
- $c_r, c_a$：形成和吸收速率
- $p, q$：非线性指数

### 24.4.3 离散细胞自动机模型

将骨组织离散为体素，每个体素状态 $s_i \in \{0, 1\}$（空/实）。

**局部规则**：
$$P(\Delta s_i = 1) = f(S_i, \rho_{\text{local}})$$

其中 $S_i$ 是体素 $i$ 的应力，$\rho_{\text{local}}$ 是邻域密度。

**Mullender-Huiskes模型**：
考虑成骨细胞和破骨细胞的信号传递：

$$\mu_i = \sum_j \exp(-d_{ij}/D) R(\sigma_j)$$

其中：
- $\mu_i$：位置 $i$ 的重塑信号
- $D$：信号衰减长度（~100μm）
- $R(\sigma)$：力学感受函数

### 24.4.4 拓扑优化与骨适应的统一框架

骨重塑可视为自然的拓扑优化过程：

**目标函数**：最小化柔度
$$C = \int_\Omega \sigma : \varepsilon \, d\Omega$$

**约束**：体积/质量守恒
$$\int_\Omega \rho \, d\Omega = M_0$$

**SIMP插值**：
$$E(\rho) = E_0 \rho^p$$

其中 $p = 2-3$ 模拟骨小梁的力学特性。

**更新方程**（OC方法）：
$$\rho^{n+1} = \begin{cases}
\max(0, \rho^n - m) & \text{if } \rho^n B^\eta \leq \max(0, \rho^n - m) \\
\min(1, \rho^n + m) & \text{if } \rho^n B^\eta \geq \min(1, \rho^n + m) \\
\rho^n B^\eta & \text{otherwise}
\end{cases}$$

其中 $B = -\frac{\partial C}{\partial \rho} / \lambda$ 是优化准则。

### 24.4.5 多尺度建模

**微观尺度**（1-10μm）：
单个骨小梁的弯曲和屈曲：

$$\sigma_{\text{crit}} = \frac{\pi^2 E I}{(KL)^2 A}$$

其中 $K$ 是有效长度系数。

**介观尺度**（100μm-1mm）：
骨小梁网络的均匀化：

$$\mathbb{C}^* = \frac{1}{|Y|} \int_Y \mathbb{C}(y) [\mathbb{I} - \mathbb{B}(y) : \mathbb{A}(y)]dy$$

使用渐近均匀化理论计算有效刚度张量。

**宏观尺度**（>1cm）：
整骨的有限元分析，使用各向异性材料模型：

$$\sigma = \mathbb{C}(\rho, \mathbf{n}) : \varepsilon$$

其中 $\mathbf{n}$ 是局部骨小梁方向。

### 24.4.6 时间演化与稳定性

**线性稳定性分析**：
扰动 $\delta\rho$ 的增长率：

$$\lambda = B \frac{\partial S}{\partial \rho} = B \frac{p}{2} \frac{\sigma^2}{E_0} \rho^{p-1}$$

稳定条件：$\lambda < 0$ 要求 $\frac{\partial S}{\partial \rho} < 0$。

**相场模型**：
使用Allen-Cahn方程描述界面演化：

$$\frac{\partial \phi}{\partial t} = -M \frac{\delta F}{\delta \phi} = M[\nabla^2 \phi - f'(\phi) + \lambda S(\phi)]$$

其中 $\phi$ 是相场变量，$f(\phi)$ 是双井势。

### 24.4.7 3D打印仿生骨结构

**设计准则**：
1. **孔隙率**：60-80%匹配天然骨
2. **孔径**：300-600μm促进骨生长
3. **连通性**：>99%确保营养输送
4. **各向异性**：沿主应力方向增强

**优化流程**：
1. 施加生理载荷
2. 计算应力场
3. 根据Wolff定律更新密度
4. 提取等值面生成STL
5. 添加最小厚度约束

**力学性能预测**：
- 杨氏模量：$E^* = C_1 \rho^{n_1}$，$n_1 \approx 2$
- 屈服强度：$\sigma_y^* = C_2 \rho^{n_2}$，$n_2 \approx 1.5$

## 24.5 群体智能优化

### 24.5.1 粒子群优化（PSO）的数学框架

粒子群算法模拟鸟群觅食行为，每个粒子 $i$ 在 $D$ 维搜索空间中的状态由位置 $\mathbf{x}_i$ 和速度 $\mathbf{v}_i$ 描述。

**速度更新方程**：
$$\mathbf{v}_i^{t+1} = w\mathbf{v}_i^t + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^t) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i^t)$$

**位置更新方程**：
$$\mathbf{x}_i^{t+1} = \mathbf{x}_i^t + \mathbf{v}_i^{t+1}$$

其中：
- $w$：惯性权重（典型值0.4-0.9）
- $c_1, c_2$：加速系数（典型值2.0）
- $r_1, r_2$：[0,1]均匀随机数
- $\mathbf{p}_i$：粒子历史最优位置
- $\mathbf{g}$：全局最优位置

**收敛性分析**：
系统稳定条件：
$$0 < w < 1, \quad 0 < c_1 + c_2 < 4$$

特征方程：
$$\lambda^2 - (1 + w - \phi)\lambda + w = 0$$

其中 $\phi = c_1 + c_2$。

### 24.5.2 蚁群算法（ACO）用于路径优化

蚁群算法通过信息素机制求解组合优化问题。对于TSP问题，蚂蚁 $k$ 从城市 $i$ 到 $j$ 的转移概率：

$$p_{ij}^k = \begin{cases}
\frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_{l \in N_i^k} [\tau_{il}]^\alpha [\eta_{il}]^\beta} & j \in N_i^k \\
0 & \text{otherwise}
\end{cases}$$

其中：
- $\tau_{ij}$：边$(i,j)$上的信息素浓度
- $\eta_{ij} = 1/d_{ij}$：启发式信息（距离倒数）
- $\alpha, \beta$：信息素和启发式信息的权重
- $N_i^k$：蚂蚁$k$在城市$i$的可行邻域

**信息素更新**：
$$\tau_{ij}^{t+1} = (1-\rho)\tau_{ij}^t + \sum_{k=1}^m \Delta\tau_{ij}^k$$

其中：
- $\rho \in (0,1)$：挥发率
- $\Delta\tau_{ij}^k = Q/L_k$：蚂蚁$k$的信息素贡献
- $L_k$：蚂蚁$k$的路径长度

**Max-Min蚂蚁系统（MMAS）**：
限制信息素范围避免过早收敛：
$$\tau_{\min} \leq \tau_{ij} \leq \tau_{\max}$$

其中：
$$\tau_{\max} = \frac{1}{\rho L^*}, \quad \tau_{\min} = \frac{\tau_{\max}}{a}$$

$L^*$ 是当前最优解，$a$ 是常数（典型值5-10）。

### 24.5.3 人工蜂群算法（ABC）

ABC算法模拟蜜蜂采蜜行为，包含三种蜂：雇佣蜂、观察蜂和侦察蜂。

**雇佣蜂阶段**：
在食物源 $\mathbf{x}_i$ 附近搜索新解：
$$\mathbf{v}_i = \mathbf{x}_i + \phi_{ij}(\mathbf{x}_i - \mathbf{x}_j)$$

其中 $\phi_{ij} \in [-1, 1]$ 是随机数，$j \neq i$ 是随机选择的邻居。

**观察蜂选择概率**：
$$p_i = \frac{\text{fit}_i}{\sum_{j=1}^{SN} \text{fit}_j}$$

其中 $\text{fit}_i$ 是适应度值：
$$\text{fit}_i = \begin{cases}
\frac{1}{1 + f_i} & f_i \geq 0 \\
1 + |f_i| & f_i < 0
\end{cases}$$

**侦察蜂机制**：
如果食物源 $i$ 在 $\text{limit}$ 次迭代后未改进，则放弃并随机初始化：
$$\mathbf{x}_i^j = \mathbf{x}_{\min}^j + \text{rand}(0,1)(\mathbf{x}_{\max}^j - \mathbf{x}_{\min}^j)$$

### 24.5.4 萤火虫算法（FA）

萤火虫算法基于萤火虫的发光吸引机制。

**吸引度函数**：
$$\beta(r) = \beta_0 e^{-\gamma r^2}$$

其中：
- $\beta_0$：$r=0$时的吸引度
- $\gamma$：光吸收系数
- $r$：两萤火虫间的笛卡尔距离

**位置更新**：
萤火虫 $i$ 被更亮的萤火虫 $j$ 吸引：
$$\mathbf{x}_i^{t+1} = \mathbf{x}_i^t + \beta_0 e^{-\gamma r_{ij}^2}(\mathbf{x}_j^t - \mathbf{x}_i^t) + \alpha \epsilon_i$$

其中 $\alpha$ 是随机参数，$\epsilon_i$ 是高斯或均匀随机向量。

**亮度计算**：
$$I_i = f(\mathbf{x}_i) e^{-\gamma r}$$

其中 $f(\mathbf{x}_i)$ 是目标函数值。

### 24.5.5 遗传算法（GA）在形状优化中的应用

**编码方案**：
- 二进制编码：适合离散变量
- 实数编码：适合连续参数
- 树形编码：适合拓扑结构

**适应度函数**：
对于多目标优化，使用Pareto排序：
$$F_i = \sum_{j \in S_i} \frac{1}{|S_j|}$$

其中 $S_i$ 是支配解 $i$ 的解集。

**交叉算子**（实数编码）：
SBX（模拟二进制交叉）：
$$c_1 = 0.5[(1 + \beta_q)p_1 + (1 - \beta_q)p_2]$$
$$c_2 = 0.5[(1 - \beta_q)p_1 + (1 + \beta_q)p_2]$$

其中：
$$\beta_q = \begin{cases}
(2u)^{1/(\eta_c + 1)} & u \leq 0.5 \\
(1/(2(1-u)))^{1/(\eta_c + 1)} & u > 0.5
\end{cases}$$

**变异算子**：
多项式变异：
$$c = p + (p_{\max} - p_{\min})\delta$$

其中：
$$\delta = \begin{cases}
(2r)^{1/(\eta_m + 1)} - 1 & r < 0.5 \\
1 - (2(1-r))^{1/(\eta_m + 1)} & r \geq 0.5
\end{cases}$$

### 24.5.6 混合群体算法

**PSO-GA混合**：
1. PSO全局搜索
2. GA局部精细化
3. 信息交换机制

**多种群协同进化**：
$$\mathbf{v}_{ij}^{t+1} = w\mathbf{v}_{ij}^t + c_1 r_1 (\mathbf{p}_{ij} - \mathbf{x}_{ij}^t) + c_2 r_2 (\mathbf{g}_j - \mathbf{x}_{ij}^t) + c_3 r_3 (\mathbf{m} - \mathbf{x}_{ij}^t)$$

其中 $\mathbf{m}$ 是所有子种群的全局最优。

### 24.5.7 在3D打印设计中的应用

**拓扑优化**：
使用GA优化材料分布：
- 染色体：密度矩阵的向量化
- 适应度：柔度与体积的加权
- 约束处理：罚函数法

**支撑结构优化**：
使用ACO优化支撑点位置：
- 节点：潜在支撑点
- 边权重：支撑材料用量
- 约束：悬垂角度限制

**路径规划**：
使用PSO优化打印路径：
- 粒子：路径序列
- 目标：最小化空行程和转角
- 约束：连续性要求

**多材料分配**：
使用ABC优化材料梯度：
- 食物源：材料分布函数
- 目标：力学性能与成本平衡
- 约束：界面兼容性

## 本章小结

本章系统介绍了仿生设计与自然算法在3D打印中的应用：

1. **Voronoi图与Delaunay三角化**：
   - 空圆性质：$\det \begin{pmatrix} x_i - x_k & y_i - y_k & (x_i - x_k)^2 + (y_i - y_k)^2 \end{pmatrix} > 0$
   - CVT能量：$E = \sum_{i=1}^n \int_{V(p_i)} \rho(x) \|x - p_i\|^2 dx$
   - 应用：多孔结构、梯度材料设计

2. **反应扩散系统**：
   - Gray-Scott方程：$\frac{\partial u}{\partial t} = D_u \nabla^2 u - uv^2 + F(1 - u)$
   - 图灵不稳定性条件：$(D_u \text{tr}(J))^2 > 4D_u D_v \det(J)$
   - 特征长度：$L = 2\pi\sqrt{\frac{D_u D_v}{\det(J)}}$

3. **最小曲面**：
   - 平均曲率：$H = \frac{\kappa_1 + \kappa_2}{2} = 0$
   - TPMS隐式方程（Gyroid）：$\sin x \cos y + \sin y \cos z + \sin z \cos x = 0$
   - 离散化：$H_i = \frac{1}{4A_i} \sum_{j} (\cot \alpha_{ij} + \cot \beta_{ij}) \|\mathbf{x}_i - \mathbf{x}_j\|$

4. **骨适应模型**：
   - Wolff定律：$\frac{\partial \rho}{\partial t} = B(S(\sigma) - S_{\text{ref}})$
   - SIMP插值：$E(\rho) = E_0 \rho^p$
   - 多尺度：微观屈曲→介观均匀化→宏观FEM

5. **群体智能**：
   - PSO更新：$\mathbf{v}_i^{t+1} = w\mathbf{v}_i^t + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^t) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i^t)$
   - ACO转移概率：$p_{ij}^k = \frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_{l} [\tau_{il}]^\alpha [\eta_{il}]^\beta}$
   - 应用：拓扑优化、路径规划、材料分配

## 练习题

### 基础题

**练习24.1** 证明二维Delaunay三角化的平均边数为 $3n - 6$（其中 $n$ 是点数）。
<details>
<summary>提示</summary>
使用Euler公式：$V - E + F = 2$，结合每个面有3条边，每条边被2个面共享。
</details>

<details>
<summary>答案</summary>
设有 $V = n$ 个顶点，$E$ 条边，$F$ 个面（包括外部无限面）。
由Euler公式：$n - E + F = 2$。
每个有限面是三角形，有3条边；每条内部边被2个面共享。
设有 $F - 1$ 个有限三角形，则 $3(F - 1) = 2E_{内} + E_{边界}$。
对于凸包上的点，边界边数约等于 $\sqrt{n}$（期望值）。
在极限情况下，忽略边界效应：$3F \approx 2E$。
代入Euler公式：$n - E + \frac{2E}{3} = 2$，解得 $E = 3n - 6$。
</details>

**练习24.2** 推导Gray-Scott模型的线性稳定性分析中，临界波数 $k_c$ 的表达式。
<details>
<summary>提示</summary>
在色散关系中，令增长率 $\lambda(k) = 0$，求解对应的波数。
</details>

<details>
<summary>答案</summary>
线性化系统的特征值：$\lambda = \text{tr}(J) - (D_u + D_v)k^2 \pm \sqrt{\Delta}$
其中 $\Delta = [\text{tr}(J) - (D_u + D_v)k^2]^2 - 4[\det(J) - (D_u \text{tr}(J) - D_v \det(J))k^2 + D_u D_v k^4]$
不稳定性开始于 $\lambda = 0$，此时判别式 $\Delta = 0$。
化简得：$(D_u - D_v)^2 k^4 - 2(D_u - D_v)\text{tr}(J)k^2 + \text{tr}(J)^2 - 4\det(J) = 0$
临界点满足 $\frac{d\lambda}{dk} = 0$，得：$k_c^2 = \frac{\det(J)}{D_u D_v}$
因此临界波数：$k_c = \sqrt{\frac{\det(J)}{D_u D_v}}$
</details>

**练习24.3** 计算Schwarz P曲面（$\cos x + \cos y + \cos z = 0$）在单位胞内的体积分数。
<details>
<summary>提示</summary>
利用对称性，只需计算第一卦限的体积。
</details>

<details>
<summary>答案</summary>
在 $[0, 2\pi]^3$ 的单位胞内，曲面将空间分为两部分。
由于对称性，体积分数为 $\phi = 0.5$。
具体计算：$V = \int_0^{2\pi} \int_0^{2\pi} \int_0^{z^*} dz dy dx$
其中 $z^* = \arccos(-\cos x - \cos y)$
数值积分得到 $V/(2\pi)^3 = 0.5$
</details>

### 挑战题

**练习24.4** 设计一个算法，在给定应力场 $\sigma(x, y, z)$ 下，生成满足Wolff定律的骨小梁结构。考虑：
a) 如何确定主应力方向？
b) 如何控制骨小梁密度？
c) 如何保证结构连通性？

<details>
<summary>提示</summary>
使用张量分解获得主方向，用各向异性Voronoi图控制结构。
</details>

<details>
<summary>答案</summary>
算法框架：
1. 计算应力张量的特征值分解：$\sigma = Q\Lambda Q^T$，主方向为 $Q$ 的列向量
2. 构造度量张量：$M = Q \text{diag}(1/\lambda_1, 1/\lambda_2, 1/\lambda_3) Q^T$
3. 使用各向异性CVT生成种子点，密度 $\rho \propto (\det M)^{1/2}$
4. 沿主应力方向生成骨小梁：参数化曲线 $\mathbf{r}(t) = \mathbf{r}_0 + t\mathbf{v}_1$
5. 使用距离场确保最小厚度：$\phi(x) = \min_i d(x, \text{strut}_i) - r_{\min}$
6. 拓扑优化保证连通性：添加约束 $\lambda_2(L) > \epsilon$（图Laplacian第二特征值）
</details>

**练习24.5** 证明对于平面上的CVT，六边形tessellation是能量最优的（蜂窝猜想）。
<details>
<summary>提示</summary>
使用等周不等式和变分原理。
</details>

<details>
<summary>答案</summary>
考虑均匀密度 $\rho = 1$ 的CVT能量：
$E = \sum_i \int_{V_i} \|x - p_i\|^2 dx$
对于规则tessellation，每个胞元面积 $A$，周长 $P$。
由等周不等式：$P^2 \geq 4\pi A$，等号成立当且仅当胞元为圆。
在平铺约束下，最接近圆的多边形是正六边形。
具体计算惯性矩：
- 正六边形：$I_6 = \frac{5\sqrt{3}}{24}A^{5/3}$
- 正方形：$I_4 = \frac{1}{6}A^{5/3}$
- 正三角形：$I_3 = \frac{\sqrt{3}}{12}A^{5/3}$
比较得 $I_6 < I_4 < I_3$，六边形最优。
</details>

**练习24.6** 推导三维TPMS的有效弹性模量与相对密度的标度关系。考虑Gibson-Ashby模型的修正。
<details>
<summary>提示</summary>
区分拉伸主导和弯曲主导的变形模式。
</details>

<details>
<summary>答案</summary>
对于TPMS结构，考虑壳单元的变形：
1. 薄壳厚度 $t$，特征尺寸 $L$，相对密度 $\bar{\rho} = t/L$
2. 拉伸主导（如Schwarz P）：$\bar{E}/E_s = C_1 \bar{\rho}$
3. 弯曲主导（薄壁极限）：$\bar{E}/E_s = C_2 \bar{\rho}^3$
4. 过渡区域使用内插：$\bar{E}/E_s = C_1 \bar{\rho} + C_2 \bar{\rho}^3$
5. 考虑曲率效应的修正：$\bar{E}/E_s = C_1 \bar{\rho}(1 + \alpha H^2 L^2)$
其中 $H$ 是平均曲率，$\alpha$ 是材料相关常数。
数值拟合得到：Gyroid的 $C_1 \approx 0.38$，$C_2 \approx 0.13$
</details>

**练习24.7** 设计一个混合PSO-GA算法优化多材料3D打印的材料分布。目标是最小化应力集中同时满足成本约束。
<details>
<summary>提示</summary>
使用PSO探索设计空间，GA进行局部优化。
</details>

<details>
<summary>答案</summary>
混合算法：
1. 初始化：
   - PSO种群：位置编码材料分布 $\mathbf{x}_i = [m_1, m_2, ..., m_n]$
   - GA种群：二进制编码材料选择
2. PSO全局搜索（前50%迭代）：
   - 目标函数：$f = \sigma_{\max}/\sigma_{\text{avg}} + \lambda \cdot \text{cost}$
   - 速度更新：标准PSO公式
   - 约束处理：投影到可行域
3. 信息迁移：
   - 选择PSO最优30%个体
   - 转换为GA编码：离散化+格雷编码
4. GA局部优化（后50%迭代）：
   - 交叉：均匀交叉，概率0.8
   - 变异：位翻转，自适应概率 $p_m = 0.01 + 0.09(1 - f_{\text{best}}/f_{\text{avg}})$
   - 选择：锦标赛选择，规模3
5. 精英保留：保持历史最优解
6. 终止条件：$|\Delta f| < 10^{-6}$ 或达到最大迭代数
</details>

**练习24.8** 分析反应扩散系统在曲面上的图灵斑图。推导球面上的临界模式。
<details>
<summary>提示</summary>
使用球谐函数作为基函数展开。
</details>

<details>
<summary>答案</summary>
在半径 $R$ 的球面上：
1. Laplace-Beltrami算子的特征函数是球谐函数 $Y_l^m(\theta, \phi)$
2. 特征值：$\lambda_l = -\frac{l(l+1)}{R^2}$
3. 线性化扰动：$\delta u = \sum_{l,m} a_{lm} Y_l^m e^{\sigma_l t}$
4. 增长率：$\sigma_l = \text{tr}(J) + \frac{l(l+1)}{R^2}(D_u + D_v) \pm \sqrt{\Delta_l}$
5. 最不稳定模式：$\frac{d\sigma_l}{dl} = 0$
6. 临界模数：$l_c = R\sqrt{\frac{\det(J)}{D_u D_v}}$
7. 对于 $R = 1$，Gray-Scott参数 $F = 0.04, k = 0.06$：$l_c \approx 6$
8. 斑图数量：$N \approx 2l_c^2 = 72$ 个斑点
</details>

## 常见陷阱与错误

1. **Delaunay三角化的退化情况**
   - 问题：四点共圆导致三角化不唯一
   - 解决：添加微小扰动或使用符号扰动法

2. **反应扩散数值不稳定**
   - 问题：显式格式的时间步长限制过严
   - 解决：使用ADI或隐式方法，或自适应时间步长

3. **最小曲面的自相交**
   - 问题：参数化方法可能产生自相交
   - 解决：使用水平集方法或检查拓扑一致性

4. **骨重塑的棋盘格现象**
   - 问题：离散化导致的数值伪影
   - 解决：使用密度过滤或高阶元素

5. **群体算法的早熟收敛**
   - 问题：多样性丧失导致陷入局部最优
   - 解决：自适应参数、多种群策略、变异算子

6. **TPMS的制造约束**
   - 问题：局部特征尺寸小于打印分辨率
   - 解决：添加最小厚度约束，使用形态学操作

## 最佳实践检查清单

### 设计阶段
- [ ] 选择合适的仿生原理（结构vs功能）
- [ ] 确定关键性能指标（强度、刚度、孔隙率）
- [ ] 验证数学模型的适用范围
- [ ] 考虑多尺度效应

### 计算阶段
- [ ] 网格独立性验证
- [ ] 数值稳定性分析
- [ ] 收敛性检查
- [ ] 参数敏感性分析

### 优化阶段
- [ ] 定义清晰的目标函数
- [ ] 处理约束条件（罚函数/拉格朗日）
- [ ] 选择合适的优化算法
- [ ] 验证全局最优性

### 制造阶段
- [ ] 检查最小特征尺寸
- [ ] 验证支撑需求
- [ ] 评估打印时间和材料用量
- [ ] 考虑后处理需求

### 验证阶段
- [ ] 与解析解或实验对比
- [ ] 不确定性量化
- [ ] 鲁棒性分析
- [ ] 性能测试方案
