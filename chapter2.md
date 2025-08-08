# 第2章：微分几何与离散算子

本章深入探讨三维曲面的微分几何性质及其离散化方法。我们将从连续理论出发，系统推导离散算子的构造方法，重点关注数值稳定性和计算效率。这些基础工具是网格处理、形状分析和物理仿真的核心，也是理解现代几何深度学习的关键。

## 2.1 曲率计算：平均曲率、高斯曲率、主曲率

### 2.1.1 连续曲率理论

对于参数曲面 $\mathbf{r}(u,v): \Omega \subset \mathbb{R}^2 \to \mathbb{R}^3$，第一基本形式和第二基本形式定义为：

$$\mathbf{I} = \begin{pmatrix} E & F \\ F & G \end{pmatrix} = \begin{pmatrix} \langle \mathbf{r}_u, \mathbf{r}_u \rangle & \langle \mathbf{r}_u, \mathbf{r}_v \rangle \\ \langle \mathbf{r}_u, \mathbf{r}_v \rangle & \langle \mathbf{r}_v, \mathbf{r}_v \rangle \end{pmatrix}$$

$$\mathbf{II} = \begin{pmatrix} L & M \\ M & N \end{pmatrix} = \begin{pmatrix} \langle \mathbf{r}_{uu}, \mathbf{n} \rangle & \langle \mathbf{r}_{uv}, \mathbf{n} \rangle \\ \langle \mathbf{r}_{uv}, \mathbf{n} \rangle & \langle \mathbf{r}_{vv}, \mathbf{n} \rangle \end{pmatrix}$$

其中 $\mathbf{n} = \frac{\mathbf{r}_u \times \mathbf{r}_v}{|\mathbf{r}_u \times \mathbf{r}_v|}$ 是单位法向量。

第一基本形式刻画了曲面的内蕴几何，决定了曲面上的长度、角度和面积元素：
- 弧长元素：$ds^2 = E du^2 + 2F du dv + G dv^2$
- 面积元素：$dA = \sqrt{EG - F^2} \, du \, dv$
- 夹角余弦：$\cos\theta = \frac{E du_1 du_2 + F(du_1 dv_2 + du_2 dv_1) + G dv_1 dv_2}{\sqrt{(E du_1^2 + 2F du_1 dv_1 + G dv_1^2)(E du_2^2 + 2F du_2 dv_2 + G dv_2^2)}}$

第二基本形式描述了曲面在三维空间中的弯曲程度，与法向量密切相关。对于曲面上的曲线 $\gamma(t) = \mathbf{r}(u(t), v(t))$，其法曲率为：
$$\kappa_n = \frac{\mathbf{II}(\dot{\gamma}, \dot{\gamma})}{\mathbf{I}(\dot{\gamma}, \dot{\gamma})} = \frac{L \dot{u}^2 + 2M \dot{u}\dot{v} + N \dot{v}^2}{E \dot{u}^2 + 2F \dot{u}\dot{v} + G \dot{v}^2}$$

形状算子（Weingarten映射）为：
$$\mathbf{S} = -\mathbf{I}^{-1}\mathbf{II} = -\frac{1}{EG-F^2}\begin{pmatrix} G & -F \\ -F & E \end{pmatrix}\begin{pmatrix} L & M \\ M & N \end{pmatrix}$$

形状算子 $\mathbf{S}$ 是切空间到自身的线性映射，描述了法向量沿切方向的变化率：
$$\mathbf{S}(\mathbf{v}) = -\nabla_\mathbf{v} \mathbf{n}$$

主曲率 $\kappa_1, \kappa_2$ 是形状算子的特征值，满足特征方程：
$$\det(\mathbf{S} - \kappa\mathbf{I}) = \kappa^2 - 2H\kappa + K = 0$$

对应的特征向量给出主方向，它们相互正交且沿这些方向法曲率达到极值。

**高斯曲率**（内蕴不变量，Theorema Egregium）：
$$K = \kappa_1 \kappa_2 = \det(\mathbf{S}) = \frac{LN - M^2}{EG - F^2}$$

**平均曲率**（外蕴量，与嵌入相关）：
$$H = \frac{\kappa_1 + \kappa_2}{2} = \frac{1}{2}\text{tr}(\mathbf{S}) = \frac{EN + GL - 2FM}{2(EG - F^2)}$$

**曲率的几何意义**：
- $K > 0$：椭圆点（局部像球面）
- $K < 0$：双曲点（局部像马鞍）
- $K = 0$：抛物点（局部像柱面）
- $H = 0$：极小曲面（局部面积最小）

### 2.1.2 离散曲率：顶点方法

对于三角网格顶点 $v_i$，设其1-邻域为 $N(i)$，相邻三角形集合为 $T(i)$。离散曲率的计算基于将连续理论离散化到分片线性函数空间。

**离散高斯曲率（角度缺陷）**：
$$K_i = \frac{1}{A_i}\left(2\pi - \sum_{j \in T(i)} \theta_j^i\right) = \frac{\Theta_i}{A_i}$$

这里 $\Theta_i$ 称为角度缺陷（angle defect），几何直观是：
- 平坦区域：角度和 = $2\pi$，$K = 0$
- 凸起区域：角度和 < $2\pi$，$K > 0$（如球面顶点）
- 凹陷区域：角度和 > $2\pi$，$K < 0$（如马鞍点）

**理论基础**：离散Gauss-Bonnet定理保证了这种定义的正确性。对于闭合网格：
$$\sum_{i \in V} K_i A_i = 2\pi \chi(M)$$
其中 $\chi(M) = |V| - |E| + |F|$ 是Euler特征数。

其中 $\theta_j^i$ 是三角形 $j$ 在顶点 $i$ 处的内角，$A_i$ 是混合Voronoi面积（mixed Voronoi area），考虑了钝角三角形的特殊情况：

$$A_i = \begin{cases}
\frac{1}{8}\sum_{j,k \in N(i)} (\cot\alpha_{jk} + \cot\beta_{jk})||v_j - v_k||^2 & \text{所有角都是锐角} \\
\frac{1}{2}\text{Area}(T) & \text{钝角在}v_i\text{处} \\
\frac{1}{4}\text{Area}(T) & \text{钝角不在}v_i\text{处}
\end{cases}$$

**Voronoi面积的几何意义**：
- 锐角情况：Voronoi区域完全在三角形内部，使用垂心分割
- 钝角在$v_i$：顶点拥有三角形的一半面积
- 钝角不在$v_i$：顶点只分配到四分之一面积

这种处理保证了面积元素的非负性和分割的完备性（所有面积都被分配）。

**离散平均曲率向量（Laplace-Beltrami）**：
$$\mathbf{H}_i = \frac{1}{2A_i}\sum_{j \in N(i)} (\cot\alpha_{ij} + \cot\beta_{ij})(v_j - v_i)$$

其中 $\alpha_{ij}$、$\beta_{ij}$ 是边 $(v_i, v_j)$ 对面的两个角。这个公式来源于离散Laplace-Beltrami算子应用于坐标函数：
$$\Delta \mathbf{r} = -2H\mathbf{n}$$

平均曲率标量：$H_i = \frac{1}{2}||\mathbf{H}_i||$，符号由 $\langle \mathbf{H}_i, \mathbf{n}_i \rangle$ 决定。

**法向量计算**：顶点法向量通常通过面积加权平均：
$$\mathbf{n}_i = \frac{\sum_{T \in T(i)} \text{Area}(T) \cdot \mathbf{n}_T}{||\sum_{T \in T(i)} \text{Area}(T) \cdot \mathbf{n}_T||}$$

其中 $\mathbf{n}_T = \frac{(v_j - v_i) \times (v_k - v_i)}{||(v_j - v_i) \times (v_k - v_i)||}$ 是三角形 $T = (v_i, v_j, v_k)$ 的单位法向量。

### 2.1.3 主曲率与主方向计算

给定平均曲率 $H$ 和高斯曲率 $K$，主曲率通过求解二次方程得到：
$$\kappa_{1,2} = H \pm \sqrt{H^2 - K}$$

注意当 $H^2 < K$ 时，上述公式可能出现复数，表明数值误差导致不一致。实际计算中需要：
$$\kappa_{1,2} = H \pm \sqrt{\max(0, H^2 - K)}$$

**主方向计算方法一：二次型拟合**

构造局部坐标系并拟合二次型。设顶点 $v_i$ 的邻域点投影到切平面，拟合二次曲面：
$$z = \frac{1}{2}(a x^2 + 2b xy + c y^2)$$

最小二乘问题：
$$\min_{a,b,c} \sum_{j \in N(i)} w_j (z_j - \frac{1}{2}(a x_j^2 + 2b x_j y_j + c y_j^2))^2$$

其中权重 $w_j = \exp(-||v_j - v_i||^2/\sigma^2)$ 或 $w_j = 1/||v_j - v_i||^2$。

形状算子在局部坐标系下为：
$$\mathbf{S}_{local} = \begin{pmatrix} a & b \\ b & c \end{pmatrix}$$

其特征向量给出主方向在切平面上的投影。

**主方向计算方法二：曲率张量**

直接构造曲率张量：
$$\mathbf{T} = \sum_{e \in E(i)} \kappa_e^n \mathbf{d}_e \otimes \mathbf{d}_e$$

其中 $\kappa_e^n$ 是边 $e$ 的法曲率，$\mathbf{d}_e$ 是边方向在切平面上的投影。张量 $\mathbf{T}$ 的特征分解给出主方向和主曲率。

### 2.1.4 张量投票方法

对于噪声数据，使用张量投票提高鲁棒性。每个三角形贡献一个曲率张量：
$$\mathbf{C}_T = \text{Area}(T) \cdot \kappa_T \mathbf{n}_T \otimes \mathbf{n}_T$$

顶点处的曲率张量为邻域加权平均：
$$\mathbf{C}_i = \frac{1}{\sum_T w_T} \sum_{T \in N(i)} w_T \mathbf{C}_T$$

其中权重 $w_T = \exp(-d^2/\sigma^2)$，$d$ 是三角形重心到顶点的距离。

## 2.2 离散Laplace-Beltrami算子

### 2.2.1 连续Laplace-Beltrami算子

对于黎曼流形 $(M, g)$ 上的光滑函数 $f$，Laplace-Beltrami算子定义为：
$$\Delta_g f = \text{div}(\nabla f) = \frac{1}{\sqrt{|g|}}\partial_i\left(\sqrt{|g|}g^{ij}\partial_j f\right)$$

在参数化曲面上：
$$\Delta f = \frac{1}{\sqrt{EG-F^2}}\left[\frac{\partial}{\partial u}\left(\frac{G f_u - F f_v}{\sqrt{EG-F^2}}\right) + \frac{\partial}{\partial v}\left(\frac{E f_v - F f_u}{\sqrt{EG-F^2}}\right)\right]$$

### 2.2.2 余切公式推导

通过有限元方法，在分片线性函数空间上离散化。考虑三角网格上的分片线性函数，其在每个三角形内是线性的。

**Galerkin离散化**：寻找弱形式解
$$\int_M \langle \nabla u, \nabla \phi_i \rangle dA = \int_M f \phi_i dA$$

其中 $\phi_i$ 是顶点 $i$ 处的帽函数（hat function），在顶点 $i$ 处为1，在其他顶点为0。

**余切权重推导**：对于边 $(v_i, v_j)$ 共享的两个三角形，计算梯度内积的积分：
$$\int_M \langle \nabla \phi_i, \nabla \phi_j \rangle dA = -\frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$$

因此，余切权重为：
$$w_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$$

其中 $\alpha_{ij}$ 和 $\beta_{ij}$ 是边 $(i,j)$ 对面的两个角。

**几何解释**：
- 余切权重反映了边的"双重性"：原始边与对偶边的关系
- 在Delaunay三角化中，所有权重非负
- 权重大小与边在Voronoi图中的对偶边长度成正比

离散Laplace-Beltrami算子矩阵：
$$L_{ij} = \begin{cases}
-\sum_{k \in N(i)} w_{ik} & i = j \\
w_{ij} & j \in N(i) \\
0 & \text{otherwise}
\end{cases}$$

### 2.2.3 质量矩阵与归一化

质量矩阵 $\mathbf{M}$ 对应于内积的离散化：
$$\langle f, g \rangle = \int_M fg \, dA \approx \mathbf{f}^T \mathbf{M} \mathbf{g}$$

**质量矩阵的构造方法**：

1. **集中质量矩阵（Lumped Mass Matrix）**：
   $$M_{ii} = A_i = \sum_{T \in T(i)} \frac{\text{Area}(T)}{3}$$
   - 对角矩阵，计算高效
   - 使用Voronoi面积可提高精度
   - 保证正定性

2. **一致质量矩阵（Consistent Mass Matrix）**：
   $$M_{ij} = \begin{cases}
   \frac{1}{6}\sum_{T \in T(i)} \text{Area}(T) & i = j \\
   \frac{1}{12}\sum_{T \in T(i,j)} \text{Area}(T) & j \in N(i) \\
   0 & \text{otherwise}
   \end{cases}$$
   - 来自有限元理论
   - 更高的精度但需要矩阵求逆
   - 保持对称正定性

3. **混合Voronoi质量（Meyer方法）**：
   使用2.1.2节中定义的混合Voronoi面积$A_i$

**归一化Laplacian**：
- 几何Laplacian：$\tilde{\mathbf{L}} = \mathbf{M}^{-1}\mathbf{L}$
- 对称归一化：$\mathbf{L}_{sym} = \mathbf{M}^{-1/2}\mathbf{L}\mathbf{M}^{-1/2}$
- 组合Laplacian：$\mathbf{L}_{comb} = \mathbf{D} - \mathbf{A}$（图理论）

**数值考虑**：
- 集中质量矩阵避免了矩阵求逆
- 一致质量矩阵提供更好的收敛性
- 选择取决于应用：谱分析通常用对称归一化

### 2.2.4 谱分析与特征函数

Laplace-Beltrami算子的特征值问题：
$$\Delta \phi_k = -\lambda_k \phi_k$$

离散形式：
$$\mathbf{L}\boldsymbol{\phi}_k = \lambda_k \mathbf{M}\boldsymbol{\phi}_k$$

特征值 $0 = \lambda_0 < \lambda_1 \leq \lambda_2 \leq \cdots$ 反映几何信息：
- $\lambda_0 = 0$ 对应常函数
- $\lambda_1$ 与等周常数相关（Cheeger不等式）
- 谱序列编码形状的多尺度信息

**Weyl渐近公式**（2D情况）：
$$\lambda_k \sim \frac{4\pi k}{\text{Area}(M)}$$

## 2.3 测地线与热核方法

### 2.3.1 测地线的变分原理

测地线是曲面上长度泛函的临界点。对于参数曲线 $\gamma(t)$，长度泛函为：
$$L[\gamma] = \int_0^1 \sqrt{\langle \dot{\gamma}, \dot{\gamma} \rangle_g} dt$$

Euler-Lagrange方程（测地线方程）：
$$\ddot{\gamma}^k + \Gamma^k_{ij}\dot{\gamma}^i\dot{\gamma}^j = 0$$

其中 $\Gamma^k_{ij}$ 是Christoffel符号。

### 2.3.2 离散测地线：Fast Marching Method

Eikonal方程描述测地距离场 $u(x)$：
$$|\nabla u| = 1, \quad u(x_0) = 0$$

这是一个Hamilton-Jacobi型偏微分方程，其解给出从源点$x_0$到任意点的测地距离。

**离散化策略**：在三角网格上，使用更新规则：
$$u_i = \min_{T \in T(i)} \text{LocalUpdate}(T, u)$$

**局部更新公式**：对于三角形$T = (v_i, v_j, v_k)$，假设$u_j$和$u_k$已知，更新$u_i$：

1. **锐角情况**：信息从三角形内部传播
   $$u_i = u_j \cos\theta_{ij} + u_k \cos\theta_{ik} + \sqrt{(u_j \cos\theta_{ij} + u_k \cos\theta_{ik})^2 - (u_j^2 + u_k^2 - d_{jk}^2)}$$
   
2. **钝角情况**：信息沿边传播
   $$u_i = \min(u_j + d_{ij}, u_k + d_{ik})$$

**算法流程**：
1. 初始化：源点距离为0，其他为∞
2. 将源点加入优先队列
3. 循环直到队列为空：
   - 提取最小距离顶点
   - 更新所有邻居
   - 将更新的顶点加入队列

**复杂度分析**：
- 时间：$O(n \log n)$，使用最小堆
- 空间：$O(n)$存储距离值
- 精度：一阶精确（可扩展到高阶）

### 2.3.3 热核与热方程

热方程连接了扩散过程与几何：
$$\frac{\partial u}{\partial t} = \Delta u$$

热核 $k_t(x, y)$ 是基本解，满足：
$$u(x, t) = \int_M k_t(x, y)u_0(y)dy$$

谱展开：
$$k_t(x, y) = \sum_{i=0}^{\infty} e^{-\lambda_i t}\phi_i(x)\phi_i(y)$$

**热核签名（HKS）**：
$$\text{HKS}(x, t) = k_t(x, x) = \sum_{i=0}^{\infty} e^{-\lambda_i t}\phi_i(x)^2$$

### 2.3.4 测地距离的热方法

Varadhan公式：
$$d(x, y)^2 = \lim_{t \to 0^+} -4t \log k_t(x, y)$$

实际计算步骤：
1. 求解热方程：$(\mathbf{M} - t\mathbf{L})\mathbf{u} = \mathbf{u}_0$
2. 计算归一化梯度场：$\mathbf{X} = -\nabla u / |\nabla u|$
3. 求解Poisson方程：$\Delta \phi = \nabla \cdot \mathbf{X}$

时间参数选择：$t = h^2$，其中 $h$ 是平均边长。

## 2.4 曲面参数化：LSCM、ABF++

### 2.4.1 参数化的数学框架

曲面参数化寻找双射 $\phi: M \to \Omega \subset \mathbb{R}^2$，使得某种失真度量最小。对于三角网格，参数化将每个顶点 $v_i \in \mathbb{R}^3$ 映射到 $u_i \in \mathbb{R}^2$。

**失真度量**：
- **共形失真**：保角映射，$\mathbf{J}^T\mathbf{J} = \sigma^2\mathbf{I}$
- **等距失真**：保长映射，$\mathbf{J}^T\mathbf{J} = \mathbf{I}$
- **等面积失真**：保面积映射，$\det(\mathbf{J}) = 1$

其中 $\mathbf{J}$ 是参数化的Jacobian矩阵。

### 2.4.2 最小二乘共形映射（LSCM）

LSCM最小化共形能量：
$$E_{LSCM} = \sum_T \text{Area}(T) \cdot ||\mathbf{J}_T - \mathbf{R}_T||_F^2$$

其中 $\mathbf{R}_T$ 是最接近 $\mathbf{J}_T$ 的旋转矩阵。

对于三角形 $T = (v_1, v_2, v_3)$，局部参数化的线性表示：
$$u(x) = u_1\phi_1(x) + u_2\phi_2(x) + u_3\phi_3(x)$$

其中 $\phi_i$ 是重心坐标。梯度为：
$$\nabla u = \sum_{i=1}^3 u_i \nabla \phi_i$$

**复数表示**：令 $w_i = u_i + iv_i$，共形条件等价于：
$$\frac{\partial w}{\partial \bar{z}} = 0$$

离散化后得到线性系统：
$$\mathbf{A}^T\mathbf{A}\mathbf{w} = \mathbf{0}$$

其中 $\mathbf{A}$ 是稀疏矩阵，每个三角形贡献一个 $2 \times 2n$ 的块。

**边界条件**：
- 自由边界：最小化总能量
- 固定边界：指定边界顶点位置（至少固定2个顶点避免平凡解）

### 2.4.3 基于角度的扁平化（ABF++）

ABF方法直接优化角度变量，保证参数化的有效性（无翻转）。

**角度约束**：
- 三角形内角和：$\alpha_1^T + \alpha_2^T + \alpha_3^T = \pi$
- 顶点周围角度和：$\sum_{T \in N(v)} \alpha_v^T = 2\pi$（内部顶点）

**能量函数**：
$$E_{ABF} = \sum_T \sum_{i=1}^3 (\alpha_i^T - \beta_i^T)^2 / \beta_i^T$$

其中 $\beta_i^T$ 是原始网格中的角度。

**ABF++改进**：
1. 使用对数角度 $\log(\alpha/\beta)$ 提高数值稳定性
2. 引入辅助变量简化约束
3. 使用牛顿法求解，Hessian矩阵具有特殊结构

**重建坐标**：给定优化后的角度，通过以下步骤重建2D坐标：
1. 计算边长比例因子（使用正弦定理）
2. 固定一个三角形的位置
3. 逐步传播确定其他顶点位置

### 2.4.4 其他参数化方法

**Mean Value Coordinates**：
$$w_{ij} = \frac{\tan(\theta_{ij}^-/2) + \tan(\theta_{ij}^+/2)}{||v_i - v_j||}$$

能量函数：
$$E_{MVC} = \sum_i ||\sum_j w_{ij}(u_j - u_i)||^2$$

**As-Rigid-As-Possible (ARAP)**：
两步迭代：
1. 局部步：固定顶点位置，优化每个三角形的旋转
2. 全局步：固定旋转，求解顶点位置

能量函数：
$$E_{ARAP} = \sum_{(i,j) \in E} w_{ij}||(u_i - u_j) - \mathbf{R}_i(v_i - v_j)||^2$$

## 2.5 离散外微分与余切权重

### 2.5.1 外微分形式基础

离散外微分（DEC）提供了微分几何的离散模拟。在单纯复形上：
- 0-形式：顶点上的标量函数
- 1-形式：边上的值（切向量的对偶）
- 2-形式：面上的值（面积元素）

**外导数算子**：
- $d^0: \Omega^0 \to \Omega^1$：梯度
- $d^1: \Omega^1 \to \Omega^2$：旋度
- $d^2: \Omega^2 \to \Omega^3$：散度

满足 $d \circ d = 0$（正合性）。

### 2.5.2 离散微分算子构造

**梯度算子**（0-形式到1-形式）：
对于边 $e = [v_i, v_j]$：
$$(d^0 f)_e = f_j - f_i$$

矩阵形式：$\mathbf{d}^0$ 是 $|E| \times |V|$ 的关联矩阵。

**旋度算子**（1-形式到2-形式）：
对于三角形 $T$ 及其边界 $\partial T = e_1 + e_2 + e_3$：
$$(d^1 \omega)_T = \sum_{e \in \partial T} \text{sign}(e, T) \cdot \omega_e$$

**散度算子**：通过对偶关系定义
$$\text{div} = -\star d \star$$

### 2.5.3 Hodge星算子与内积

Hodge星算子 $\star: \Omega^k \to \Omega^{n-k}$ 建立对偶关系。

**离散Hodge星**：
- 0-形式：$\star_0 = \mathbf{M}$（质量矩阵）
- 1-形式：$(\star_1)_{ee} = (\cot\alpha_e + \cot\beta_e)/2$
- 2-形式：$(\star_2)_{TT} = 1/\text{Area}(T)$

**内积结构**：
$$\langle \alpha, \beta \rangle = \int_M \alpha \wedge \star\beta$$

离散版本：
$$\langle \alpha, \beta \rangle_d = \alpha^T \star_k \beta$$

### 2.5.4 余切权重的几何意义

余切权重出现在多个背景下：
1. **Dirichlet能量最小化**：$E[f] = \frac{1}{2}\int_M |\nabla f|^2$
2. **调和映射**：临界点满足 $\Delta u = 0$
3. **离散共形理论**：保角变换的线性化

**与其他权重的比较**：
- **组合Laplacian**：$w_{ij} = 1$（图Laplacian）
- **归一化权重**：$w_{ij} = 1/||v_i - v_j||$
- **Mean value权重**：更稳定但非对称

**数值考虑**：
- 钝角三角形导致负权重
- Delaunay三角化最大化最小角
- 内在Delaunay通过边翻转改善权重

### 2.5.5 向量场设计与Helmholtz-Hodge分解

**离散向量场**：定义在面上或边上
- 面向量场：每个三角形一个切向量
- 边向量场：1-形式的原始表示

**Helmholtz-Hodge分解**：
任意向量场 $\mathbf{v}$ 可分解为：
$$\mathbf{v} = \nabla f + \nabla \times \mathbf{A} + \mathbf{h}$$

其中：
- $\nabla f$：无旋分量（梯度场）
- $\nabla \times \mathbf{A}$：无散分量（旋度场）
- $\mathbf{h}$：调和分量

离散计算：
1. 投影到无旋空间：求解 $\Delta f = \nabla \cdot \mathbf{v}$
2. 投影到无散空间：求解 $\Delta \mathbf{A} = \nabla \times \mathbf{v}$
3. 调和分量：$\mathbf{h} = \mathbf{v} - \nabla f - \nabla \times \mathbf{A}$

## 本章小结

本章系统介绍了微分几何在离散网格上的实现方法：

**核心概念**：
1. **离散曲率**：通过角度缺陷和Laplace-Beltrami算子计算高斯曲率和平均曲率
2. **离散Laplacian**：余切权重公式 $w_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$
3. **测地距离**：Fast Marching ($O(n\log n)$) 和热方法 ($O(n)$稀疏求解)
4. **曲面参数化**：LSCM（共形）、ABF++（保角度）、ARAP（尽可能刚性）
5. **离散外微分**：提供微分几何的离散框架

**关键公式汇总**：
- 离散高斯曲率：$K_i = \frac{1}{A_i}(2\pi - \sum_j \theta_j^i)$
- 离散平均曲率向量：$\mathbf{H}_i = \frac{1}{2A_i}\sum_j (\cot\alpha_{ij} + \cot\beta_{ij})(v_j - v_i)$
- 热核签名：$\text{HKS}(x,t) = \sum_i e^{-\lambda_i t}\phi_i(x)^2$
- LSCM能量：$E_{LSCM} = \sum_T \text{Area}(T) \cdot ||\mathbf{J}_T - \mathbf{R}_T||_F^2$

**计算复杂度**：
- 曲率计算：$O(n)$
- 稀疏线性系统求解：$O(n^{1.5})$（典型情况）
- 特征值分解：$O(kn^2)$（求前$k$个特征值）
- Fast Marching：$O(n\log n)$

## 练习题

### 基础题

**练习2.1** 证明离散高斯曲率的角度缺陷公式满足Gauss-Bonnet定理：
$$\sum_{i \in V} K_i A_i + \sum_{j \in \partial M} \kappa_g^j l_j = 2\pi\chi(M)$$
其中 $\kappa_g^j$ 是边界测地曲率，$l_j$ 是边界边长，$\chi(M)$ 是Euler特征数。

*提示*：利用三角形内角和为$\pi$，统计每个角的贡献。

<details>
<summary>答案</summary>

考虑闭合曲面的情况。总角度缺陷为：
$$\sum_i K_i A_i = \sum_i (2\pi - \sum_j \theta_j^i) = 2\pi|V| - \sum_T (\alpha_T + \beta_T + \gamma_T)$$
$$= 2\pi|V| - \pi|F| = \pi(2|V| - |F|)$$

由Euler公式 $|V| - |E| + |F| = 2 - 2g$，其中 $|E| = \frac{3|F|}{2}$（闭合曲面），得：
$$2|V| - |F| = 2|V| - |E| + |E| - |F| = 2\chi(M) + |E| - |F| = 2\chi(M) + \frac{|F|}{2}$$

因此 $\sum_i K_i A_i = 2\pi\chi(M)$。
</details>

**练习2.2** 推导三角形上分片线性函数的Dirichlet能量，验证余切权重公式。

*提示*：使用重心坐标表示梯度，计算 $\int_T |\nabla f|^2 dA$。

<details>
<summary>答案</summary>

设三角形 $T$ 的顶点为 $v_1, v_2, v_3$，函数值为 $f_1, f_2, f_3$。在重心坐标下：
$$f = f_1\lambda_1 + f_2\lambda_2 + f_3\lambda_3$$

梯度：
$$\nabla f = f_1\nabla\lambda_1 + f_2\nabla\lambda_2 + f_3\nabla\lambda_3$$

其中 $\nabla\lambda_i = \frac{\mathbf{n} \times \mathbf{e}_i}{2\text{Area}(T)}$，$\mathbf{e}_i$ 是顶点 $i$ 对面的边向量。

Dirichlet能量：
$$E_T = \int_T |\nabla f|^2 dA = \text{Area}(T) \sum_{i,j} f_i f_j \langle\nabla\lambda_i, \nabla\lambda_j\rangle$$

计算内积：
$$\langle\nabla\lambda_i, \nabla\lambda_j\rangle = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{4\text{Area}(T)^2} = -\frac{\cot\theta_k}{2\text{Area}(T)}$$

其中 $k$ 是 $i,j$ 之外的顶点，$\theta_k$ 是顶点 $k$ 处的角。
</details>

**练习2.3** 实现热方法计算测地距离的算法框架（伪代码）。

*提示*：三个主要步骤：热扩散、归一化梯度、求解Poisson方程。

<details>
<summary>答案</summary>

```
function HeatMethodGeodesics(mesh, source_vertex):
    // Step 1: 热扩散
    t = mean_edge_length^2
    u0 = zeros(n_vertices)
    u0[source_vertex] = 1
    A = M - t*L  // M: 质量矩阵, L: Laplacian
    u = solve(A, u0)
    
    // Step 2: 计算归一化梯度场
    for each triangle T:
        grad_u_T = compute_gradient(u, T)
        X_T = -grad_u_T / ||grad_u_T||
    
    // Step 3: 求解Poisson方程
    div_X = compute_divergence(X)
    phi = solve(L, M * div_X)
    
    // 归一化使源点距离为0
    phi = phi - phi[source_vertex]
    return phi
```
</details>

**练习2.4** 证明LSCM参数化的共形能量等价于Cauchy-Riemann方程的离散化。

*提示*：使用复数表示，共形映射满足 $\frac{\partial w}{\partial \bar{z}} = 0$。

<details>
<summary>答案</summary>

设参数化 $w = u + iv$，在复坐标下 $z = x + iy$。共形条件：
$$\frac{\partial w}{\partial \bar{z}} = \frac{1}{2}\left(\frac{\partial w}{\partial x} + i\frac{\partial w}{\partial y}\right) = 0$$

展开得Cauchy-Riemann方程：
$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$

离散化：对每个三角形，线性插值给出常梯度。设 $\mathbf{J} = [\nabla u, \nabla v]$，共形条件等价于：
$$\mathbf{J}^T\mathbf{J} = \lambda^2 \mathbf{I}$$

LSCM能量 $||\mathbf{J} - \mathbf{R}||^2$ 最小化时，$\mathbf{J}$ 最接近旋转矩阵（乘以缩放），即满足共形条件。
</details>

### 挑战题

**练习2.5** 设计一个算法计算曲面上的Voronoi图，给定种子点集合。考虑：
a) 如何处理非凸区域？
b) 如何保证计算效率？
c) 如何处理退化情况？

*提示*：可以使用Fast Marching的多源版本，或者热方法的叠加原理。

<details>
<summary>答案</summary>

**算法1：多源Fast Marching**
```
1. 初始化所有种子点距离为0，其他为∞
2. 将所有种子点加入优先队列
3. 同时传播所有波前，记录最近种子点ID
4. 使用标签避免重复更新

复杂度：O(n log n)
处理退化：等距点处可能出现数值误差，使用容差判断
```

**算法2：热核叠加**
```
1. 对每个种子点计算热核
2. 找到每个顶点的最大响应种子点
3. 使用行进立方体提取Voronoi边界

优势：自然光滑，可并行计算
缺点：需要多次求解线性系统
```

**非凸区域处理**：
- 测地Voronoi自动处理非凸性
- 边界处需要特殊处理反射条件
</details>

**练习2.6** 分析不同参数化方法的适用场景：
a) 何时使用LSCM vs ABF++？
b) 如何量化参数化质量？
c) 如何选择边界条件？

*提示*：考虑共形失真、面积失真、数值稳定性、计算成本。

<details>
<summary>答案</summary>

**方法选择准则**：

1. **LSCM**：
   - 优点：线性系统，快速，保角性好
   - 缺点：可能产生翻转，需要固定边界
   - 适用：纹理映射，共形几何处理

2. **ABF++**：
   - 优点：保证无翻转，角度保持好
   - 缺点：非线性优化，较慢
   - 适用：高质量参数化，鲁棒性要求高

**质量度量**：
- 共形失真：$\sigma_{max}/\sigma_{min}$（奇异值比）
- 面积失真：$\text{Area}_{2D}/\text{Area}_{3D}$
- 等距失真：MIPS能量
- 组合质量：翻转三角形数量

**边界选择**：
- 自由边界：最小化总体失真
- 凸边界：圆形或方形，简化后续处理
- 特征对齐：沿主方向或对称轴
</details>

**练习2.7** 推导离散外微分框架下的Hodge分解算法，并分析其在向量场设计中的应用。

*提示*：利用正合序列 $0 \to \Omega^0 \xrightarrow{d} \Omega^1 \xrightarrow{d} \Omega^2 \to 0$。

<details>
<summary>答案</summary>

**Hodge分解**：
$$\Omega^1 = \text{Im}(d^0) \oplus \text{Ker}(\Delta^1) \oplus \text{Im}(\star d^1 \star)$$

**算法**：
1. 梯度分量：$\omega_g = d^0(L^{-1}d^{0T}\star_1\omega)$
2. 旋度分量：$\omega_r = \star_1^{-1}d^{1T}(d^1\star_1^{-1}d^{1T})^{-1}d^1\omega$
3. 调和分量：$\omega_h = \omega - \omega_g - \omega_r$

**应用**：
- 无旋场：流体势流，静电场
- 无散场：不可压流体，磁场
- 调和场：拓扑特征，周期结构

**数值考虑**：
- 使用QR分解找到调和空间基
- 预条件共轭梯度法求解大规模系统
- 边界条件影响分解唯一性
</details>

**练习2.8** 设计一个多分辨率的曲率计算方案，能够在不同尺度下捕获几何特征。

*提示*：可以使用谱方法或者多尺度的邻域。

<details>
<summary>答案</summary>

**方法1：谱多分辨率**
```
1. 计算Laplacian特征函数 φ_i
2. 低通滤波：f_σ = Σ_{λ_i < σ} <f, φ_i> φ_i
3. 在滤波后的几何上计算曲率
4. 尺度空间：K(x, σ) 随σ变化
```

**方法2：测地球邻域**
```
1. 对每个顶点，提取半径r的测地球
2. 在邻域内拟合二次曲面
3. 从拟合参数提取曲率
4. 改变r得到多尺度曲率
```

**特征检测**：
- 尺度选择：曲率变化率最大处
- 特征追踪：不同尺度下的对应关系
- 应用：自适应网格简化，特征保持平滑
</details>

## 常见陷阱与错误

### 数值稳定性问题

1. **余切权重的负值**
   - 问题：钝角三角形产生负权重，导致矩阵非正定
   - 解决：使用Mean Value坐标或投影到最近正定矩阵
   - 预防：网格预处理，改善三角形质量

2. **小角度的数值误差**
   - 问题：接近0或π的角度导致余切值爆炸
   - 解决：设置阈值，钳制极端值
   - 更好方案：使用稳定的公式 $\cot\theta = \cos\theta/\sin\theta$

3. **质量矩阵的条件数**
   - 问题：极小三角形导致病态矩阵
   - 解决：使用集中质量矩阵或预条件器
   - 诊断：监控条件数，自适应选择求解器

### 几何退化情况

1. **零面积三角形**
   - 检测：叉积范数 < ε
   - 处理：合并顶点或删除退化元素
   - 影响：曲率计算无定义，参数化失败

2. **非流形结构**
   - T-junction：边连接超过2个三角形
   - 非流形顶点：邻域不同胚于圆盘
   - 处理：局部重网格化或分离处理

3. **边界处理不当**
   - 曲率计算：需要虚拟延拓或特殊公式
   - 参数化：边界约束影响全局质量
   - 建议：使用自然边界条件或特征对齐

### 算法选择错误

1. **不适当的离散化**
   - 症状：结果对网格分辨率敏感
   - 原因：未考虑收敛阶
   - 改进：使用高阶方法或自适应细化

2. **忽略对称性**
   - 问题：数值误差破坏理论对称性
   - 例子：Laplacian应该对称
   - 解决：强制对称化 $(L + L^T)/2$

3. **错误的特征值计算**
   - 使用密集方法计算大规模问题
   - 未利用矩阵稀疏性和对称性
   - 正确：使用Lanczos或Arnoldi迭代

## 最佳实践检查清单

### 网格预处理
- [ ] 检查并修复非流形结构
- [ ] 移除退化三角形（面积 < 1e-10）
- [ ] 改善三角形质量（最小角 > 15°）
- [ ] 统一网格朝向（法向量一致）
- [ ] 处理孤立顶点和悬挂边

### 数值计算
- [ ] 选择合适的线性求解器（直接vs迭代）
- [ ] 使用预条件器加速收敛
- [ ] 监控残差和收敛历史
- [ ] 验证矩阵性质（对称性、正定性）
- [ ] 处理数值零（使用相对容差）

### 算法验证
- [ ] 在简单几何上验证（球、环面）
- [ ] 检查尺度不变性和旋转不变性
- [ ] 验证守恒量（面积、体积、拓扑）
- [ ] 与解析解或其他方法对比
- [ ] 测试极端情况和边界条件

### 性能优化
- [ ] 利用稀疏矩阵结构
- [ ] 预计算不变量（如余切权重）
- [ ] 使用空间数据结构加速查询
- [ ] 考虑并行化（OpenMP、CUDA）
- [ ] 平衡精度与计算成本

### 结果验证
- [ ] 可视化中间结果
- [ ] 统计分析（直方图、分位数）
- [ ] 检查异常值和离群点
- [ ] 验证理论性质（如Gauss-Bonnet）
- [ ] 评估对输入扰动的敏感性