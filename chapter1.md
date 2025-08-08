# 第1章：几何表示与变换

本章介绍3D打印中的基础几何表示方法和空间变换理论。我们将从网格、点云、隐式曲面等多种表示形式出发，深入探讨它们的数学基础、相互转换以及在3D打印中的应用。特别关注离散化带来的数值问题、拓扑保持性以及计算效率。通过本章学习，读者将掌握选择合适几何表示的准则、实现高效空间变换的技巧，以及处理大规模几何数据的方法。

## 1.1 网格表示：半边结构、四边形网格、体素化

### 1.1.1 三角网格与拓扑

三角网格是3D打印中最常用的几何表示。设 $M = (V, E, F)$ 为三角网格，其中 $V = \{v_i \in \mathbb{R}^3\}_{i=1}^{n_v}$ 为顶点集，$E$ 为边集，$F$ 为面集。

**欧拉-庞加莱公式**：对于亏格为 $g$ 的闭合曲面：
$$\chi(M) = |V| - |E| + |F| = 2 - 2g$$

其中 $\chi(M)$ 为欧拉特征数。对于三角网格，有约束：$3|F| = 2|E|$（每个面有3条边，每条边被2个面共享）。

**顶点度数分布**：平均顶点度数 $\bar{d} = \frac{2|E|}{|V|}$。对于大型三角网格，根据欧拉公式和边面关系：
$$\bar{d} \approx 6$$

这个结果来源于：$|E| \approx 3|V|$（当 $|V| \gg 1$ 时），因此每个顶点平均连接6条边。

**网格的定向性**：网格 $M$ 是可定向的当且仅当存在一致的面法向分配。对于三角形 $(v_i, v_j, v_k)$，法向定义为：
$$\vec{n} = \frac{(v_j - v_i) \times (v_k - v_i)}{||(v_j - v_i) \times (v_k - v_i)||}$$

**Möbius带检测**：非定向曲面（如Möbius带）的特征是存在一条路径，沿着它行走会反转法向。数学上，这对应于 $H_1(M; \mathbb{Z}_2) \neq 0$。

### 1.1.2 半边数据结构

半边结构将每条边分解为两个有向半边，实现 $O(1)$ 时间复杂度的局部拓扑查询。

**半边结构定义**：
```
HalfEdge {
    vertex: 指向起点
    twin: 对偶半边
    next: 同面下一条半边
    prev: 同面上一条半边  
    face: 所属面
}
```

**1-环邻域遍历算法**：给定顶点 $v$，遍历其1-环邻域的时间复杂度为 $O(d)$，其中 $d$ 为顶点度数。

```
function traverse_1ring(v):
    h = v.halfedge  // 出发半边
    h_start = h
    do:
        yield h.twin.vertex  // 邻接顶点
        h = h.twin.next  // 逆时针旋转
    while h != h_start
```

**拓扑算子复杂度分析**：
- 边收缩(edge collapse)：$O(d_1 + d_2)$
- 边翻转(edge flip)：$O(1)$
- 顶点分裂(vertex split)：$O(d)$

**边收缩的拓扑有效性检查**：
收缩边 $(v_1, v_2) \to v_{new}$ 前需验证：
1. **Link条件**：$link(v_1) \cap link(v_2) = link(e)$
2. **流形保持**：收缩后不产生非流形边或顶点
3. **几何有效性**：不产生翻转三角形（法向反转）

翻转检测：对每个受影响三角形，检查：
$$\text{sign}(\vec{n}_{old} \cdot \vec{n}_{new}) > 0$$

**内存布局优化**：
- **面向数组的半边结构**：使用索引替代指针，提高缓存局部性
- **内存对齐**：确保结构体大小为缓存行的倍数（通常64字节）
- **冷热数据分离**：频繁访问的拓扑信息与几何坐标分开存储

### 1.1.3 四边形网格与张量积曲面

四边形网格在CAD/CAM中广泛应用，支持张量积参数化：

$$S(u,v) = \sum_{i=0}^{m}\sum_{j=0}^{n} P_{ij} B_i^p(u) B_j^q(v)$$

其中 $B_i^p$ 为 $p$ 次B样条基函数，$P_{ij}$ 为控制点。

**B样条基函数递归定义**（Cox-de Boor公式）：
$$B_i^p(u) = \frac{u - u_i}{u_{i+p} - u_i}B_i^{p-1}(u) + \frac{u_{i+p+1} - u}{u_{i+p+1} - u_{i+1}}B_{i+1}^{p-1}(u)$$

基函数满足：
- 非负性：$B_i^p(u) \geq 0$
- 局部支撑：$B_i^p(u) = 0$ 当 $u \notin [u_i, u_{i+p+1}]$
- 单位分解：$\sum_i B_i^p(u) = 1$

**四边形网格质量度量**：
- 正交性：$Q_{orth} = \min_f \cos\theta_f$，$\theta_f$ 为四边形内角
- 长宽比：$Q_{ar} = \min_f \frac{\min(l_1, l_2)}{\max(l_1, l_2)}$
- 翘曲度：$Q_{warp} = \min_f \frac{h}{\max(d_1, d_2)}$，$h$ 为四点到拟合平面的最大距离
- Jacobian质量：$Q_{jac} = \min_f \min_{(u,v) \in f} \det(J)$，其中：
  $$J = \begin{bmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\ \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} \end{bmatrix}$$

**四边形网格生成方法**：
1. **前进前沿法**：从边界向内逐层生成，复杂度 $O(n\log n)$
2. **中轴变换法**：基于Voronoi图的对偶，生成高质量四边形
3. **整数规划法**：全局优化框架，求解混合整数规划问题

### 1.1.4 体素化与稀疏表示

体素化将连续几何离散到规则网格 $\mathbb{Z}^3$。设分辨率为 $N^3$，则内存需求为 $O(N^3)$。

**6-分离体素化**：保证原始曲面的6-连通性
$$V_{6sep}(p) = \begin{cases} 1 & \text{if } \exists v \in S, ||p - v||_\infty < 0.5 \\ 0 & \text{otherwise} \end{cases}$$

**26-连通性分析**：
- 6-邻接：共享面的体素，$\Delta = \{(\pm1,0,0), (0,\pm1,0), (0,0,\pm1)\}$
- 18-邻接：共享边的体素，增加12个方向
- 26-邻接：共享顶点的体素，全部27-1=26个方向

拓扑一致性定理：6-连通物体的补集是26-连通的，反之亦然。

**稀疏八叉树表示**：
- 空间复杂度：$O(N^2)$（仅存储表面体素）
- 查询复杂度：$O(\log N)$
- Morton编码：$(x,y,z) \mapsto$ 交错位表示，支持空间局部性

Morton编码计算：
$$M(x,y,z) = \sum_{i=0}^{b-1} (x_i \cdot 2^{3i} + y_i \cdot 2^{3i+1} + z_i \cdot 2^{3i+2})$$
其中 $x_i, y_i, z_i$ 为坐标的第 $i$ 位。

**距离场体素化**：存储到最近表面的有符号距离
$$D(p) = \text{sign}(p) \cdot \min_{q \in S} ||p - q||_2$$

**窄带优化**：仅在表面附近 $|D(p)| < \delta$ 的区域存储精确距离：
$$D_{narrow}(p) = \begin{cases}
    D(p) & \text{if } |D(p)| < \delta \\
    \text{sign}(D(p)) \cdot \delta & \text{otherwise}
\end{cases}$$

内存节省：从 $O(N^3)$ 降至 $O(N^2 \cdot \delta/h)$，其中 $h$ 为网格间距。

## 1.2 点云与隐式曲面

### 1.2.1 点云表示与处理

点云 $P = \{p_i \in \mathbb{R}^3\}_{i=1}^n$ 配合法向 $N = \{n_i \in S^2\}_{i=1}^n$。

**k-d树构建**：递归分割，时间复杂度 $O(n\log n)$，查询 $O(\log n)$。

分割策略：
1. **循环分割**：按 $x, y, z$ 轴循环
2. **最大方差分割**：选择方差最大的维度
3. **表面积启发式(SAH)**：最小化预期遍历代价

**法向估计**（主成分分析）：
对邻域点集 $\{p_j\}_{j \in N(i)}$，协方差矩阵：
$$C = \frac{1}{k}\sum_{j \in N(i)} (p_j - \bar{p})(p_j - \bar{p})^T$$

特征分解 $C = V\Lambda V^T$，法向为最小特征值对应的特征向量。

**移动最小二乘(MLS)曲面**：
给定查询点 $x$，拟合局部多项式：
$$f(x) = \arg\min_f \sum_{p_i \in N(x)} w(||x - p_i||) ||f(p_i) - 0||^2$$

其中权重函数 $w(d) = \exp(-d^2/h^2)$，$h$ 为带宽参数。

**投影算子**：将点 $q$ 投影到MLS曲面：
1. 初始化：$x_0 = q$
2. 迭代：$x_{k+1} = x_k - f(x_k) \cdot \nabla f(x_k) / ||\nabla f(x_k)||^2$
3. 收敛条件：$||x_{k+1} - x_k|| < \epsilon$

收敛速率：二次收敛，通常3-5次迭代达到机器精度。

### 1.2.2 有符号距离场(SDF)

SDF定义：$\phi: \mathbb{R}^3 \rightarrow \mathbb{R}$，满足：
- $|\nabla\phi| = 1$ （Eikonal方程）
- 零水平集 $\{x: \phi(x) = 0\}$ 为目标曲面
- 符号约定：$\phi < 0$ 在物体内部，$\phi > 0$ 在外部

**SDF的几何性质**：
- 法向：$\vec{n} = \nabla\phi$
- 平均曲率：$\kappa = \Delta\phi$
- 最近点：$\vec{p}_{closest} = \vec{x} - \phi(\vec{x}) \cdot \nabla\phi(\vec{x})$

**快速行进法(FMM)**求解Eikonal方程：
$$|\nabla T| = \frac{1}{F(x)}$$

离散化更新规则（一阶精度）：
$$\max(D^{-x}_{ijk}T, -D^{+x}_{ijk}T, 0)^2 + \max(D^{-y}_{ijk}T, -D^{+y}_{ijk}T, 0)^2 + \max(D^{-z}_{ijk}T, -D^{+z}_{ijk}T, 0)^2 = h^2/F_{ijk}^2$$

**二阶精度FMM**：
使用二阶差分：
$$D^{-x}T = \frac{3T_{ijk} - 4T_{i-1,jk} + T_{i-2,jk}}{2h}$$

误差分析：一阶 $O(h)$，二阶 $O(h^2)$。

**快速扫描法(FSM)**：
更简单的并行化算法，交替执行8个方向的扫描：
```
for direction in [(+,+,+), (+,+,-), (+,-,+), ..., (-,-,-)]:
    sweep_grid(direction)
```
收敛速度：$O(N)$ 次扫描，每次 $O(N^3)$。

### 1.2.3 水平集方法与演化

水平集演化方程：
$$\frac{\partial\phi}{\partial t} + V|\nabla\phi| = 0$$

其中 $V$ 为法向速度。对于曲率流：$V = -\kappa$，其中平均曲率：
$$\kappa = \nabla \cdot \left(\frac{\nabla\phi}{|\nabla\phi|}\right) = \frac{\phi_{xx}(\phi_y^2 + \phi_z^2) + \phi_{yy}(\phi_x^2 + \phi_z^2) + \phi_{zz}(\phi_x^2 + \phi_y^2) - 2\phi_x\phi_y\phi_{xy} - 2\phi_x\phi_z\phi_{xz} - 2\phi_y\phi_z\phi_{yz}}{(\phi_x^2 + \phi_y^2 + \phi_z^2)^{3/2}}$$

**数值格式**：
1. **迎风格式**：根据速度方向选择差分
   $$\phi_x \approx \begin{cases}
       D^-_x\phi & \text{if } V > 0 \\
       D^+_x\phi & \text{if } V < 0
   \end{cases}$$

2. **ENO/WENO格式**：高阶精度，避免数值振荡
   - ENO：选择最平滑的模板
   - WENO：加权组合多个模板

**重新初始化**保持距离场性质：
$$\frac{\partial\phi}{\partial\tau} = \text{sign}(\phi_0)(1 - |\nabla\phi|)$$

止损函数稳定化：
$$\text{sign}(\phi_0) \approx \frac{\phi_0}{\sqrt{\phi_0^2 + \epsilon^2}}$$
其中 $\epsilon = \Delta x$ 为网格间距。

### 1.2.4 RBF隐式曲面

径向基函数插值：
$$f(x) = \sum_{i=1}^n \lambda_i \varphi(||x - c_i||) + p(x)$$

其中 $\varphi(r) = r^3$（三次RBF）或 $\varphi(r) = r^2\log r$（薄板样条）。

**常用RBF核函数**：
- 高斯：$\varphi(r) = e^{-(r/\sigma)^2}$
- 多二次：$\varphi(r) = \sqrt{r^2 + c^2}$
- 逆多二次：$\varphi(r) = 1/\sqrt{r^2 + c^2}$
- Wendland紧支撑：$\varphi(r) = (1-r)^4_+(4r+1)$

约束条件：
- 插值：$f(c_i) = d_i$
- 正交性：$\sum_{i=1}^n \lambda_i p_j(c_i) = 0$

线性系统：
$$\begin{bmatrix} A & P \\ P^T & 0 \end{bmatrix} \begin{bmatrix} \lambda \\ c \end{bmatrix} = \begin{bmatrix} d \\ 0 \end{bmatrix}$$

其中 $A_{ij} = \varphi(||c_i - c_j||)$，$P$ 为多项式基矩阵。

**快速多极子方法(FMM)**：
将计算复杂度从 $O(n^2)$ 降至 $O(n\log n)$：
1. 将远场作用通过多极展开近似
2. 使用八叉树分层计算
3. 近场直接计算，远场使用多极展开

**正则化**避免过拟合：
$$f(x) = \sum_{i=1}^n \lambda_i \varphi(||x - c_i||) + \alpha||\lambda||^2$$

正则化参数 $\alpha$ 通过交叉验证选择。

## 1.3 齐次坐标与射影几何

### 1.3.1 齐次坐标系统

点 $(x, y, z) \in \mathbb{R}^3$ 的齐次坐标：$(x, y, z, 1) \in \mathbb{P}^3$。
等价类：$[x:y:z:w] = \{(\lambda x, \lambda y, \lambda z, \lambda w) : \lambda \neq 0\}$。

**仿射变换矩阵**：
$$T = \begin{bmatrix} R & t \\ 0^T & 1 \end{bmatrix} = \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**基本变换矩阵**：

平移：
$$T_{trans} = \begin{bmatrix} 1 & 0 & 0 & \Delta x \\ 0 & 1 & 0 & \Delta y \\ 0 & 0 & 1 & \Delta z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

缩放：
$$T_{scale} = \begin{bmatrix} s_x & 0 & 0 & 0 \\ 0 & s_y & 0 & 0 \\ 0 & 0 & s_z & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

剪切（xy平面）：
$$T_{shear} = \begin{bmatrix} 1 & sh_y & 0 & 0 \\ sh_x & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**变换的复合**：
注意矩阵乘法的顺序（右乘）：
$$T_{total} = T_3 \cdot T_2 \cdot T_1$$
先执行 $T_1$，再 $T_2$，最后 $T_3$。

### 1.3.2 透视投影

透视投影矩阵（针孔相机模型）：
$$P = K[R|t] = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} R & t \end{bmatrix}$$

其中 $K$ 为内参矩阵，$(f_x, f_y)$ 为焦距，$(c_x, c_y)$ 为主点，$s$ 为偷斜参数。

**完整投影矩阵**（OpenGL风格）：
$$P_{proj} = \begin{bmatrix}
\frac{2n}{r-l} & 0 & \frac{r+l}{r-l} & 0 \\
0 & \frac{2n}{t-b} & \frac{t+b}{t-b} & 0 \\
0 & 0 & -\frac{f+n}{f-n} & -\frac{2fn}{f-n} \\
0 & 0 & -1 & 0
\end{bmatrix}$$

其中 $n, f$ 为近远裁剪面，$(l,r,b,t)$ 为视锥边界。

**深度恢复**：给定图像点 $(u, v)$ 和深度 $Z$：
$$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = Z \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

**畸变校正**：
径向畸变：
$$\begin{bmatrix} u' \\ v' \end{bmatrix} = \begin{bmatrix} u \\ v \end{bmatrix} (1 + k_1r^2 + k_2r^4 + k_3r^6)$$

切向畸变：
$$\begin{bmatrix} u' \\ v' \end{bmatrix} = \begin{bmatrix} u \\ v \end{bmatrix} + \begin{bmatrix} 2p_1uv + p_2(r^2 + 2u^2) \\ p_1(r^2 + 2v^2) + 2p_2uv \end{bmatrix}$$

其中 $r^2 = u^2 + v^2$，$(k_1, k_2, k_3)$ 为径向畸变系数，$(p_1, p_2)$ 为切向畸变系数。

### 1.3.3 交比不变性

四共线点 $A, B, C, D$ 的交比：
$$(A, B; C, D) = \frac{AC \cdot BD}{AD \cdot BC}$$

在射影变换下交比不变。这是射影几何的基本不变量。

**调和共轭**：当 $(A, B; C, D) = -1$ 时，称 $C, D$ 调和分离 $A, B$。

**应用：单应性矩阵正
给定四对对应点，计算单应性矩阵 $H$：
$$\vec{p}' = H\vec{p}$$

使用DLT（直接线性变换）算法：
1. 构建线性系统 $A\vec{h} = 0$
2. SVD分解：$A = U\Sigma V^T$
3. 解为 $V$ 的最后一列

**无穷远点与无穷远线**：
- 无穷远点：$[x:y:z:0]$，表示方向
- 无穷远线：$z = 0$（在2D射影平面中）
- 平行线交于无穷远点

### 1.3.4 对偶原理

射影平面中点与线的对偶：
- 点 $(a, b, c)$ ↔ 线 $ax + by + c = 0$
- 共线点 ↔ 共点线
- 两点连线 ↔ 两线交点

**Plücker坐标**表示3D直线：
$$L = (l:m:n:p:q:r) \in \mathbb{P}^5$$
满足Plücker关系：$lp + mq + nr = 0$。

由两点 $P_1 = (x_1, y_1, z_1, w_1)$ 和 $P_2 = (x_2, y_2, z_2, w_2)$ 确定的直线：
$$\begin{aligned}
l &= x_1w_2 - w_1x_2 \\
m &= y_1w_2 - w_1y_2 \\
n &= z_1w_2 - w_1z_2 \\
p &= y_1z_2 - z_1y_2 \\
q &= z_1x_2 - x_1z_2 \\
r &= x_1y_2 - y_1x_2
\end{aligned}$$

**两直线的位置关系**：
设两直线 $L_1 = (l_1:m_1:n_1:p_1:q_1:r_1)$ 和 $L_2 = (l_2:m_2:n_2:p_2:q_2:r_2)$：
- 相交：$l_1p_2 + l_2p_1 + m_1q_2 + m_2q_1 + n_1r_2 + n_2r_1 = 0$
- 异面距离：$d = \frac{|l_1p_2 + l_2p_1 + m_1q_2 + m_2q_1 + n_1r_2 + n_2r_1|}{\sqrt{(l_1^2+m_1^2+n_1^2)(l_2^2+m_2^2+n_2^2)}}$

## 1.4 四元数与旋转表示

### 1.4.1 四元数代数

四元数 $q = w + xi + yj + zk$，其中：
- $i^2 = j^2 = k^2 = ijk = -1$
- $ij = k, jk = i, ki = j$
- $ji = -k, kj = -i, ik = -j$

**乘法规则**：
$$q_1 q_2 = (w_1 w_2 - \vec{v_1} \cdot \vec{v_2}) + (w_1 \vec{v_2} + w_2 \vec{v_1} + \vec{v_1} \times \vec{v_2})$$

**基本运算**：
- 共轭：$q^* = w - xi - yj - zk$
- 模长：$||q|| = \sqrt{w^2 + x^2 + y^2 + z^2}$
- 逆元：$q^{-1} = q^*/||q||^2$
- 单位四元数：$||q|| = 1$

**四元数与复数的关系**：
四元数可看作复数的推广：
$$q = (a + bi) + (c + di)j$$
其中 $a, b, c, d \in \mathbb{R}$。

### 1.4.2 旋转表示

单位四元数表示绕轴 $\vec{n}$ 旋转角度 $\theta$：
$$q = \cos\frac{\theta}{2} + \sin\frac{\theta}{2}(n_x i + n_y j + n_z k)$$

**旋转向量**：$p' = qpq^*$，其中 $p = (0, \vec{v})$。

展开公式：
$$\vec{v}' = \vec{v} + 2\vec{q} \times (\vec{q} \times \vec{v} + w\vec{v})$$
其中 $q = (w, \vec{q})$。

展开为矩阵形式：
$$R(q) = \begin{bmatrix} 
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix}$$

**从旋转矩阵提取四元数**：
$$\begin{aligned}
w &= \frac{1}{2}\sqrt{1 + R_{11} + R_{22} + R_{33}} \\
x &= \frac{R_{32} - R_{23}}{4w} \\
y &= \frac{R_{13} - R_{31}}{4w} \\
z &= \frac{R_{21} - R_{12}}{4w}
\end{aligned}$$

注意：当 $w \approx 0$ 时需要特殊处理，选择最大对角元素。

### 1.4.3 球面线性插值(SLERP)

两个单位四元数 $q_0, q_1$ 之间的插值（$t \in [0,1]$）：
$$\text{slerp}(q_0, q_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}q_0 + \frac{\sin(t\Omega)}{\sin\Omega}q_1$$

其中 $\cos\Omega = q_0 \cdot q_1$。

**处理反向旋转**：
当 $q_0 \cdot q_1 < 0$ 时，将 $q_1$ 替换为 $-q_1$ 以选择短路径。

**数值稳定版本**（当 $\Omega$ 很小时）：
$$\text{slerp}(q_0, q_1, t) \approx (1-t)q_0 + tq_1 + \frac{\Omega^2}{6}t(1-t)(q_1 - q_0)$$

**三次球面插值(SQUAD)**：
给定四个控制点 $q_0, q_1, q_2, q_3$，计算中间控制点：
$$s_i = q_i \exp\left(-\frac{\log(q_i^{-1}q_{i-1}) + \log(q_i^{-1}q_{i+1})}{4}\right)$$

然后：
$$\text{squad}(q_i, q_{i+1}, s_i, s_{i+1}, t) = \text{slerp}(\text{slerp}(q_i, q_{i+1}, t), \text{slerp}(s_i, s_{i+1}, t), 2t(1-t))$$

### 1.4.4 旋转的其他表示

**欧拉角**（ZYX顺序）：
$$R = R_z(\gamma)R_y(\beta)R_x(\alpha)$$

具体展开：
$$R = \begin{bmatrix}
c\beta c\gamma & s\alpha s\beta c\gamma - c\alpha s\gamma & c\alpha s\beta c\gamma + s\alpha s\gamma \\
c\beta s\gamma & s\alpha s\beta s\gamma + c\alpha c\gamma & c\alpha s\beta s\gamma - s\alpha c\gamma \\
-s\beta & s\alpha c\beta & c\alpha c\beta
\end{bmatrix}$$

其中 $c = \cos, s = \sin$。

万向锁条件：当 $\beta = \pm\pi/2$ 时，失去一个自由度。

**轴角表示**：Rodriguez公式
$$R = I + \sin\theta [\vec{n}]_\times + (1-\cos\theta)[\vec{n}]_\times^2$$

其中反对称矩阵：
$$[\vec{n}]_\times = \begin{bmatrix} 0 & -n_z & n_y \\ n_z & 0 & -n_x \\ -n_y & n_x & 0 \end{bmatrix}$$

**指数映射**：
$$R = \exp([\vec{\omega}]_\times) = \sum_{k=0}^{\infty} \frac{[\vec{\omega}]_\times^k}{k!}$$

闭形式（Rodrigues公式）：
$$\exp([\vec{\omega}]_\times) = I + \frac{\sin||\vec{\omega}||}{||\vec{\omega}||} [\vec{\omega}]_\times + \frac{1-\cos||\vec{\omega}||}{||\vec{\omega}||^2} [\vec{\omega}]_\times^2$$

**对数映射**：
$$\log(R) = \frac{\theta}{2\sin\theta}(R - R^T)$$
其中 $\theta = \arccos\left(\frac{\text{tr}(R) - 1}{2}\right)$。

## 1.5 几何哈希与空间索引结构

### 1.5.1 空间哈希

将3D空间离散化为网格，哈希函数：
$$h(x, y, z) = (x \cdot p_1 \oplus y \cdot p_2 \oplus z \cdot p_3) \mod M$$

其中 $p_1, p_2, p_3$ 为大素数，$M$ 为哈希表大小。

**完美空间哈希**：两阶段方法
1. 第一级：粗网格哈希
   $$h_1(x, y, z) = \text{hash}(\lfloor x/s \rfloor, \lfloor y/s \rfloor, \lfloor z/s \rfloor) \mod M_1$$
2. 第二级：每个槽位的完美哈希
   $$h_2(x, y, z) = a_i x + b_i y + c_i z \mod p_i$$
   其中系数 $(a_i, b_i, c_i, p_i)$ 通过预计算保证无冲突。

空间复杂度：$O(n)$，查询时间：$O(1)$。

**局部敏感哈希(LSH)**用于近似最近邻：
$$h(p) = \lfloor \frac{\vec{a} \cdot \vec{p} + b}{w} \rfloor$$
其中 $\vec{a}$ 为随机向量，$b \in [0, w]$ 为随机偏移。

### 1.5.2 几何哈希用于形状匹配

**离线阶段**：
1. 选择基准对 $(p_i, p_j)$
2. 构建局部坐标系
   - 原点：$O = p_i$
   - $x$ 轴：$\vec{x} = (p_j - p_i)/||p_j - p_i||$
   - $z$ 轴：$\vec{z} = \vec{n}_i$（法向）
   - $y$ 轴：$\vec{y} = \vec{z} \times \vec{x}$
3. 变换其他点到局部坐标
   $$p'_k = R^T(p_k - O)$$
4. 量化并存入哈希表
   $$\text{key} = (\lfloor x'/\delta \rfloor, \lfloor y'/\delta \rfloor, \lfloor z'/\delta \rfloor, \lfloor \theta/\Delta\theta \rfloor)$$

**在线匹配**：
1. 枚举场景中的基准对
2. 查询哈希表投票
3. 验证高票数假设（RANSAC）

时间复杂度：$O(n^2m)$，其中 $n$ 为模型点数，$m$ 为场景点数。

**提高鲁棒性**：
- 使用多个基准对
- 投票时考虑邻域
- 采用加权投票（根据特征显著性）

### 1.5.3 R-树与包围盒层次

**R-树性质**：
- 每个节点包含 $[m, M]$ 个条目（根除外），通常 $m = 0.4M$
- 所有叶节点在同一层
- 每个条目的MBR（最小包围矩形）包含其所有子节点

**插入算法**（选择子树）：
1. 最小扩展准则：$\arg\min_i \text{volume}(MBR_i \cup object) - \text{volume}(MBR_i)$
2. 最小重叠准则（R*-树）：
   $$\arg\min_i \sum_{j \neq i} \text{overlap}(MBR_i', MBR_j)$$
   其中 $MBR_i'$ 为插入后的新MBR。

**节点分裂算法**：
1. **线性分裂**：$O(n)$，选择最远的两个对象作为种子
2. **二次分裂**：$O(n^2)$，考虑所有可能的分组
3. **R*-树分裂**：按每个维度排序，选择最优分割

**包围盒层次(BVH)**：
- **AABB**：轴对齐包围盒，更新 $O(1)$
- **OBB**：有向包围盒，更紧致但更新慢
- **球形包围**：旋转不变，适合动态场景

### 1.5.4 BSP树与kd-树

**BSP树构建**：递归分割，选择分割平面使两侧平衡。

分割平面选择准则：
$$\text{cost} = C_t + P_l \cdot C_l \cdot n_l + P_r \cdot C_r \cdot n_r$$

其中 $P_l, P_r$ 为命中概率，$n_l, n_r$ 为子节点三角形数。

**SAH（表面积启发式）**：
$$P_l = \frac{SA(V_l)}{SA(V)}, \quad P_r = \frac{SA(V_r)}{SA(V)}$$

**kd-树优化**：
1. **空间中位数分割**：简单但可能不平衡
2. **对象中位数分割**：平衡但可能不紧致
3. **SAH分割**：最优但构建慢

**并行构建**：
- **自顶向下**：递归并行分割
- **Morton编码**：先排序后分割，$O(n\log n)$
- **LBVH**：线性BVH构建，$O(n)$但质量稍差

**查询优化**：
- **提前终止**：找到第一个交点即返回
- **邮箱算法**：避免重复计算射线-平面交点
- **路径压缩**：动态调整树结构

## 本章小结

本章系统介绍了3D打印中的基础几何表示方法：
1. **网格表示**提供了离散曲面的拓扑结构，半边结构实现高效的局部操作
2. **隐式曲面**适合拓扑变化和布尔运算，SDF在物理仿真中广泛应用
3. **齐次坐标**统一了仿射和射影变换，是计算机图形学的数学基础
4. **四元数**避免了欧拉角的奇异性，提供稳定的旋转插值
5. **空间索引**加速几何查询，是处理大规模3D数据的关键

核心数学工具：
- 欧拉公式：$\chi = V - E + F = 2 - 2g$
- Eikonal方程：$|\nabla\phi| = 1$
- Rodriguez公式：$R = I + \sin\theta [\vec{n}]_\times + (1-\cos\theta)[\vec{n}]_\times^2$
- SLERP插值：$\text{slerp}(q_0, q_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}q_0 + \frac{\sin(t\Omega)}{\sin\Omega}q_1$

## 练习题

### 基础题

**习题1.1** 证明对于闭合三角网格，满足 $3|F| = 2|E|$ 的关系。

<details>
<summary>提示</summary>
考虑每个三角形有3条边，每条内部边被两个三角形共享。
</details>

<details>
<summary>答案</summary>

每个三角形贡献3条半边，总共 $3|F|$ 条半边。每条边对应2条半边（除了边界），因此 $3|F| = 2|E|$。

对于有边界的网格，设边界边数为 $|E_b|$，则：
$$3|F| = 2|E| - |E_b|$$
</details>

**习题1.2** 给定四元数 $q_1 = \frac{1}{\sqrt{2}}(1 + k)$ 和 $q_2 = \frac{1}{\sqrt{2}}(1 + i)$，计算：
a) $q_1 \cdot q_2$
b) 它们分别代表的旋转轴和角度
c) 复合旋转 $q_1 q_2$

<details>
<summary>提示</summary>
四元数 $q = \cos\frac{\theta}{2} + \sin\frac{\theta}{2}\vec{n}$ 表示绕轴 $\vec{n}$ 旋转 $\theta$。
</details>

<details>
<summary>答案</summary>

a) $q_1 \cdot q_2 = \frac{1}{2}(1 \cdot 1 + 0 \cdot 0 + 0 \cdot 0 + 1 \cdot 0) = \frac{1}{2}$

b) 对于 $q_1 = \frac{1}{\sqrt{2}}(1 + k)$：
   - $\cos\frac{\theta_1}{2} = \frac{1}{\sqrt{2}}$，所以 $\theta_1 = 90°$
   - 旋转轴：$z$ 轴
   
   对于 $q_2 = \frac{1}{\sqrt{2}}(1 + i)$：
   - $\theta_2 = 90°$
   - 旋转轴：$x$ 轴

c) $q_1 q_2 = \frac{1}{2}(1 + k)(1 + i) = \frac{1}{2}(1 + i + k + ki) = \frac{1}{2}(1 + i + j + k)$
</details>

**习题1.3** 设计一个算法，将点云转换为有符号距离场。给出算法步骤和复杂度分析。

<details>
<summary>提示</summary>
考虑使用kd-树加速最近邻查询，法向用于确定符号。
</details>

<details>
<summary>答案</summary>

算法步骤：
1. 构建点云的kd-树，$O(n\log n)$
2. 估计每个点的法向（PCA或MLS），$O(kn)$，$k$ 为邻域大小
3. 法向定向（最小生成树传播），$O(n\log n)$
4. 对每个查询点：
   - kd-树找最近点，$O(\log n)$
   - 计算距离和符号
5. 总复杂度：构建 $O(n\log n)$，查询 $O(m\log n)$，$m$ 为网格点数
</details>

### 挑战题

**习题1.4** 推导并实现Marching Cubes算法的二义性解决方案。考虑如何保证生成网格的流形性质。

<details>
<summary>提示</summary>
二义性出现在面二义性和体二义性两种情况。考虑使用渐近决策者(asymptotic decider)。
</details>

<details>
<summary>答案</summary>

面二义性：当立方体某个面的4个顶点呈对角分布（2正2负）时，有两种连接方式。

解决方案：
1. 计算面中心的双线性插值：
   $$f_c = \frac{1}{4}(f_{00} + f_{01} + f_{10} + f_{11})$$
   
2. 计算鞍点判别式：
   $$\Delta = f_{00}f_{11} - f_{01}f_{10}$$
   
3. 决策规则：
   - 若 $\text{sign}(f_c) = \text{sign}(\Delta)$：分离连接
   - 否则：连通连接

体二义性：使用三线性插值的临界点分析，保证拓扑一致性。

流形保证：确保每条边最多被两个三角形共享，顶点邻域同胚于圆盘或半圆盘。
</details>

**习题1.5** 证明任意旋转可以分解为不超过4次反射的复合。这与四元数的4维表示有何联系？

<details>
<summary>提示</summary>
Cartan-Dieudonné定理：$n$ 维正交变换最多需要 $n$ 次反射。
</details>

<details>
<summary>答案</summary>

3D旋转的反射分解：

1. 任意3D旋转 $R \in SO(3)$ 可表示为偶数次反射的复合
2. 最少需要2次反射（绕交线的旋转）
3. 一般情况需要4次反射

四元数联系：
- 单位四元数 $\mathbb{S}^3$ 是 $SO(3)$ 的双覆盖
- 四元数乘法对应Clifford代数中的几何积
- 反射 $r$ 对应四元数 $q = -\vec{n}$（纯虚四元数）
- 旋转 $R(v) = qvq^*$ 是两次反射的复合

具体分解：设旋转轴为 $\vec{n}$，角度 $\theta$：
1. 选择垂直于 $\vec{n}$ 的平面 $P_1$
2. 旋转 $\theta/2$ 后的平面 $P_2$
3. $R = R_{P_2} \circ R_{P_1}$

这解释了为什么四元数用半角 $\theta/2$：它直接编码了反射平面的关系。
</details>

**习题1.6** 设计一个自适应八叉树，使得叶节点大小与局部特征（曲率）成反比。推导误差界并分析存储效率。

<details>
<summary>提示</summary>
使用Hausdorff距离度量逼近误差，考虑曲率的二阶Taylor展开。
</details>

<details>
<summary>答案</summary>

自适应准则：
设叶节点尺寸 $h$，局部最大主曲率 $\kappa_{max}$。

误差界（Hausdorff距离）：
$$d_H \leq \frac{1}{8}\kappa_{max} h^2$$

自适应策略：
$$h = \min\left(\sqrt{\frac{8\epsilon}{\kappa_{max}}}, h_{max}\right)$$

其中 $\epsilon$ 为目标误差阈值。

存储分析：
- 平坦区域：大叶节点，节点数 $\sim O(1/h^2)$
- 高曲率区域：小叶节点，节点数 $\sim O(1/h^3)$
- 总存储：$O(A/\epsilon + L/\sqrt{\epsilon})$
  其中 $A$ 为表面积，$L$ 为特征边长度

优化：使用截断八叉树，仅在表面附近细分，内存降至 $O(n^{2/3})$。
</details>

**习题1.7** 推导3D Delaunay三角化的期望复杂度。在什么条件下会退化为 $O(n^2)$？

<details>
<summary>提示</summary>
考虑随机点分布和最坏情况（如圆柱表面的点）。
</details>

<details>
<summary>答案</summary>

随机点分布的期望复杂度：
- 四面体数：$E[|T|] = O(n)$
- 边数：$E[|E|] = O(n)$
- 构建时间：$E[T_{build}] = O(n\log n)$

证明要点：
1. Euler公式：$|V| - |E| + |F| - |T| = 1$
2. 每个四面体4个面，每个内部面被2个四面体共享：$4|T| \approx 2|F|$
3. 代入得：$|T| \approx n$

退化情况：
圆柱表面均匀分布的点：
- 所有点近似共圆
- Delaunay四面体数：$\Theta(n^2)$
- 构建复杂度：$O(n^2)$

其他退化配置：
- 螺旋线上的点
- 抛物面 $z = x^2 + y^2$ 上的点（投影后共圆）
</details>

## 常见陷阱与错误

### 1. 网格拓扑陷阱

**陷阱1**：假设网格总是流形
- 问题：T-junction、非流形边、孤立顶点
- 检测：每条边应恰好被2个面共享（边界边为1个）
- 修复：使用 `mesh.is_manifold()` 检查，`mesh.repair()` 修复

**陷阱2**：欧拉公式误用
- 错误：对非闭合曲面使用 $\chi = 2 - 2g$
- 正确：带边界曲面 $\chi = 2 - 2g - b$，$b$ 为边界环数

### 2. 数值精度问题

**陷阱3**：浮点数比较
- 问题：`if (dot == 1.0)` 判断向量平行
- 解决：使用容差 `if (abs(dot - 1.0) < 1e-6)`

**陷阱4**：四元数归一化
- 问题：累积误差导致四元数模长偏离1
- 解决：定期归一化，但注意归一化频率（过频影响性能）

### 3. 算法选择错误

**陷阱5**：SDF计算用暴力法
- 错误：对每个网格点遍历所有三角形，$O(mn)$
- 正确：使用Fast Marching或距离变换，$O(n\log n)$

**陷阱6**：点云法向估计不一致
- 问题：法向指向随机，内外不一致
- 解决：使用最小生成树传播或泊松重建的一致定向

### 4. 内存管理陷阱

**陷阱7**：体素化分辨率过高
- 问题：$1024^3$ 体素需要1GB内存（仅存储占用标记）
- 解决：使用稀疏体素、八叉树或距离场

**陷阱8**：半边结构的指针失效
- 问题：vector重分配导致指针/迭代器失效
- 解决：使用索引而非指针，或预留足够空间

## 最佳实践检查清单

### 几何表示选择

- [ ] **网格 vs 隐式**：是否需要拓扑操作？布尔运算？
- [ ] **精度要求**：CAD级精度用NURBS，扫描数据用网格
- [ ] **内存限制**：大模型考虑LOD或流式处理
- [ ] **查询类型**：最近点查询多用kd-树，相交测试用BVH

### 数据结构设计

- [ ] **拓扑完整性**：是否需要快速的邻接查询？
- [ ] **动态更新**：静态用数组，动态用半边结构
- [ ] **并行友好**：避免指针追逐，使用面向数据设计
- [ ] **缓存友好**：数据局部性，考虑内存布局

### 数值稳定性

- [ ] **条件数检查**：矩阵求逆前检查条件数
- [ ] **归一化时机**：四元数、法向量定期归一化
- [ ] **容差设置**：几何谓词使用自适应精度算术
- [ ] **溢出预防**：大坐标值先平移到原点附近

### 算法效率

- [ ] **渐进复杂度**：确认算法复杂度符合数据规模
- [ ] **常数因子**：简单算法在小数据上可能更快
- [ ] **并行化潜力**：数据并行优于任务并行
- [ ] **内存访问模式**：顺序访问优于随机访问

### 鲁棒性检查

- [ ] **退化输入**：共线点、共面点、零面积三角形
- [ ] **边界情况**：空输入、单点、重复点
- [ ] **数值极端**：很大/很小的数值、接近零的除数
- [ ] **拓扑异常**：自相交、非流形、孤立元素
