# 第3章：计算共形几何

共形几何研究保角变换下的不变量，在3D打印中具有重要应用：参数化、纹理映射、形状分析和变形设计。本章从Riemann映射定理出发，深入探讨离散共形几何的计算方法，包括离散Ricci流、圆堆积理论和Teichmüller空间。我们将重点关注如何将连续理论离散化并保持其关键性质，以及在实际计算中的数值稳定性问题。

## 3.1 共形映射与Riemann映射定理

### 3.1.1 共形映射的定义与性质

设 $f: \Omega \subset \mathbb{C} \to \mathbb{C}$ 是一个全纯函数，若 $f'(z) \neq 0$ 对所有 $z \in \Omega$ 成立，则 $f$ 是共形映射。共形映射的核心性质是保角性：

$$\lim_{r \to 0} \frac{\angle(f(z + re^{i\theta_1}), f(z), f(z + re^{i\theta_2}))}{\angle(z + re^{i\theta_1}, z, z + re^{i\theta_2})} = 1$$

在局部坐标下，共形映射的Jacobian矩阵具有形式：

$$J_f = \lambda(z) \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

其中 $\lambda(z) = |f'(z)|$ 称为共形因子，表征局部的缩放率。

**关键性质**：
1. **保圆性**：无穷小圆映射为无穷小圆
2. **调和共轭**：若 $f = u + iv$ 共形，则 $u, v$ 满足Cauchy-Riemann方程：
   $$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$
3. **最大模原理**：非常数全纯函数的模在区域内部无局部极大值
4. **保持定向**：共形映射保持曲线的定向（$f'(z) \neq 0$ 保证）

**度量变换**：若原度量为 $ds^2 = |dz|^2$，则在共形映射下新度量为：
$$ds'^2 = |f'(z)|^2 |dz|^2 = \lambda(z)^2 ds^2$$

这表明共形映射诱导的度量变换是各向同性的局部缩放。

### 3.1.2 Riemann映射定理

**定理3.1** (Riemann映射定理) 设 $\Omega \subset \mathbb{C}$ 是单连通域且 $\Omega \neq \mathbb{C}$，则存在共形映射 $f: \Omega \to \mathbb{D}$，其中 $\mathbb{D} = \{z \in \mathbb{C}: |z| < 1\}$ 是单位圆盘。

证明思路基于极值原理：考虑函数族
$$\mathcal{F} = \{f: \Omega \to \mathbb{D} \text{ 全纯}, f(z_0) = 0, f'(z_0) > 0\}$$

通过Montel定理证明 $\mathcal{F}$ 是正规族，存在使 $|f'(z_0)|$ 达到最大值的函数。

**唯一性定理**：固定 $z_0 \in \Omega$ 和 $\arg f'(z_0)$，则Riemann映射唯一确定。这给出3个实参数的自由度（$z_0$ 的2个坐标 + 旋转角度）。

**构造性证明要素**：
1. **Montel正规性**：一致有界的全纯函数族在紧子集上等度连续
2. **Hurwitz定理**：全纯函数序列的极限若非常数，则保持单叶性
3. **Schwarz引理**：$f: \mathbb{D} \to \mathbb{D}$ 全纯且 $f(0) = 0$，则 $|f(z)| \leq |z|$ 且 $|f'(0)| \leq 1$

**边界对应定理** (Carathéodory)：若 $\partial\Omega$ 是Jordan曲线，则共形映射 $f$ 可连续延拓到边界，给出 $\bar{\Omega} \to \bar{\mathbb{D}}$ 的同胚。

### 3.1.3 Schwarz-Christoffel映射

对于多边形域，Schwarz-Christoffel公式给出显式的共形映射：

$$f(z) = A + C \int_0^z \prod_{k=1}^n (\zeta - a_k)^{\alpha_k - 1} d\zeta$$

其中 $a_k$ 是单位圆上的点，$\alpha_k\pi$ 是多边形在第 $k$ 个顶点的内角。

参数确定通过求解非线性方程组：
$$\sum_{k=1}^n \alpha_k = n - 2, \quad \prod_{k=1}^n a_k^{\alpha_k} = 1$$

**几何解释**：
- 导数 $f'(z) = C\prod_{k=1}^n (z - a_k)^{\alpha_k - 1}$ 在 $a_k$ 处有奇点
- 穿过 $a_k$ 时，$\arg f'(z)$ 跳变 $(\alpha_k - 1)\pi$，对应多边形转角
- 积分路径绕过奇点时产生多值性，需要选择适当的分支切割

**特殊情况**：
1. **矩形映射** ($n=4, \alpha_k = 1/2$)：涉及椭圆积分
   $$f(z) = \int_0^z \frac{d\zeta}{\sqrt{(1-\zeta^2)(1-k^2\zeta^2)}}$$
   模数 $k$ 决定矩形的纵横比

2. **正多边形**：利用对称性，$a_k = e^{2\pi ik/n}$，$\alpha_k = 1 - 2/n$

3. **半平面到多边形**：使用扩展的Schwarz-Christoffel公式
   $$f(z) = A + C \int_{-\infty}^z \prod_{k=1}^n (t - x_k)^{\alpha_k - 1} dt$$

**数值挑战**：
- **Crowding现象**：顶点在圆周上聚集，导致数值不稳定
- **参数问题**：$n-3$ 个自由参数的非线性优化
- **奇异积分**：靠近顶点时被积函数的奇异性

### 3.1.4 数值计算方法

**Zipper算法**：基于测地线的迭代构造
1. 将边界离散为点集 $\{z_1, ..., z_n\}$
2. 迭代应用 Möbius 变换：
   $$\phi_k(z) = \sqrt{\frac{z - z_k}{z - \bar{z}_k}}$$
3. 累积变换得到近似共形映射

算法复杂度：$O(n^2)$，其中 $n$ 是边界点数。收敛速度依赖于边界的光滑性。

**CRDT (Cross-Ratio Dynamics Toolbox)**：
- 利用交比不变性
- 定义离散共形能量：
  $$E = \sum_{i,j,k,l} |cr(z_i, z_j, z_k, z_l) - cr(w_i, w_j, w_k, w_l)|^2$$
- 通过梯度下降优化

**圆域迭代法** (Wegmann方法)：
1. 初始化：$f_0(z) = z$
2. 迭代步骤：
   - 计算边界对应：$w_k = f_n(e^{i\theta_k})$
   - 求解共轭函数：$\tilde{h}$ 使得 $h + i\tilde{h}$ 全纯
   - 更新：$f_{n+1} = e^{h + i\tilde{h}} \circ f_n$
3. 收敛准则：$\max_k |w_k - e^{i\phi_k}| < \epsilon$

**有限元方法**：
- 变分形式：最小化Dirichlet能量
  $$E[u] = \int_\Omega |\nabla u|^2 dA$$
- 约束条件：边界值和共形性约束
- 离散化：分片线性基函数
- 求解：稀疏线性系统

**快速多极子方法** (FMM)：
- 用于加速边界积分方程求解
- 复杂度从 $O(n^2)$ 降至 $O(n\log n)$
- 适用于复杂边界的高精度计算

### 3.1.5 拟共形映射推广

当允许有界的角度畸变时，得到拟共形映射。Beltrami系数刻画局部畸变：

$$\mu(z) = \frac{\bar{\partial}f}{\partial f} = \frac{f_{\bar{z}}}{f_z}$$

满足 $|\mu(z)| < 1$ 的映射称为拟共形映射。最大畸变为：
$$K = \frac{1 + |\mu|_\infty}{1 - |\mu|_\infty}$$

**几何意义**：
- $|\mu(z)| = 0$：共形映射（无畸变）
- $|\mu(z)| = k < 1$：椭圆畸变，长短轴比为 $\frac{1+k}{1-k}$
- $\arg\mu(z)$：主畸变方向

**Beltrami方程**：
$$f_{\bar{z}} = \mu(z) f_z$$

这是拟共形映射的基本方程。给定 $\mu$ 满足 $|\mu|_\infty < 1$，存在唯一（相差共形映射）的拟共形映射满足此方程。

**可测Riemann映射定理**：对于可测的Beltrami系数 $\mu$ 满足 $|\mu|_\infty < 1$，存在拟共形同胚 $f: \mathbb{C} \to \mathbb{C}$ 满足Beltrami方程，且归一化条件 $f(0) = 0, f(1) = 1, f(\infty) = \infty$ 唯一确定 $f$。

**应用于3D打印**：
- **容错参数化**：允许小的角度畸变以改善其他性质（如面积畸变）
- **渐进共形**：通过迭代减小 $|\mu|$ 逐步改善共形性
- **约束变形**：控制 $\mu$ 的分布实现特定的变形效果

## 3.2 离散Ricci流

### 3.2.1 连续Ricci流回顾

Hamilton的Ricci流方程：
$$\frac{\partial g_{ij}}{\partial t} = -2R_{ij}$$

其中 $R_{ij}$ 是Ricci曲率张量。对于曲面，简化为：
$$\frac{\partial g}{\partial t} = -2Kg$$

这里 $K$ 是高斯曲率。

**归一化Ricci流**：为保持总面积不变
$$\frac{\partial g}{\partial t} = -2Kg + \frac{2\bar{K}}{A}g$$

其中 $\bar{K} = \int_M K dA$ 是平均曲率，$A$ 是总面积。

**共形坐标下的表示**：设度量 $g = e^{2u}g_0$，则
$$\frac{\partial u}{\partial t} = -K = \Delta_{g_0} u - K_0$$

这将Ricci流转化为热方程类型，其中 $\Delta_{g_0}$ 是背景度量的Laplacian。

**长时间行为**：
- **正曲率**：收缩到点（有限时间奇点）
- **零曲率**：保持不变（平坦度量）
- **负曲率**：演化到常负曲率度量

### 3.2.2 离散曲率定义

对于三角网格，顶点 $v_i$ 的离散高斯曲率：
$$K_i = \begin{cases}
2\pi - \sum_{jk \in N(i)} \theta_{jk}^i & \text{内部顶点} \\
\pi - \sum_{jk \in N(i)} \theta_{jk}^i & \text{边界顶点}
\end{cases}$$

其中 $\theta_{jk}^i$ 是三角形 $(i,j,k)$ 在顶点 $i$ 处的角度。

**几何意义**：
- $K_i > 0$：顶点处角度亏量（锥形奇点）
- $K_i < 0$：顶点处角度盈余（鞍点）
- $K_i = 0$：局部平坦

**离散平均曲率**：
$$H_i = \frac{1}{4A_i} \sum_{j \in N(i)} (\cot\alpha_{ij} + \cot\beta_{ij}) l_{ij}$$

其中 $A_i$ 是顶点 $i$ 的Voronoi面积，$\alpha_{ij}, \beta_{ij}$ 是边 $(i,j)$ 对角。

**标量曲率**：
$$R_i = \frac{2K_i}{A_i}$$

满足离散的Gauss-Bonnet定理：
$$\sum_{i \in V} K_i A_i = 2\pi\chi(M) \cdot \text{Area}(M)$$

### 3.2.3 离散Ricci流方程

引入共形因子 $u_i$，边长通过以下关系更新：
$$l_{ij}^{new} = e^{\frac{u_i + u_j}{2}} l_{ij}^{old}$$

离散Ricci流的演化方程：
$$\frac{du_i}{dt} = (\bar{K}_i - K_i(u))$$

其中 $\bar{K}_i$ 是目标曲率。

### 3.2.4 Ricci能量与优化

定义Ricci能量：
$$\mathcal{E}(u) = \int_0^u \sum_{i=1}^n (K_i(v) - \bar{K}_i) dv_i$$

其Hessian矩阵（也称为离散Laplace矩阵）：
$$H_{ij} = \frac{\partial^2 \mathcal{E}}{\partial u_i \partial u_j} = \begin{cases}
-\frac{\partial \theta_{jk}^i}{\partial u_j} & i \neq j, (i,j) \in E \\
\sum_{k \in N(i)} \frac{\partial \theta_{ki}^j}{\partial u_i} & i = j \\
0 & \text{otherwise}
\end{cases}$$

使用Newton法求解：
$$H \Delta u = \bar{K} - K(u)$$

### 3.2.5 奇异性处理

当三角形退化时，需要特殊处理：
1. **虚拟边翻转**：当 $l_{ij} + l_{jk} < l_{ik}$ 时
2. **能量正则化**：添加势垒函数
   $$\mathcal{E}_{reg} = \mathcal{E} + \epsilon \sum_{ijk} \log(\text{Area}_{ijk})$$

### 3.2.6 双曲与球面情况

对于非欧几何，边长关系修改为：

**双曲情况**：
$$\cosh l_{ij} = \cosh r_i \cosh r_j + \sinh r_i \sinh r_j \cos \theta_{ij}$$

**球面情况**：
$$\cos l_{ij} = \cos r_i \cos r_j + \sin r_i \sin r_j \cos \theta_{ij}$$

其中 $r_i = e^{u_i}$ 是离散共形因子。

**统一框架**：使用背景几何参数 $\kappa$
- $\kappa = 0$：欧氏几何
- $\kappa > 0$：球面几何（曲率半径 $1/\sqrt{\kappa}$）
- $\kappa < 0$：双曲几何（曲率 $-|\kappa|$）

**离散Yamabe流**：寻找常标量曲率度量
$$\frac{du_i}{dt} = (\bar{R} - R_i(u))$$

其中 $\bar{R}$ 是目标标量曲率，由拓扑决定：
- 球面：$\bar{R} > 0$
- 环面：$\bar{R} = 0$
- 高亏格曲面：$\bar{R} < 0$

**数值实现要点**：
- 双曲计算使用Poincaré圆盘模型避免溢出
- 球面计算需要处理反极点奇异性
- 自适应选择背景几何以改善条件数

## 3.3 圆堆积与离散共形因子

### 3.3.1 圆堆积定理

**定理3.2** (Koebe-Andreev-Thurston) 对于任意平面三角剖分 $G$，存在唯一（相差Möbius变换）的圆堆积 $P$，使得：
1. 圆 $C_i$ 与 $C_j$ 相切当且仅当 $(i,j) \in E$
2. 三个圆 $C_i, C_j, C_k$ 围成三角形当且仅当 $(i,j,k) \in F$

**推广形式**：
- **相交角圆堆积**：圆可以相交，指定交角 $\Phi_{ij} \in [0, \pi]$
- **广义圆堆积**：允许圆退化为点或直线
- **高维球堆积**：推广到 $n$ 维球面堆积

**存在性证明要点**：
1. 构造凸函数（Colin de Verdière泛函）
2. 证明临界点唯一
3. 使用不动点定理或变分方法

**离散共形等价**：两个圆堆积共形等价当且仅当对应的半径满足：
$$\frac{r_i'}{r_i} = \lambda \cdot \rho(v_i)$$
其中 $\lambda$ 是全局缩放，$\rho$ 是离散共形密度。

### 3.3.2 离散共形因子计算

给定目标角度 $\Theta_{ij}$，寻找半径 $r_i$ 使得：
$$\theta_{ij}(r) = \Theta_{ij}$$

其中 $\theta_{ij}$ 由余弦定理决定：
$$\cos \theta_{ij} = \frac{(r_i + r_j)^2 + (r_i + r_k)^2 - (r_j + r_k)^2}{2(r_i + r_j)(r_i + r_k)}$$

### 3.3.3 Thurston迭代算法

1. 初始化半径 $r_i^{(0)} = 1$
2. 迭代更新：
   $$r_i^{(n+1)} = r_i^{(n)} \left(\frac{\Theta_i}{\theta_i^{(n)}}\right)^\lambda$$
   其中 $\lambda \in (0,1]$ 是步长，$\Theta_i = 2\pi$ （内部顶点）

收敛性由Perron-Frobenius定理保证。

### 3.3.4 变分原理

定义能量函数：
$$\mathcal{F}(r) = \sum_{(i,j,k) \in F} \left( \sum_{cyclic} l_{ij} \theta_k - \pi l_{ij} \right) + \sum_{i \in V} \Theta_i \log r_i$$

其中 $l_{ij} = r_i + r_j$。能量最小化等价于圆堆积方程。

### 3.3.5 逆距离圆堆积

对于一般度量，使用逆距离定义：
$$\gamma_{ij} = \frac{1}{l_{ij}}$$

修改的圆堆积条件：
$$r_i + r_j = \frac{1}{\gamma_{ij}}$$

这导致优化问题：
$$\min_r \sum_{ij} (r_i + r_j - \frac{1}{\gamma_{ij}})^2$$

受约束 $r_i > 0$。

## 3.4 全纯微分与共形不变量

### 3.4.1 全纯1-形式

在Riemann面 $M$ 上，全纯1-形式 $\omega$ 局部表示为：
$$\omega = f(z)dz$$

其中 $f(z)$ 是全纯函数。全纯1-形式空间 $\Omega^{1,0}(M)$ 的维数由Riemann-Roch定理给出：
$$\dim \Omega^{1,0}(M) = g$$

其中 $g$ 是曲面的亏格。

### 3.4.2 Abel-Jacobi映射

定义周期矩阵：
$$\Pi_{ij} = \oint_{A_j} \omega_i$$

其中 $\{A_j\}$ 是同调基，$\{\omega_i\}$ 是全纯微分基。Abel-Jacobi映射：
$$\mathcal{A}: M \to \mathbb{C}^g/\Lambda, \quad p \mapsto \left(\int_{p_0}^p \omega_1, ..., \int_{p_0}^p \omega_g\right)$$

### 3.4.3 离散全纯形式

在三角网格上，离散全纯1-形式定义在对偶边上：
$$\omega_{ij}^* = \alpha_{ij} + i\beta_{ij}$$

满足离散Cauchy-Riemann条件：
$$\sum_{j \in N(i)} \omega_{ij}^* = 0$$

### 3.4.4 共形模与极值长度

两个环形区域之间的共形模定义为：
$$\text{Mod}(A) = \frac{1}{2\pi} \log \frac{R_2}{R_1}$$

极值长度提供了另一种刻画：
$$\lambda(\Gamma) = \sup_\rho \frac{L_\rho(\Gamma)^2}{A_\rho}$$

其中 $\Gamma$ 是曲线族，$\rho$ 是度量。

### 3.4.5 谱共形不变量

Laplace-Beltrami算子的谱：
$$\Delta \phi_k = \lambda_k \phi_k$$

热核的迹给出共形不变量：
$$Z(t) = \sum_{k=0}^\infty e^{-\lambda_k t}$$

渐近展开：
$$Z(t) \sim \frac{\text{Area}}{4\pi t} + \frac{\chi(M)}{6} + O(t)$$

## 3.5 Teichmüller理论应用

### 3.5.1 Teichmüller空间

亏格 $g$ 曲面的Teichmüller空间 $\mathcal{T}_g$ 是所有共形结构的模空间。维数为：
$$\dim_\mathbb{R} \mathcal{T}_g = \begin{cases}
2 & g = 0 \\
2 & g = 1 \\
6g - 6 & g \geq 2
\end{cases}$$

### 3.5.2 Fenchel-Nielsen坐标

通过裤子分解，引入坐标 $(l_1, \tau_1, ..., l_{3g-3}, \tau_{3g-3})$：
- $l_i$：测地线长度
- $\tau_i$：扭转参数

度量形式：
$$ds^2 = \sum_{i=1}^{3g-3} (dl_i^2 + l_i^2 d\tau_i^2)$$

### 3.5.3 极值映射

Teichmüller映射最小化最大畸变：
$$K(f) = \inf_{g \in \text{Homeo}} K(g)$$

其Beltrami系数具有形式：
$$\mu = k \frac{|\phi|}{\phi}$$

其中 $\phi$ 是全纯二次微分，$k = \frac{K-1}{K+1}$。

### 3.5.4 离散化方法

**边长坐标**：使用边长 $\{l_{ij}\}$ 参数化
- 约束：三角不等式
- 维数：$|E| - |V| + \chi(M)$

**角度坐标**：使用角度 $\{\theta_{ijk}\}$ 参数化
- 约束：$\sum_k \theta_{ijk} = \pi$（每条边）
- 维数：$|F| - |E| + \chi(M)$

### 3.5.5 形状空间测地线

在形状空间中，测地线满足：
$$\frac{d^2 x}{dt^2} + \Gamma_{jk}^i \frac{dx^j}{dt} \frac{dx^k}{dt} = 0$$

离散化为：
$$x^{n+1} = 2x^n - x^{n-1} - h^2 \Gamma(x^n, \dot{x}^n)$$

应用：形状插值、变形动画。

## 本章小结

本章系统介绍了计算共形几何的核心概念和算法：

**关键数学工具**：
- Riemann映射定理及其数值实现
- 离散Ricci流与曲率流动
- 圆堆积理论与Thurston迭代
- 全纯微分与共形不变量
- Teichmüller空间与极值映射

**主要计算方法**：
- Schwarz-Christoffel公式：$O(n^2)$ 复杂度
- Newton法求解Ricci流：$O(n^3)$ 每次迭代
- Thurston算法：线性收敛率
- 谱方法计算共形不变量：$O(n^2)$

**数值稳定性要点**：
- 三角形退化检测与处理
- 能量正则化避免奇异性
- 步长自适应控制收敛
- 条件数监控与预处理

**3D打印应用**：
- 曲面参数化实现纹理映射
- 共形变形保持局部细节
- 度量优化减少打印畸变
- 形状分析用于质量控制

## 练习题

### 基础题

**习题3.1** 证明Möbius变换保持交比不变，即对于四个不同的复数 $z_1, z_2, z_3, z_4$ 和Möbius变换 $f(z) = \frac{az+b}{cz+d}$，有：
$$[f(z_1), f(z_2), f(z_3), f(z_4)] = [z_1, z_2, z_3, z_4]$$

其中交比定义为 $[z_1, z_2, z_3, z_4] = \frac{(z_1-z_3)(z_2-z_4)}{(z_1-z_4)(z_2-z_3)}$。

<details>
<summary>提示</summary>
利用Möbius变换的分式线性性质，直接计算并化简。
</details>

<details>
<summary>答案</summary>

设 $f(z) = \frac{az+b}{cz+d}$，其中 $ad-bc \neq 0$。

计算 $f(z_i) - f(z_j)$：
$$f(z_i) - f(z_j) = \frac{az_i+b}{cz_i+d} - \frac{az_j+b}{cz_j+d} = \frac{(ad-bc)(z_i-z_j)}{(cz_i+d)(cz_j+d)}$$

因此：
$$\frac{f(z_1)-f(z_3)}{f(z_1)-f(z_4)} = \frac{z_1-z_3}{z_1-z_4} \cdot \frac{(cz_1+d)(cz_4+d)}{(cz_1+d)(cz_3+d)} = \frac{z_1-z_3}{z_1-z_4} \cdot \frac{cz_4+d}{cz_3+d}$$

类似地：
$$\frac{f(z_2)-f(z_4)}{f(z_2)-f(z_3)} = \frac{z_2-z_4}{z_2-z_3} \cdot \frac{cz_3+d}{cz_4+d}$$

两式相乘，$(cz_3+d)$ 和 $(cz_4+d)$ 项相消，得到原交比。
</details>

**习题3.2** 给定三角形网格的边长 $\{l_{12}, l_{23}, l_{31}\}$，推导顶点 $v_1$ 处的角度 $\theta_1$ 关于边长的偏导数。

<details>
<summary>提示</summary>
使用余弦定理和隐函数定理。
</details>

<details>
<summary>答案</summary>

由余弦定理：
$$\cos\theta_1 = \frac{l_{12}^2 + l_{31}^2 - l_{23}^2}{2l_{12}l_{31}}$$

对 $l_{12}$ 求偏导：
$$\frac{\partial \theta_1}{\partial l_{12}} = -\frac{1}{\sin\theta_1} \cdot \frac{\partial \cos\theta_1}{\partial l_{12}}$$

$$= -\frac{1}{\sin\theta_1} \cdot \frac{2l_{12} \cdot 2l_{31} - (l_{12}^2 + l_{31}^2 - l_{23}^2) \cdot 2l_{31}}{4l_{12}^2l_{31}^2}$$

$$= \frac{1}{\sin\theta_1} \cdot \frac{l_{31}^2 - l_{12}^2 + l_{23}^2}{2l_{12}^2l_{31}}$$

$$= \frac{\cos\theta_3}{l_{12}\sin\theta_1}$$

其中使用了 $l_{31}^2 - l_{12}^2 + l_{23}^2 = 2l_{12}l_{23}\cos\theta_3$。

类似可得：
$$\frac{\partial \theta_1}{\partial l_{23}} = -\frac{l_{23}}{2A}, \quad \frac{\partial \theta_1}{\partial l_{31}} = \frac{\cos\theta_2}{l_{31}\sin\theta_1}$$

其中 $A$ 是三角形面积。
</details>

**习题3.3** 计算单位圆盘到上半平面的共形映射，并验证其保角性。

<details>
<summary>提示</summary>
使用Cayley变换。
</details>

<details>
<summary>答案</summary>

Cayley变换：
$$w = i\frac{1+z}{1-z}$$

将单位圆盘 $|z| < 1$ 映射到上半平面 $\text{Im}(w) > 0$。

验证：当 $|z| = 1$ 时，设 $z = e^{i\theta}$：
$$w = i\frac{1+e^{i\theta}}{1-e^{i\theta}} = i\frac{e^{-i\theta/2}(e^{i\theta/2} + e^{-i\theta/2})}{e^{-i\theta/2}(e^{i\theta/2} - e^{-i\theta/2})} = i\frac{2\cos(\theta/2)}{2i\sin(\theta/2)} = \cot(\theta/2)$$

这是实数，即边界映射到实轴。

保角性：计算导数
$$\frac{dw}{dz} = i\frac{(1-z) - (1+z)(-1)}{(1-z)^2} = \frac{2i}{(1-z)^2} \neq 0$$

对于 $|z| < 1$，导数非零，故映射共形。
</details>

**习题3.4** 设三角网格的离散高斯曲率为 $K_i$，证明 Gauss-Bonnet 定理的离散版本：
$$\sum_{i \in V} K_i = 2\pi\chi(M)$$

<details>
<summary>提示</summary>
利用角度和公式与Euler特征数。
</details>

<details>
<summary>答案</summary>

对于闭曲面，离散高斯曲率：
$$K_i = 2\pi - \sum_{jk \in N(i)} \theta_{jk}^i$$

求和：
$$\sum_{i \in V} K_i = 2\pi|V| - \sum_{i \in V} \sum_{jk \in N(i)} \theta_{jk}^i$$

注意每个三角形的三个角恰好被计算一次：
$$\sum_{i \in V} \sum_{jk \in N(i)} \theta_{jk}^i = \sum_{f \in F} \pi = \pi|F|$$

因此：
$$\sum_{i \in V} K_i = 2\pi|V| - \pi|F|$$

由Euler公式 $|V| - |E| + |F| = \chi(M)$ 和每个三角形有3条边（每条边被两个三角形共享）得 $3|F| = 2|E|$：

$$\sum_{i \in V} K_i = 2\pi|V| - \pi|F| = 2\pi(|V| - |F|/2) = 2\pi(|V| - |E| + |F|) = 2\pi\chi(M)$$
</details>

### 挑战题

**习题3.5** 考虑离散Ricci流的收敛性。设初始度量为 $g^0$，目标曲率为 $\bar{K}$，证明如果 $\bar{K}$ 满足Gauss-Bonnet条件和正性条件，则离散Ricci流收敛到唯一解。

<details>
<summary>提示</summary>
构造Lyapunov函数并证明其单调性。
</details>

<details>
<summary>答案</summary>

定义Ricci势能：
$$\mathcal{E}(u) = \sum_{i,j} \int_0^{u_i+u_j} \theta_{ij}(s) ds - \sum_i \bar{K}_i u_i$$

其梯度：
$$\nabla_i \mathcal{E} = K_i(u) - \bar{K}_i$$

Hessian矩阵：
$$H_{ij} = \frac{\partial K_i}{\partial u_j}$$

由角度和约束，$H$ 是对称半正定的，且核空间为常数向量。

在约束 $\sum_i u_i = 0$ 下，$H$ 正定。因此 $\mathcal{E}$ 是严格凸的。

Ricci流等价于梯度流：
$$\frac{du}{dt} = -\nabla \mathcal{E}(u)$$

沿流的能量变化：
$$\frac{d\mathcal{E}}{dt} = \langle \nabla \mathcal{E}, \frac{du}{dt} \rangle = -|\nabla \mathcal{E}|^2 \leq 0$$

能量单调递减且有下界（凸函数），故收敛到临界点。

唯一性由严格凸性保证。
</details>

**习题3.6** 设计一个算法，将亏格 $g > 0$ 的曲面共形映射到标准的 Riemann 面。分析算法复杂度和数值稳定性。

<details>
<summary>提示</summary>
使用全纯微分构造周期矩阵。
</details>

<details>
<summary>答案</summary>

算法步骤：

1. **计算同调基**：
   - 使用树-余树分解找到 $2g$ 个生成元
   - 复杂度：$O(|E| + |V|\log|V|)$

2. **构造全纯微分基**：
   - 求解 $g$ 个线性独立的调和1-形式
   - 离散Hodge分解：$\Delta \omega = 0$
   - 复杂度：$O(g \cdot |V|^3)$（稀疏线性系统）

3. **计算周期矩阵**：
   $$\Pi_{ij} = \oint_{A_j} \omega_i$$
   - 沿同调环积分
   - 复杂度：$O(g^2 \cdot |E|)$

4. **归一化到标准形式**：
   - Siegel上半空间归一化
   - 使用 $Sp(2g, \mathbb{Z})$ 变换
   - 复杂度：$O(g^3)$

5. **构造Abel-Jacobi映射**：
   $$\mathcal{A}(p) = \left(\sum_{e \in \gamma_{p_0}^p} \omega_1(e), ..., \sum_{e \in \gamma_{p_0}^p} \omega_g(e)\right) \mod \Lambda$$

数值稳定性分析：
- 条件数：$\kappa(H) = O(h^{-2})$，其中 $h$ 是网格尺寸
- 使用多重网格预处理：$\kappa_{MG} = O(1)$
- 周期积分使用高精度求积公式
- 模运算需要处理浮点误差累积

总复杂度：$O(g \cdot |V|^3)$，瓶颈在求解调和形式。
</details>

**习题3.7** 推导三维情况下的离散共形几何。考虑四面体网格，定义合适的共形结构和Ricci流。

<details>
<summary>提示</summary>
使用双曲几何和理想四面体。
</details>

<details>
<summary>答案</summary>

三维离散共形几何基于双曲理想四面体：

1. **共形结构**：每条边赋予复数参数 $z_{ij} \in \mathbb{C} \setminus \{0, 1\}$

2. **相容性条件**（边方程）：
   围绕边 $e$ 的四面体满足：
   $$\prod_{t \in \text{star}(e)} z_t^e = 1$$

3. **完备性条件**（顶点方程）：
   $$\sum_{t \in \text{star}(v)} \arg(z_t^v) = 2\pi$$

4. **体积公式**（Lobachevsky函数）：
   $$V(z) = \text{Im}\left(\text{Li}_2(z) + \log|z| \cdot \log\frac{1-z}{1-\bar{z}}\right)$$

5. **离散Ricci流**：
   $$\frac{dz_{ij}}{dt} = -\frac{\partial \mathcal{V}}{\partial \kappa_{ij}} \cdot \frac{\partial \kappa_{ij}}{\partial z_{ij}}$$
   其中 $\mathcal{V} = \sum_t V_t$ 是总体积，$\kappa_{ij}$ 是边 $(i,j)$ 的固角亏量。

6. **变分原理**：
   最大化体积泛函
   $$\mathcal{V}(z) = \sum_{t} V_t(z_t)$$
   受约束于相容性和完备性条件。

7. **数值方法**：
   - Newton-Raphson求解非线性方程组
   - Jacobian矩阵稀疏，使用迭代求解器
   - 复杂度：$O(|T|^2)$ 每次迭代

应用：三维网格参数化、体积优化、共形不变量计算。
</details>

**习题3.8** 探讨机器学习方法学习共形映射。设计一个神经网络架构，输入曲面网格，输出共形参数化。讨论如何保证共形性约束。

<details>
<summary>提示</summary>
使用图神经网络和物理信息损失函数。
</details>

<details>
<summary>答案</summary>

神经网络架构设计：

1. **编码器**（Graph Attention Network）：
   ```
   输入：顶点特征 X ∈ R^{|V|×3}，边连接 E
   层结构：
   - GAT层：h_i^{(l+1)} = σ(∑_{j∈N(i)} α_{ij} W h_j^{(l)})
   - 注意力：α_{ij} = softmax(LeakyReLU(a^T[Wh_i || Wh_j]))
   输出：潜在表示 Z ∈ R^{|V|×d}
   ```

2. **共形预测器**：
   ```
   输入：潜在表示 Z
   输出：共形因子 u ∈ R^{|V|}
   约束：∑_i u_i = 0（规范化）
   ```

3. **损失函数设计**：

   a) **共形能量损失**：
   $$L_{conf} = \sum_{(i,j,k)} \left|\frac{\sin\theta_{jk}^i}{\sin\hat{\theta}_{jk}^i} - \frac{l_{jk}}{\hat{l}_{jk}}\right|^2$$
   
   b) **曲率匹配损失**：
   $$L_{curv} = \sum_i (K_i(u) - \bar{K}_i)^2$$
   
   c) **正则化损失**：
   $$L_{reg} = \lambda_1 \|u\|^2 + \lambda_2 \sum_{ij} (\nabla u)_{ij}^2$$
   
   d) **物理约束损失**（三角形非退化）：
   $$L_{phys} = \sum_{ijk} \max(0, \epsilon - \text{Area}_{ijk})$$

4. **训练策略**：
   - 数据增强：随机旋转、缩放、网格细分
   - 课程学习：从简单曲面到复杂拓扑
   - 多尺度训练：不同分辨率网格

5. **保证共形性的技术**：
   
   a) **硬约束投影**：
   ```python
   u = u - mean(u)  # 中心化
   u = project_to_feasible(u)  # 投影到可行域
   ```
   
   b) **拉格朗日乘子法**：
   $$L_{total} = L_{conf} + \mu \cdot \text{constraints}$$
   
   c) **残差连接到经典算法**：
   $$u_{final} = u_{classical} + \alpha \cdot u_{neural}$$

6. **评估指标**：
   - 角度畸变：$\max_{ijk} |\theta_{ijk} - \hat{\theta}_{ijk}|$
   - 面积畸变：$\max_{ijk} |A_{ijk}/\hat{A}_{ijk} - 1|$
   - 共形因子平滑度：$\|\nabla u\|_2$

优势：
- 快速推理：$O(|V|)$ vs 传统迭代 $O(|V|^3)$
- 泛化能力：学习曲面先验
- 鲁棒性：对噪声和不规则网格

局限：
- 理论保证较弱
- 需要大量训练数据
- 对新拓扑泛化困难
</details>

## 常见陷阱与错误

### 数值计算陷阱

1. **三角形退化**
   - 错误：忽略接近退化的三角形
   - 正确：添加最小角度约束 $\theta_{min} > \epsilon$
   - 实践：使用自适应步长和回溯线搜索

2. **浮点精度损失**
   - 错误：直接计算 $\cos\theta = \frac{a \cdot b}{|a||b|}$
   - 正确：使用稳定的公式 $\sin(\theta/2) = \frac{|a \times b|}{2|a||b|}$
   - 实践：避免极小角度的余弦值

3. **矩阵条件数**
   - 错误：直接求解病态线性系统
   - 正确：使用预处理共轭梯度法
   - 实践：监控条件数，必要时正则化

### 算法实现陷阱

4. **收敛判定**
   - 错误：仅检查残差范数
   - 正确：同时检查相对变化和KKT条件
   - 实践：设置多重收敛准则

5. **边界处理**
   - 错误：内部顶点公式直接用于边界
   - 正确：边界顶点使用修正的曲率定义
   - 实践：分别处理内部和边界情况

6. **拓扑一致性**
   - 错误：参数化后不检查折叠
   - 正确：验证雅可比行列式符号
   - 实践：使用双射性约束

### 理论理解误区

7. **共形 vs 等距**
   - 误解：共形映射保持距离
   - 事实：仅保持角度，不保持长度
   - 区别：等距 ⊂ 共形 ⊂ 微分同胚

8. **离散 vs 连续**
   - 误解：离散算法总是收敛到连续解
   - 事实：需要适当的离散化方案
   - 关键：保持关键的几何/拓扑性质

## 最佳实践检查清单

### 算法选择
- [ ] 曲面拓扑是否适合所选方法？
- [ ] 网格质量是否满足算法要求？
- [ ] 计算资源是否充足？
- [ ] 精度要求是否明确？

### 预处理
- [ ] 网格是否流形？
- [ ] 是否存在退化三角形？
- [ ] 边界是否正确标记？
- [ ] 网格是否需要重新采样？

### 数值计算
- [ ] 是否选择了合适的线性求解器？
- [ ] 是否设置了合理的收敛准则？
- [ ] 是否添加了必要的正则化？
- [ ] 是否处理了数值奇异性？

### 验证与调试
- [ ] 是否验证了Gauss-Bonnet定理？
- [ ] 是否检查了能量单调性？
- [ ] 是否测试了简单案例？
- [ ] 是否与解析解对比（如适用）？

### 性能优化
- [ ] 是否利用了稀疏性？
- [ ] 是否使用了合适的数据结构？
- [ ] 是否可以并行化？
- [ ] 是否缓存了重复计算？

### 鲁棒性
- [ ] 是否处理了所有边界情况？
- [ ] 是否对输入添加了合理性检查？
- [ ] 是否有错误恢复机制？
- [ ] 是否记录了详细的诊断信息？