# 第5章：网格处理算法

网格处理是3D打印工作流中的核心环节，直接影响打印质量、效率和可制造性。本章深入探讨网格简化、重新网格化、修复、细分和变形等关键算法的数学原理。我们将从误差度量、能量优化、微分几何等角度分析这些算法，强调数值稳定性和计算效率，并通过大量练习帮助读者掌握算法设计和分析技巧。

## 5.1 网格简化：QEM、渐进网格

网格简化旨在减少网格复杂度同时保持几何特征。这在3D打印中至关重要：降低切片计算量、减少文件大小、适应打印精度限制。

### 5.1.1 二次误差度量（QEM）

#### 基本原理

对于顶点 $\mathbf{v}$，定义其到相关平面集合的二次误差：

$$E(\mathbf{v}) = \sum_{p \in \text{planes}(\mathbf{v})} (\mathbf{n}_p^T \mathbf{v} + d_p)^2$$

其中平面 $p$ 由法向量 $\mathbf{n}_p$ 和距离 $d_p$ 定义。

这个误差度量的几何意义是顶点到其相关平面的距离平方和。对于网格顶点，相关平面通常是其相邻三角形所在的平面。平面方程 $\mathbf{n}^T\mathbf{x} + d = 0$ 中，$\mathbf{n}$ 是单位法向量，$d$ 是原点到平面的有向距离。

#### 矩阵形式

将误差改写为二次形式：

$$E(\mathbf{v}) = \mathbf{v}^T \mathbf{Q} \mathbf{v} + 2\mathbf{b}^T \mathbf{v} + c$$

其中：
- $\mathbf{Q} = \sum_p \mathbf{n}_p \mathbf{n}_p^T$ （$3 \times 3$ 矩阵）
- $\mathbf{b} = \sum_p d_p \mathbf{n}_p$ （$3 \times 1$ 向量）
- $c = \sum_p d_p^2$ （标量）

更紧凑地，可以使用齐次坐标表示：

$$E(\tilde{\mathbf{v}}) = \tilde{\mathbf{v}}^T \mathbf{K}_p \tilde{\mathbf{v}}$$

其中 $\tilde{\mathbf{v}} = [\mathbf{v}^T, 1]^T$，基础二次矩阵：

$$\mathbf{K}_p = \begin{pmatrix}
\mathbf{n}\mathbf{n}^T & d\mathbf{n} \\
d\mathbf{n}^T & d^2
\end{pmatrix} = \begin{pmatrix}
n_x^2 & n_xn_y & n_xn_z & dn_x \\
n_xn_y & n_y^2 & n_yn_z & dn_y \\
n_xn_z & n_yn_z & n_z^2 & dn_z \\
dn_x & dn_y & dn_z & d^2
\end{pmatrix}$$

顶点的总误差矩阵是所有相关平面误差矩阵的和：$\mathbf{K} = \sum_p \mathbf{K}_p$

#### 边收缩操作

对于边 $(v_1, v_2) \to \bar{v}$，新顶点的误差矩阵为：

$$\mathbf{Q}_{\bar{v}} = \mathbf{Q}_{v_1} + \mathbf{Q}_{v_2}$$

最优位置通过求解获得：

$$\bar{\mathbf{v}} = -\mathbf{Q}^{-1} \mathbf{b}$$

当 $\mathbf{Q}$ 奇异时，使用伪逆或选择端点/中点。

#### 最优位置的推导

最小化误差 $E(\mathbf{v})$ 需要满足：

$$\nabla E = 2\mathbf{Q}\mathbf{v} + 2\mathbf{b} = \mathbf{0}$$

因此最优位置为：

$$\mathbf{v}^* = -\mathbf{Q}^{-1}\mathbf{b}$$

在齐次坐标下，最优位置满足：

$$\begin{pmatrix}
\mathbf{Q} & \mathbf{b} \\
\mathbf{b}^T & c
\end{pmatrix} \begin{pmatrix}
\mathbf{v}^* \\
1
\end{pmatrix} = \begin{pmatrix}
\mathbf{0} \\
1
\end{pmatrix}$$

当 $\mathbf{Q}$ 奇异（秩 < 3）时，解不唯一，表示最优位置在一条线或平面上。实践中的处理策略：

1. **线性搜索**：在边 $(v_1, v_2)$ 上搜索最小误差点
2. **中点选择**：简单选择 $(v_1 + v_2)/2$
3. **端点选择**：选择误差较小的端点
4. **伪逆求解**：使用Moore-Penrose伪逆

#### 特征保持

通过加权保持尖锐特征：

$$\mathbf{Q}_{\text{weighted}} = \mathbf{Q}_{\text{basic}} + w_{\text{feature}} \cdot \mathbf{Q}_{\text{feature}}$$

其中特征矩阵基于二面角：

$$w_{\text{feature}} = \begin{cases}
\lambda \cdot (1 - \cos\theta) & \text{if } \theta > \theta_{\text{sharp}} \\
0 & \text{otherwise}
\end{cases}$$

#### 边界和约束处理

对于边界边，构造虚拟约束平面：

$$\mathbf{K}_{\text{boundary}} = w_{\text{boundary}} \cdot \begin{pmatrix}
\mathbf{t}\mathbf{t}^T & \mathbf{0} \\
\mathbf{0}^T & 0
\end{pmatrix}$$

其中 $\mathbf{t}$ 是边界边的切向量，$w_{\text{boundary}}$ 是边界保持权重（通常设为 $10^3$ 到 $10^6$）。

对于特征边（如折痕），添加垂直于特征的约束平面：

$$\mathbf{K}_{\text{crease}} = w_{\text{crease}} \cdot \sum_{i} \mathbf{K}_{\perp,i}$$

其中 $\mathbf{K}_{\perp,i}$ 是垂直于特征边的约束平面。

### 5.1.2 渐进网格（Progressive Meshes）

#### 边收缩序列

渐进网格通过记录边收缩序列实现多分辨率表示：

$$M^n \xrightarrow{\text{ecol}_{n-1}} M^{n-1} \xrightarrow{\text{ecol}_{n-2}} \cdots \xrightarrow{\text{ecol}_0} M^0$$

逆操作（顶点分裂）：

$$M^0 \xrightarrow{\text{vsplit}_0} M^1 \xrightarrow{\text{vsplit}_1} \cdots \xrightarrow{\text{vsplit}_{n-1}} M^n$$

#### 几何形态插值

参数化边收缩 $\text{ecol}(v_1, v_2, t)$：

$$\mathbf{v}(t) = (1-t)\mathbf{v}_1 + t\mathbf{v}_2$$

支持连续LOD（细节层次）：

$$M(t) = M^{\lfloor t \rfloor} + \text{geomorph}(t - \lfloor t \rfloor)$$

#### 视相关简化

基于视点的误差度量：

$$E_{\text{view}}(\mathbf{v}) = E_{\text{geometric}}(\mathbf{v}) \cdot \psi(\mathbf{v}, \mathbf{e})$$

其中 $\psi$ 是视相关权重函数：

$$\psi(\mathbf{v}, \mathbf{e}) = \frac{\max(0, \mathbf{n} \cdot (\mathbf{e} - \mathbf{v}))}{\|\mathbf{e} - \mathbf{v}\|^2}$$

### 5.1.3 拓扑约束

#### 连通性保持

防止拓扑改变的链接条件（Link Condition）：

$$\text{link}(v_1) \cap \text{link}(v_2) = \text{link}(\text{edge}(v_1, v_2))$$

其中：
- $\text{link}(v)$ = $\{w : (v,w) \in E\}$ 是顶点 $v$ 的链接
- $\text{link}(e)$ = $\{w : (v_1,w) \in F \wedge (v_2,w) \in F\}$ 是边 $e=(v_1,v_2)$ 的链接

链接条件保证边收缩不改变网格的拓扑类型（亏格、边界数、连通分量数）。

#### 流形性检查

确保简化后保持2-流形性质：
- 每条边最多被两个三角形共享
- 顶点的1-邻域同胚于圆盘或半圆盘

具体检查包括：

1. **边流形性**：边 $(v_1, v_2)$ 收缩后，检查新顶点 $\bar{v}$ 的每条边是否满足流形条件
2. **顶点流形性**：$\bar{v}$ 的1-邻域应形成单个连通的扇形或圆盘
3. **面翻转检测**：确保收缩不会导致三角形法向翻转

$$\text{valid} = \bigwedge_{f \in N(\bar{v})} (\mathbf{n}_f^{\text{before}} \cdot \mathbf{n}_f^{\text{after}} > \epsilon)$$

### 5.1.4 数值稳定性

#### 条件数分析

矩阵 $\mathbf{Q}$ 的条件数：

$$\kappa(\mathbf{Q}) = \frac{\lambda_{\max}}{\lambda_{\min}}$$

当 $\kappa > 10^6$ 时使用正则化：

$$\mathbf{Q}_{\text{reg}} = \mathbf{Q} + \epsilon \mathbf{I}$$

其中 $\epsilon$ 的选择策略：
- 绝对正则化：$\epsilon = 10^{-6}$
- 相对正则化：$\epsilon = 10^{-6} \cdot \text{trace}(\mathbf{Q})/3$
- 自适应正则化：$\epsilon = \max(10^{-8}, 10^{-3} \cdot \lambda_{\min})$

#### 浮点精度考虑

使用Kahan求和减少累积误差：

```
procedure KahanSum(values[])
  sum = 0.0
  c = 0.0  // 补偿项
  for value in values:
    y = value - c
    t = sum + y
    c = (t - sum) - y
    sum = t
  return sum
```

对于QEM矩阵累加，使用双精度（double）而非单精度（float），并在关键计算中使用扩展精度。

#### 优先队列更新策略

使用懒惰删除避免频繁的堆重构：

1. **时间戳方法**：每个边存储时间戳，过期的边在出队时忽略
2. **标记方法**：标记已删除的边，出队时跳过
3. **延迟更新**：批量更新受影响的边

## 5.2 重新网格化：各向同性、各向异性

重新网格化改善网格质量，对3D打印的结构分析和切片效率至关重要。

### 5.2.1 各向同性重新网格化

#### Delaunay三角化

最大化最小角准则：

$$\max \min_{T \in \mathcal{T}} \min_{i=1,2,3} \angle_i(T)$$

空圆性质：三角形外接圆内不包含其他顶点。

**Delaunay三角化的数学性质**：

1. **最大化最小角**：在所有可能的三角化中，Delaunay三角化最大化最小角
2. **对偶性**：Delaunay三角化是Voronoi图的对偶
3. **唯一性**：当没有四点共圆时，Delaunay三角化唯一
4. **凸包性质**：Delaunay三角化的边界是点集的凸包

**增量插入算法**（Bowyer-Watson）：

1. 初始化超级三角形包含所有点
2. 对每个点 $p$：
   - 找到包含 $p$ 的三角形
   - 删除所有外接圆包含 $p$ 的三角形（坏三角形）
   - 重新三角化形成的空腔
3. 删除与超级三角形相关的三角形

时间复杂度：$O(n \log n)$ 期望，$O(n^2)$ 最坏情况

#### 质心Voronoi图（CVT）

Lloyd迭代优化能量函数：

$$E_{\text{CVT}} = \sum_{i=1}^n \int_{\Omega_i} \|\mathbf{x} - \mathbf{c}_i\|^2 d\mathbf{x}$$

其中 $\Omega_i$ 是Voronoi区域，$\mathbf{c}_i$ 是质心：

$$\mathbf{c}_i = \frac{\int_{\Omega_i} \mathbf{x} d\mathbf{x}}{\int_{\Omega_i} d\mathbf{x}}$$

**Lloyd算法**：

1. 初始化种子点 $\{\mathbf{p}_i\}$
2. 重复直到收敛：
   - 计算Voronoi图 $\{\Omega_i\}$
   - 更新种子点为质心：$\mathbf{p}_i \leftarrow \mathbf{c}_i$

**收敛性分析**：

- Lloyd算法单调降低CVT能量
- 收敛到局部最小值（不保证全局最优）
- 收敛速度：线性收敛，收敛率依赖于初始配置

**加速技术**：

1. **准牛顿方法**（L-BFGS）：
   $$\mathbf{p}^{(k+1)} = \mathbf{p}^{(k)} - \alpha \mathbf{H}^{-1} \nabla E$$
   其中 $\mathbf{H}$ 是近似Hessian矩阵

2. **多重网格方法**：
   - 在粗网格上求解
   - 插值到细网格
   - 局部优化

3. **GPU并行化**：
   - 并行计算Voronoi区域
   - 并行质心计算

#### 边翻转优化

通过边翻转最大化质量度量：

$$Q(T) = \frac{A}{\sum_{i=1}^3 l_i^2}$$

其中 $A$ 是三角形面积，$l_i$ 是边长。

**常用质量度量**：

1. **半径比（Radius Ratio）**：
   $$Q_{RR} = \frac{r_{in}}{r_{circ}}$$
   其中 $r_{in}$ 是内切圆半径，$r_{circ}$ 是外接圆半径

2. **角度偏差（Angle Deviation）**：
   $$Q_{AD} = \max_i |\angle_i - 60°|$$

3. **边长比（Edge Ratio）**：
   $$Q_{ER} = \frac{\max_i l_i}{\min_i l_i}$$

**边翻转准则**：

对于共享边 $e$ 的三角形对 $(T_1, T_2)$，翻转为 $(T_1', T_2')$ 当且仅当：

$$Q(T_1') + Q(T_2') > Q(T_1) + Q(T_2) + \epsilon$$

**局部Delaunay准则**：

边 $e$ 满足局部Delaunay当且仅当：

$$\angle_{\text{opposite}}^1 + \angle_{\text{opposite}}^2 \leq \pi$$

不满足时进行翻转。

### 5.2.2 各向异性重新网格化

#### 度量张量场

定义黎曼度量 $\mathbf{M}(\mathbf{x})$：

$$\mathbf{M} = \mathbf{R}^T \begin{pmatrix} h_1^{-2} & 0 \\ 0 & h_2^{-2} \end{pmatrix} \mathbf{R}$$

其中 $\mathbf{R}$ 是主方向旋转矩阵，$h_1, h_2$ 是期望尺寸。

**度量张量的几何意义**：

在度量 $\mathbf{M}$ 下，单位圆变换为椭圆：

$$\mathbf{x}^T \mathbf{M} \mathbf{x} = 1$$

椭圆的主轴方向是 $\mathbf{R}$ 的列向量，主轴长度是 $h_1, h_2$。

**度量插值**：

给定两个度量 $\mathbf{M}_1, \mathbf{M}_2$，线性插值：

$$\mathbf{M}(t) = (1-t)\mathbf{M}_1 + t\mathbf{M}_2$$

对数插值（保持正定性）：

$$\mathbf{M}(t) = \exp((1-t)\log\mathbf{M}_1 + t\log\mathbf{M}_2)$$

其中矩阵对数通过特征分解计算：

$$\log\mathbf{M} = \mathbf{V}\log\mathbf{\Lambda}\mathbf{V}^T$$

#### 曲率自适应

基于主曲率构造度量：

$$h_i = \min\left(\frac{\epsilon}{|\kappa_i|}, h_{\max}\right)$$

其中 $\epsilon$ 是近似误差容限。

**Hessian度量**：

基于曲面的二阶导数构造度量：

$$\mathbf{H} = \begin{pmatrix}
\frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\
\frac{\partial^2 f}{\partial x \partial y} & \frac{\partial^2 f}{\partial y^2}
\end{pmatrix}$$

度量张量：

$$\mathbf{M} = \frac{1}{\epsilon^{2/3}} |\mathbf{H}|^{2/3}$$

其中 $|\mathbf{H}| = \mathbf{V}|\mathbf{\Lambda}|\mathbf{V}^T$，$|\lambda_i|$ 取绝对值。

**各向异性误差估计**：

线性插值误差的上界：

$$\|f - \Pi_h f\|_{L^\infty} \leq C \max_{T} \max_{\mathbf{e} \in T} \mathbf{e}^T \mathbf{H} \mathbf{e} \cdot \|\mathbf{e}\|^2$$

其中 $\Pi_h$ 是分片线性插值算子。

#### 各向异性Delaunay

在度量空间中的Delaunay条件：

$$\text{incircle}_{\mathbf{M}}(p, q, r, s) = \det\begin{pmatrix}
p_x - s_x & p_y - s_y & \|\mathbf{p} - \mathbf{s}\|_{\mathbf{M}}^2 \\
q_x - s_x & q_y - s_y & \|\mathbf{q} - \mathbf{s}\|_{\mathbf{M}}^2 \\
r_x - s_x & r_y - s_y & \|\mathbf{r} - \mathbf{s}\|_{\mathbf{M}}^2
\end{pmatrix} > 0$$

### 5.2.3 特征对齐

#### 主曲率方向场

计算形状算子 $\mathbf{S}$：

$$\mathbf{S} = -\mathbf{dN} \cdot \mathbf{dX}^{-1}$$

特征值分解得主曲率和主方向。

**离散形状算子**：

对于顶点 $v$，使用最小二乘拟合：

$$\min_{\mathbf{S}} \sum_{i \in N(v)} w_i \|\mathbf{S}(\mathbf{x}_i - \mathbf{x}_v) - (\mathbf{n}_i - \mathbf{n}_v)\|^2$$

权重选择：
- 均匀权重：$w_i = 1$
- 余切权重：$w_i = \cot\alpha_i + \cot\beta_i$
- 面积权重：$w_i = A_i/3$

解为：

$$\mathbf{S} = \left(\sum_i w_i (\mathbf{x}_i - \mathbf{x}_v)(\mathbf{n}_i - \mathbf{n}_v)^T\right) \left(\sum_i w_i (\mathbf{x}_i - \mathbf{x}_v)(\mathbf{x}_i - \mathbf{x}_v)^T\right)^{-1}$$

**主方向的奇点**：

脐点（umbilical point）：$\kappa_1 = \kappa_2$，主方向不确定

奇点指数：

$$\text{index} = \frac{1}{2\pi} \oint_{\gamma} d\theta$$

其中 $\theta$ 是主方向场的角度。

#### 交叉场引导

4-RoSy场（4-旋转对称场）：

$$\mathbf{u}(\mathbf{x}) = e^{i\theta(\mathbf{x})}$$

满足边界对齐约束：

$$\mathbf{u}|_{\partial\Omega} \parallel \mathbf{t}|_{\partial\Omega}$$

### 5.2.4 全局参数化

#### 整数网格映射

寻找参数化 $f: M \to \mathbb{R}^2$，使得：

$$\nabla f \approx \mathbf{u}$$

通过Poisson方程求解：

$$\Delta f = \nabla \cdot \mathbf{u}$$

#### 奇点放置

使用Index理论确定奇点：

$$\sum_{v \in V} \text{index}(v) = \chi(M)$$

其中 $\chi(M)$ 是欧拉特征数。

## 5.3 网格修复：孔洞填充、自相交检测

3D扫描和建模常产生缺陷网格，修复算法是打印前处理的必要步骤。

### 5.3.1 孔洞检测与分类

#### 边界环提取

识别边界边（只属于一个三角形）：

$$\partial M = \{e \in E : |\text{faces}(e)| = 1\}$$

通过深度优先搜索提取连通边界环。

**边界环算法**：

```
procedure ExtractBoundaryLoops(mesh)
  boundary_edges = 找到所有边界边
  visited = 初始化为false
  loops = []
  
  for e in boundary_edges:
    if not visited[e]:
      loop = []
      current = e
      while not visited[current]:
        visited[current] = true
        loop.append(current.vertex)
        current = 找到下一条未访问的边界边
      loops.append(loop)
  
  return loops
```

**半边结构优化**：

使用半边结构可以 $O(1)$ 时间找到下一条边界边：

```
next_boundary = current.twin.next
while next_boundary.twin != null:
  next_boundary = next_boundary.twin.next
```

#### 孔洞特征分析

计算边界环的几何特征：
- 周长：$L = \sum_{e \in \partial} \|e\|$
- 面积（投影）：$A = \frac{1}{2}|\sum_{i} (\mathbf{x}_i \times \mathbf{x}_{i+1})|$
- 平面性：$\sigma = \frac{\lambda_3}{\lambda_1 + \lambda_2 + \lambda_3}$

其中 $\lambda_i$ 是边界点协方差矩阵的特征值。

### 5.3.2 孔洞填充算法

#### 最小面积三角化

动态规划求解最小面积：

$$A(i, j) = \min_{i < k < j} \{A(i, k) + A(k, j) + A_{\triangle}(v_i, v_k, v_j)\}$$

时间复杂度 $O(n^3)$。

**动态规划详细推导**：

定义 $A(i, j)$ 为顶点 $v_i$ 到 $v_j$ 的子多边形的最小三角化面积。

边界条件：
- $A(i, i+1) = 0$ （相邻顶点无需三角化）
- $A(i, i+2) = A_{\triangle}(v_i, v_{i+1}, v_{i+2})$

递推关系：
对于 $j - i > 2$，枚举中间顶点 $k$，形成三角形 $(v_i, v_k, v_j)$。

**三角形面积计算**：

使用叉积：

$$A_{\triangle} = \frac{1}{2}\|(\mathbf{v}_k - \mathbf{v}_i) \times (\mathbf{v}_j - \mathbf{v}_i)\|$$

或使用Heron公式：

$$A = \sqrt{s(s-a)(s-b)(s-c)}$$

其中 $s = (a+b+c)/2$ 是半周长。

#### 推进波前法（Advancing Front）

迭代添加三角形，选择最优候选：

$$\text{score}(v_i, v_j, v_k) = \alpha \cdot \text{angle} + \beta \cdot \text{area} + \gamma \cdot \text{normal}$$

角度项倾向等边三角形：

$$\text{angle} = \sum_{i=1}^3 |\angle_i - 60°|$$

#### Poisson曲面重建

求解Poisson方程：

$$\Delta \chi = \nabla \cdot \vec{V}$$

其中 $\vec{V}$ 是从采样点法向构造的向量场，$\chi$ 是指示函数。

使用八叉树自适应离散化：

$$\chi(\mathbf{x}) = \sum_{o \in \text{octree}} c_o B_o(\mathbf{x})$$

其中 $B_o$ 是基函数。

**向量场构造**：

从有向点云 $\{(\mathbf{p}_i, \mathbf{n}_i)\}$ 构造向量场：

$$\vec{V}(\mathbf{x}) = \sum_i W_i(\mathbf{x}) \mathbf{n}_i$$

其中权重函数：

$$W_i(\mathbf{x}) = \frac{1}{\|\mathbf{x} - \mathbf{p}_i\|^2} \cdot G_{\sigma}(\|\mathbf{x} - \mathbf{p}_i\|)$$

$G_{\sigma}$ 是高斯核。

**八叉树基函数**：

使用三维B样条作为基函数：

$$B_o(\mathbf{x}) = B\left(\frac{x - c_x}{w}\right) B\left(\frac{y - c_y}{w}\right) B\left(\frac{z - c_z}{w}\right)$$

其中 $(c_x, c_y, c_z)$ 是八叉树节点中心，$w$ 是节点宽度。

**离散Laplacian**：

在八叉树上的离散Laplacian：

$$[\mathbf{L}]_{ij} = \int_{\Omega} \nabla B_i \cdot \nabla B_j d\mathbf{x}$$

由于B样条的局部支撑性，矩阵是稀疏的。

#### RBF隐式曲面

使用径向基函数插值：

$$f(\mathbf{x}) = \sum_{i=1}^n \lambda_i \phi(\|\mathbf{x} - \mathbf{x}_i\|) + p(\mathbf{x})$$

常用核函数：
- 薄板样条：$\phi(r) = r^2 \log r$
- 多重二次：$\phi(r) = \sqrt{1 + (r/c)^2}$
- 高斯：$\phi(r) = e^{-(r/c)^2}$

### 5.3.3 自相交检测

#### 空间哈希加速

将三角形映射到空间网格：

$$h(T) = \{\text{hash}(i, j, k) : (i, j, k) \in \text{bbox}(T)\}$$

期望复杂度从 $O(n^2)$ 降至 $O(n)$。

**空间哈希函数**：

对于网格单元 $(i, j, k)$：

$$h(i, j, k) = (i \cdot p_1 \oplus j \cdot p_2 \oplus k \cdot p_3) \mod m$$

其中 $p_1, p_2, p_3$ 是大质数，$\oplus$ 是异或操作，$m$ 是哈希表大小。

**网格单元大小选择**：

根据平均三角形大小选择：

$$\text{cell\_size} = k \cdot \sqrt[3]{\frac{\text{volume}(\text{bbox})}{n}}$$

其中 $k \approx 2\text{-}3$ 是调节参数。

**层次包围盒（BVH）**：

递归构建BVH树：

```
procedure BuildBVH(triangles)
  if |triangles| ≤ threshold:
    return LeafNode(triangles)
  
  axis = 选择最长轴
  pivot = 中位数或SAH启发式
  left, right = 划分三角形
  
  return InternalNode(
    BuildBVH(left),
    BuildBVH(right),
    合并包围盒
  )
```

#### 精确相交测试

Möller-Trumbore算法检测三角形相交：

$$\begin{pmatrix} t \\ u \\ v \end{pmatrix} = \frac{1}{\mathbf{e}_1 \cdot (\mathbf{d} \times \mathbf{e}_2)} \begin{pmatrix} \mathbf{q} \cdot \mathbf{e}_2 \\ \mathbf{p} \cdot \mathbf{q} \\ \mathbf{d} \cdot \mathbf{p} \end{pmatrix}$$

其中：
- $\mathbf{e}_1 = \mathbf{v}_1 - \mathbf{v}_0$，$\mathbf{e}_2 = \mathbf{v}_2 - \mathbf{v}_0$
- $\mathbf{p} = \mathbf{d} \times \mathbf{e}_2$，$\mathbf{q} = \mathbf{s} \times \mathbf{e}_1$

相交条件：$t > 0$，$u \geq 0$，$v \geq 0$，$u + v \leq 1$。

**三角形-三角形相交**：

使用分离轴定理（SAT）：

测试15个潜在分离轴：
- 3个三角形A的法向
- 3个三角形B的法向
- 9个边-边叉积方向

对每个轴 $\mathbf{a}$，检查投影区间是否重叠：

$$[\min_i(\mathbf{a} \cdot \mathbf{v}_i^A), \max_i(\mathbf{a} \cdot \mathbf{v}_i^A)] \cap [\min_j(\mathbf{a} \cdot \mathbf{v}_j^B), \max_j(\mathbf{a} \cdot \mathbf{v}_j^B)] \neq \emptyset$$

**鲁棒性处理**：

使用精确谓词（Exact Predicates）：

```
function Orient3D(a, b, c, d)
  // 计算行列式的符号
  return sign(det[
    [ax-dx, ay-dy, az-dz],
    [bx-dx, by-dy, bz-dz],
    [cx-dx, cy-dy, cz-dz]
  ])
```

使用自适应精度算术或区间算术保证正确性。

#### 连续碰撞检测（CCD）

对于运动三角形，求解：

$$\min t \in [0, 1] : T_1(t) \cap T_2(t) \neq \emptyset$$

使用三次多项式根求解。

### 5.3.4 自相交消除

#### 布尔运算方法

通过自并集消除自相交：

$$M_{\text{clean}} = M \cup M$$

使用BSP树或Exact Predicates确保鲁棒性。

#### 网格手术（Mesh Surgery）

局部移除和重新三角化：
1. 识别相交区域
2. 删除涉及三角形
3. 重新三角化空洞
4. 光滑过渡区域

#### 收缩膨胀法

通过Mean Curvature Flow收缩：

$$\frac{\partial \mathbf{x}}{\partial t} = -H\mathbf{n}$$

然后反向膨胀恢复体积。

### 5.3.5 水密性保证

#### 定向一致性

使用传播算法统一法向：

$$\mathbf{n}_j = \begin{cases}
\mathbf{n}_j & \text{if } \mathbf{n}_i \cdot \mathbf{n}_j > 0 \\
-\mathbf{n}_j & \text{otherwise}
\end{cases}$$

**全局定向算法**：

1. **构建邻接图**：三角形作为节点，共享边的三角形相连
2. **宽度优先搜索**：
   ```
   queue = [选择一个种子三角形]
   visited = set()
   
   while queue:
     tri = queue.pop()
     visited.add(tri)
     
     for neighbor in tri.neighbors:
       if neighbor not in visited:
         调整neighbor使其与tri一致
         queue.append(neighbor)
   ```

3. **处理多个连通分量**：对每个分量独立定向

**法向一致性检查**：

对于共享边 $(v_i, v_j)$ 的两个三角形，检查边的方向：
- 一致定向：一个三角形中是 $(v_i, v_j)$，另一个中是 $(v_j, v_i)$
- 不一致：两个三角形中都是同一方向

#### 射线法内外判定

点 $\mathbf{p}$ 在网格内部当且仅当：

$$\text{winding number}(\mathbf{p}) = \frac{1}{4\pi} \sum_{T \in M} \Omega_T(\mathbf{p}) \neq 0$$

其中 $\Omega_T(\mathbf{p})$ 是三角形 $T$ 对点 $\mathbf{p}$ 的立体角。

## 5.4 细分曲面：Loop、Catmull-Clark

细分算法通过递归细化产生光滑曲面，在3D打印中用于提高模型质量。

### 5.4.1 Loop细分（三角网格）

#### 顶点更新规则

对于内部顶点（度为 $n$）：

$$\mathbf{v}' = (1 - n\beta)\mathbf{v} + \beta \sum_{i=1}^n \mathbf{v}_i$$

其中 $\beta$ 的选择：

$$\beta = \begin{cases}
\frac{3}{8n} & \text{Warren's choice} \\
\frac{1}{n}\left(\frac{5}{8} - \left(\frac{3}{8} + \frac{1}{4}\cos\frac{2\pi}{n}\right)^2\right) & \text{Loop's choice}
\end{cases}$$

#### 边点插入

新边点位置：

$$\mathbf{e} = \frac{3}{8}(\mathbf{v}_1 + \mathbf{v}_2) + \frac{1}{8}(\mathbf{v}_3 + \mathbf{v}_4)$$

其中 $\mathbf{v}_1, \mathbf{v}_2$ 是边的端点，$\mathbf{v}_3, \mathbf{v}_4$ 是对顶点。

#### 极限位置

顶点的极限位置：

$$\mathbf{v}_{\infty} = \frac{1}{1 + \frac{3n}{8}}\left(\mathbf{v} + \frac{3}{8}\sum_{i=1}^n \mathbf{v}_i\right)$$

#### 特征值分析

细分矩阵 $\mathbf{S}$ 的特征值：
- $\lambda_0 = 1$（保持仿射不变性）
- $\lambda_1 = \lambda_2 = \frac{3 + 2\cos(2\pi/n)}{8}$（切平面）
- $|\lambda_i| < 1$ for $i > 2$（收敛性）

### 5.4.2 Catmull-Clark细分（四边形网格）

#### 面点规则

$$\mathbf{f} = \frac{1}{n}\sum_{i=1}^n \mathbf{v}_i$$

#### 边点规则

$$\mathbf{e} = \frac{1}{4}(\mathbf{v}_1 + \mathbf{v}_2 + \mathbf{f}_1 + \mathbf{f}_2)$$

#### 顶点更新

对于度为 $n$ 的顶点：

$$\mathbf{v}' = \frac{n-2}{n}\mathbf{v} + \frac{1}{n^2}\sum_{i=1}^n \mathbf{e}_i + \frac{1}{n^2}\sum_{i=1}^n \mathbf{f}_i$$

#### 奇异点处理

非四边形面产生的奇异点，使用特殊规则：

$$\mathbf{v}' = \frac{1}{n}\left(\mathbf{v} + \frac{2}{n}\sum_{i=1}^n \mathbf{e}_i + \frac{1}{n}\sum_{i=1}^n \mathbf{f}_i\right)$$

### 5.4.3 $\sqrt{3}$-细分

#### 拓扑规则

每个三角形分成三个，顶点度趋向6：

$$|F'| = 3|F|, \quad |V'| = |V| + |F|$$

#### 几何规则

新面点：

$$\mathbf{f} = \frac{1}{3}(\mathbf{v}_1 + \mathbf{v}_2 + \mathbf{v}_3)$$

顶点更新：

$$\mathbf{v}' = (1 - \alpha_n)\mathbf{v} + \alpha_n \bar{\mathbf{v}}$$

其中：

$$\alpha_n = \frac{4 - 2\cos(2\pi/n)}{9}$$

### 5.4.4 自适应细分

#### 误差驱动细分

基于曲率的细分准则：

$$\text{subdivide}(T) = \begin{cases}
\text{true} & \text{if } \kappa_{\max} \cdot h > \tau \\
\text{false} & \text{otherwise}
\end{cases}$$

#### T-顶点处理

避免裂缝的红绿细分：
- 红色分裂：1→4
- 绿色分裂：1→2（过渡）

保持一级邻域细分差异 ≤ 1。

### 5.4.5 细分曲面的解析性质

#### $C^2$连续性

除奇异点外处处 $C^2$ 连续。奇异点处：
- Loop：$C^1$ 连续（$n \neq 6$）
- Catmull-Clark：$C^1$ 连续（$n \neq 4$）

#### 极限曲面评估

使用Stam的方法直接计算极限曲面：

$$\mathbf{S}(u, v) = \mathbf{b}(u, v)^T \mathbf{A}^n \mathbf{P}$$

其中 $\mathbf{A}$ 是细分矩阵，$\mathbf{b}$ 是基函数。

## 5.5 网格变形：ARAP、笼形变形

网格变形在3D打印中用于设计修改、形状优化和适应性调整。

### 5.5.1 As-Rigid-As-Possible (ARAP)

#### 能量函数

ARAP最小化局部非刚性变形：

$$E(\mathbf{p}') = \sum_{i \in V} w_i \sum_{j \in N(i)} w_{ij} \|\mathbf{p}'_i - \mathbf{p}'_j - \mathbf{R}_i(\mathbf{p}_i - \mathbf{p}_j)\|^2$$

其中 $\mathbf{R}_i$ 是顶点 $i$ 的局部旋转矩阵。

#### 交替优化

##### 步骤1：固定位置求旋转

使用SVD求解Procrustes问题：

$$\mathbf{S}_i = \sum_{j \in N(i)} w_{ij}(\mathbf{p}_i - \mathbf{p}_j)(\mathbf{p}'_i - \mathbf{p}'_j)^T$$

分解 $\mathbf{S}_i = \mathbf{U}_i \boldsymbol{\Sigma}_i \mathbf{V}_i^T$，则：

$$\mathbf{R}_i = \mathbf{V}_i \mathbf{U}_i^T$$

处理反射：若 $\det(\mathbf{R}_i) < 0$，翻转最小奇异值对应的列。

##### 步骤2：固定旋转求位置

求解线性系统：

$$\mathbf{L}\mathbf{p}' = \mathbf{b}$$

其中Laplacian矩阵：

$$L_{ij} = \begin{cases}
\sum_{k \in N(i)} w_{ik} & i = j \\
-w_{ij} & j \in N(i) \\
0 & \text{otherwise}
\end{cases}$$

右端项：

$$\mathbf{b}_i = \sum_{j \in N(i)} \frac{w_{ij}}{2}(\mathbf{R}_i + \mathbf{R}_j)(\mathbf{p}_i - \mathbf{p}_j)$$

#### 权重选择

余切权重（保持共形性）：

$$w_{ij} = \frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij})$$

均匀权重（简单快速）：

$$w_{ij} = 1$$

### 5.5.2 笼形变形（Cage-based Deformation）

#### Mean Value Coordinates (MVC)

对于点 $\mathbf{p}$ 在笼子内的坐标：

$$\lambda_i(\mathbf{p}) = \frac{w_i}{\sum_j w_j}$$

其中：

$$w_i = \frac{\tan(\alpha_{i-1}/2) + \tan(\alpha_i/2)}{\|\mathbf{v}_i - \mathbf{p}\|}$$

$\alpha_i$ 是从 $\mathbf{p}$ 看顶点 $i$ 和 $i+1$ 的夹角。

#### Harmonic Coordinates

通过求解Laplace方程获得：

$$\Delta \phi_i = 0 \text{ in } \Omega$$
$$\phi_i = \delta_{ij} \text{ on } \partial\Omega$$

离散化为线性系统：

$$\mathbf{L}_{\text{int}} \boldsymbol{\phi}_{\text{int}} + \mathbf{L}_{\text{bnd}} \boldsymbol{\phi}_{\text{bnd}} = \mathbf{0}$$

#### Green Coordinates

包含法向分量的坐标：

$$\mathbf{p} = \sum_{i} \phi_i(\mathbf{p})\mathbf{v}_i + \sum_{j} \psi_j(\mathbf{p})\mathbf{n}_j$$

其中 $\psi_j$ 是面的贡献：

$$\psi_j = -\frac{1}{4\pi} \int_{T_j} \frac{\mathbf{n}_j \cdot (\mathbf{x} - \mathbf{p})}{\|\mathbf{x} - \mathbf{p}\|^3} dA$$

### 5.5.3 基于物理的变形

#### 线弹性模型

应变能密度：

$$W = \frac{\lambda}{2}(\text{tr}(\boldsymbol{\varepsilon}))^2 + \mu \text{tr}(\boldsymbol{\varepsilon}^2)$$

其中应变张量：

$$\boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)$$

#### 共旋线性模型

分解变形梯度：

$$\mathbf{F} = \mathbf{R}\mathbf{S}$$

应力计算：

$$\boldsymbol{\sigma} = \mathbf{R}(2\mu \boldsymbol{\varepsilon} + \lambda \text{tr}(\boldsymbol{\varepsilon})\mathbf{I})$$

#### Neo-Hookean模型

超弹性能量：

$$W = \frac{\mu}{2}(I_C - 3) - \mu \log J + \frac{\lambda}{2}(\log J)^2$$

其中 $I_C = \text{tr}(\mathbf{F}^T\mathbf{F})$，$J = \det(\mathbf{F})$。

### 5.5.4 自由形式变形（FFD）

#### B样条体

变形函数：

$$\mathbf{p}'(\mathbf{u}) = \sum_{i,j,k} B_i(u)B_j(v)B_k(w) \mathbf{P}_{ijk}$$

其中 $B_i$ 是B样条基函数，$\mathbf{P}_{ijk}$ 是控制点。

#### NURBS变形

加权版本：

$$\mathbf{p}'(\mathbf{u}) = \frac{\sum_{i,j,k} w_{ijk}B_i(u)B_j(v)B_k(w) \mathbf{P}_{ijk}}{\sum_{i,j,k} w_{ijk}B_i(u)B_j(v)B_k(w)}$$

#### T样条变形

支持局部细化的变形：

$$\mathbf{p}'(\mathbf{u}) = \sum_{\alpha \in \mathcal{A}} N_{\alpha}(\mathbf{u}) \mathbf{P}_{\alpha}$$

其中 $N_{\alpha}$ 是T样条基函数。

### 5.5.5 约束变形

#### 位置约束

硬约束（拉格朗日乘子）：

$$\min_{\mathbf{x}} E(\mathbf{x}) \text{ s.t. } \mathbf{C}\mathbf{x} = \mathbf{d}$$

KKT系统：

$$\begin{pmatrix} \mathbf{H} & \mathbf{C}^T \\ \mathbf{C} & \mathbf{0} \end{pmatrix} \begin{pmatrix} \mathbf{x} \\ \boldsymbol{\lambda} \end{pmatrix} = \begin{pmatrix} -\mathbf{g} \\ \mathbf{d} \end{pmatrix}$$

软约束（罚函数）：

$$E_{\text{total}} = E_{\text{deform}} + \alpha \|\mathbf{C}\mathbf{x} - \mathbf{d}\|^2$$

#### 体积保持

局部体积约束：

$$V_T = \frac{1}{6}|(\mathbf{p}_1 - \mathbf{p}_0) \cdot ((\mathbf{p}_2 - \mathbf{p}_0) \times (\mathbf{p}_3 - \mathbf{p}_0))|$$

全局体积：

$$V = \sum_{T \in \mathcal{T}} V_T = V_0$$

#### 长度保持

边长约束：

$$\|\mathbf{p}'_i - \mathbf{p}'_j\| = l_{ij}$$

使用投影动力学求解：

$$\mathbf{p}^{(k+1)} = \arg\min_{\mathbf{p}} \|\mathbf{p} - \mathbf{q}^{(k)}\|_{\mathbf{M}}^2$$

其中 $\mathbf{q}^{(k)}$ 是投影到约束流形的位置。

## 本章小结

本章系统介绍了网格处理的核心算法，从简化、重新网格化到修复、细分和变形。关键要点：

1. **网格简化**：QEM提供了高效的误差度量，通过二次形式优雅地累积和最小化几何误差。渐进网格支持多分辨率表示和视相关渲染。

2. **重新网格化**：各向同性方法追求均匀三角形，各向异性方法根据曲率自适应调整尺寸和方向。CVT和各向异性Delaunay提供了理论保证。

3. **网格修复**：孔洞填充需要平衡几何质量和计算效率。自相交检测和消除是确保3D打印可制造性的关键步骤。

4. **细分曲面**：Loop和Catmull-Clark提供了 $C^2$ 连续的极限曲面（除奇异点外）。$\sqrt{3}$-细分提供了更好的顶点度分布。

5. **网格变形**：ARAP通过交替优化实现保形变形。笼形变形提供了直观的控制方式。物理模拟提供了真实的变形效果。

### 核心数学工具

- **二次形式**：$\mathbf{x}^T\mathbf{Q}\mathbf{x}$ 用于误差度量和能量优化
- **SVD分解**：求解最优旋转和Procrustes问题
- **Laplace算子**：离散微分几何的基础工具
- **有限元方法**：物理变形的数值求解
- **优化理论**：交替方向法、投影动力学

### 算法复杂度总结

| 算法 | 时间复杂度 | 空间复杂度 | 备注 |
|------|------------|------------|------|
| QEM简化 | $O(n \log n)$ | $O(n)$ | 使用优先队列 |
| Lloyd迭代 | $O(kn)$ | $O(n)$ | k次迭代 |
| 孔洞填充(DP) | $O(n^3)$ | $O(n^2)$ | n为边界顶点数 |
| Loop细分 | $O(n)$ | $O(n)$ | 线性增长 |
| ARAP变形 | $O(n + m)$ | $O(n^2)$ | 稀疏矩阵 |

## 练习题

### 基础题

**习题5.1** 证明QEM中的误差矩阵 $\mathbf{Q}$ 是半正定的。

<details>
<summary>提示</summary>
考虑 $\mathbf{Q} = \sum_p \mathbf{n}_p \mathbf{n}_p^T$ 的形式，利用外积矩阵的性质。
</details>

<details>
<summary>答案</summary>
对任意向量 $\mathbf{v} \neq \mathbf{0}$：
$$\mathbf{v}^T\mathbf{Q}\mathbf{v} = \mathbf{v}^T\left(\sum_p \mathbf{n}_p \mathbf{n}_p^T\right)\mathbf{v} = \sum_p (\mathbf{n}_p^T\mathbf{v})^2 \geq 0$$
等号成立当且仅当 $\mathbf{v}$ 垂直于所有平面法向量。因此 $\mathbf{Q}$ 半正定。
</details>

**习题5.2** 推导Loop细分中边点权重 $3/8$ 和 $1/8$ 的来源。

<details>
<summary>提示</summary>
考虑立方B样条的细分规则和张量积。
</details>

<details>
<summary>答案</summary>
Loop细分基于三向Box样条。对于均匀B样条，细分规则为：
- 边点：$(1/2, 1/2)$ 在一维
- 三角形的张量积给出：$(3/8, 3/8, 1/8, 1/8)$
这保证了 $C^2$ 连续性和仿射不变性。
</details>

**习题5.3** 计算正四面体网格经过一次Catmull-Clark细分后的顶点数、边数和面数。

<details>
<summary>提示</summary>
使用欧拉公式 $V - E + F = 2$ 验证结果。
</details>

<details>
<summary>答案</summary>
原始：$V = 4$，$E = 6$，$F = 4$
细分后：
- 面点：4个
- 边点：6个  
- 原顶点：4个
- 总顶点：$V' = 4 + 6 + 4 = 14$
- 每个三角面产生3个四边形：$F' = 4 \times 3 = 12$
- 由欧拉公式：$E' = V' + F' - 2 = 14 + 12 - 2 = 24$
</details>

**习题5.4** 给定2D多边形边界，使用最小面积三角化的动态规划算法，计算凸四边形ABCD的最优三角化。

<details>
<summary>提示</summary>
枚举中间顶点，递归计算子问题。
</details>

<details>
<summary>答案</summary>
设顶点按逆时针排列。两种可能：
1. 对角线AC：面积 = $|△ABC| + |△ACD|$
2. 对角线BD：面积 = $|△ABD| + |△BCD|$

选择面积较小的方案。面积计算：
$$|△| = \frac{1}{2}|x_1(y_2-y_3) + x_2(y_3-y_1) + x_3(y_1-y_2)|$$
</details>

### 挑战题

**习题5.5** 设计一个保体积的网格简化算法。给出能量函数和优化策略。

<details>
<summary>提示</summary>
在QEM基础上添加体积约束项，使用拉格朗日乘子法。
</details>

<details>
<summary>答案</summary>
增强的能量函数：
$$E(\mathbf{v}) = E_{QEM}(\mathbf{v}) + \lambda |V(\mathbf{v}) - V_0|$$

其中体积变化：
$$\Delta V = \frac{1}{6} \sum_{T \in \text{affected}} \text{sign}(T) \cdot \mathbf{v} \cdot (\mathbf{v}_i \times \mathbf{v}_j)$$

使用二阶近似：
$$V(\mathbf{v}) \approx V_0 + \nabla V^T(\mathbf{v} - \mathbf{v}_0) + \frac{1}{2}(\mathbf{v} - \mathbf{v}_0)^T\mathbf{H}_V(\mathbf{v} - \mathbf{v}_0)$$

通过迭代调整 $\lambda$ 保持体积。
</details>

**习题5.6** 分析ARAP变形的收敛性。在什么条件下算法可能不收敛？

<details>
<summary>提示</summary>
考虑能量函数的下界和每步迭代的单调性。
</details>

<details>
<summary>答案</summary>
ARAP能量函数有下界0（完全刚性变形）。每步迭代：
1. 固定 $\mathbf{p}'$ 求 $\mathbf{R}$：闭式解最优
2. 固定 $\mathbf{R}$ 求 $\mathbf{p}'$：凸二次问题，全局最优

因此能量单调递减，算法收敛到局部最小值。

不收敛情况：
- 约束不相容（如冲突的位置约束）
- 数值问题（矩阵病态、浮点误差累积）
- 退化配置（所有点共线）

改进：添加正则项 $\epsilon\|\mathbf{p}' - \mathbf{p}\|^2$
</details>

**习题5.7** 推导各向异性Voronoi图的对偶是各向异性Delaunay三角化。定义合适的度量空间。

<details>
<summary>提示</summary>
在Riemannian度量下重新定义距离和外接椭圆。
</details>

<details>
<summary>答案</summary>
定义度量张量场 $\mathbf{M}(\mathbf{x})$，距离：
$$d_{\mathbf{M}}(\mathbf{x}, \mathbf{y}) = \int_0^1 \sqrt{\dot{\gamma}(t)^T \mathbf{M}(\gamma(t)) \dot{\gamma}(t)} dt$$

其中 $\gamma$ 是连接 $\mathbf{x}$ 和 $\mathbf{y}$ 的测地线。

Voronoi区域：
$$V_i = \{\mathbf{x} : d_{\mathbf{M}}(\mathbf{x}, \mathbf{p}_i) \leq d_{\mathbf{M}}(\mathbf{x}, \mathbf{p}_j), \forall j\}$$

对偶的Delaunay条件：三点 $(p_i, p_j, p_k)$ 形成三角形当且仅当存在点 $\mathbf{c}$ 使得：
$$d_{\mathbf{M}}(\mathbf{c}, \mathbf{p}_i) = d_{\mathbf{M}}(\mathbf{c}, \mathbf{p}_j) = d_{\mathbf{M}}(\mathbf{c}, \mathbf{p}_k) \leq d_{\mathbf{M}}(\mathbf{c}, \mathbf{p}_l), \forall l$$

这定义了各向异性空圆性质。
</details>

**习题5.8** 设计一个检测和修复非流形顶点的算法。考虑顶点的邻域不同胚于圆盘的情况。

<details>
<summary>提示</summary>
分析顶点1-环的连通分量，考虑"蝴蝶结"和"沙漏"配置。
</details>

<details>
<summary>答案</summary>
算法步骤：

1. **检测非流形顶点**：
   ```
   对每个顶点v：
     构建1-环面集合F(v)
     构建边-面关联图G
     计算G的连通分量数k
     if k > 1: v是非流形顶点
   ```

2. **分类非流形类型**：
   - 蝴蝶结：两个或多个圆盘在顶点相接
   - 非流形边：边被超过2个面共享

3. **修复策略**：
   - **顶点分裂**：
     ```
     创建k个新顶点v₁...vₖ
     将第i个连通分量的面关联到vᵢ
     更新拓扑连接
     ```
   
   - **局部重网格化**：
     ```
     删除非流形配置的面
     使用推进波前法重新三角化
     确保流形性约束
     ```

4. **验证**：
   检查Link条件：$\text{link}(v)$ 应同胚于圆或线段

时间复杂度：$O(n \cdot d_{max})$，其中 $d_{max}$ 是最大顶点度。
</details>

## 常见陷阱与错误

### 1. QEM简化的数值不稳定

**问题**：当多个平面近似共面时，矩阵 $\mathbf{Q}$ 接近奇异。

**症状**：
- 顶点位置计算出现NaN或极大值
- 简化后网格出现尖刺

**解决方案**：
- 使用SVD代替直接求逆
- 添加正则项：$\mathbf{Q}_{reg} = \mathbf{Q} + \epsilon \mathbf{I}$
- 回退到端点或中点

### 2. 自相交检测的浮点误差

**问题**：浮点运算导致的几何谓词不一致。

**症状**：
- 明显相交的三角形未被检测
- 错误的拓扑判断

**解决方案**：
- 使用精确谓词（Shewchuk's predicates）
- 实现符号扰动（Simulation of Simplicity）
- 使用区间算术

### 3. 细分曲面的奇异点处理

**问题**：非正则顶点（度≠6 for Loop，度≠4 for Catmull-Clark）导致的曲率问题。

**症状**：
- 奇异点处出现褶皱或平坦区域
- 曲率不连续

**解决方案**：
- 使用特殊的奇异点规则
- 局部重参数化
- 自适应细分避免过度细化

### 4. ARAP变形的局部最小值

**问题**：交替优化陷入次优解。

**症状**：
- 大变形时出现不自然的扭曲
- 不同初始化导致不同结果

**解决方案**：
- 使用多分辨率策略
- 添加动量项
- 结合全局优化方法

### 5. 孔洞填充的拓扑歧义

**问题**：复杂孔洞可能有多种拓扑上不同的填充方式。

**症状**：
- 填充结果与预期拓扑不符
- 产生自相交

**解决方案**：
- 用户指导的拓扑选择
- 基于上下文的启发式
- 多个候选方案的评分

## 最佳实践检查清单

### 网格简化
- [ ] 选择合适的误差度量（QEM、Hausdorff距离、法向偏差）
- [ ] 保持边界和特征边
- [ ] 验证拓扑不变性（亏格、连通分量）
- [ ] 检查三角形质量（避免退化）
- [ ] 实现渐进式存储（支持LOD）

### 重新网格化
- [ ] 根据应用选择各向同性或各向异性
- [ ] 设置合适的目标边长/密度
- [ ] 保持特征线和约束
- [ ] 优化三角形质量指标
- [ ] 处理边界和孔洞

### 网格修复
- [ ] 完整的缺陷检测（孔洞、自相交、非流形）
- [ ] 选择合适的填充算法（平面、曲率感知）
- [ ] 确保水密性和定向一致性
- [ ] 验证修复后的拓扑正确性
- [ ] 保存修复日志用于追溯

### 细分曲面
- [ ] 选择合适的细分方案（Loop for 三角形、Catmull-Clark for 四边形）
- [ ] 处理边界和折痕
- [ ] 控制细分层数（避免过度细化）
- [ ] 实现自适应细分（基于曲率或视角）
- [ ] 优化内存使用（增量计算）

### 网格变形
- [ ] 选择合适的变形方法（ARAP保形、FFD全局控制）
- [ ] 设置合理的约束（位置、体积、长度）
- [ ] 实现多分辨率编辑
- [ ] 提供实时预览
- [ ] 支持撤销/重做操作