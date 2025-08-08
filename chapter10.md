# 第10章：路径规划与填充

本章深入探讨3D打印中的路径规划算法和填充策略。从基础的轮廓偏置算法开始，逐步介绍各种高级填充模式的数学原理，包括Voronoi图、空间填充曲线等。最后讨论路径优化问题和功能梯度材料的填充策略。这些技术直接影响打印效率、结构强度和材料分布。

## 10.1 轮廓偏置与Clipper库原理

### 10.1.1 多边形偏置的数学基础

轮廓偏置（Polygon Offsetting）是生成填充路径的基础操作。给定平面多边形 $P$，其偏置多边形定义为：

$$P_{\delta} = \{x \in \mathbb{R}^2 : d(x, P) = \delta\}$$

其中 $d(x, P) = \min_{p \in P} \|x - p\|$ 是点到多边形的最短距离。

**Minkowski和表示**：
偏置操作等价于Minkowski和：
$$P_{\delta} = P \oplus B_{\delta}$$

其中 $B_{\delta} = \{x \in \mathbb{R}^2 : \|x\| \leq \delta\}$ 是半径为 $\delta$ 的圆盘。

对于简单多边形的每条边 $e_i$，偏置后的边 $e'_i$ 满足：
- 与原边平行
- 法向距离为 $\delta$

偏置方向由边的法向量决定：
$$\mathbf{n}_i = \frac{1}{\|e_i\|}(y_{i+1} - y_i, x_i - x_{i+1})$$

**符号距离函数**：
多边形的符号距离函数 $\phi: \mathbb{R}^2 \to \mathbb{R}$：
$$\phi(x) = \begin{cases}
-d(x, \partial P) & x \in P \\
+d(x, \partial P) & x \notin P
\end{cases}$$

偏置轮廓即为水平集 $\{x : \phi(x) = \delta\}$。

### 10.1.2 顶点处理与斜接限制

在多边形顶点处，相邻边的偏置可能产生不同的交点位置。设顶点 $v_i$ 处的内角为 $\theta_i$，两条相邻边的单位法向量为 $\mathbf{n}_{i-1}$ 和 $\mathbf{n}_i$。

**斜接（Miter）连接**：偏置顶点位置为：
$$v'_i = v_i + \frac{\delta}{\sin(\theta_i/2)} \cdot \frac{\mathbf{n}_{i-1} + \mathbf{n}_i}{\|\mathbf{n}_{i-1} + \mathbf{n}_i\|}$$

**角平分线方法**：
设角平分线方向 $\mathbf{b}_i = \frac{\mathbf{n}_{i-1} + \mathbf{n}_i}{\|\mathbf{n}_{i-1} + \mathbf{n}_i\|}$

偏置距离：
$$d_i = \frac{\delta}{\mathbf{n}_{i-1} \cdot \mathbf{b}_i} = \frac{\delta}{\cos((\pi - \theta_i)/2)}$$

当 $\theta_i$ 很小时，斜接长度 $\frac{\delta}{\sin(\theta_i/2)}$ 可能非常大。定义斜接限制因子：
$$m_{limit} = \frac{1}{\sin(\theta_{min}/2)}$$

当超过限制时，采用**斜角（Bevel）**或**圆角（Round）**连接。

**圆角连接的参数化**：
圆弧段参数方程（$t \in [0, 1]$）：
$$\mathbf{p}(t) = v_i + \delta \cdot (\cos(\alpha(t)), \sin(\alpha(t)))$$
其中 $\alpha(t) = \alpha_{start} + t(\alpha_{end} - \alpha_{start})$

### 10.1.3 自相交检测与处理

偏置操作可能产生自相交，特别是在凹多边形的情况下。检测自相交需要：

1. **扫描线算法**：复杂度 $O(n \log n)$
2. **Bentley-Ottmann算法**：处理线段相交
3. **布尔运算**：通过并集操作消除自相交

**Bentley-Ottmann算法详解**：

维护两个数据结构：
- **事件队列** $Q$：按y坐标排序的端点和交点
- **状态结构** $T$：当前扫描线与线段的交点，按x坐标排序

算法步骤：
```
1. 初始化Q为所有线段端点
2. while Q非空:
   p = Q.pop()
   if p是上端点: HandleUpperEndpoint(p)
   if p是下端点: HandleLowerEndpoint(p)  
   if p是交点: HandleIntersection(p)
3. 返回所有找到的交点
```

**Vatti裁剪算法**：

核心数据结构：
- **边表**（ET）：按y_{min}排序的边
- **活动边表**（AET）：与当前扫描线相交的边
- **输出多边形表**：构建结果

状态转换：
- 局部最小值：开始新的多边形
- 左边界：进入多边形内部
- 右边界：离开多边形内部
- 相交点：交换边的左右关系

### 10.1.4 鲁棒性问题

浮点运算的数值误差可能导致拓扑不一致。Clipper库采用整数坐标：

$$x_{int} = \lfloor x_{float} \cdot scale + 0.5 \rfloor$$

其中 $scale$ 通常取 $10^6$ 或更大，保证精度同时避免浮点误差。

**精确算术谓词**：对于关键的几何判断（如点在线的哪一侧），使用：
$$\text{orient2d}(a, b, c) = \det\begin{bmatrix} a_x - c_x & a_y - c_y \\ b_x - c_x & b_y - c_y \end{bmatrix}$$

**Shewchuk的自适应精度算术**：

1. **快速浮点计算**：首先用标准浮点
2. **误差估计**：计算舍入误差界
3. **精确计算**：仅在误差可能影响符号时

误差界估计（对于2D orientation）：
$$|\epsilon| \leq (3 + 16\epsilon_{mach}) \epsilon_{mach} \cdot (|a_x - c_x| \cdot |b_y - c_y| + |a_y - c_y| \cdot |b_x - c_x|)$$

**区间算术方法**：
将每个数表示为区间 $[a_{low}, a_{high}]$：
- 加法：$[a_l, a_h] + [b_l, b_h] = [a_l + b_l, a_h + b_h]$
- 乘法：考虑所有端点组合的最小/最大值
- 如果结果区间不包含0，符号确定

## 10.2 Voronoi填充与Fermat螺旋

### 10.2.1 Voronoi图的计算几何

给定平面上的点集 $S = \{p_1, ..., p_n\}$，点 $p_i$ 的Voronoi单元定义为：
$$V(p_i) = \{x \in \mathbb{R}^2 : \|x - p_i\| \leq \|x - p_j\|, \forall j \neq i\}$$

**对偶性质**：
Voronoi图与Delaunay三角化对偶：
- Voronoi顶点 ↔ Delaunay三角形的外接圆心
- Voronoi边 ↔ Delaunay边的垂直平分线
- Voronoi单元 ↔ Delaunay顶点

**Fortune扫描线算法**：

核心观察：点 $p$ 到扫描线 $\ell$ 上方已处理点的最近距离等于到某个抛物线的距离。

海滩线（beach line）是这些抛物线的下包络线。

海滩线上的抛物线方程（扫描线位置为 $y = \ell$）：
$$y = \frac{(x - p_x)^2}{2(p_y - \ell)} + \frac{p_y + \ell}{2}$$

**两个抛物线的交点**：
给定焦点 $p_i = (x_i, y_i)$ 和 $p_j = (x_j, y_j)$，交点x坐标：
$$x = \frac{x_j(y_i - \ell) - x_i(y_j - \ell) \pm \sqrt{D}}{y_i - y_j}$$

其中 $D = (y_i - \ell)(y_j - \ell)[(x_i - x_j)^2 + (y_i - y_j)^2]$

**圆事件检测**：
三个连续抛物线 $p_i, p_j, p_k$ 的收敛点是其外接圆的最低点。

### 10.2.2 中轴变换与Voronoi骨架

多边形的中轴（Medial Axis）是内部所有具有多个最近边界点的点的集合。对于多边形 $P$：

$$MA(P) = \{x \in P : |\{y \in \partial P : \|x - y\| = d(x, \partial P)\}| \geq 2\}$$

**中轴的数学性质**：

1. **火烧草原类比**：中轴是边界同时点火后火锋相遇的轨迹

2. **最大内切圆中心**：中轴上每点是某个最大内切圆的中心

3. **形态学骨架**：
   $$MA(P) = \bigcup_{r > 0} \{\text{centers of maximal } B_r \subseteq P\}$$

中轴与Voronoi图的关系：
- 多边形边的Voronoi图包含中轴
- 中轴的分支对应Voronoi边
- 分支点对应Voronoi顶点

**Lee算法（计算方法）**：
1. 构建约束Delaunay三角化（CDT）
2. 分类三角形：
   - Type 0: 三个顶点都在边界上
   - Type 1: 两个顶点在同一边上
   - Type 2: 两个顶点在不同边上
   - Type 3: 一个顶点在角点
3. 提取中轴：
   - Type 0: 外接圆心
   - Type 1: 到对边的垂线
   - Type 2,3: 角平分线段
4. 修剪外部分支（使用circumradius阈值）

### 10.2.3 Fermat螺旋填充

Fermat螺旋的参数方程：
$$r = a\sqrt{\theta}, \quad \theta \in [0, \theta_{max}]$$

笛卡尔坐标：
$$\begin{cases}
x(\theta) = a\sqrt{\theta} \cos(\theta) \\
y(\theta) = a\sqrt{\theta} \sin(\theta)
\end{cases}$$

**微分几何性质**：

1. **弧长参数化**：
   $$s(\theta) = \int_0^\theta \sqrt{r^2 + (\frac{dr}{d\theta})^2} d\theta = \frac{a}{2} \int_0^\theta \frac{\sqrt{4\theta + 1}}{\sqrt{\theta}} d\theta$$

2. **曲率计算**：
   使用极坐标曲率公式：
   $$\kappa = \frac{|r^2 + 2(\frac{dr}{d\theta})^2 - r\frac{d^2r}{d\theta^2}|}{(r^2 + (\frac{dr}{d\theta})^2)^{3/2}}$$
   
   对Fermat螺旋：
   $$\kappa = \frac{2 + \theta}{2a(1 + \theta)^{3/2}}$$

3. **等间距性分析**：
   相邻螺旋臂（相位差$2\pi$）的径向距离：
   $$\Delta r = a(\sqrt{\theta + 2\pi} - \sqrt{\theta}) \approx \frac{\pi a}{\sqrt{\theta}}$$
   
   当 $\theta >> 2\pi$ 时，间距趋于常数 $\pi a$。

**填充密度控制**：
通过调整参数 $a$ 控制填充密度：
$$\rho = \frac{\text{path length}}{\text{area}} \approx \frac{1}{\pi a}$$

### 10.2.4 连续路径生成

将Voronoi骨架转换为连续填充路径：

**基于中轴的偏置算法**：

1. **偏置层生成**：
   从中轴开始，生成等距偏置曲线：
   $$C_k = \{x : d(x, MA) = k\delta\}, \quad k = 0, 1, 2, ...$$

2. **拓扑事件处理**：
   - **分裂事件**：一条曲线分为多条
   - **合并事件**：多条曲线合为一条
   - **消失事件**：曲线收缩为点

3. **连接优化**（层间过渡）：
   
   构建二分图 $G = (U \cup V, E)$：
   - $U$：第$i$层的端点
   - $V$：第$i+1$层的端点
   - 权重：$w(u,v) = \|u - v\|$
   
   求解最小权匹配。

**路径平滑**：

使用变分方法最小化能量泛函：
$$E[\gamma] = \int_0^L \left[ \alpha \|\gamma'(s)\|^2 + \beta \|\gamma''(s)\|^2 \right] ds$$

其中：
- 第一项：长度惩罚
- 第二项：曲率惩罚

Euler-Lagrange方程：
$$\alpha \gamma'' - \beta \gamma^{(4)} = 0$$

离散化后使用五点差分求解。

## 10.3 空间填充曲线：Hilbert、Peano

### 10.3.1 Hilbert曲线的递归构造

Hilbert曲线是一种分形空间填充曲线，通过递归定义：

**基础模式**（$n=1$）：
```
┌─┐
│ │
└─┘
```

**递归规则**（$n \to n+1$）：
1. 将正方形分为 $2^n \times 2^n$ 个子正方形
2. 对每个象限应用旋转/反射的基础模式
3. 连接相邻象限

**数学表示**：
定义映射 $H_n: [0, 1] \to [0, 1]^2$，满足：
- $H_n$ 是连续的
- $H_n$ 是满射的（空间填充）
- 保持局部性：$|t_1 - t_2|$ 小 $\Rightarrow$ $\|H_n(t_1) - H_n(t_2)\|$ 小

**L-系统表示**：
Hilbert曲线可用L-系统描述：
- 公理：A
- 规则：
  - A → +BF−AFA−FB+
  - B → −AF+BFB+FA−
- 符号：F=前进，+=左转90°，−=右转90°

**Hölder连续性**：
Hilbert曲线满足Hölder条件：
$$\|H(t_1) - H(t_2)\| \leq C|t_1 - t_2|^{1/2}$$

证明：第$n$级曲线将单位区间分为$4^n$段，每段映射到边长$2^{-n}$的正方形。

### 10.3.2 Gray码与位操作

Hilbert曲线的高效计算基于Gray码：

$$g = b \oplus (b >> 1)$$

其中 $\oplus$ 是异或操作，$>>$ 是右移。

**坐标到Hilbert索引算法**：

```
function xy2h(x, y, n):
    h = 0
    for i from n-1 to 0:
        rx = (x >> i) & 1  // 第i位
        ry = (y >> i) & 1
        h = (h << 2) | (3 * rx) ⊕ ry  // 2位Gray码
        // 坐标变换
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            swap(x, y)
    return h
```

**逆变换（Hilbert索引到坐标）**：

```
function h2xy(h, n):
    x = y = 0
    for i from 0 to n-1:
        rx = 1 & (h >> 1)
        ry = 1 & (h ⊕ rx)
        rot(n, x, y, rx, ry)
        x += rx * n
        y += ry * n
        h >>= 2
        n >>= 1
    return (x, y)
```

**位操作优化**：
使用查找表加速小块计算：
$$LUT[i] = H_2(i), \quad i \in [0, 15]$$

### 10.3.3 Peano曲线与三进制系统

Peano曲线基于三进制分割：

$$P_n: [0, 1] \to [0, 1]^2$$

**递归构造**：

第$n$级Peano曲线将单位正方形分为$3^n \times 3^n$个子正方形。

基础模式（3×3）：
```
1→2→3
    ↓
6←5←4
↓    
7→8→9
```

**参数化表示**：
设 $t \in [0, 1]$ 的三进制展开为 $t = 0.t_1t_2t_3...$

定义状态转移函数 $\sigma: \{0,1,2\} \times S \to \{0,1,2\}^2 \times S$：
$$\sigma(d_i, s_i) = ((x_i, y_i), s_{i+1})$$

其中$s_i \in \{\text{normal}, \text{flip_x}, \text{flip_y}, \text{flip_xy}\}$表示变换状态。

则坐标为：
$$\begin{cases}
x(t) = \sum_{i=1}^{\infty} x_i \cdot 3^{-i} \\
y(t) = \sum_{i=1}^{\infty} y_i \cdot 3^{-i}
\end{cases}$$

**与Hilbert曲线的比较**：
- Peano：更好的各向同性，$3^n$分割
- Hilbert：更好的局部性，$2^n$分割，二进制友好
- 两者的Hausdorff维数都是2（真正空间填充）

### 10.3.4 填充质量分析

**局部性度量**（Locality Measure）：
$$L = \frac{1}{n} \sum_{i=1}^{n} \frac{d_{path}(i, i+1)}{d_{Euclidean}(i, i+1)}$$

理想值为1，实际值通常在1.5-3之间。

**聚类性质**（Box-counting维数）：
覆盖曲线所需的边长为$\epsilon$的正方形数量：
$$N(\epsilon) \sim \epsilon^{-D}$$

对于Hilbert和Peano曲线，$D = 2$（空间填充）。

**各向同性分析**：

定义方向相关的相关函数：
$$C(r, \theta) = \langle \rho(\mathbf{x}) \rho(\mathbf{x} + r\mathbf{e}_\theta) \rangle$$

各向同性度量：
$$I = 1 - \frac{\max_\theta C(r, \theta) - \min_\theta C(r, \theta)}{\langle C(r, \theta) \rangle_\theta}$$

Peano曲线的各向同性优于Hilbert曲线。

**热力学性能**：

稳态热传导方程：
$$\nabla \cdot (k \nabla T) + Q = 0$$

空间填充曲线路径的有效热导率（均匀化理论）：
$$k_{eff} = k_{material} \cdot f + k_{air} \cdot (1-f)$$

其中$f$是填充率。曲线的连续性减少了热点。

**机械性能预测**：
使用Gibson-Ashby模型：
$$\frac{E_{eff}}{E_s} = C(\frac{\rho_{eff}}{\rho_s})^n$$

其中$n \approx 2$对于弯曲主导的结构。

## 10.4 连续路径规划与TSP问题

### 10.4.1 旅行商问题的数学形式

给定 $n$ 个点 $\{p_1, ..., p_n\}$ 和距离矩阵 $D_{ij} = \|p_i - p_j\|$，TSP寻找排列 $\pi$：

$$\min_{\pi} \sum_{i=1}^{n} D_{\pi(i), \pi(i+1 \mod n)}$$

**整数规划形式**：
$$\begin{align}
\min \quad & \sum_{i,j} c_{ij} x_{ij} \\
s.t. \quad & \sum_{j} x_{ij} = 1, \forall i \\
& \sum_{i} x_{ij} = 1, \forall j \\
& \sum_{i,j \in S} x_{ij} \leq |S| - 1, \forall S \subset V, 2 \leq |S| \leq n-2
\end{align}$$

最后一个约束（子回路消除）数量为指数级。

**Miller-Tucker-Zemlin (MTZ) 公式化**：
使用额外变量$u_i$避免指数级约束：
$$\begin{align}
u_i - u_j + nx_{ij} &\leq n - 1, \quad \forall i \neq j, i,j \geq 2 \\
2 \leq u_i &\leq n, \quad \forall i \geq 2
\end{align}$$

**分支定界方法**：
- 下界：使用线性松弛或Held-Karp下界
- 上界：启发式解
- 分支策略：选择分数$x_{ij}$最接近0.5的变量

### 10.4.2 启发式算法

**最近邻算法**：
```
1. 从任意点开始
2. 访问最近的未访问点
3. 重复直到所有点被访问
```
时间复杂度：$O(n^2)$，近似比：$O(\log n)$

**Christofides算法详解**：

1. **构建最小生成树**：Prim或Kruskal，$O(n^2)$或$O(n\log n)$

2. **奇度顶点匹配**：
   - 识别奇度顶点集$V_{odd}$（偶数个，握手定理）
   - 构建完全图$K_{V_{odd}}$
   - 求解最小权完美匹配（Blossom算法，$O(n^3)$）

3. **欧拉回路构造**：
   MST + 匹配 = 所有顶点偶度图
   使用Hierholzer算法找欧拉回路

4. **短路化**：跳过重复顶点

**证明1.5-近似**：
- $w(MST) \leq OPT$（删除TSP一条边得生成树）
- $w(M) \leq \frac{1}{2}OPT$（奇度顶点在OPT中的子回路）
- 总长度 $\leq 1.5 \cdot OPT$

**贪心插入算法**：
```
1. 初始化为三角形
2. 重复：找到插入成本最小的点
3. 插入该点到最佳位置
```
近似比：$2 \cdot OPT$

### 10.4.3 局部搜索与2-opt

**2-opt操作**：

删除两条边$(i, i+1)$和$(j, j+1)$，添加$(i, j)$和$(i+1, j+1)$：

原路径：$... \to i \to i+1 \to ... \to j \to j+1 \to ...$
新路径：$... \to i \to j \to ... \to i+1 \to j+1 \to ...$

注意：$i+1$到$j$之间的段被反转。

**改进量计算**：
$$\Delta = d_{ij} + d_{i+1,j+1} - d_{i,i+1} - d_{j,j+1}$$

**加速技巧**：
1. **Don't Look Bits**：标记近期没有改进的顶点
2. **邻域列表**：只考虑k个最近邻
3. **几何剪枝**：$d_{ij} > d_{i,i+1} + d_{j,j+1}$时跳过

**k-opt扩展**：
- 3-opt：删除3条边，7种重连方式
- 时间复杂度：$O(n^k)$

**Lin-Kernighan算法**：

核心思想：动态选择k值，通过交替序列构建改进。

交替序列：$X = \{x_1, x_2, ..., x_k\}$（删除边），$Y = \{y_1, y_2, ..., y_k\}$（添加边）

约束：
1. 保持连通性
2. 增益条件：$\sum_{i=1}^{j} (w(x_i) - w(y_i)) > 0$，$\forall j$
3. 顺序性：$x_i$和$y_i$共享一个端点

### 10.4.4 打印路径的特殊约束

**层间过渡优化**：

定义层间过渡问题：
$$\min \sum_{l=1}^{L-1} d(C_l, C_{l+1}) + \lambda \sum_{l=1}^{L} T(C_l)$$

其中：
- $C_l$：第$l$层的轮廓
- $d(C_l, C_{l+1})$：层间过渡距离
- $T(C_l)$：层内打印时间

**动态规划求解**：
$$f(l, p) = \min_{q \in C_{l-1}} \{f(l-1, q) + \|p - q\| + T(l, p)\}$$

**缝隙位置优化**：

评价函数：
$$S(p) = \alpha \cdot V(p) + \beta \cdot C(p) + \gamma \cdot O(p)$$

其中：
- $V(p)$：可见性分数（视角隐藏）
- $C(p)$：曲率分数（尖角优先）
- $O(p)$：悬垂分数（避免悬垂区）

**速度规划**：

运动学约束：
$$\begin{cases}
\|v\| \leq v_{max} \\
\|a\| \leq a_{max} \\
\|j\| \leq j_{max} \quad \text{(jerk限制)}
\end{cases}$$

曲率自适应速度：
$$v(s) = \min\left(v_{max}, \sqrt{\frac{a_{cent}}{\kappa(s)}}, \sqrt[3]{\frac{j_{max}}{|\dot{\kappa}(s)|}}\right)$$

**S形速度曲线**：
加速阶段：
$$v(t) = \begin{cases}
\frac{1}{2}jt^2 & 0 \leq t < t_1 \\
v_0 + a_{max}(t - t_1) & t_1 \leq t < t_2 \\
v_{cruise} - \frac{1}{2}j(t_3 - t)^2 & t_2 \leq t < t_3
\end{cases}$$

## 10.5 梯度填充与功能梯度材料

### 10.5.1 密度场定义

功能梯度的密度场 $\rho: \Omega \to [0, 1]$，满足：

$$\begin{cases}
\nabla \cdot (\sigma(\rho) \nabla u) = f & \text{in } \Omega \\
u = 0 & \text{on } \Gamma_D \\
\sigma(\rho) \nabla u \cdot n = g & \text{on } \Gamma_N
\end{cases}$$

**材料插值模型**：

1. **SIMP（Solid Isotropic Material with Penalization）**：
   $$E(\rho) = \rho^p E_0, \quad p > 1$$
   通常$p = 3$以惩罚中间密度。

2. **RAMP（Rational Approximation of Material Properties）**：
   $$E(\rho) = E_0 \frac{\rho(1 + q)}{\rho + q}, \quad q > 0$$
   避免零密度奇异性。

3. **Heaviside投影**：
   $$\tilde{\rho} = \frac{\tanh(\beta \eta) + \tanh(\beta(\rho - \eta))}{\tanh(\beta \eta) + \tanh(\beta(1 - \eta))}$$
   其中$\beta$控制陡峭度，$\eta$是阈值。

### 10.5.2 填充图案映射

将连续密度场离散化为填充图案：

**线性映射**：
$$spacing(x) = s_{min} + (s_{max} - s_{min})(1 - \rho(x))$$

**非线性映射**（考虑制造约束）：
$$spacing(x) = s_{min} \cdot \exp\left(\ln\left(\frac{s_{max}}{s_{min}}\right) \cdot (1 - \rho(x))^{\gamma}\right)$$

**抓抖（Dithering）技术**：

1. **Floyd-Steinberg误差扩散**：
   ```
   当前像素  7/16
   3/16  5/16  1/16
   ```
   误差$e = \rho_{target} - \rho_{actual}$按权重分配。

2. **蓝噪声抓抖**：
   使用高频噪声图案，避免低频伪影：
   $$\rho_{dithered}(x,y) = \rho(x,y) + \alpha \cdot B(x,y)$$
   其中$B(x,y)$是蓝噪声模板。

**填充模式选择**：
- 低密度（$\rho < 0.3$）：蛋糕或六角形
- 中密度（$0.3 \leq \rho < 0.7$）：回形或网格
- 高密度（$\rho \geq 0.7$）：同心轮廓

### 10.5.3 各向异性填充

对于各向异性材料，定义方向场 $\theta: \Omega \to [0, 2\pi]$：

**主应力方向**：
$$\theta = \frac{1}{2} \arctan\left(\frac{2\tau_{xy}}{\sigma_{xx} - \sigma_{yy}}\right)$$

**路径生成**：
沿流线积分：
$$\frac{dx}{ds} = \cos\theta(x, y), \quad \frac{dy}{ds} = \sin\theta(x, y)$$

使用Runge-Kutta方法数值求解。

### 10.5.4 多材料梯度

对于 $m$ 种材料，定义体积分数场 $\phi_i: \Omega \to [0, 1]$：

$$\sum_{i=1}^{m} \phi_i(x) = 1, \quad \forall x \in \Omega$$

**材料插值**：
$$E(x) = \sum_{i=1}^{m} \phi_i(x) E_i$$

**路径分配策略**：
1. 时间分割：按比例分配挤出时间
2. 空间分割：Voronoi划分或条纹图案
3. 概率分配：随机抖动（dithering）

**优化问题**：
$$\min_{\phi} \int_{\Omega} f(\phi, \nabla\phi) dx$$

subject to manufacturing constraints。

## 本章小结

本章系统介绍了3D打印中路径规划与填充的核心算法：

**关键概念**：
1. **轮廓偏置**：基于Minkowski和的多边形偏置，处理自相交和数值鲁棒性
2. **Voronoi填充**：利用中轴变换生成自适应填充路径
3. **空间填充曲线**：Hilbert和Peano曲线的递归构造与局部性保持
4. **路径优化**：TSP问题的启发式求解和局部搜索
5. **梯度填充**：功能梯度材料的密度场映射与各向异性路径

**核心公式汇总**：

- 多边形偏置：$P_{\delta} = \{x : d(x, P) = \delta\}$
- Voronoi单元：$V(p_i) = \{x : \|x - p_i\| \leq \|x - p_j\|, \forall j \neq i\}$
- Fermat螺旋：$r = a\sqrt{\theta}$
- Hilbert映射：$H_n: [0, 1] \to [0, 1]^2$（递归构造）
- TSP目标：$\min \sum_{i} D_{\pi(i), \pi(i+1)}$
- 密度插值：$\sigma(\rho) = \rho^p E_0$（SIMP方法）

**算法复杂度**：
- Bentley-Ottmann线段相交：$O(n \log n)$
- Fortune扫描线（Voronoi）：$O(n \log n)$
- Christofides算法（TSP）：1.5-近似
- 2-opt局部搜索：$O(n^2)$每次迭代

## 练习题

### 基础题

**习题10.1** 给定三角形顶点 $A(0,0)$，$B(4,0)$，$C(2,3)$，计算向内偏置距离 $\delta = 0.5$ 后的三角形顶点坐标。

<details>
<summary>提示</summary>
计算每条边的内法向量，然后求偏置边的交点。注意内角的影响。
</details>

<details>
<summary>答案</summary>

1. 计算各边的内法向量（归一化）：
   - 边AB：$\mathbf{n}_{AB} = (0, 1)$
   - 边BC：$\mathbf{n}_{BC} = (\frac{3}{2\sqrt{13}}, \frac{2}{2\sqrt{13}})$
   - 边CA：$\mathbf{n}_{CA} = (\frac{3}{\sqrt{13}}, -\frac{2}{\sqrt{13}})$

2. 偏置后的边方程：
   - $A'B'$：$y = 0.5$
   - $B'C'$：法向偏置0.5
   - $C'A'$：法向偏置0.5

3. 求交点得到新顶点：
   - $A' \approx (0.62, 0.5)$
   - $B' \approx (3.38, 0.5)$
   - $C' \approx (2, 2.13)$
</details>

**习题10.2** 证明Hilbert曲线在 $n \to \infty$ 时是空间填充的，即其像集在 $[0,1]^2$ 中稠密。

<details>
<summary>提示</summary>
利用一致连续性和紧集的性质。证明任意点都是曲线的极限点。
</details>

<details>
<summary>答案</summary>

证明：
1. Hilbert曲线 $H_n$ 将单位区间 $[0,1]$ 映射到 $4^n$ 个边长为 $2^{-n}$ 的正方形
2. 对任意 $(x,y) \in [0,1]^2$ 和 $\epsilon > 0$，选择 $n$ 使得 $2^{-n} < \epsilon/\sqrt{2}$
3. 存在某个正方形包含 $(x,y)$，其中心到 $(x,y)$ 的距离 $< \epsilon$
4. 由于 $H_n$ 经过每个正方形的中心，存在 $t \in [0,1]$ 使得 $\|H_n(t) - (x,y)\| < \epsilon$
5. 因此 $\bigcup_{n=1}^{\infty} H_n([0,1])$ 在 $[0,1]^2$ 中稠密
</details>

**习题10.3** 设计一个算法，将给定的Voronoi图转换为连续的螺旋填充路径。分析算法的时间复杂度。

<details>
<summary>提示</summary>
从中轴开始，逐层向外偏置，并在每层末端连接到下一层。
</details>

<details>
<summary>答案</summary>

算法：
1. 计算Voronoi图的中轴（$O(n \log n)$）
2. 从中轴向外偏置，步长为 $\delta$
3. 对每个偏置轮廓：
   - 参数化为 $s \in [0, L]$（弧长参数）
   - 在末端找最近的下一层起点
4. 使用样条曲线连接层间过渡
5. 总复杂度：$O(n \log n + km)$，其中 $k$ 是层数，$m$ 是每层的顶点数
</details>

### 挑战题

**习题10.4** 推导Fermat螺旋相邻臂之间的精确距离公式，并分析其随半径的变化规律。

<details>
<summary>提示</summary>
考虑相位差 $2\pi$ 的两点之间的距离，使用泰勒展开近似。
</details>

<details>
<summary>答案</summary>

设Fermat螺旋 $r = a\sqrt{\theta}$，考虑角度 $\theta$ 和 $\theta + 2\pi$ 的两点：

1. 两点的极坐标：
   - $P_1: (a\sqrt{\theta}, \theta)$
   - $P_2: (a\sqrt{\theta + 2\pi}, \theta + 2\pi)$

2. 在同一射线上的径向距离：
   $$d_r = a\sqrt{\theta + 2\pi} - a\sqrt{\theta} = a\sqrt{\theta}\left(\sqrt{1 + \frac{2\pi}{\theta}} - 1\right)$$

3. 当 $\theta >> 2\pi$ 时，使用泰勒展开：
   $$\sqrt{1 + x} \approx 1 + \frac{x}{2} - \frac{x^2}{8} + O(x^3)$$

4. 得到：
   $$d_r \approx \frac{a\pi}{\sqrt{\theta}} = \frac{\pi a^2}{r}$$

5. 因此间距与半径成反比，在中心附近间距大，外围间距趋于常数 $\pi a$。
</details>

**习题10.5** 对于TSP问题，证明最近邻算法的近似比为 $O(\log n)$，并构造达到此界的实例。

<details>
<summary>提示</summary>
构造一个"梳子"形状的点集，使最近邻算法产生最坏情况。
</details>

<details>
<summary>答案</summary>

证明上界：
1. 设最优解长度为 $OPT$
2. 最近邻选择的每条边长度 $\leq$ 当前未访问点到已访问点的最小距离
3. 使用概率论证明期望长度 $\leq H_n \cdot OPT$，其中 $H_n = \sum_{i=1}^{n} \frac{1}{i} = O(\log n)$

构造最坏实例：
```
点集排列如下（"梳子"结构）：
     1   3   5   7   ...  2n-1
     |   |   |   |        |
0 -- 2 - 4 - 6 - 8 - ... - 2n
```

- 最优路径：$0 \to 1 \to 2 \to 3 \to ... \to 2n$，长度 $= 2n + 1$
- 最近邻从0开始：$0 \to 2 \to 1 \to 3 \to 4 \to ...$
- 产生"之字形"路径，长度 $\approx n \log n$
- 近似比 $= \Theta(\log n)$
</details>

**习题10.6** 设计一个算法，给定应力场 $\sigma(x,y)$，生成沿主应力方向的各向异性填充路径。讨论数值稳定性。

<details>
<summary>提示</summary>
计算主应力方向场，然后积分流线。注意奇点处理。
</details>

<details>
<summary>答案</summary>

算法设计：

1. **主应力方向计算**：
   $$\theta = \frac{1}{2} \arctan\left(\frac{2\tau_{xy}}{\sigma_{xx} - \sigma_{yy}}\right)$$
   
   当 $\sigma_{xx} = \sigma_{yy}$ 时需要特殊处理（各向同性点）

2. **流线积分**（RK4方法）：
   ```
   给定起点 (x₀, y₀)
   for i = 0 to N:
       k₁ = h * [cos(θ(xᵢ, yᵢ)), sin(θ(xᵢ, yᵢ))]
       k₂ = h * [cos(θ(xᵢ + k₁/2)), sin(θ(xᵢ + k₁/2))]
       k₃ = h * [cos(θ(xᵢ + k₂/2)), sin(θ(xᵢ + k₂/2))]
       k₄ = h * [cos(θ(xᵢ + k₃)), sin(θ(xᵢ + k₃))]
       (xᵢ₊₁, yᵢ₊₁) = (xᵢ, yᵢ) + (k₁ + 2k₂ + 2k₃ + k₄)/6
   ```

3. **数值稳定性问题**：
   - **奇点检测**：$|\nabla \theta| > threshold$ 表示奇点附近
   - **步长自适应**：在高曲率区域减小步长
   - **方向场平滑**：使用Gaussian滤波预处理
   - **终止条件**：到达边界或形成闭环

4. **填充策略**：
   - 从种子点开始生成流线
   - 横向偏置距离 $d$ 生成平行流线
   - 使用Voronoi图确定流线覆盖区域
</details>

**习题10.7**（开放性问题）如何将机器学习方法应用于路径规划优化？设计一个基于强化学习的填充策略。

<details>
<summary>提示</summary>
定义状态空间（当前位置、已填充区域）、动作空间（下一步方向）、奖励函数（效率、质量）。
</details>

<details>
<summary>答案</summary>

强化学习框架设计：

1. **状态空间** $s \in S$：
   - 当前喷头位置 $(x, y)$
   - 局部已填充密度图（如 $32 \times 32$ 网格）
   - 剩余未填充区域的几何特征
   - 当前层的打印时间

2. **动作空间** $a \in A$：
   - 离散：8个方向 + 跳转
   - 连续：$(dx, dy)$ 速度向量

3. **奖励函数** $R(s, a, s')$：
   $$R = \alpha \cdot coverage + \beta \cdot (-time) + \gamma \cdot quality + \delta \cdot smoothness$$
   
   其中：
   - $coverage$：新填充面积
   - $time$：路径长度（负奖励）
   - $quality$：局部密度均匀性
   - $smoothness$：路径曲率惩罚

4. **网络架构**（PPO算法）：
   - 输入：状态编码（CNN处理密度图）
   - Actor网络：输出动作概率分布
   - Critic网络：估计值函数 $V(s)$

5. **训练策略**：
   - 课程学习：从简单形状到复杂形状
   - 域随机化：不同的几何形状
   - 模仿学习：使用传统算法生成的路径作为初始示范

6. **优势**：
   - 自适应不同几何形状
   - 考虑全局最优而非贪心
   - 可学习特定打印机的特性
</details>

## 常见陷阱与错误

1. **浮点精度问题**
   - **错误**：直接使用浮点坐标进行几何运算
   - **正确**：转换为整数坐标或使用精确算术谓词
   - **调试**：检查 orient2d 测试的一致性

2. **自相交处理**
   - **错误**：忽略偏置操作产生的自相交
   - **正确**：使用布尔运算或Vatti算法处理
   - **调试**：可视化中间结果，检查拓扑一致性

3. **Voronoi图退化情况**
   - **错误**：假设Voronoi图总是良好定义的
   - **正确**：处理共圆点、共线点等退化情况
   - **调试**：添加微小扰动打破对称性

4. **空间填充曲线的边界处理**
   - **错误**：直接裁剪Hilbert曲线到任意形状
   - **正确**：自适应四叉树分解，只在内部单元生成曲线
   - **调试**：检查边界单元的连通性

5. **TSP近似算法的初始解**
   - **错误**：随机选择起始点
   - **正确**：选择几何中心或进行多次尝试
   - **调试**：比较不同起始点的结果

6. **梯度填充的离散化误差**
   - **错误**：直接映射连续密度到离散线距
   - **正确**：使用误差扩散或抖动技术
   - **调试**：计算实际密度与目标密度的偏差

## 最佳实践检查清单

### 算法选择
- [ ] 根据几何复杂度选择合适的填充模式
- [ ] 考虑打印时间vs质量的权衡
- [ ] 评估各向同性vs各向异性需求
- [ ] 检查是否需要支撑结构

### 数值鲁棒性
- [ ] 使用整数坐标或精确算术
- [ ] 处理所有退化情况
- [ ] 添加epsilon容差但保持一致性
- [ ] 验证拓扑不变量（欧拉特征数等）

### 性能优化
- [ ] 使用空间索引结构（KD树、R树）
- [ ] 实现增量式算法避免重复计算
- [ ] 并行化独立的区域填充
- [ ] 缓存中间结果（如Voronoi图）

### 质量保证
- [ ] 检查路径连续性（无跳跃）
- [ ] 验证覆盖完整性（无空隙）
- [ ] 限制最大/最小线距
- [ ] 平滑急转弯（限制加速度）

### 工程实践
- [ ] 参数化所有阈值和常数
- [ ] 提供可视化调试输出
- [ ] 记录算法失败情况和原因
- [ ] 支持增量式修改（局部重新规划）
