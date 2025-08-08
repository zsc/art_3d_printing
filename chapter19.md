# 第19章：符号几何与程序化建模

本章探讨3D打印中的符号计算和程序化建模方法。我们将深入研究构造实体几何(CSG)的数学基础、L-系统的分形生成原理、符号计算系统的几何表示、参数化设计的约束求解，以及形状文法的规则系统。这些方法提供了从高层抽象描述到具体几何实现的桥梁，是现代CAD系统和程序化设计的核心。

## 19.1 CSG树与布尔运算

### 19.1.1 CSG的数学基础

构造实体几何(Constructive Solid Geometry)通过基本几何体的集合运算构建复杂形状。设 $\mathcal{S}$ 为三维空间中的实体集合，定义布尔运算：

$$\begin{aligned}
A \cup B &= \{x \in \mathbb{R}^3 : x \in A \text{ 或 } x \in B\} \\
A \cap B &= \{x \in \mathbb{R}^3 : x \in A \text{ 且 } x \in B\} \\
A \setminus B &= \{x \in \mathbb{R}^3 : x \in A \text{ 且 } x \notin B\}
\end{aligned}$$

CSG树的隐式表示通过符号距离场(SDF)实现。对于基本体 $\Omega_i$ 的SDF $f_i(x)$：

$$f_{\cup}(x) = \min(f_A(x), f_B(x))$$
$$f_{\cap}(x) = \max(f_A(x), f_B(x))$$
$$f_{\setminus}(x) = \max(f_A(x), -f_B(x))$$

**基本原语的SDF表示**

球体（中心$c$，半径$r$）：
$$f_{sphere}(x) = \|x - c\| - r$$

长方体（中心$c$，半尺寸$h$）：
$$f_{box}(x) = \|\max(|x - c| - h, 0)\|$$

圆柱体（轴向$z$，半径$r$，高度$h$）：
$$f_{cylinder}(x) = \max(\sqrt{x^2 + y^2} - r, |z| - h/2)$$

圆环体（主半径$R$，管半径$r$）：
$$f_{torus}(x) = \sqrt{(\sqrt{x^2 + y^2} - R)^2 + z^2} - r$$

**光滑布尔运算**

为避免尖锐边缘，使用光滑最小/最大函数：

$$\text{smin}(a, b, k) = -\frac{1}{k}\log(e^{-ka} + e^{-kb})$$

或多项式版本：
$$\text{smin}(a, b, k) = \min(a, b) - \frac{k}{4}\max(0, k - |a - b|)^2$$

光滑并集：
$$f_{\cup}^{smooth}(x) = \text{smin}(f_A(x), f_B(x), k)$$

**梯度计算与法向量**

SDF梯度即为法向量：
$$\nabla f(x) = \vec{n}(x)$$

对于布尔运算：
$$\nabla f_{\cup} = \begin{cases}
\nabla f_A & \text{if } f_A < f_B \\
\nabla f_B & \text{if } f_B < f_A \\
\frac{\nabla f_A + \nabla f_B}{2} & \text{if } f_A = f_B
\end{cases}$$

### 19.1.2 正则化布尔运算

为避免悬挂边和孤立点，使用正则化运算：

$$A \cup^* B = \overline{\text{int}(A \cup B)}$$
$$A \cap^* B = \overline{\text{int}(A \cap B)}$$

其中 $\text{int}(\cdot)$ 表示内部，$\overline{\cdot}$ 表示闭包。

### 19.1.3 BSP树与空间分割

二叉空间分割(BSP)树通过超平面递归分割空间：

$$H(n, d) = \{x \in \mathbb{R}^3 : n \cdot x = d\}$$

分割产生半空间：
$$H^+(n, d) = \{x : n \cdot x \geq d\}, \quad H^-(n, d) = \{x : n \cdot x < d\}$$

BSP树的构建算法复杂度为 $O(n^2)$，查询复杂度为 $O(\log n)$。

**BSP树构建算法**

选择分割平面的启发式：
1. **最小分割数**：选择切割最少多边形的平面
2. **平衡树**：使两侧多边形数量相近
3. **SAH (Surface Area Heuristic)**：
   $$\text{Cost} = C_t + \frac{A_L}{A} \cdot N_L \cdot C_i + \frac{A_R}{A} \cdot N_R \cdot C_i$$
   
   其中$A_L, A_R$为左右子空间表面积，$N_L, N_R$为多边形数，$C_t, C_i$为遍历和相交测试成本。

**多边形分类算法**

给定平面$H(n, d)$和多边形$P$，计算顶点到平面的符号距离：
$$d_i = n \cdot v_i - d$$

分类规则：
- 若所有$d_i > \epsilon$：$P \in H^+$
- 若所有$d_i < -\epsilon$：$P \in H^-$
- 若存在$d_i > \epsilon$且$d_j < -\epsilon$：$P$被分割
- 若所有$|d_i| \leq \epsilon$：$P$共面

**分割算法（Sutherland-Hodgman）**

对于被分割的多边形，计算交点：
$$t = \frac{d - n \cdot v_1}{n \cdot (v_2 - v_1)}$$
$$p_{intersect} = v_1 + t(v_2 - v_1)$$

生成两个新多边形，保持顶点顺序和法向量一致性。

### 19.1.4 鲁棒性与精度问题

浮点运算的数值误差导致拓扑不一致。使用区间算术保证鲁棒性：

$$[a, b] \oplus [c, d] = [a+c-\epsilon, b+d+\epsilon]$$

其中 $\epsilon$ 为机器精度。采用精确几何计算(CGAL的Exact_predicates_exact_constructions_kernel)避免退化情况。

### 19.1.5 光线追踪与CSG

CSG树的光线求交通过递归遍历：

```
对于并集：取最近交点
对于交集：取重叠区间
对于差集：A的区间减去B的区间
```

区间合并算法的时间复杂度为 $O(n \log n)$。

**光线-CSG相交算法**

光线参数方程：$r(t) = o + td$，其中$o$为起点，$d$为方向。

对于每个原语，计算进入点$t_{in}$和离开点$t_{out}$：

球体相交：
$$\|o + td - c\|^2 = r^2$$
$$t^2 + 2t(d \cdot (o-c)) + \|o-c\|^2 - r^2 = 0$$

使用判别式$\Delta = b^2 - 4ac$判断相交情况。

**区间运算规则**

设$[a_1, b_1]$和$[a_2, b_2]$为两个相交区间：

并集：$\text{Union}([a_1, b_1], [a_2, b_2]) = [\min(a_1, a_2), \max(b_1, b_2)]$（若重叠）

交集：$\text{Intersect}([a_1, b_1], [a_2, b_2]) = [\max(a_1, a_2), \min(b_1, b_2)]$（若非空）

差集：$\text{Difference}([a_1, b_1], [a_2, b_2])$可能产生0、1或2个区间

**优化策略**

1. **包围盒剔除**：先测试AABB，避免复杂计算
2. **空间一致性**：利用相邻光线的相似性
3. **层次包围盒**：构建BVH加速结构
4. **懒惰求值**：只计算必要的子树

## 19.2 L-系统与分形几何

### 19.2.1 L-系统的形式定义

L-系统是一个四元组 $G = (V, \omega, P, \delta)$：
- $V$：字母表（符号集合）
- $\omega \in V^+$：公理（初始串）
- $P \subset V \times V^*$：产生式规则
- $\delta$：终结符的几何解释

确定性上下文无关L-系统(D0L)的产生式：
$$a \rightarrow \chi, \quad a \in V, \chi \in V^*$$

### 19.2.2 龟图形解释

龟状态 $s = (x, y, z, \vec{H}, \vec{L}, \vec{U})$，其中位置 $(x,y,z)$，方向基 $(\vec{H}, \vec{L}, \vec{U})$。

符号的几何操作：
- $F$：前进并画线，$\vec{p}_{new} = \vec{p} + d\vec{H}$
- $+/-$：偏航角旋转 $\pm\delta$
- $\wedge/\&$：俯仰角旋转 $\pm\delta$
- $\backslash//$：滚转角旋转 $\pm\delta$
- $[/]$：压栈/出栈

旋转矩阵：
$$R_{\vec{U}}(\theta) = \begin{pmatrix}
\cos\theta + u_x^2(1-\cos\theta) & u_xu_y(1-\cos\theta) - u_z\sin\theta & u_xu_z(1-\cos\theta) + u_y\sin\theta \\
u_yu_x(1-\cos\theta) + u_z\sin\theta & \cos\theta + u_y^2(1-\cos\theta) & u_yu_z(1-\cos\theta) - u_x\sin\theta \\
u_zu_x(1-\cos\theta) - u_y\sin\theta & u_zu_y(1-\cos\theta) + u_x\sin\theta & \cos\theta + u_z^2(1-\cos\theta)
\end{pmatrix}$$

**扩展符号集**

几何符号：
- $f$：前进不画线
- $G$：画到指定点
- $.$：画点
- $\{/\}$：开始/结束多边形
- $|$：旋转180度

颜色和材质：
- $;(n)$：设置材质索引
- $!(w)$：设置线宽
- $'$：增加颜色索引

**向量操作与对齐**

重力向量对齐（模拟向地性/向光性）：
$$\vec{H}_{new} = \vec{H} + \epsilon \vec{g}$$
$$\vec{H}_{new} = \frac{\vec{H}_{new}}{\|\vec{H}_{new}\|}$$

其中$\epsilon$为向性强度，$\vec{g}$为重力/光照方向。

保持正交基：使用Gram-Schmidt正交化
$$\vec{L} = \vec{L} - (\vec{L} \cdot \vec{H})\vec{H}$$
$$\vec{U} = \vec{H} \times \vec{L}$$

### 19.2.3 参数化L-系统

参数化产生式允许连续变量：
$$A(t) : t > 5 \rightarrow B(t/2)C(t-1)$$

条件判断和参数传递实现生长模型：
$$A(l, w) : l > l_{min} \rightarrow F(l)[+(\theta)A(l \cdot r_1, w \cdot q)][-(\theta)A(l \cdot r_2, w \cdot (1-q))]$$

其中 $r_1, r_2$ 为长度衰减因子，$q$ 为分支比例。

### 19.2.4 随机L-系统

引入概率产生式：
$$A \xrightarrow{p_1} \alpha_1, \quad A \xrightarrow{p_2} \alpha_2, \quad \sum_i p_i = 1$$

随机性通过伪随机数生成器控制，种子固定保证可重复性。

**概率分布设计**

生物学动机的概率模型：
- **分支概率**：$P(branch) = \sigma(a \cdot age - b)$，其中$\sigma$为sigmoid函数
- **生长速率**：$growth \sim \mathcal{N}(\mu, \sigma^2)$，正态分布模拟自然变异
- **死亡概率**：$P(death) = 1 - e^{-\lambda t}$，指数分布

**上下文敏感随机L-系统**

产生式依赖于前驱和后继：
$$\langle L \rangle A \langle R \rangle \xrightarrow{p} \omega$$

其中$L$和$R$分别为左右上下文。

**马尔可夫链L-系统**

状态转移概率矩阵：
$$P_{ij} = P(S_{t+1} = j | S_t = i)$$

稳态分布满足：
$$\pi = \pi P$$

用于模拟长期生长模式的统计特性。

### 19.2.5 分形维数计算

Hausdorff维数定义：
$$d_H = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}$$

其中 $N(\epsilon)$ 为覆盖集合所需的半径为 $\epsilon$ 的球数。

对于自相似分形，盒计数维数：
$$d_B = \frac{\log N}{\log r}$$

其中 $N$ 为相似副本数，$r$ 为缩放比。

**计算方法**

1. **盒计数法实现**：
   - 将空间划分为边长$\epsilon$的立方体网格
   - 计数包含分形部分的盒子数$N(\epsilon)$
   - 对不同$\epsilon$值，拟合$\log N(\epsilon)$对$\log(1/\epsilon)$的斜率

2. **相关维数**：
   $$d_2 = \lim_{r \to 0} \frac{\log C(r)}{\log r}$$
   其中$C(r) = \frac{1}{N^2}\sum_{i,j} \Theta(r - \|x_i - x_j\|)$

3. **信息维数**：
   $$d_1 = \lim_{\epsilon \to 0} \frac{\sum_i p_i \log p_i}{\log \epsilon}$$
   其中$p_i$为第$i$个盒子中点的比例

**L-系统生成分形的维数**

Koch曲线（$F \rightarrow F+F--F+F$，角度60°）：
$$d = \frac{\log 4}{\log 3} \approx 1.262$$

Sierpinski三角形：
$$d = \frac{\log 3}{\log 2} \approx 1.585$$

植物分支结构的经验维数：
- 树冠：$d \approx 2.5-2.8$
- 根系：$d \approx 1.4-1.7$
- 血管网络：$d \approx 2.3-2.7$

## 19.3 Mathematica Graphics3D原理

### 19.3.1 符号几何表示

Mathematica使用符号表达式树表示几何：

```
Graphics3D[{
  Sphere[{0,0,0}, 1],
  Cylinder[{{0,0,-1}, {0,0,1}}, 0.5]
}]
```

内部表示为S-表达式：
$$\text{Graphics3D}[\text{List}[\text{Sphere}[\ldots], \text{Cylinder}[\ldots]]]$$

### 19.3.2 符号求值与简化

几何运算的符号简化规则：
$$\text{Translate}[\text{Translate}[g, v_1], v_2] \rightarrow \text{Translate}[g, v_1 + v_2]$$
$$\text{Scale}[\text{Scale}[g, s_1], s_2] \rightarrow \text{Scale}[g, s_1 \cdot s_2]$$

模式匹配实现自动优化。

### 19.3.3 区域函数与隐式曲面

RegionFunction定义隐式几何：
$$\text{RegionFunction} \rightarrow (f(x,y,z) \leq 0)$$

多项式隐式曲面的代数几何处理：
$$f(x,y,z) = \sum_{i+j+k \leq n} a_{ijk}x^i y^j z^k$$

使用Gröbner基计算交线和奇异点。

### 19.3.4 网格生成与离散化

DiscretizeRegion的自适应网格算法：
1. 八叉树空间分割
2. Marching Cubes提取等值面
3. Delaunay优化提高网格质量

误差估计：
$$\|f - f_h\|_{L^2} \leq Ch^{p+1}\|f\|_{H^{p+1}}$$

其中 $h$ 为网格尺寸，$p$ 为插值阶数。

### 19.3.5 符号积分与物理量计算

体积计算通过符号积分：
$$V = \iiint_{\Omega} dx\,dy\,dz$$

质心：
$$\vec{c} = \frac{1}{V}\iiint_{\Omega} \vec{r}\,dV$$

惯性张量：
$$I_{ij} = \iiint_{\Omega} \rho(r^2\delta_{ij} - r_ir_j)dV$$

使用蒙特卡洛积分处理复杂区域。

## 19.4 参数化设计与约束求解

### 19.4.1 几何约束系统

约束类型：
- 距离约束：$\|p_1 - p_2\| = d$
- 角度约束：$\cos^{-1}(\vec{v}_1 \cdot \vec{v}_2) = \theta$
- 相切约束：$\vec{n}_1 \cdot \vec{n}_2 = 0$
- 共面约束：$(p - p_0) \cdot \vec{n} = 0$

约束系统的代数表示：
$$F(x) = 0, \quad F: \mathbb{R}^n \rightarrow \mathbb{R}^m$$

### 19.4.2 自由度分析

使用图论分析结构自由度。约束图 $G = (V, E)$：
- 顶点：几何元素
- 边：约束关系

Laman定理：2D中刚性的充要条件是 $|E| = 2|V| - 3$ 且任意子图满足 $|E'| \leq 2|V'| - 3$。

3D推广（Maxwell计数）：
$$\text{DOF} = 6n - \sum_i c_i$$

其中 $n$ 为刚体数，$c_i$ 为约束 $i$ 移除的自由度。

### 19.4.3 数值求解方法

Newton-Raphson迭代：
$$x_{k+1} = x_k - J_F(x_k)^{-1}F(x_k)$$

雅可比矩阵：
$$J_F = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}$$

欠定系统使用最小二乘：
$$\min_x \|F(x)\|^2 = \min_x F(x)^TF(x)$$

过定系统使用Moore-Penrose伪逆。

### 19.4.4 符号求解与Gröbner基

多项式约束系统的Gröbner基方法：

给定理想 $I = \langle f_1, \ldots, f_m \rangle$，Gröbner基 $G = \{g_1, \ldots, g_k\}$ 满足：
$$\text{LT}(I) = \langle \text{LT}(g_1), \ldots, \text{LT}(g_k) \rangle$$

Buchberger算法计算Gröbner基，复杂度最坏为双指数。

消元理想求解：
$$I_k = I \cap k[x_{k+1}, \ldots, x_n]$$

### 19.4.5 连续性与导数约束

$C^0$连续：$f(t_0^-) = f(t_0^+)$
$C^1$连续：$f'(t_0^-) = f'(t_0^+)$
$G^1$连续：$\exists \lambda > 0, f'(t_0^-) = \lambda f'(t_0^+)$

曲率连续($C^2$)约束：
$$\kappa = \frac{\|f' \times f''\|}{\|f'\|^3}$$

使用B-spline保证参数连续性。

## 19.5 形状文法与规则系统

### 19.5.1 形状文法定义

形状文法 $SG = (S, L, R, I)$：
- $S$：形状词汇表
- $L$：标签集
- $R$：形状规则 $\alpha \rightarrow \beta$
- $I$：初始形状

形状规则应用需要子形状匹配：
$$\text{match}(\alpha, S) = \{T : T(\alpha) \subseteq S\}$$

其中 $T$ 为允许的变换（平移、旋转、缩放）。

### 19.5.2 参数化形状文法

引入参数控制：
$$\alpha(p_1, \ldots, p_k) \rightarrow \beta(f_1(p_1, \ldots, p_k), \ldots, f_m(p_1, \ldots, p_k))$$

参数传递实现上下文敏感性。条件规则：
$$\alpha : P(p_1, \ldots, p_k) \rightarrow \beta$$

其中 $P$ 为谓词。

### 19.5.3 建筑生成文法

Split文法操作：
$$\text{Split}_x(S, r_1:r_2:\ldots:r_n) \rightarrow S_1, S_2, \ldots, S_n$$

其中 $r_i$ 为相对或绝对尺寸。

CGA Shape的主要操作：
- Extrude(h)：挤出高度h
- Comp(f)：组件分解（面、边、顶点）
- Subdiv(n)：细分
- Repeat(s)：重复模式

### 19.5.4 属性文法与语义

属性分为：
- 综合属性(Synthesized)：自下而上传递
- 继承属性(Inherited)：自上而下传递

属性计算的依赖图必须无环。使用拓扑排序确定求值顺序。

语义函数：
$$\sigma: \text{Rules} \times \text{Attributes} \rightarrow \text{Values}$$

### 19.5.5 随机文法与概率模型

概率上下文无关文法(PCFG)：
$$A \xrightarrow{p} \alpha, \quad \sum_{\alpha} P(A \rightarrow \alpha) = 1$$

最大似然参数估计：
$$\hat{p}(A \rightarrow \alpha) = \frac{\text{Count}(A \rightarrow \alpha)}{\text{Count}(A)}$$

使用Inside-Outside算法进行无监督学习。

## 本章小结

本章系统介绍了符号几何与程序化建模的核心数学方法：

**CSG与布尔运算**：
- 正则化布尔运算：$A \cup^* B = \overline{\text{int}(A \cup B)}$
- SDF布尔运算：$f_{\cup} = \min(f_A, f_B)$, $f_{\cap} = \max(f_A, f_B)$
- BSP树空间分割，查询复杂度$O(\log n)$

**L-系统与分形**：
- 形式定义：$G = (V, \omega, P, \delta)$
- 龟图形解释与3D旋转矩阵
- 分形维数：$d_H = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}$
- 参数化与随机L-系统

**Mathematica符号计算**：
- S-表达式树表示
- 符号简化规则与模式匹配
- RegionFunction隐式曲面
- 符号积分计算物理量

**参数化设计**：
- 几何约束系统：$F(x) = 0$
- 自由度分析：Laman定理
- Newton-Raphson数值求解
- Gröbner基符号求解

**形状文法**：
- 形式定义：$SG = (S, L, R, I)$
- 参数化规则与条件判断
- Split文法与CGA Shape
- 属性文法与PCFG

关键数学工具：集合论、群论、代数几何、形式语言理论、计算复杂度分析。

## 练习题

### 基础题

**练习19.1** CSG布尔运算性质  
证明对于符号距离场，以下等式成立：
$$f_{A \setminus B}(x) = \max(f_A(x), -f_B(x))$$

*提示：考虑点$x$在$A \setminus B$内部、边界和外部的情况。*

<details>
<summary>答案</summary>

对于差集$A \setminus B = \{x : x \in A \wedge x \notin B\}$：

1. 若$x \in A \setminus B$内部：
   - $f_A(x) < 0$（在A内部）
   - $f_B(x) > 0$（在B外部）
   - $-f_B(x) < 0$
   - $\max(f_A(x), -f_B(x)) < 0$（在结果内部）

2. 若$x$在边界上：
   - 要么$f_A(x) = 0$（A的边界）
   - 要么$f_B(x) = 0$（B的边界，此时$-f_B(x) = 0$）
   - $\max(f_A(x), -f_B(x)) = 0$

3. 若$x$在外部：
   - 要么$f_A(x) > 0$（A外部）
   - 要么$f_B(x) < 0$（B内部，此时$-f_B(x) > 0$）
   - $\max(f_A(x), -f_B(x)) > 0$

因此公式正确表示了差集的SDF。
</details>

**练习19.2** L-系统串长度增长  
给定D0L系统：
- 公理：$\omega = A$
- 规则：$A \rightarrow AB$, $B \rightarrow A$

计算第$n$次迭代后串的长度$L_n$，并找出通项公式。

*提示：建立递推关系，识别斐波那契数列。*

<details>
<summary>答案</summary>

设$a_n$和$b_n$分别为第$n$次迭代后A和B的数量。

初始：$a_0 = 1, b_0 = 0$

递推关系：
- $a_{n+1} = a_n + b_n$（A产生A，B产生A）
- $b_{n+1} = a_n$（只有A产生B）

串长度：$L_n = a_n + b_n$

推导：
- $L_0 = 1$
- $L_1 = 2$
- $L_{n+1} = a_{n+1} + b_{n+1} = (a_n + b_n) + a_n = L_n + a_n$

由于$a_n = L_{n-1}$（从递推关系可得），所以：
$$L_{n+1} = L_n + L_{n-1}$$

这是斐波那契数列，通项公式：
$$L_n = F_{n+2} = \frac{\phi^{n+2} - \psi^{n+2}}{\sqrt{5}}$$

其中$\phi = \frac{1+\sqrt{5}}{2}$，$\psi = \frac{1-\sqrt{5}}{2}$。
</details>

**练习19.3** 约束系统自由度  
一个2D机构包含4个刚体通过5个转动副连接成闭链。计算系统的自由度，并判断是否过约束。

*提示：使用Grübler-Kutzbach公式。*

<details>
<summary>答案</summary>

Grübler-Kutzbach公式（2D）：
$$F = 3(n-1) - 2j_1 - j_2$$

其中：
- $n = 4$（刚体数，包括固定基座）
- $j_1 = 5$（转动副，每个约束2个自由度）
- $j_2 = 0$（没有移动副）

计算：
$$F = 3(4-1) - 2 \times 5 = 9 - 10 = -1$$

自由度为-1，系统过约束。这意味着并非所有约束都是独立的，存在一个冗余约束。实际上这是一个四杆机构加一个额外约束，需要特殊的几何关系才能装配。
</details>

### 挑战题

**练习19.4** CSG树优化  
给定CSG表达式：$(A \cup B) \cap (C \cup D) \cap E$，其中对象的包围盒体积为：$V_A = 1$, $V_B = 4$, $V_C = 2$, $V_D = 3$, $V_E = 5$。设计一个启发式算法重新排列CSG树以最小化光线追踪的平均计算成本。

*提示：考虑提前剔除和包围盒测试的成本。*

<details>
<summary>答案</summary>

优化策略：
1. 交集运算应优先测试小体积对象（提前剔除）
2. 并集运算顺序影响较小

原始树的期望测试次数（假设均匀分布的光线）：
- 必须测试$(A \cup B)$和$(C \cup D)$的包围盒
- 包围盒体积：$V_{A \cup B} \approx \max(1,4) = 4$，$V_{C \cup D} \approx \max(2,3) = 3$

优化方案：
1. 将E提前，因为它可能剔除最多光线
2. 重排为：$E \cap ((A \cup B) \cap (C \cup D))$

更优方案（考虑分配律）：
$(A \cap C \cap E) \cup (A \cap D \cap E) \cup (B \cap C \cap E) \cup (B \cap D \cap E)$

但这会增加树的大小。实践中的平衡方案：
$$E \cap (A \cup B) \cap (C \cup D)$$

启发式算法：
1. 对交集节点，按包围盒体积升序排列子节点
2. 对并集节点，考虑空间相关性分组
3. 使用动态规划优化深度较大的树
</details>

**练习19.5** L-系统逆向工程  
观察到一个分形图案具有以下特征：
- 每次迭代，线段数量变为原来的4倍
- 总长度变为原来的$4/3$倍
- 具有自相似性，分为4个子部分

推导可能的L-系统规则，并计算其分形维数。

*提示：这是Koch曲线的变体。*

<details>
<summary>答案</summary>

根据特征分析：
- 线段数$N = 4$
- 缩放比$r = 3/4$（因为4段总长为原来的4/3，每段为1/3）

可能的L-系统：
- 公理：$F$
- 规则：$F \rightarrow F+F--F+F$
- 角度：$\delta = 60°$

这产生Koch曲线。验证：
- 原始线段替换为4段
- 每段长度为原来的1/3
- 总长度：$4 \times \frac{1}{3} = \frac{4}{3}$✓

分形维数计算：
$$d = \frac{\log N}{\log(1/r)} = \frac{\log 4}{\log 3} = \frac{2\log 2}{\log 3} \approx 1.262$$

物理意义：维数介于1（线）和2（面）之间，表示曲线填充空间的程度。

其他可能的规则（产生不同形状但相同维数）：
- $F \rightarrow F-F++F-F$（另一种Koch变体）
- $F \rightarrow FF+F+F-F$（需调整角度）
</details>

**练习19.6** 形状文法设计  
设计一个参数化形状文法生成中国传统建筑的屋顶层级结构。要求：
1. 支持单檐、重檐、三重檐
2. 檐的宽度递减
3. 包含翘角参数

*提示：使用递归规则和参数传递。*

<details>
<summary>答案</summary>

形状文法定义：

初始形状：
```
Roof(n, w, h, curve)
```

参数：
- n：檐数
- w：底层宽度
- h：层高
- curve：翘角曲率

规则：

R1：多重檐递归
```
Roof(n, w, h, c) : n > 1 →
  Eave(w, c) 
  T(0, 0, h)
  Roof(n-1, w*0.8, h*0.9, c*1.1)
```

R2：终止条件
```
Roof(1, w, h, c) →
  Eave(w, c)
  RoofTop(w*0.5)
```

R3：檐生成
```
Eave(w, c) →
  Split_x(-w/2, w/2) {
    CornerL(c) | Middle(w-2c) | CornerR(c)
  }
```

R4：翘角
```
Corner*(c) →
  Bezier((0,0), (c/3, c/2), (c, c))
```

参数约束：
- $0.7 \leq \text{width\_ratio} \leq 0.85$
- $1.0 \leq \text{curve\_ratio} \leq 1.2$
- $h_{min} = 0.3w$

这个文法可以生成符合传统比例的多层檐结构，通过调整参数产生不同风格（如南方vs北方建筑）。
</details>

**练习19.7** 约束求解收敛性  
考虑平面四杆机构的位置分析，给定输入角$\theta$，求解输出角$\phi$。约束方程：
$$l_1\cos\theta + l_2\cos\alpha = l_3\cos\phi + l_4$$
$$l_1\sin\theta + l_2\sin\alpha = l_3\sin\phi$$

其中$\alpha$是耦合角。分析Newton-Raphson方法的收敛条件。

*提示：计算雅可比矩阵的条件数。*

<details>
<summary>答案</summary>

消去$\alpha$得到关于$\phi$的方程：
$$f(\phi) = (l_1\cos\theta - l_3\cos\phi - l_4)^2 + (l_1\sin\theta - l_3\sin\phi)^2 - l_2^2 = 0$$

展开：
$$f(\phi) = A\cos\phi + B\sin\phi + C = 0$$

其中：
- $A = -2l_1l_3\cos\theta + 2l_3l_4$
- $B = -2l_1l_3\sin\theta$
- $C = l_1^2 + l_3^2 + l_4^2 - l_2^2 - 2l_1l_4\cos\theta$

Newton-Raphson迭代：
$$\phi_{n+1} = \phi_n - \frac{f(\phi_n)}{f'(\phi_n)}$$

导数：
$$f'(\phi) = -A\sin\phi + B\cos\phi$$

收敛条件：
1. **存在性**：$|C| \leq \sqrt{A^2 + B^2}$（有实数解）
2. **唯一性**：避免$f'(\phi) = 0$的点（奇异配置）
3. **收敛速度**：条件数$\kappa = \frac{\max|f''|}{min|f'|}$

奇异配置发生在：
$$\tan\phi = \frac{B}{A} = \frac{l_1\sin\theta}{l_1\cos\theta - l_4}$$

这对应于机构的死点位置。在死点附近，雅可比矩阵接近奇异，收敛变慢或发散。

改进策略：
1. 使用延拓法从已知解开始
2. 加入阻尼：$\phi_{n+1} = \phi_n - \lambda\frac{f}{f'}$，$0 < \lambda \leq 1$
3. 切换到区间方法（如二分法）保证全局收敛
</details>

**练习19.8** 分形维数的多重分形推广  
对于一个非均匀分形（如DLA聚集体），单一分形维数不足以描述其复杂性。定义广义维数：
$$D_q = \frac{1}{1-q}\lim_{\epsilon \to 0}\frac{\log\sum_i p_i^q}{\log\epsilon}$$

其中$p_i$是第$i$个盒子中的测度。证明$D_0$是盒维数，$D_1$是信息维数，$D_2$是关联维数。

*提示：使用L'Hôpital法则处理$q=1$的情况。*

<details>
<summary>答案</summary>

**情况1：$q=0$**
$$D_0 = \lim_{\epsilon \to 0}\frac{\log\sum_i p_i^0}{\log\epsilon} = \lim_{\epsilon \to 0}\frac{\log N(\epsilon)}{\log(1/\epsilon)}$$

这是盒计数维数，$N(\epsilon)$是非空盒子数。

**情况2：$q=1$（使用L'Hôpital）**

令$S_q = \sum_i p_i^q$，对$q$求导：
$$\frac{\partial S_q}{\partial q} = \sum_i p_i^q \log p_i$$

在$q=1$：
$$D_1 = \lim_{\epsilon \to 0}\frac{-\sum_i p_i\log p_i}{\log(1/\epsilon)}$$

这是信息维数，分子是Shannon熵。

**情况3：$q=2$**
$$D_2 = \lim_{\epsilon \to 0}\frac{\log\sum_i p_i^2}{-\log\epsilon}$$

$\sum_i p_i^2$是两点落在同一盒子的概率，因此$D_2$是关联维数。

**物理意义**：
- $D_0$：几何结构
- $D_1$：信息分布
- $D_2$：动力学关联
- $D_q, q<0$：稀有事件
- $D_q, q>0$：密集区域

对于均匀分形，所有$D_q$相等；对于多重分形，$D_q$是$q$的递减函数。

**谱函数**：
$$f(\alpha) = q\alpha - (q-1)D_q$$

其中$\alpha = \frac{d[(q-1)D_q]}{dq}$是局部Hölder指数。
</details>

## 常见陷阱与错误

### CSG布尔运算
1. **数值鲁棒性**：浮点误差导致拓扑不一致
   - 使用精确算术或区间算术
   - 实现epsilon-几何决策
   
2. **自相交处理**：非流形结果
   - 进行正则化布尔运算
   - 检测并修复自相交

3. **性能瓶颈**：复杂CSG树遍历缓慢
   - 使用空间数据结构加速
   - 实现CSG树优化算法

### L-系统实现
1. **内存爆炸**：字符串指数增长
   - 使用延迟求值
   - 实现增量渲染

2. **浮点累积误差**：长分支末端位置偏差
   - 使用双精度或任意精度
   - 定期重新正交化方向矩阵

3. **随机性不可重复**：每次运行结果不同
   - 固定随机种子
   - 使用确定性伪随机数生成器

### 约束求解
1. **过约束检测失败**：系统无解但未检测到
   - 进行自由度分析
   - 使用符号方法验证可解性

2. **收敛到错误解**：多解情况选择了非预期解
   - 提供良好初值
   - 使用全局优化方法

3. **数值不稳定**：近奇异配置
   - 添加正则化项
   - 使用SVD求解

### 形状文法
1. **规则冲突**：多个规则同时适用
   - 定义优先级
   - 使用冲突消解策略

2. **无限递归**：终止条件缺失
   - 添加深度限制
   - 确保递归参数单调变化

3. **计算复杂度爆炸**：组合爆炸
   - 实现剪枝策略
   - 使用启发式搜索

## 最佳实践检查清单

### 设计审查要点

#### CSG建模
- [ ] 布尔运算是否正则化？
- [ ] 是否处理了所有退化情况？
- [ ] CSG树是否优化（平衡、重排）？
- [ ] 是否实现了包围盒层次结构？
- [ ] 数值鲁棒性是否经过测试？

#### L-系统设计
- [ ] 规则是否产生预期的拓扑结构？
- [ ] 参数范围是否合理？
- [ ] 是否避免了指数增长？
- [ ] 分形维数是否计算正确？
- [ ] 渲染性能是否可接受？

#### 约束系统
- [ ] 自由度分析是否正确？
- [ ] 是否识别了所有奇异配置？
- [ ] 数值方法是否稳定收敛？
- [ ] 是否提供了多解选择机制？
- [ ] 容差设置是否合理？

#### 形状文法
- [ ] 文法是否完备（覆盖所有情况）？
- [ ] 规则是否一致（无冲突）？
- [ ] 终止条件是否明确？
- [ ] 参数传递是否正确？
- [ ] 生成结果是否满足约束？

#### 性能优化
- [ ] 是否使用了适当的数据结构？
- [ ] 是否实现了必要的缓存？
- [ ] 是否进行了并行化？
- [ ] 内存使用是否优化？
- [ ] 是否有增量计算策略？

#### 验证测试
- [ ] 是否有单元测试覆盖边界情况？
- [ ] 是否进行了数值稳定性测试？
- [ ] 是否验证了算法复杂度？
- [ ] 是否测试了大规模输入？
- [ ] 是否比较了不同方法的结果？
