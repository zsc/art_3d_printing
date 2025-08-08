# 第11章：误差分析与补偿

3D打印过程中的误差来源复杂多样，从CAD模型到最终打印件的每个环节都会引入不同类型的误差。本章系统介绍误差的数学建模、传播分析、预测补偿以及鲁棒设计方法。我们将深入探讨离散化带来的几何误差、热物理过程导致的变形、制造不确定性的量化方法，以及如何通过优化理论设计对误差鲁棒的结构。特别地，我们将介绍多项式混沌展开(PCE)、贝叶斯优化等现代数学工具在3D打印质量控制中的应用。

## 11.1 离散化误差与阶梯效应

### 11.1.1 STL离散化的几何误差分析

当连续曲面 $S$ 被三角网格 $M$ 近似时，主要误差来源于弦高误差(chordal error)。对于曲率半径为 $\rho$ 的局部曲面，边长为 $h$ 的三角形产生的最大弦高误差为：

$$\epsilon_{chord} = \rho - \sqrt{\rho^2 - \frac{h^2}{4}} \approx \frac{h^2}{8\rho}$$

这表明误差与网格尺寸的平方成正比，与曲率成反比。对于整个模型，Hausdorff距离提供了全局误差度量：

$$d_H(S, M) = \max\left\{\sup_{p \in S} d(p, M), \sup_{q \in M} d(q, S)\right\}$$

其中 $d(p, M) = \inf_{q \in M} \|p - q\|$ 是点到网格的距离。

#### 误差的方向性分析

弦高误差具有明显的方向性特征。对于法向量为 $\mathbf{n}$ 的曲面片，误差主要沿法向分布：

$$\mathbf{e}_{chord} = \epsilon_{chord} \cdot \mathbf{n} + O(h^3)$$

在主曲率方向 $\mathbf{e}_1, \mathbf{e}_2$ 上，误差分别为：

$$\epsilon_1 = \frac{h^2 \cos^2\phi}{8R_1}, \quad \epsilon_2 = \frac{h^2 \sin^2\phi}{8R_2}$$

其中 $\phi$ 是网格边与第一主方向的夹角，$R_1, R_2$ 是主曲率半径。

#### 自适应网格生成策略

基于曲率的网格密度函数：

$$\rho_{mesh}(\mathbf{x}) = \rho_0 \cdot \max\left\{1, \frac{|\kappa_1(\mathbf{x})| + |\kappa_2(\mathbf{x})|}{2\kappa_{ref}}\right\}^{-\alpha}$$

其中 $\rho_0$ 是基准密度，$\kappa_{ref}$ 是参考曲率，$\alpha \in [0.5, 1]$ 控制自适应强度。使用Delaunay细化算法时，终止准则为：

$$\max_{T \in M} \frac{|T|}{Q(T)} < \tau_{quality}$$

其中 $|T|$ 是三角形面积，$Q(T)$ 是质量度量（如外接圆半径与内切圆半径之比）。

### 11.1.2 层高与阶梯效应的数学模型

对于倾斜角度为 $\theta$ 的表面，层高 $h$ 导致的阶梯误差(staircase error)为：

$$\epsilon_{stair}(\theta) = \begin{cases}
h \sin \theta & \text{if } 0 \leq \theta < \pi/2 \\
h & \text{if } \theta = \pi/2
\end{cases}$$

考虑表面法向量 $\mathbf{n} = (n_x, n_y, n_z)$，垂直误差分布为：

$$\epsilon_{\perp} = h \cdot |n_z| = h \cdot |\mathbf{n} \cdot \mathbf{e}_z|$$

对于复杂几何，我们可以定义加权平均误差：

$$\bar{\epsilon} = \frac{1}{A} \int_S \epsilon_{stair}(\theta(x,y)) \, dA$$

其中 $A$ 是表面积，$\theta(x,y)$ 是局部倾斜角。

### 11.1.3 自适应切片的误差控制

为了控制阶梯误差在容差 $\tau$ 内，层高应满足：

$$h_i \leq \frac{\tau}{\max_{\theta \in [\theta_{min}, \theta_{max}]} \sin \theta}$$

其中 $[\theta_{min}, \theta_{max}]$ 是第 $i$ 层的倾斜角范围。基于曲率的自适应切片策略使用：

$$h_i = \min\left\{\frac{\tau}{\kappa_{max}}, h_{max}\right\}$$

其中 $\kappa_{max}$ 是局部最大曲率，$h_{max}$ 是设备限制的最大层高。

#### 多目标优化的层高选择

实际应用中需要平衡多个目标：打印时间、表面质量、结构强度。多目标优化问题：

$$\begin{aligned}
\min_{\{h_i\}} \quad & \left(T_{print} = \sum_i \frac{V_i}{v \cdot w \cdot h_i}, \quad E_{surface} = \sum_i A_i \cdot h_i \sin\theta_i\right) \\
\text{s.t.} \quad & h_{min} \leq h_i \leq h_{max} \\
& |h_i - h_{i-1}| \leq \Delta h_{max}
\end{aligned}$$

其中 $V_i$ 是第 $i$ 层体积，$v$ 是打印速度，$w$ 是线宽，$A_i$ 是表面积。使用Pareto前沿分析：

$$\mathcal{P} = \{(T, E) : \nexists (T', E') \text{ s.t. } T' \leq T \land E' \leq E \land (T', E') \neq (T, E)\}$$

#### 局部特征保持策略

对于细节特征（如尖角、薄壁），需要特殊处理：

$$h_{feature} = \min\left\{\frac{t_{min}}{n_{layers}}, \frac{r_{corner}}{\tan(\alpha_{overhang})}\right\}$$

其中 $t_{min}$ 是最小特征厚度，$n_{layers} \geq 3$ 确保特征强度，$r_{corner}$ 是尖角半径，$\alpha_{overhang}$ 是悬垂角限制。

### 11.1.4 体素化误差与分辨率分析

对于体素尺寸 $\Delta x$ 的离散化，表面重建的误差界为：

$$\epsilon_{voxel} \leq \sqrt{3} \cdot \Delta x$$

这是因为体素中心到其顶点的最大距离为对角线长度。采用符号距离场(SDF)表示时，三线性插值的误差为：

$$\epsilon_{SDF} = O(\Delta x^2)$$

当使用高阶插值(如三次样条)时，误差可以降至 $O(\Delta x^4)$。

#### 各向异性体素化

对于具有主导方向的几何，使用各向异性体素：

$$\Delta \mathbf{x} = (\Delta x_1, \Delta x_2, \Delta x_3) = \Delta x_{base} \cdot (\lambda_1^{-1/2}, \lambda_2^{-1/2}, \lambda_3^{-1/2})$$

其中 $\lambda_i$ 是几何张量 $\mathbf{G} = \int_V \nabla \phi \otimes \nabla \phi \, dV$ 的特征值，$\phi$ 是水平集函数。

#### 八叉树自适应细分

使用八叉树结构实现空间自适应分辨率。细分准则基于局部特征尺度：

$$\text{subdivide}(cell) = \begin{cases}
true & \text{if } \max(|\nabla \phi|, |\nabla^2 \phi| \cdot \Delta x) > \tau_{adapt} \\
false & \text{otherwise}
\end{cases}$$

内存复杂度从 $O(N^3)$ 降至 $O(N^2 \log N)$，其中 $N$ 是最细分辨率。

#### 亚体素精度技术

使用双重轮廓(Dual Contouring)方法，在每个体素内部放置一个顶点，位置通过最小化误差函数确定：

$$\mathbf{v}^* = \arg\min_{\mathbf{v}} \sum_{i} \|\mathbf{n}_i \cdot (\mathbf{v} - \mathbf{p}_i)\|^2$$

其中 $\mathbf{p}_i$ 是边界采样点，$\mathbf{n}_i$ 是对应法向量。这可以达到亚体素精度的表面重建。

## 11.2 热变形预测与补偿

### 11.2.1 热弹性变形的有限元建模

打印过程中的温度场 $T(\mathbf{x}, t)$ 满足热传导方程：

$$\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q$$

其中 $\rho$ 是密度，$c_p$ 是比热容，$k$ 是热导率，$Q$ 是热源项。热应变为：

$$\boldsymbol{\epsilon}_{th} = \alpha (T - T_{ref}) \mathbf{I}$$

其中 $\alpha$ 是热膨胀系数，$T_{ref}$ 是参考温度。总应变分解为：

$$\boldsymbol{\epsilon} = \boldsymbol{\epsilon}_{el} + \boldsymbol{\epsilon}_{th} + \boldsymbol{\epsilon}_{pl}$$

包含弹性、热和塑性成分。应力通过本构关系计算：

$$\boldsymbol{\sigma} = \mathbf{C} : (\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{th} - \boldsymbol{\epsilon}_{pl})$$

### 11.2.2 残余应力与翘曲变形

冷却过程中的残余应力可通过固有应变法(inherent strain method)估算：

$$\boldsymbol{\epsilon}^* = \beta \cdot \alpha \cdot \Delta T \cdot \mathbf{I}$$

其中 $\beta$ 是约束因子(0到1之间)。翘曲变形的解析解(对于简单梁结构)：

$$w(x) = \frac{M x^2}{2 E I}$$

其中 $M = \int_A \sigma_r y \, dA$ 是残余应力产生的弯矩，$E$ 是弹性模量，$I$ 是惯性矩。

#### 层间应力累积模型

考虑逐层打印的应力累积效应：

$$\boldsymbol{\sigma}_n = \sum_{i=1}^{n} \mathbf{C}_i : \left(\boldsymbol{\epsilon}_{mech}^{(i)} - \boldsymbol{\epsilon}_{th}^{(i)} \cdot H(t - t_i)\right)$$

其中 $H(t)$ 是Heaviside阶跃函数，$t_i$ 是第 $i$ 层的沉积时间。考虑材料的时变特性：

$$\mathbf{C}(T, t) = \mathbf{C}_0 \cdot \left(1 - \exp\left(-\frac{t}{\tau_{solid}(T)}\right)\right)$$

其中 $\tau_{solid}(T)$ 是温度相关的固化时间常数。

#### 翘曲的几何非线性分析

对于大变形情况，需要考虑几何非线性。von Kármán板理论给出：

$$\nabla^4 w + \frac{h}{D} \left[\frac{\partial^2 F}{\partial y^2} \frac{\partial^2 w}{\partial x^2} - 2\frac{\partial^2 F}{\partial x \partial y} \frac{\partial^2 w}{\partial x \partial y} + \frac{\partial^2 F}{\partial x^2} \frac{\partial^2 w}{\partial y^2}\right] = \frac{q}{D}$$

其中 $w$ 是挠度，$F$ 是Airy应力函数，$D = Eh^3/(12(1-\nu^2))$ 是板的弯曲刚度。

#### 多尺度应力分析

使用渐进均匀化方法处理微观结构的影响：

$$\boldsymbol{\sigma}^{macro} = \mathbf{C}^{eff} : \boldsymbol{\epsilon}^{macro} + \boldsymbol{\sigma}^{eigen}$$

有效刚度张量通过单胞问题求解：

$$\mathbf{C}^{eff}_{ijkl} = \frac{1}{|Y|} \int_Y C_{ijpq} \left(\delta_{pk}\delta_{ql} + \frac{\partial \chi^{kl}_p}{\partial y_q}\right) dY$$

其中 $\chi^{kl}$ 是特征位移场，$Y$ 是代表性体积单元。

### 11.2.3 逆变形补偿策略

给定期望形状 $\mathbf{X}_{target}$ 和预测变形 $\mathbf{u}_{pred}$，补偿后的形状为：

$$\mathbf{X}_{comp} = \mathbf{X}_{target} - \mathbf{u}_{pred}$$

考虑到非线性效应，迭代补偿方案：

$$\mathbf{X}_{comp}^{(k+1)} = \mathbf{X}_{target} - \gamma \cdot \mathbf{u}_{pred}(\mathbf{X}_{comp}^{(k)})$$

其中 $\gamma \in (0, 1]$ 是松弛因子。收敛条件：

$$\|\mathbf{X}_{comp}^{(k+1)} - \mathbf{X}_{comp}^{(k)}\| < \epsilon_{tol}$$

### 11.2.4 机器学习辅助的变形预测

使用神经网络预测变形场：

$$\mathbf{u}_{pred} = f_{\theta}(\mathbf{X}, \mathbf{p}_{process}, \mathbf{p}_{material})$$

其中 $\mathbf{p}_{process}$ 是工艺参数(温度、速度等)，$\mathbf{p}_{material}$ 是材料参数。损失函数结合了数据拟合和物理约束：

$$\mathcal{L} = \|\mathbf{u}_{pred} - \mathbf{u}_{true}\|^2 + \lambda \|\mathcal{R}(\mathbf{u}_{pred})\|^2$$

其中 $\mathcal{R}$ 是物理残差算子(如平衡方程)。

#### 物理信息神经网络(PINN)

将偏微分方程约束直接嵌入损失函数：

$$\mathcal{L}_{PINN} = \mathcal{L}_{data} + \lambda_1 \mathcal{L}_{PDE} + \lambda_2 \mathcal{L}_{BC} + \lambda_3 \mathcal{L}_{IC}$$

其中PDE残差为：

$$\mathcal{L}_{PDE} = \frac{1}{N_{col}} \sum_{i=1}^{N_{col}} \left\|\nabla \cdot \boldsymbol{\sigma}(\mathbf{x}_i) + \mathbf{f}(\mathbf{x}_i)\right\|^2$$

边界条件残差：

$$\mathcal{L}_{BC} = \frac{1}{N_{bc}} \sum_{j=1}^{N_{bc}} \|\mathbf{u}(\mathbf{x}_j) - \mathbf{u}_{BC}(\mathbf{x}_j)\|^2 + \|\boldsymbol{\sigma}(\mathbf{x}_j) \cdot \mathbf{n}_j - \mathbf{t}_{BC}(\mathbf{x}_j)\|^2$$

#### 图神经网络表示

使用图神经网络(GNN)处理不规则网格：

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\mathbf{W}_{self}^{(l)} \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}_{msg}^{(l)} \mathbf{h}_j^{(l)}\right)$$

其中 $\alpha_{ij}$ 是注意力权重：

$$\alpha_{ij} = \frac{\exp(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j])}{\sum_{k \in \mathcal{N}(i)} \exp(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_k])}$$

#### 迁移学习策略

从仿真数据预训练，然后在实验数据上微调：

$$\theta^* = \arg\min_{\theta} \mathcal{L}_{exp}(\theta; \mathcal{D}_{exp}) + \gamma \|\theta - \theta_{sim}\|^2$$

其中 $\theta_{sim}$ 是仿真预训练参数，$\gamma$ 控制正则化强度。使用域适应技术处理仿真-实验差异：

$$\mathcal{L}_{DA} = \mathcal{L}_{task} - \lambda \mathcal{L}_{domain}$$

其中域分类器的损失 $\mathcal{L}_{domain}$ 通过梯度反转层(GRL)优化。

## 11.3 不确定性量化：多项式混沌展开

### 11.3.1 随机输入的参数化

设随机输入参数 $\boldsymbol{\xi} = (\xi_1, ..., \xi_d)$ 具有联合概率密度 $\rho(\boldsymbol{\xi})$。标准化到 $[-1, 1]^d$ 或其他标准域：

$$\xi_i = \frac{2(X_i - a_i)}{b_i - a_i} - 1$$

其中 $X_i \in [a_i, b_i]$ 是原始随机变量。

### 11.3.2 多项式混沌展开

输出响应 $Y(\boldsymbol{\xi})$ 的PCE表示：

$$Y(\boldsymbol{\xi}) = \sum_{\boldsymbol{\alpha} \in \mathbb{N}^d} c_{\boldsymbol{\alpha}} \Psi_{\boldsymbol{\alpha}}(\boldsymbol{\xi})$$

其中 $\Psi_{\boldsymbol{\alpha}}$ 是正交多项式基，$c_{\boldsymbol{\alpha}}$ 是展开系数。对于均匀分布，使用Legendre多项式；对于正态分布，使用Hermite多项式。

截断到总阶数 $p$：

$$Y(\boldsymbol{\xi}) \approx \sum_{|\boldsymbol{\alpha}| \leq p} c_{\boldsymbol{\alpha}} \Psi_{\boldsymbol{\alpha}}(\boldsymbol{\xi})$$

展开项数为：

$$P = \binom{d + p}{p} = \frac{(d + p)!}{d! \cdot p!}$$

### 11.3.3 系数计算与稀疏性

投影法计算系数：

$$c_{\boldsymbol{\alpha}} = \frac{\mathbb{E}[Y \Psi_{\boldsymbol{\alpha}}]}{\mathbb{E}[\Psi_{\boldsymbol{\alpha}}^2]} = \frac{1}{\gamma_{\boldsymbol{\alpha}}} \int Y(\boldsymbol{\xi}) \Psi_{\boldsymbol{\alpha}}(\boldsymbol{\xi}) \rho(\boldsymbol{\xi}) d\boldsymbol{\xi}$$

其中 $\gamma_{\boldsymbol{\alpha}} = \mathbb{E}[\Psi_{\boldsymbol{\alpha}}^2]$ 是归一化常数。

使用稀疏网格积分(Smolyak quadrature)：

$$\mathcal{Q}^{(l)}_d = \sum_{l-d+1 \leq |\mathbf{i}| \leq l} (-1)^{l-|\mathbf{i}|} \binom{d-1}{l-|\mathbf{i}|} (\mathcal{U}^{i_1} \otimes \cdots \otimes \mathcal{U}^{i_d})$$

### 11.3.4 统计矩与灵敏度分析

均值和方差直接从PCE系数获得：

$$\mu_Y = c_{\mathbf{0}}, \quad \sigma_Y^2 = \sum_{|\boldsymbol{\alpha}| > 0} c_{\boldsymbol{\alpha}}^2 \gamma_{\boldsymbol{\alpha}}$$

Sobol指数用于全局灵敏度分析：

$$S_{\mathcal{I}} = \frac{\sum_{\boldsymbol{\alpha} \in \mathcal{A}_{\mathcal{I}}} c_{\boldsymbol{\alpha}}^2 \gamma_{\boldsymbol{\alpha}}}{\sigma_Y^2}$$

其中 $\mathcal{A}_{\mathcal{I}}$ 是仅包含索引集 $\mathcal{I}$ 中变量的多重指标集。

## 11.4 鲁棒优化与最坏情况设计

### 11.4.1 鲁棒优化的数学框架

考虑不确定参数 $\mathbf{u} \in \mathcal{U}$ 的优化问题，鲁棒对应式(robust counterpart)为：

$$\begin{aligned}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{s.t.} \quad & g_j(\mathbf{x}, \mathbf{u}) \leq 0, \quad \forall \mathbf{u} \in \mathcal{U}_j, \quad j = 1, ..., m
\end{aligned}$$

对于椭球不确定集 $\mathcal{U} = \{\mathbf{u} : \|\mathbf{P}^{-1/2}(\mathbf{u} - \bar{\mathbf{u}})\| \leq 1\}$，线性约束的鲁棒对应式为：

$$\mathbf{a}^T \mathbf{x} + \|\mathbf{P}^{1/2} \mathbf{x}\| \leq b$$

其中 $\mathbf{P}$ 是不确定性的协方差矩阵。

### 11.4.2 最坏情况分析与优化

最坏情况性能指标：

$$J_{worst}(\mathbf{x}) = \max_{\mathbf{u} \in \mathcal{U}} J(\mathbf{x}, \mathbf{u})$$

使用对偶理论，内层最大化问题可转化为：

$$\max_{\mathbf{u} \in \mathcal{U}} J(\mathbf{x}, \mathbf{u}) = \min_{\lambda \geq 0} \left\{\lambda \delta^2 + \max_{\mathbf{u}} [J(\mathbf{x}, \mathbf{u}) - \lambda \|\mathbf{u} - \bar{\mathbf{u}}\|^2]\right\}$$

其中 $\delta$ 是不确定集的半径。

### 11.4.3 机会约束与风险度量

机会约束优化(chance-constrained optimization)：

$$\mathbb{P}[g(\mathbf{x}, \boldsymbol{\xi}) \leq 0] \geq 1 - \alpha$$

其中 $\alpha$ 是可接受的失败概率。使用CVaR(条件风险值)重新表述：

$$\text{CVaR}_{\alpha}[g(\mathbf{x}, \boldsymbol{\xi})] = \mathbb{E}[g(\mathbf{x}, \boldsymbol{\xi}) | g(\mathbf{x}, \boldsymbol{\xi}) \geq \text{VaR}_{\alpha}]$$

Sample Average Approximation(SAA)将机会约束转化为确定性约束：

$$\frac{1}{N} \sum_{i=1}^N \mathbb{I}[g(\mathbf{x}, \boldsymbol{\xi}^{(i)}) \leq 0] \geq 1 - \alpha$$

### 11.4.4 分布鲁棒优化

当概率分布 $\mathbb{P}$ 本身不确定时，考虑模糊集 $\mathcal{P}$：

$$\min_{\mathbf{x}} \max_{\mathbb{P} \in \mathcal{P}} \mathbb{E}_{\mathbb{P}}[f(\mathbf{x}, \boldsymbol{\xi})]$$

使用Wasserstein距离定义模糊集：

$$\mathcal{P} = \{\mathbb{Q} : W_p(\mathbb{Q}, \hat{\mathbb{P}}_N) \leq \epsilon\}$$

其中 $\hat{\mathbb{P}}_N$ 是经验分布，$W_p$ 是 $p$-Wasserstein距离：

$$W_p(\mathbb{P}, \mathbb{Q}) = \left(\inf_{\gamma \in \Pi(\mathbb{P}, \mathbb{Q})} \int \|\mathbf{x} - \mathbf{y}\|^p d\gamma(\mathbf{x}, \mathbf{y})\right)^{1/p}$$

## 11.5 贝叶斯优化与实验设计

### 11.5.1 高斯过程建模

目标函数 $f(\mathbf{x})$ 的高斯过程先验：

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

其中均值函数 $m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]$，协方差函数(如Matérn核)：

$$k(\mathbf{x}, \mathbf{x}') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}r}{\ell}\right)^{\nu} K_{\nu}\left(\frac{\sqrt{2\nu}r}{\ell}\right)$$

其中 $r = \|\mathbf{x} - \mathbf{x}'\|$，$K_{\nu}$ 是修正贝塞尔函数，$\ell$ 是长度尺度，$\nu$ 控制平滑度。

### 11.5.2 获取函数与探索-利用权衡

期望改进(Expected Improvement)获取函数：

$$\text{EI}(\mathbf{x}) = \mathbb{E}[\max(0, f(\mathbf{x}) - f^*)]$$

闭式解为：

$$\text{EI}(\mathbf{x}) = (\mu(\mathbf{x}) - f^*) \Phi\left(\frac{\mu(\mathbf{x}) - f^*}{\sigma(\mathbf{x})}\right) + \sigma(\mathbf{x}) \phi\left(\frac{\mu(\mathbf{x}) - f^*}{\sigma(\mathbf{x})}\right)$$

其中 $\mu(\mathbf{x})$ 和 $\sigma(\mathbf{x})$ 是后验均值和标准差，$\Phi$ 和 $\phi$ 分别是标准正态的CDF和PDF。

上置信界(UCB)策略：

$$\text{UCB}(\mathbf{x}) = \mu(\mathbf{x}) + \beta^{1/2} \sigma(\mathbf{x})$$

其中 $\beta$ 控制探索程度。

### 11.5.3 多目标贝叶斯优化

对于多目标 $\mathbf{f} = (f_1, ..., f_k)$，使用期望超体积改进(EHVI)：

$$\text{EHVI}(\mathbf{x}) = \mathbb{E}[\text{HV}(\mathcal{P} \cup \{\mathbf{f}(\mathbf{x})\}) - \text{HV}(\mathcal{P})]$$

其中 $\mathcal{P}$ 是当前Pareto前沿，$\text{HV}$ 是超体积指标：

$$\text{HV}(\mathcal{S}) = \lambda^k\left(\bigcup_{\mathbf{s} \in \mathcal{S}} [\mathbf{r}, \mathbf{s}]\right)$$

$\mathbf{r}$ 是参考点，$\lambda^k$ 是 $k$ 维Lebesgue测度。

### 11.5.4 最优实验设计

D-优化设计最大化信息矩阵的行列式：

$$\mathbf{X}^*_D = \arg\max_{\mathbf{X}} \det(\mathbf{X}^T \mathbf{X})$$

A-优化最小化参数估计的平均方差：

$$\mathbf{X}^*_A = \arg\min_{\mathbf{X}} \text{tr}[(\mathbf{X}^T \mathbf{X})^{-1}]$$

E-优化最小化最大特征值：

$$\mathbf{X}^*_E = \arg\min_{\mathbf{X}} \lambda_{max}[(\mathbf{X}^T \mathbf{X})^{-1}]$$

对于非线性模型，使用Fisher信息矩阵：

$$\mathbf{F}(\boldsymbol{\theta}) = \mathbb{E}\left[\left(\frac{\partial \log p(\mathbf{y}|\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right) \left(\frac{\partial \log p(\mathbf{y}|\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right)^T\right]$$

## 本章小结

本章系统介绍了3D打印中的误差分析与补偿方法：

1. **离散化误差**：弦高误差 $\epsilon_{chord} \approx h^2/(8\rho)$ 与曲率成反比；阶梯效应 $\epsilon_{stair} = h\sin\theta$ 与层高和倾斜角相关

2. **热变形补偿**：通过求解耦合的热-力学方程预测变形，使用迭代补偿 $\mathbf{X}_{comp}^{(k+1)} = \mathbf{X}_{target} - \gamma \cdot \mathbf{u}_{pred}$ 

3. **不确定性量化**：多项式混沌展开 $Y = \sum c_{\alpha}\Psi_{\alpha}$ 提供高效的不确定性传播；Sobol指数量化参数敏感度

4. **鲁棒优化**：通过最坏情况设计 $\min_x \max_{u \in \mathcal{U}} J(x,u)$ 和机会约束 $\mathbb{P}[g \leq 0] \geq 1-\alpha$ 处理不确定性

5. **贝叶斯优化**：使用高斯过程建模未知函数，通过EI或UCB获取函数平衡探索与利用

关键公式汇总：
- Hausdorff距离：$d_H(S,M) = \max\{\sup_{p \in S} d(p,M), \sup_{q \in M} d(q,S)\}$
- PCE方差：$\sigma_Y^2 = \sum_{|\alpha| > 0} c_{\alpha}^2 \gamma_{\alpha}$
- 期望改进：$\text{EI}(x) = (μ(x) - f^*)\Phi(z) + σ(x)\phi(z)$，其中 $z = (μ(x) - f^*)/σ(x)$

## 练习题

### 基础题

**习题11.1** 对于半径 $R = 50$ mm的球面，使用边长 $h = 2$ mm的三角形网格近似。计算最大弦高误差，并估算需要多少个三角形才能使误差小于0.1 mm。

<details>
<summary>提示</summary>
使用弦高误差公式 $\epsilon = h^2/(8R)$，反解得到所需边长，然后估算三角形数量。
</details>

<details>
<summary>答案</summary>
最大弦高误差：$\epsilon = 2^2/(8 \times 50) = 0.01$ mm。
要使 $\epsilon < 0.1$ mm，需要 $h < \sqrt{8 \times 50 \times 0.1} = 6.32$ mm。
球面积约 $4\pi R^2 = 31416$ mm²，每个三角形面积约 $\sqrt{3}h^2/4$。
对于 $h = 6$ mm，需要约 $31416/(15.6) \approx 2014$ 个三角形。
</details>

**习题11.2** 打印一个倾斜角为30°的平面，层高为0.2 mm。计算阶梯误差，并推导使误差减半所需的层高。

<details>
<summary>提示</summary>
使用阶梯误差公式 $\epsilon = h \sin\theta$。
</details>

<details>
<summary>答案</summary>
阶梯误差：$\epsilon = 0.2 \times \sin(30°) = 0.1$ mm。
要使误差减半到0.05 mm，需要层高 $h = 0.05/\sin(30°) = 0.1$ mm。
</details>

**习题11.3** 给定二维随机变量 $(X_1, X_2)$ 均匀分布在 $[-1,1]^2$，使用2阶PCE展开函数 $f(x_1, x_2) = x_1^2 + x_1x_2$。计算展开系数和输出方差。

<details>
<summary>提示</summary>
使用Legendre多项式基，利用正交性计算系数。
</details>

<details>
<summary>答案</summary>
Legendre基：$\{1, x_1, x_2, (3x_1^2-1)/2, x_1x_2, (3x_2^2-1)/2\}$。
非零系数：$c_{(0,0)} = 1/3$（来自 $x_1^2$ 的常数项），$c_{(2,0)} = 2/3$，$c_{(1,1)} = 1$。
方差：$\sigma^2 = (2/3)^2 \times (1/5) + 1^2 \times (1/3) \times (1/3) = 4/45 + 1/9 = 9/45 = 1/5$。
</details>

### 挑战题

**习题11.4** 考虑热弹性问题，材料的热膨胀系数 $\alpha$ 不确定，服从正态分布 $\alpha \sim \mathcal{N}(\bar{\alpha}, \sigma_{\alpha}^2)$。推导长度为 $L$ 的杆在温升 $\Delta T$ 下的伸长量的均值和方差。

<details>
<summary>提示</summary>
伸长量 $\Delta L = \alpha L \Delta T$，利用线性变换的性质。
</details>

<details>
<summary>答案</summary>
伸长量：$\Delta L = \alpha L \Delta T$。
均值：$\mathbb{E}[\Delta L] = \mathbb{E}[\alpha] L \Delta T = \bar{\alpha} L \Delta T$。
方差：$\text{Var}[\Delta L] = \text{Var}[\alpha] (L \Delta T)^2 = \sigma_{\alpha}^2 L^2 (\Delta T)^2$。
标准差：$\sigma_{\Delta L} = \sigma_{\alpha} L |\Delta T|$。
</details>

**习题11.5** 设计一个鲁棒优化问题：最小化梁的重量，约束最大应力不超过许用应力，考虑载荷 $F$ 在 $[F_{min}, F_{max}]$ 范围内变化。写出鲁棒对应式。

<details>
<summary>提示</summary>
梁的应力 $\sigma = FL/(Wh^2)$，其中 $W$ 是宽度，$h$ 是高度。重量正比于 $Wh$。
</details>

<details>
<summary>答案</summary>
原问题：$\min_{W,h} WLh$，约束：$FL/(Wh^2) \leq \sigma_{allow}$。
鲁棒对应式：$\min_{W,h} WLh$，约束：$F_{max}L/(Wh^2) \leq \sigma_{allow}$。
简化得：$Wh^2 \geq F_{max}L/\sigma_{allow}$。
这保证了在最坏情况（最大载荷）下约束仍满足。
</details>

**习题11.6** 使用贝叶斯优化寻找函数 $f(x) = -\sin(3x) - x^2 + 0.7x$ 在 $[-1, 1]$ 上的最大值。假设已有3个观测点：$(-0.5, f(-0.5))$，$(0, f(0))$，$(0.5, f(0.5))$。计算下一个最优采样点的EI值。

<details>
<summary>提示</summary>
先计算高斯过程的后验均值和方差，然后使用EI公式。
</details>

<details>
<summary>答案</summary>
观测值：$f(-0.5) = 0.729$，$f(0) = 0$，$f(0.5) = -0.351$。
当前最大值：$f^* = 0.729$。
使用RBF核建立GP模型，后验预测在 $x = -0.8$ 附近有较大不确定性。
EI在 $x \approx -0.8$ 或 $x \approx 0.8$ 处较大（具体值需数值计算）。
真实最大值在 $x \approx -0.63$ 处，$f_{max} \approx 0.77$。
</details>

**习题11.7** 分析多项式混沌展开的收敛性：对于解析函数 $f(\xi) = e^{\xi}$，$\xi \sim \mathcal{U}[-1,1]$，推导PCE系数的衰减率。

<details>
<summary>提示</summary>
计算Legendre系数 $c_n = \langle e^{\xi}, P_n(\xi) \rangle / \langle P_n, P_n \rangle$，分析其渐近行为。
</details>

<details>
<summary>答案</summary>
Legendre系数：$c_n = \frac{2n+1}{2} \int_{-1}^1 e^{\xi} P_n(\xi) d\xi$。
使用Rodrigues公式和分部积分：$c_n = \frac{2n+1}{2} \frac{(-1)^n}{2^n n!} \int_{-1}^1 e^{\xi} \frac{d^n}{d\xi^n}(\xi^2-1)^n d\xi$。
渐近行为：$c_n \sim \frac{2\sinh(1)}{n!}$，呈指数衰减。
这说明对于解析函数，PCE具有谱收敛性（比代数收敛快）。
</details>

**习题11.8** 设计一个多材料3D打印的鲁棒拓扑优化问题，考虑两种材料的弹性模量都有10%的不确定性。推导优化问题的数学形式，并讨论求解策略。

<details>
<summary>提示</summary>
使用SIMP插值，考虑材料属性的随机场表示，建立期望柔度最小化问题。
</details>

<details>
<summary>答案</summary>
设密度场 $\rho_1, \rho_2$，弹性模量 $E_1 = \bar{E}_1(1 + 0.1\xi_1)$，$E_2 = \bar{E}_2(1 + 0.1\xi_2)$。
局部刚度：$E(\mathbf{x}) = \rho_1^p E_1 + \rho_2^p E_2$。
鲁棒优化：$\min_{\rho_1,\rho_2} \mathbb{E}[C(\rho_1, \rho_2, \xi)] + \beta \cdot \text{Var}[C]$。
约束：$\rho_1 + \rho_2 = 1$（材料分配），$\int \rho_i dV \leq V_i$（体积约束）。
求解：使用PCE或蒙特卡洛估计期望和方差，MMA优化器更新设计变量。
关键挑战：高维随机空间的高效采样，梯度计算的准确性。
</details>

## 常见陷阱与错误

### 1. 离散化误差的错误估计
- **陷阱**：仅考虑局部误差，忽略误差累积
- **正确做法**：使用全局误差度量如Hausdorff距离，考虑最坏情况

### 2. 热补偿的过度修正
- **陷阱**：直接使用预测变形的负值作为补偿，导致过补偿
- **正确做法**：使用松弛因子 $\gamma < 1$，迭代收敛到正确补偿

### 3. PCE截断阶数选择
- **陷阱**：盲目增加展开阶数，导致过拟合或数值不稳定
- **正确做法**：使用交叉验证或留一法(LOO)选择最优截断阶数

### 4. 鲁棒优化的保守性
- **陷阱**：不确定集设置过大，导致解过于保守
- **正确做法**：基于数据估计合理的不确定集大小，使用分布鲁棒方法

### 5. 贝叶斯优化的核函数选择
- **陷阱**：使用不合适的核函数，如对非平滑函数使用RBF核
- **正确做法**：根据函数特性选择核(Matérn族提供不同平滑度)

### 6. 实验设计的样本效率
- **陷阱**：使用全因子设计导致维度灾难
- **正确做法**：使用拉丁超立方采样(LHS)或Sobol序列

## 最佳实践检查清单

### 误差分析阶段
- [ ] 识别所有误差源：离散化、制造、材料、环境
- [ ] 量化各误差源的大小和分布
- [ ] 分析误差传播路径和累积效应
- [ ] 确定主导误差源，优先处理

### 补偿设计阶段
- [ ] 建立误差预测模型（物理或数据驱动）
- [ ] 验证预测模型的准确性（交叉验证）
- [ ] 设计补偿策略（几何补偿、工艺优化）
- [ ] 考虑补偿的可制造性约束

### 不确定性处理
- [ ] 收集足够的数据估计不确定性分布
- [ ] 选择合适的不确定性量化方法（PCE、蒙特卡洛）
- [ ] 平衡计算成本和精度要求
- [ ] 进行敏感性分析，识别关键参数

### 优化求解阶段
- [ ] 明确优化目标：性能、鲁棒性、成本
- [ ] 选择合适的优化框架（确定性、随机、鲁棒）
- [ ] 设置合理的收敛准则和计算预算
- [ ] 验证优化结果的鲁棒性

### 实验验证阶段
- [ ] 设计高效的实验方案（DOE）
- [ ] 考虑测量误差和重复性
- [ ] 使用统计方法分析结果
- [ ] 根据实验反馈更新模型和补偿策略