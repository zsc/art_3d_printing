# 第18章：高级可微渲染与逆向问题

本章深入探讨可微渲染的高级技术及其在逆向问题中的应用。我们将从路径空间的梯度计算开始，深入研究边界积分方法、体积渲染的可微性，以及如何通过可微渲染解决材质、几何和光照的分解问题。最后，我们将探讨神经渲染器的混合表示方法和物理仿真的可微性框架。这些技术在3D打印的质量预测、材料优化和逆向工程中有重要应用。

## 18.1 路径空间梯度与边界积分

### 18.1.1 路径空间的数学框架

在渲染中，一条光路可以表示为顶点序列 $\bar{x} = (x_0, x_1, ..., x_k)$，其中 $x_0$ 是相机位置，$x_k$ 是光源位置。路径空间 $\Omega$ 是所有有效光路的集合，其维度为 $\dim(\Omega) = 3(k+1)$。

渲染积分可以写为：
$$I = \int_{\Omega} f(\bar{x}) d\mu(\bar{x})$$

其中测度 $d\mu(\bar{x})$ 是路径空间上的乘积测度：
$$d\mu(\bar{x}) = dA(x_0) \prod_{i=1}^{k-1} dA(x_i) dA(x_k)$$

路径贡献函数 $f(\bar{x})$ 包含了BSDF、几何项和光源发射：
$$f(\bar{x}) = L_e(x_k \to x_{k-1}) \prod_{i=1}^{k-1} \rho(x_{i+1} \to x_i \to x_{i-1}) G(x_i, x_{i+1})$$

其中BSDF项 $\rho$ 满足互易性：
$$\rho(\omega_i \to x \to \omega_o) = \rho(\omega_o \to x \to \omega_i)$$

几何项定义为：
$$G(x, y) = \frac{|\cos\theta_x \cos\theta_y|}{||x - y||^2} V(x, y)$$

其中 $V(x,y)$ 是可见性函数，$\theta_x$ 是 $x$ 处法向量与连线 $xy$ 的夹角。

路径的概率密度函数（用于重要性采样）：
$$p(\bar{x}) = p(x_0) \prod_{i=0}^{k-1} p(x_{i+1}|x_i)$$

其中转移概率 $p(x_{i+1}|x_i)$ 通常基于BSDF或光源采样。

### 18.1.2 梯度计算的挑战

当场景参数 $\theta$ 变化时（如物体位置、材质参数），我们需要计算：
$$\frac{\partial I}{\partial \theta} = \frac{\partial}{\partial \theta} \int_{\Omega} f(\bar{x}, \theta) d\mu(\bar{x})$$

主要挑战来自于：
1. **可见性不连续**：$V(x,y)$ 在遮挡边界处不连续，导致 $f$ 不可微
2. **积分域变化**：当几何变化时，有效路径集合 $\Omega(\theta)$ 也会变化
3. **高维积分**：路径空间维度为 $3(k+1)$，Monte Carlo方差高
4. **相关性耦合**：路径各顶点通过BSDF和几何项耦合

数学上，问题的核心是交换微分和积分算子：
$$\frac{\partial}{\partial \theta} \int_{\Omega(\theta)} f(\bar{x}, \theta) d\mu \stackrel{?}{=} \int_{\Omega(\theta)} \frac{\partial f}{\partial \theta} d\mu$$

当 $f$ 包含不连续（如可见性函数）时，上式不成立，需要额外的边界项。

### 18.1.3 边界积分方法

Reynolds运输定理给出了移动域上积分的导数：
$$\frac{d}{dt} \int_{\Omega(t)} f(\bar{x}, t) d\mu = \int_{\Omega(t)} \frac{\partial f}{\partial t} d\mu + \int_{\partial\Omega(t)} f v_n ds$$

其中 $v_n$ 是边界 $\partial\Omega(t)$ 的法向速度。

应用到渲染积分，边界项对应于轮廓边缘（silhouette edges）的贡献：
$$\frac{\partial I}{\partial \theta} = \int_{\Omega} \frac{\partial f}{\partial \theta} d\mu + \int_{\partial\Omega} f \cdot \nu_{\theta} ds$$

其中 $\nu_{\theta} = \frac{\partial\Omega}{\partial\theta} \cdot n$ 是参数变化引起的边界法向速度。

对于多边形网格，轮廓边满足：
$$e \in \text{silhouette} \iff (n_1 \cdot v)(n_2 \cdot v) < 0$$

其中 $n_1, n_2$ 是边 $e$ 两侧三角形的法向量，$v$ 是视线方向。

### 18.1.4 轮廓积分的计算

对于一个轮廓边 $e$，其对梯度的贡献通过线积分计算：
$$\Delta I_e = \int_e \left( f^+ - f^- \right) \langle v_e, n_e \rangle ds$$

其中：
- $f^+, f^-$ 是边两侧的辐射度值
- $v_e$ 是边上点的速度场 $v_e = \frac{\partial x_e}{\partial \theta}$
- $n_e$ 是边在图像平面上的法向量

具体计算步骤：

1. **轮廓边检测**：对每条边 $e = (v_1, v_2)$，计算
   $$\text{is\_silhouette}(e) = \begin{cases}
   \text{true} & \text{if } (n_1 \cdot \omega)(n_2 \cdot \omega) < 0 \\
   \text{false} & \text{otherwise}
   \end{cases}$$
   其中 $\omega$ 是从边到相机的视线方向。

2. **轮廓曲线参数化**：将边投影到图像平面
   $$p(t) = (1-t)\pi(v_1) + t\pi(v_2), \quad t \in [0,1]$$
   其中 $\pi$ 是投影函数。

3. **辐射度差计算**：
   $$\Delta f = L(x_e, \omega^+) - L(x_e, \omega^-)$$
   其中 $\omega^+, \omega^-$ 是边两侧的出射方向。

4. **速度投影**：计算参数速度在边法向的投影
   $$v_n = \langle \frac{\partial p}{\partial \theta}, n_e \rangle$$

5. **数值积分**：使用Gauss-Legendre求积
   $$\Delta I_e \approx \sum_{i=1}^{N_q} w_i \Delta f(t_i) v_n(t_i) ||p'(t_i)||$$

### 18.1.5 重参数化技巧

对于包含不连续的函数，重参数化可以实现可微化：

**硬阈值到软阈值**：
原始形式（不可微）：
$$I = \int_{\Omega} f(\bar{x}) \mathbf{1}[\phi(\bar{x}) > 0] d\mu$$

重参数化（可微）：
$$I_{\epsilon} = \int_{\Omega} f(\bar{x}) \sigma_{\epsilon}(\phi(\bar{x})) d\mu$$

其中 $\sigma_{\epsilon}(x) = \frac{1}{1 + e^{-x/\epsilon}}$ 是温度参数为 $\epsilon$ 的sigmoid函数。

**边界模糊化**：
对于尖锐边界，引入模糊带宽 $\delta$：
$$V_{\delta}(x,y) = \begin{cases}
1 & \text{if } d(x,y) > \delta \\
\frac{1}{2} + \frac{d(x,y)}{2\delta} & \text{if } |d(x,y)| \leq \delta \\
0 & \text{if } d(x,y) < -\delta
\end{cases}$$

其中 $d(x,y)$ 是有符号距离函数。

**随机重参数化**：
使用Gumbel-Softmax技巧处理离散选择：
$$\text{sample}(\pi) \approx \text{softmax}((g + \log\pi)/\tau)$$
其中 $g \sim \text{Gumbel}(0,1)$，$\tau$ 是温度参数。

## 18.2 可微体积渲染与射线积分

### 18.2.1 体积渲染方程

体积渲染描述光线穿过参与介质时的辐射传输过程。完整的辐射传输方程（RTE）为：
$$(\omega \cdot \nabla) L(x, \omega) = -\sigma_t(x) L(x, \omega) + \sigma_s(x) \int_{S^2} p(x, \omega', \omega) L(x, \omega') d\omega' + Q(x, \omega)$$

其中：
- $\sigma_t = \sigma_a + \sigma_s$ 是消光系数（吸收+散射）
- $p(x, \omega', \omega)$ 是相函数
- $Q(x, \omega)$ 是发射项

沿射线积分的解（忽略多次散射）：
$$L(x, \omega) = \int_0^{t_{max}} T(0,t) \sigma_s(r(t)) L_s(r(t), \omega) dt + T(0,t_{max}) L_{bg}$$

其中：
- 射线参数化：$r(t) = x + t\omega$
- 透射率：$T(s,t) = \exp\left(-\int_s^t \sigma_t(r(u))du\right)$
- 内散射辐射：$L_s = \sigma_s \int_{S^2} p(\omega', \omega) L_i(x, \omega') d\omega' + L_e$

对于各向同性介质，相函数常用Henyey-Greenstein模型：
$$p_{HG}(\cos\theta) = \frac{1}{4\pi} \frac{1 - g^2}{(1 + g^2 - 2g\cos\theta)^{3/2}}$$

其中 $g \in (-1, 1)$ 是各向异性参数。

### 18.2.2 离散化与正交采样

**分层采样方案**：
将射线区间 $[t_{near}, t_{far}]$ 均匀分为 $N$ 个子区间，每个区间随机采样：
$$t_i \sim \mathcal{U}\left[t_{near} + \frac{i-1}{N}\Delta t, t_{near} + \frac{i}{N}\Delta t\right]$$
其中 $\Delta t = t_{far} - t_{near}$。

**离散积分近似**：
$$\hat{L} = \sum_{i=1}^N T_i \alpha_i c_i + T_{N+1} L_{bg}$$

其中：
- 不透明度：$\alpha_i = 1 - \exp(-\sigma_i \delta_i)$
- 区间长度：$\delta_i = t_{i+1} - t_i$
- 累积透射率：$T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$
- 颜色贡献：$c_i = \frac{\sigma_{s,i}}{\sigma_{t,i}} L_{s,i} + \frac{\sigma_{a,i}}{\sigma_{t,i}} L_{e,i}$

**误差分析**：
离散化误差为 $O(\Delta t^2)$，可通过Richardson外推提高精度：
$$L_{exact} \approx \frac{4L(N/2) - L(N)}{3}$$

### 18.2.3 梯度计算

**自动微分方法**：
对密度场参数 $\theta$ 的梯度可通过反向传播计算：
$$\frac{\partial \hat{L}}{\partial \theta} = \sum_{i=1}^N \left[ \frac{\partial T_i}{\partial \theta} \alpha_i c_i + T_i \frac{\partial \alpha_i}{\partial \theta} c_i + T_i \alpha_i \frac{\partial c_i}{\partial \theta} \right]$$

关键梯度项的展开：

1. **透射率梯度**（通过递归关系）：
   $$\frac{\partial T_i}{\partial \alpha_j} = \begin{cases}
   -T_i / (1 - \alpha_j) & \text{if } j < i \\
   0 & \text{otherwise}
   \end{cases}$$

2. **不透明度梯度**：
   $$\frac{\partial \alpha_i}{\partial \sigma_i} = \delta_i \exp(-\sigma_i \delta_i) = \delta_i (1 - \alpha_i)$$

3. **复合链式法则**：
   $$\frac{\partial L}{\partial \sigma_i} = \sum_{j=i}^N \frac{\partial L}{\partial T_j} \frac{\partial T_j}{\partial \alpha_i} \frac{\partial \alpha_i}{\partial \sigma_i}$$

**梯度的数值稳定性**：
为避免数值下溢，使用对数空间计算：
$$\log T_i = \sum_{j=1}^{i-1} \log(1 - \alpha_j) \approx -\sum_{j=1}^{i-1} \alpha_j \quad (\text{when } \alpha_j \ll 1)$$

### 18.2.4 重要性采样与分层采样

**逆CDF采样**：
给定累积密度函数 $F(t) = 1 - T(t)$，采样点通过求解获得：
$$t_i = F^{-1}(u_i), \quad u_i \sim \mathcal{U}(0,1)$$

对于分段常数密度，逆CDF可解析计算：
$$F^{-1}(u) = -\frac{1}{\sigma}\log(1 - u(1 - e^{-\sigma t_{max}}))$$

**层次采样策略**：
1. **粗采样**：$N_c$ 个均匀样本估计密度分布
2. **重要性权重**：$w_i = T_i \alpha_i / \sum_j T_j \alpha_j$
3. **细采样**：根据权重 $w_i$ 分配 $N_f$ 个额外样本
4. **合并**：组合粗细样本进行最终积分

**俄罗斯轮盘赌**：
当透射率 $T_i < \epsilon$ 时，以概率 $p = T_i/\epsilon$ 继续追踪：
$$\hat{L}_{RR} = \begin{cases}
\hat{L}/p & \text{with probability } p \\
0 & \text{with probability } 1-p
\end{cases}$$

### 18.2.5 射线积分的解析解

特殊密度分布的解析透射率：

**指数密度** $\sigma(t) = \sigma_0 e^{-\lambda t}$：
$$T(t) = \exp\left(-\frac{\sigma_0}{\lambda}(1 - e^{-\lambda t})\right)$$

**多项式密度** $\sigma(t) = \sum_{k=0}^n a_k t^k$：
$$T(t) = \exp\left(-\sum_{k=0}^n \frac{a_k t^{k+1}}{k+1}\right)$$

**球形高斯** $\sigma(x) = A\exp(-||x-c||^2/2r^2)$：
沿射线 $x = o + td$ 的光学深度：
$$\tau = A\sqrt{\frac{\pi r^2}{2}} \exp\left(-\frac{b^2}{2r^2}\right) \left[\text{erf}\left(\frac{t_2 - a}{\sqrt{2}r}\right) - \text{erf}\left(\frac{t_1 - a}{\sqrt{2}r}\right)\right]$$
其中 $a = d \cdot (c - o)$，$b = ||d \times (c - o)||$。

**分段线性密度**：
对于线性插值的体素网格，每个体素内：
$$\sigma(t) = (1-\alpha)s_0 + \alpha s_1, \quad \alpha = \frac{t - t_0}{t_1 - t_0}$$
$$T_{cell} = \exp\left(-\frac{(s_0 + s_1)(t_1 - t_0)}{2}\right)$$

## 18.3 逆向渲染：材质、几何、光照分解

### 18.3.1 逆向渲染的数学建模

逆向渲染是从观察图像推断场景属性的病态问题。完整的贝叶斯建模：
$$p(\theta|I_{obs}) \propto p(I_{obs}|\theta) p(\theta)$$

其中 $\theta = (\theta_g, \theta_m, \theta_l, \theta_c)$ 包含：
- 几何 $\theta_g$：形状、拓扑
- 材质 $\theta_m$：BRDF参数、纹理
- 光照 $\theta_l$：环境光、点光源
- 相机 $\theta_c$：内参、外参

**MAP估计**转化为优化问题：
$$\min_{\theta} \mathcal{L}(\theta) = \mathcal{L}_{data}(\theta) + \mathcal{L}_{reg}(\theta)$$

数据项（支持多视角）：
$$\mathcal{L}_{data} = \sum_{v} \sum_{p} \rho(I_{obs}^v(p) - I_{render}^v(p; \theta))$$

其中 $\rho$ 是鲁棒损失函数（如Huber损失）：
$$\rho_{Huber}(x) = \begin{cases}
\frac{1}{2}x^2 & |x| \leq \delta \\
\delta(|x| - \frac{\delta}{2}) & |x| > \delta
\end{cases}$$

### 18.3.2 材质分解

**微表面BRDF模型**（Cook-Torrance）：
$$f_r(\omega_i, \omega_o) = k_d \frac{\rho_d}{\pi} + k_s \frac{D(\omega_h) G(\omega_i, \omega_o) F(\omega_i, \omega_h)}{4(\omega_i \cdot n)(\omega_o \cdot n)}$$

其中：
- 能量守恒：$k_d + k_s = 1$
- 漫反射项：$\rho_d \in [0,1]^3$
- 镜面反射权重：$k_s = F_0 + (1 - F_0)(1 - \omega_i \cdot \omega_h)^5$

**GGX法线分布**：
$$D(\omega_h) = \frac{\alpha^2}{\pi((n \cdot \omega_h)^2(\alpha^2 - 1) + 1)^2}$$

**几何遮蔽函数**（Smith G）：
$$G(\omega_i, \omega_o) = G_1(\omega_i) G_1(\omega_o)$$
$$G_1(\omega) = \frac{2(n \cdot \omega)}{(n \cdot \omega) + \sqrt{\alpha^2 + (1 - \alpha^2)(n \cdot \omega)^2}}$$

**Fresnel项**（Schlick近似）：
$$F(\omega_i, \omega_h) = F_0 + (1 - F_0)(1 - \omega_i \cdot \omega_h)^5$$

其中 $F_0$ 由金属度 $m$ 和基础反射率 $\rho_0$ 决定：
$$F_0 = (1 - m)\cdot 0.04 + m \cdot \rho_0$$

**参数空间优化**：
使用重参数化确保参数有效范围：
- 粗糙度：$\alpha = \sigma^2$，优化 $\log\sigma$
- 金属度：$m = \text{sigmoid}(\tilde{m})$
- 反射率：在HSV空间优化避免负值

### 18.3.3 几何优化

使用隐式表示 $\phi: \mathbb{R}^3 \to \mathbb{R}$，表面定义为零水平集：
$$\mathcal{S} = \{x \in \mathbb{R}^3 : \phi(x) = 0\}$$

法向量：
$$n = \frac{\nabla\phi}{||\nabla\phi||}$$

几何更新使用水平集演化：
$$\frac{\partial\phi}{\partial t} + v|\nabla\phi| = 0$$

其中速度场 $v$ 由渲染梯度决定：
$$v = -\frac{\partial\mathcal{L}}{\partial\phi}$$

### 18.3.4 光照估计

环境光照使用球谐函数表示：
$$L(\omega) = \sum_{l=0}^{L_{max}} \sum_{m=-l}^{l} c_{lm} Y_l^m(\omega)$$

对于 $L_{max} = 2$（9个系数），已经能捕获大部分低频光照。

点光源和面光源的联合优化：
$$L_{total} = L_{env} + \sum_i I_i \frac{\delta(\omega - \omega_i)}{||\omega_i||^2} + \int_A L_a dA$$

### 18.3.5 联合优化策略

交替优化框架：
1. **固定G,L，优化M**：材质参数通常收敛最快
2. **固定M,L，优化G**：几何优化需要正则化
3. **固定G,M，优化L**：光照估计是线性问题

正则化项设计：
$$\mathcal{R}(\theta) = \lambda_s ||\nabla\phi||_2^2 + \lambda_r TV(\alpha) + \lambda_l ||c_{lm}||_1$$

其中：
- 几何平滑项：$||\nabla\phi||_2^2$
- 粗糙度全变分：$TV(\alpha) = \int |\nabla\alpha| dx$
- 光照稀疏性：$||c_{lm}||_1$

## 18.4 神经渲染器与混合表示

### 18.4.1 神经渲染的基本架构

神经渲染器将传统渲染管线的某些部分替换为神经网络：
$$I = R_{\theta}(G, M, L, \xi)$$

其中 $R_{\theta}$ 是参数化的神经网络，$\xi$ 是相机参数。

典型架构包括：
1. **特征提取**：$F = \text{Encoder}(G, M)$
2. **神经渲染**：$I' = \text{Renderer}_{\theta}(F, L, \xi)$
3. **后处理**：$I = \text{Postprocess}(I')$

### 18.4.2 混合表示方法

结合显式和隐式表示的优势：

**网格+神经纹理**：
$$c(x) = \text{MLP}_{\theta}(f_{uv}(x), h(x))$$

其中 $f_{uv}$ 是UV坐标，$h(x)$ 是学习的特征。

**体素+神经场**：
$$\sigma(x) = \sum_i w_i(x) \cdot \text{MLP}_{\theta}(x - x_i, f_i)$$

使用三线性插值权重 $w_i(x)$。

**点云+神经splat**：
$$I(p) = \sum_i \alpha_i(p) \cdot \text{CNN}_{\theta}(\mathcal{P}_i(p))$$

其中 $\mathcal{P}_i$ 是点 $i$ 的投影。

### 18.4.3 可微光栅化

软光栅化使用sigmoid混合：
$$C(p) = \frac{\sum_i c_i \cdot \sigma(d_i(p)/\epsilon)}{\sum_i \sigma(d_i(p)/\epsilon)}$$

其中 $d_i(p)$ 是像素 $p$ 到三角形 $i$ 的距离。

梯度计算：
$$\frac{\partial C}{\partial v_j} = \sum_p \frac{\partial C(p)}{\partial d_i} \cdot \frac{\partial d_i}{\partial v_j}$$

### 18.4.4 神经辐射缓存

预计算和缓存神经特征以加速渲染：
$$\mathcal{C} = \{(x_i, f_i, n_i, t_i)\}$$

查询时使用k-NN插值：
$$f(x) = \sum_{i \in \mathcal{N}_k(x)} w_i \cdot f_i$$

权重基于距离和法向相似度：
$$w_i = \exp\left(-\frac{||x - x_i||^2}{2\sigma_x^2} - \frac{(1 - n \cdot n_i)}{2\sigma_n^2}\right)$$

### 18.4.5 时序一致性

对于动态场景，添加时序正则化：
$$\mathcal{L}_{temp} = ||I_t - \mathcal{W}(I_{t-1}, F_{t-1,t})||^2$$

其中 $\mathcal{W}$ 是基于光流 $F$ 的warping操作。

## 18.5 物理仿真的可微性：DiffTaichi

### 18.5.1 可微物理仿真框架

可微物理仿真将前向仿真和反向传播结合：
$$s_{t+1} = f(s_t, u_t, \theta)$$

其中 $s_t$ 是状态，$u_t$ 是控制，$\theta$ 是物理参数。

损失函数：
$$\mathcal{L} = \sum_t L_t(s_t, s_t^*) + R(u_t)$$

梯度通过伴随方法计算：
$$\lambda_t = \frac{\partial L_t}{\partial s_t} + \left(\frac{\partial f}{\partial s_t}\right)^T \lambda_{t+1}$$

### 18.5.2 连续体力学的可微实现

弹性能量密度函数：
$$\Psi(F) = \frac{\mu}{2}(||F||_F^2 - 3) - \mu\log(J) + \frac{\lambda}{2}\log^2(J)$$

其中 $F$ 是变形梯度，$J = \det(F)$。

第一Piola-Kirchhoff应力：
$$P = \frac{\partial\Psi}{\partial F} = \mu(F - F^{-T}) + \lambda\log(J)F^{-T}$$

力的计算：
$$f_i = -\sum_e V_e \frac{\partial\Psi_e}{\partial x_i}$$

### 18.5.3 隐式积分的可微性

隐式Euler方法：
$$M(v_{t+1} - v_t) = \Delta t \cdot f(x_{t+1})$$
$$x_{t+1} = x_t + \Delta t \cdot v_{t+1}$$

Newton-Raphson迭代：
$$\Delta x = -H^{-1}g$$

其中Hessian矩阵：
$$H = M + \Delta t^2 K$$
$$K = -\frac{\partial f}{\partial x}$$

反向传播需要计算：
$$\frac{\partial x_{t+1}}{\partial x_t} = (I + \Delta t^2 M^{-1}K)^{-1}$$

### 18.5.4 接触和碰撞的可微处理

使用软约束处理接触：
$$f_c = k_n \max(0, -d)^2 n + k_t ||v_t||^2 \frac{v_t}{||v_t||}$$

其中 $d$ 是穿透深度，$n$ 是接触法向。

摩擦锥约束：
$$||f_t|| \leq \mu ||f_n||$$

可微化使用smooth max：
$$\text{smax}(x, 0) = \frac{1}{\beta}\log(1 + e^{\beta x})$$

### 18.5.5 拓扑优化中的应用

结合可微仿真的拓扑优化：
$$\min_{\rho} c(\rho) = U^T K(\rho) U$$
$$\text{s.t. } K(\rho)U = F, \quad \sum_i \rho_i v_i \leq V_{max}$$

灵敏度分析：
$$\frac{\partial c}{\partial \rho_e} = -U_e^T \frac{\partial K_e}{\partial \rho_e} U_e$$

使用SIMP插值：
$$E_e = E_{min} + \rho_e^p (E_0 - E_{min})$$

更新规则（MMA或OC方法）：
$$\rho_e^{new} = \max(0, \rho_e - m) + \frac{\max(0, B_e\eta)}{1 + \lambda}$$

## 本章小结

本章深入探讨了可微渲染的高级技术和应用：

1. **路径空间梯度计算**：通过边界积分方法处理可见性不连续，使用Reynolds运输定理计算轮廓贡献，重参数化技巧平滑不连续函数。

2. **可微体积渲染**：体积渲染方程的离散化和梯度传播，重要性采样和分层采样策略，特殊密度分布的解析解。

3. **逆向渲染框架**：材质、几何、光照的联合优化，交替优化策略和正则化设计，球谐函数表示环境光照。

4. **神经渲染器**：混合显式-隐式表示的优势，可微光栅化的软化方法，神经辐射缓存加速技术。

5. **可微物理仿真**：连续体力学的可微实现，隐式积分的反向传播，接触碰撞的软约束处理。

关键数学工具：
- 测度论和积分变换
- 伴随方法和自动微分
- 水平集方法和Hamilton-Jacobi方程
- 球谐函数和正交基展开
- 有限元方法和变分原理

## 练习题

### 基础题

**练习18.1** 推导二维情况下矩形遮挡物的边界积分
给定矩形遮挡物顶点 $(x_1, y_1), (x_2, y_2)$，光源在 $(x_s, y_s)$，相机在原点。计算当矩形沿x方向移动时的梯度贡献。

*Hint*：考虑四条边分别的贡献，注意判断哪些边是轮廓边。

<details>
<summary>答案</summary>

轮廓边判断：边的两个相邻面法向量与视线方向点积符号相反。
对于垂直边$(x_1, y_1)-(x_1, y_2)$：
$$\Delta I = \int_{y_1}^{y_2} (L^+ - L^-) v_x dy$$
其中$L^+, L^-$是边两侧的辐射度，$v_x$是x方向速度。

水平边类似处理。总梯度为四条轮廓边贡献之和。
</details>

**练习18.2** 体积渲染的透射率计算
对于密度场 $\sigma(t) = \sigma_0 e^{-at}$，推导透射率 $T(t)$ 的解析表达式。

*Hint*：使用透射率定义 $T(t) = \exp(-\int_0^t \sigma(s)ds)$。

<details>
<summary>答案</summary>

$$T(t) = \exp\left(-\int_0^t \sigma_0 e^{-as} ds\right) = \exp\left(-\frac{\sigma_0}{a}(1 - e^{-at})\right)$$

当$a \to 0$时，退化为常数密度情况：$T(t) = \exp(-\sigma_0 t)$。
</details>

**练习18.3** Cook-Torrance BRDF的粗糙度梯度
给定GGX分布 $D(\alpha) = \frac{\alpha^2}{\pi((n \cdot h)^2(\alpha^2 - 1) + 1)^2}$，推导 $\frac{\partial D}{\partial \alpha}$。

*Hint*：使用链式法则，注意 $\alpha$ 出现在分子和分母中。

<details>
<summary>答案</summary>

令 $\chi = (n \cdot h)^2(\alpha^2 - 1) + 1$，则：
$$\frac{\partial D}{\partial \alpha} = \frac{2\alpha}{\pi \chi^2} - \frac{2\alpha^2}{\pi \chi^3} \cdot 2(n \cdot h)^2 \alpha$$

简化后：
$$\frac{\partial D}{\partial \alpha} = \frac{2\alpha}{\pi \chi^2}\left(1 - \frac{2\alpha^2(n \cdot h)^2}{\chi}\right)$$
</details>

### 挑战题

**练习18.4** 路径空间的方差减少
设计一种重要性采样策略，减少路径追踪中的方差。考虑BSDF和光源的联合分布。

*Hint*：使用多重重要性采样(MIS)，平衡BSDF采样和光源采样。

<details>
<summary>答案</summary>

MIS权重使用平衡启发式：
$$w_s(x) = \frac{p_s(x)}{p_s(x) + p_b(x)}$$

其中$p_s$是光源采样PDF，$p_b$是BSDF采样PDF。

组合估计器：
$$I = \frac{f(x_s)}{p_s(x_s)}w_s(x_s) + \frac{f(x_b)}{p_b(x_b)}w_b(x_b)$$

这种方法在光源采样和BSDF采样各有优势的情况下都能获得低方差。
</details>

**练习18.5** 神经场的频谱分析
分析位置编码 $\gamma(x) = [\sin(2^0\pi x), \cos(2^0\pi x), ..., \sin(2^{L-1}\pi x), \cos(2^{L-1}\pi x)]$ 对神经场表达能力的影响。

*Hint*：考虑神经正切核(NTK)理论，分析不同频率成分的学习速度。

<details>
<summary>答案</summary>

NTK在无限宽度极限下：
$$K(x, x') = \langle \nabla_\theta f(x), \nabla_\theta f(x') \rangle$$

位置编码增加了高频特征，使得：
$$K_{PE}(x, x') = \sum_{l=0}^{L-1} 4^l \cos(2^l\pi(x - x'))$$

高频成分($2^l$较大)的核值更大，因此学习更快。这解释了为什么位置编码能帮助学习高频细节。
</details>

**练习18.6** 可微仿真的稳定性分析
分析隐式Euler方法在可微仿真中的数值稳定性。考虑刚度矩阵的条件数对梯度传播的影响。

*Hint*：分析雅可比矩阵的谱半径。

<details>
<summary>答案</summary>

隐式Euler的放大因子：
$$G = (I + \Delta t A)^{-1}$$

谱半径$\rho(G) < 1$当且仅当$A$的所有特征值实部为正。

梯度传播的条件数：
$$\kappa(\frac{\partial x_{t+1}}{\partial x_t}) = \kappa((I + \Delta t^2 M^{-1}K)^{-1})$$

当$K$病态时，梯度可能爆炸或消失。使用预条件子或自适应时间步长可以改善稳定性。
</details>

## 常见陷阱与错误

1. **边界积分的符号错误**
   - 错误：忽略轮廓边的方向性
   - 正确：根据视角方向确定积分符号

2. **体积渲染的数值不稳定**
   - 错误：步长过大导致采样不足
   - 正确：自适应步长或重要性采样

3. **逆向渲染的局部最优**
   - 错误：同时优化所有参数
   - 正确：交替优化或多尺度策略

4. **神经渲染的过拟合**
   - 错误：网络容量过大
   - 正确：使用正则化和数据增强

5. **可微仿真的梯度爆炸**
   - 错误：时间步长过大
   - 正确：梯度裁剪或自适应步长

## 最佳实践检查清单

### 可微渲染实现
- [ ] 验证梯度计算的正确性（有限差分检查）
- [ ] 处理数值不稳定（梯度裁剪、正则化）
- [ ] 优化内存使用（检查点技术）
- [ ] 支持批处理加速
- [ ] 实现自适应采样策略

### 逆向渲染优化
- [ ] 设计合适的损失函数
- [ ] 选择适当的正则化项
- [ ] 实现多尺度优化策略
- [ ] 验证收敛性
- [ ] 评估重建质量指标

### 物理仿真集成
- [ ] 确保能量守恒
- [ ] 验证时间积分稳定性
- [ ] 实现碰撞检测优化
- [ ] 支持多物理场耦合
- [ ] 提供参数敏感性分析
