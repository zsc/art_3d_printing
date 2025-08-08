# 第22章：形状分析与检索

本章深入探讨三维形状的数学分析方法和检索技术。我们将从经典的形状描述子出发，通过谱分析揭示形状的内在几何特性，利用最优传输理论度量形状之间的相似性，并研究功能驱动的形状分析方法。这些技术构成了现代3D打印设计检索、质量评估和智能推荐系统的数学基础。

## 22.1 形状描述子：D2、光场描述子

形状描述子是将三维几何编码为紧凑特征向量的数学方法，在3D打印设计检索、质量控制和相似性分析中起关键作用。理想的描述子应满足：判别性（区分不同形状）、鲁棒性（对噪声不敏感）、不变性（对刚体变换不变）和紧凑性（低维表示）。

### 22.1.1 全局形状描述子理论

**形状描述子的数学框架**

设 $\mathcal{S} \subset \mathbb{R}^3$ 为三维形状，形状描述子是映射：
$$\Phi: \mathcal{S} \rightarrow \mathbb{R}^d$$

其中 $d$ 是描述子维度。两个形状 $\mathcal{S}_1, \mathcal{S}_2$ 的相似度通过描述子空间的距离度量：
$$d(\mathcal{S}_1, \mathcal{S}_2) = \|\Phi(\mathcal{S}_1) - \Phi(\mathcal{S}_2)\|_p$$

典型的 $p$-范数选择：
- $p=1$：Manhattan距离，对异常值鲁棒
- $p=2$：欧氏距离，最常用
- $p=\infty$：Chebyshev距离，关注最大差异

**不变性要求的群论表述**

对于刚体变换群 $SE(3) = SO(3) \ltimes \mathbb{R}^3$，理想描述子应满足：
$$\Phi(g \cdot \mathcal{S}) = \Phi(\mathcal{S}), \quad \forall g \in SE(3)$$

更一般地，对于变换群 $G$ 的作用，不变描述子满足等变性图表：
$$\begin{CD}
\mathcal{S} @>{\Phi}>> \mathbb{R}^d \\
@V{g}VV @V{\rho(g)}VV \\
g \cdot \mathcal{S} @>{\Phi}>> \mathbb{R}^d
\end{CD}$$

其中 $\rho(g) = I$ 对于不变描述子。

对于缩放不变性，引入相似变换群 $\text{Sim}(3) = \mathbb{R}^+ \times SE(3)$：
$$\Phi(s \cdot g \cdot \mathcal{S}) = \Phi(\mathcal{S}), \quad s \in \mathbb{R}^+, g \in SE(3)$$

实现方法：
- 显式归一化：$\tilde{\mathcal{S}} = \mathcal{S} / \text{scale}(\mathcal{S})$
- 比值特征：$\phi_i/\phi_j$ 自然消除尺度
- 对数域表示：$\log(\phi_i) - \log(\phi_j)$

**信息论视角与率失真理论**

从信息论角度，描述子提取过程是有损压缩，目标是最大化互信息：
$$I(\mathcal{S}; \Phi(\mathcal{S})) = H(\mathcal{S}) - H(\mathcal{S}|\Phi(\mathcal{S}))$$

率失真函数定义最优压缩边界：
$$R(D) = \min_{p(\hat{s}|s): \mathbb{E}[d(S,\hat{S})] \leq D} I(S; \hat{S})$$

其中 $D$ 是允许的失真度。拉格朗日形式：
$$\mathcal{L} = I(S; \hat{S}) + \beta \mathbb{E}[d(S,\hat{S})]$$

最优描述子在率失真曲线的拐点处取得信息-紧凑性平衡。

**描述子的判别能力分析**

Fisher判别率定义类间/类内距离比：
$$J(\Phi) = \frac{\text{tr}(S_B)}{\text{tr}(S_W)}$$

其中：
- 类间散度矩阵：$S_B = \sum_i n_i (\mu_i - \mu)(\mu_i - \mu)^T$
- 类内散度矩阵：$S_W = \sum_i \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T$

最优投影通过广义特征值问题求解：
$$S_B v = \lambda S_W v$$

**维度约简与流形学习**

形状空间往往位于高维空间的低维流形上。设形状流形 $\mathcal{M} \subset \mathbb{R}^N$，内在维度 $d \ll N$。

局部线性嵌入（LLE）保持局部几何：
$$\Phi^* = \arg\min_\Phi \sum_i \left\|\Phi(S_i) - \sum_{j \in N(i)} w_{ij} \Phi(S_j)\right\|^2$$

等距映射（Isomap）保持测地距离：
$$\Phi^* = \arg\min_\Phi \sum_{i,j} (d_{\mathbb{R}^d}(\Phi(S_i), \Phi(S_j)) - d_{\mathcal{M}}(S_i, S_j))^2$$

扩散映射利用随机游走：
$$\Phi_t(x) = (\lambda_1^t \psi_1(x), \lambda_2^t \psi_2(x), \ldots)$$

其中 $\lambda_i, \psi_i$ 是转移算子的特征值和特征函数。

### 22.1.2 D2形状分布

**D2描述子定义**

D2形状分布基于形状内随机点对距离的统计分布。对于形状 $\mathcal{S}$，随机采样两点 $p, q \in \mathcal{S}$，其欧氏距离：
$$d_{pq} = \|p - q\|_2$$

D2描述子是距离分布的直方图：
$$h_i = P(d_{pq} \in [r_i, r_{i+1}])$$

归一化形式考虑形状的最大距离：
$$\tilde{h}_i = P\left(\frac{d_{pq}}{d_{\max}} \in \left[\frac{r_i}{d_{\max}}, \frac{r_{i+1}}{d_{\max}}\right]\right)$$

其中 $d_{\max} = \max_{p,q \in \mathcal{S}} \|p - q\|_2$ 是形状直径。

**概率密度估计与核选择**

使用核密度估计（KDE）得到连续分布：
$$f_D(r) = \frac{1}{N} \sum_{i=1}^{N} K\left(\frac{r - d_i}{h}\right)$$

核函数选择：
- 高斯核：$K(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2}$，平滑但可能过度模糊
- Epanechnikov核：$K(u) = \frac{3}{4}(1-u^2) \mathbb{1}_{|u| \leq 1}$，最优MISE
- 三角核：$K(u) = (1-|u|) \mathbb{1}_{|u| \leq 1}$，计算高效

带宽选择（Silverman规则）：
$$h = 1.06 \cdot \sigma \cdot N^{-1/5}$$

交叉验证优化：
$$h^* = \arg\min_h \text{CV}(h) = \arg\min_h \sum_{i=1}^N \left[\hat{f}_{-i}(d_i; h) - f(d_i)\right]^2$$

**统计矩特征的几何意义**

D2分布的统计矩提供紧凑描述：
- 均值：$\mu = \mathbb{E}[d_{pq}] \approx 0.6 \cdot d_{\max}$（紧致形状）
- 方差：$\sigma^2 = \text{Var}[d_{pq}]$，反映形状的伸展程度
- 偏度：$\gamma_1 = \mathbb{E}\left[\left(\frac{d_{pq} - \mu}{\sigma}\right)^3\right]$，检测形状不对称性
- 峰度：$\gamma_2 = \mathbb{E}\left[\left(\frac{d_{pq} - \mu}{\sigma}\right)^4\right] - 3$，识别outlier结构

中心矩的递推计算：
$$m_k = \mathbb{E}[(d_{pq} - \mu)^k] = \frac{1}{N} \sum_{i=1}^N (d_i - \mu)^k$$

标准化矩（尺度不变）：
$$\tilde{m}_k = \frac{m_k}{m_2^{k/2}}$$

**采样策略优化**

均匀采样的方差分析：
$$\text{Var}[\hat{h}_i] = \frac{h_i(1-h_i)}{N}$$

最优采样数（Chernoff界）：
$$N \geq \frac{3}{\epsilon^2} \log\left(\frac{2B}{\delta}\right)$$

保证 $P(|\hat{h}_i - h_i| > \epsilon) < \delta$，其中 $B$ 是直方图箱数。

重要性采样减少方差：
$$\rho^*(p) \propto \sqrt{\text{Var}[d_{p\cdot}]}$$

即在距离方差大的区域密集采样。

自适应采样算法：
1. 初始均匀采样 $N_0$ 点
2. 估计局部方差 $\hat{\sigma}^2(p)$
3. 更新采样密度：$\rho_{t+1}(p) = \rho_t(p) \cdot (1 + \alpha \hat{\sigma}(p))$
4. 迭代直到收敛

**形状族的D2签名**

不同几何类别的D2特征：
- 球体：单峰分布，峰值在 $\sqrt{2}r$
- 立方体：双峰分布，对应面内和对角距离
- 细长形状：右偏分布，$\gamma_1 > 0$
- 分支结构：多峰分布，反映不同尺度

### 22.1.3 光场描述子

**光场表示理论**

光场描述子将形状编码为从各个视角观察的2D投影集合。定义观察球面 $\mathbb{S}^2$，视点 $v \in \mathbb{S}^2$ 的投影图像：
$$I_v = \Pi_v(\mathcal{S})$$

其中 $\Pi_v$ 是沿方向 $v$ 的正交投影算子。

投影算子的数学定义：
$$\Pi_v(p) = p - (p \cdot v)v$$

投影到像平面的坐标变换：
$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} e_1^T \\ e_2^T \end{bmatrix} \Pi_v(p)$$

其中 $\{e_1, e_2, v\}$ 构成正交基。

**Zernike矩描述**

对每个投影图像计算Zernike矩：
$$Z_{nm} = \frac{n+1}{\pi} \int\int_{x^2+y^2 \leq 1} I(x,y) V_{nm}^*(x,y) \, dx dy$$

Zernike多项式的显式形式：
$$V_{nm}(r,\theta) = R_{nm}(r) e^{im\theta}$$

径向多项式通过递推关系计算：
$$R_{nm}(r) = \sum_{k=0}^{(n-|m|)/2} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+|m|}{2}-k\right)! \left(\frac{n-|m|}{2}-k\right)!} r^{n-2k}$$

正交性条件：
$$\int_0^1 R_{nm}(r) R_{n'm}(r) r \, dr = \frac{\delta_{nn'}}{2n+1}$$
$$\int_0^{2\pi} e^{im\theta} e^{-im'\theta} \, d\theta = 2\pi\delta_{mm'}$$

Zernike矩的旋转不变性：投影旋转 $\alpha$ 导致：
$$Z'_{nm} = Z_{nm} e^{-im\alpha}$$

因此 $|Z_{nm}|$ 是旋转不变量。

**球面采样策略**

均匀球面采样（Fibonacci格点）：
$$\theta_i = \arccos(1 - 2i/N)$$
$$\phi_i = 2\pi i \cdot \frac{\sqrt{5}-1}{2}$$

其中 $i = 0, 1, \ldots, N-1$。

测地网格采样（icosahedron细分）：
1. 从正二十面体开始
2. 递归细分每个三角形
3. 投影到单位球面：$v' = v/\|v\|$

采样密度分析（Voronoi面积）：
$$A_i = \int_{\text{Vor}(v_i)} d\Omega \approx \frac{4\pi}{N}$$

**球面积分与调和分析**

光场在观察球面上的积分：
$$L(\mathcal{S}) = \int_{\mathbb{S}^2} \|I_v\|^2 \, dv$$

球谐展开的完整形式：
$$I_v(\theta,\phi) = \sum_{l=0}^{\infty} \sum_{m=-l}^{l} a_{lm} Y_l^m(\theta,\phi)$$

球谐函数的显式表达：
$$Y_l^m(\theta,\phi) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}} P_l^m(\cos\theta) e^{im\phi}$$

展开系数通过内积计算：
$$a_{lm} = \int_{\mathbb{S}^2} I_v(\theta,\phi) Y_l^m(\theta,\phi)^* \sin\theta \, d\theta d\phi$$

Parseval定理：
$$\int_{\mathbb{S}^2} |I_v|^2 \, d\Omega = \sum_{l=0}^{\infty} \sum_{m=-l}^{l} |a_{lm}|^2$$

旋转不变能量谱：
$$E_l = \sum_{m=-l}^{l} |a_{lm}|^2$$

功率谱密度：
$$S(l) = \frac{E_l}{2l+1}$$

**多分辨率光场分析**

带限逼近误差：
$$\epsilon_L = \sum_{l>L} E_l = \int_{\mathbb{S}^2} |I_v - I_v^L|^2 \, d\Omega$$

最优截断选择（保留95%能量）：
$$L^* = \min\left\{L : \sum_{l=0}^L E_l \geq 0.95 \sum_{l=0}^{\infty} E_l\right\}$$

小波光场分析（球面小波）：
$$\psi_{j,k}(\theta,\phi) = 2^j \psi(2^j\theta - k_\theta, 2^j\phi - k_\phi)$$

多尺度系数：
$$w_{j,k} = \langle I_v, \psi_{j,k} \rangle$$

### 22.1.4 球谐描述子

**球谐函数基础**

球谐函数 $Y_l^m(\theta, \phi)$ 是球面上的正交基：
$$Y_l^m(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}} P_l^m(\cos\theta) e^{im\phi}$$

其中 $P_l^m$ 是关联Legendre多项式。

**形状的球谐展开**

将形状表面参数化到单位球面，半径函数：
$$r(\theta, \phi) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} Y_l^m(\theta, \phi)$$

展开系数：
$$c_{lm} = \int_{\mathbb{S}^2} r(\theta, \phi) Y_l^m(\theta, \phi)^* \, d\Omega$$

**旋转不变性构造**

在 $SO(3)$ 群作用下，球谐系数变换为：
$$c'_{lm} = \sum_{m'=-l}^{l} D^l_{mm'}(R) c_{lm'}$$

其中 $D^l$ 是Wigner D-矩阵。构造不变量：
$$I_l = \|c_l\|^2 = \sum_{m=-l}^{l} |c_{lm}|^2$$

**多分辨率分析**

不同阶数 $l$ 捕获不同尺度特征：
- $l=0$：平均半径（体积信息）
- $l=1$：质心偏移
- $l=2$：主轴方向
- $l>2$：细节特征

带限重构误差：
$$\epsilon_L = \int_{\mathbb{S}^2} \left|r(\theta,\phi) - \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} Y_l^m\right|^2 d\Omega$$

### 22.1.5 描述子的不变性分析

**群论视角**

形状空间 $\mathcal{M}$ 在变换群 $G$ 作用下的商空间：
$$\mathcal{M}/G = \{[S] : S \in \mathcal{M}\}$$

其中 $[S] = \{g \cdot S : g \in G\}$ 是等价类。不变描述子是商映射：
$$\Phi: \mathcal{M}/G \rightarrow \mathbb{R}^d$$

**Haar积分与不变性**

对于紧致李群 $G$，使用Haar测度构造不变量：
$$\Phi_{\text{inv}}(S) = \int_G \Phi(g \cdot S) \, dg$$

例如，对 $SO(3)$ 旋转群：
$$\Phi_{\text{rot-inv}}(S) = \frac{1}{8\pi^2} \int_{SO(3)} \Phi(R \cdot S) \, dR$$

**主成分对齐**

通过主成分分析（PCA）实现姿态归一化：
1. 计算惯性张量：
   $$I_{ij} = \int_{\mathcal{S}} (x_i - \bar{x}_i)(x_j - \bar{x}_j) \, dV$$
   
2. 特征分解：$I = V\Lambda V^T$

3. 对齐到主轴：$S' = V^T(S - \bar{x})$

**度量学习与不变性**

学习度量张量 $M$ 使得：
$$d_M(S_1, S_2) = \sqrt{(\Phi(S_1) - \Phi(S_2))^T M (\Phi(S_1) - \Phi(S_2))}$$

优化目标包含不变性约束：
$$\min_M \sum_{i,j} \ell(d_M(S_i, S_j), y_{ij}) + \lambda \sum_{g \in G} d_M(S, g \cdot S)$$

## 22.2 谱分析与Laplacian特征

谱几何将形状分析转化为线性算子的特征值问题，揭示形状的内在几何性质。Laplace-Beltrami算子的谱包含了从全局到局部的多尺度几何信息，在3D打印的形状匹配、分割和变形分析中有重要应用。

### 22.2.1 Laplace-Beltrami算子谱理论

**连续Laplace-Beltrami算子**

对于黎曼流形 $(M, g)$，Laplace-Beltrami算子定义为：
$$\Delta_g f = -\text{div}(\text{grad} f) = -\frac{1}{\sqrt{\det g}} \partial_i \left(\sqrt{\det g} g^{ij} \partial_j f\right)$$

在局部坐标系下，对于曲面嵌入 $\mathbb{R}^3$：
$$\Delta_M f = \frac{1}{\sqrt{EG-F^2}} \left[\frac{\partial}{\partial u}\left(\frac{G f_u - F f_v}{\sqrt{EG-F^2}}\right) + \frac{\partial}{\partial v}\left(\frac{E f_v - F f_u}{\sqrt{EG-F^2}}\right)\right]$$

其中 $E, F, G$ 是第一基本形式系数。

**离散Laplacian矩阵**

余切权重公式（Cotangent Laplacian）：
$$L_{ij} = \begin{cases}
-\frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij}) & \text{if } (i,j) \in E \\
\sum_{k \neq i} -L_{ik} & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}$$

其中 $\alpha_{ij}, \beta_{ij}$ 是边 $(i,j)$ 对角。

归一化形式：
$$\tilde{L} = D^{-1}L$$
其中 $D_{ii} = \frac{1}{3}\sum_{j \in N(i)} A_j$ 是Voronoi面积。

**特征值问题**

广义特征值问题：
$$L\phi = \lambda M\phi$$

其中 $M$ 是质量矩阵（对角矩阵，$M_{ii}$ 为顶点 $i$ 的Voronoi面积）。

特征值满足：
$$0 = \lambda_0 < \lambda_1 \leq \lambda_2 \leq \cdots$$

**Weyl渐近定理**

特征值的渐近行为：
$$\lambda_k \sim \frac{4\pi k}{\text{Area}(M)} \quad \text{as } k \to \infty$$

计数函数：
$$N(\lambda) = \#\{k : \lambda_k \leq \lambda\} \sim \frac{\text{Area}(M)}{4\pi} \lambda + O(\sqrt{\lambda})$$

**谱与几何的关系**

听鼓问题（Can one hear the shape of a drum?）：
- 面积：$\text{Area} = \lim_{t \to 0^+} \sum_{k} e^{-\lambda_k t}$
- 周长：通过热核迹的渐近展开获得
- 亏格：通过谱的多重性结构推断

### 22.2.2 特征函数与振动模态

**特征函数的物理意义**

Helmholtz方程：
$$\Delta \phi + \lambda \phi = 0$$

对应于弹性膜的振动模态，$\sqrt{\lambda_k}$ 是第 $k$ 个固有频率。

**节点集与节点域**

第 $k$ 个特征函数的节点集：
$$N_k = \{x \in M : \phi_k(x) = 0\}$$

Courant节点域定理：$\phi_k$ 将流形分割成至多 $k$ 个连通分量。

**特征函数的正交性**

$$\int_M \phi_i \phi_j \, dV = \delta_{ij}$$
$$\int_M \nabla \phi_i \cdot \nabla \phi_j \, dV = \lambda_i \delta_{ij}$$

**Fiedler向量与谱分割**

第二特征函数（Fiedler向量）最小化：
$$\phi_1 = \arg\min_{\phi \perp \mathbf{1}} \frac{\int_M |\nabla \phi|^2 \, dV}{\int_M \phi^2 \, dV}$$

用于形状的二分分割：
$$M = M^+ \cup M^-, \quad M^{\pm} = \{x : \pm\phi_1(x) > 0\}$$

**高阶特征函数的对称性**

利用群表示理论分析特征空间的对称性：
- 如果 $g \in \text{Isom}(M)$，则 $\phi_k \circ g$ 也是特征函数
- 特征空间按对称群的不可约表示分解

### 22.2.3 Heat Kernel Signature

**热方程与热核**

热方程：
$$\frac{\partial u}{\partial t} = -\Delta u$$

热核 $k_t(x,y)$ 是基本解：
$$u(x,t) = \int_M k_t(x,y) u_0(y) \, dy$$

谱展开：
$$k_t(x,y) = \sum_{i=0}^{\infty} e^{-\lambda_i t} \phi_i(x) \phi_i(y)$$

**Heat Kernel Signature (HKS)**

点 $x$ 的HKS定义为：
$$h_t(x) = k_t(x,x) = \sum_{i=0}^{\infty} e^{-\lambda_i t} \phi_i(x)^2$$

物理意义：从点 $x$ 出发的热量经时间 $t$ 后返回的概率。

**多尺度特性**

不同时间尺度捕获不同几何特征：
- 小 $t$：局部几何（曲率）
- 大 $t$：全局几何（体积、连通性）

短时渐近展开：
$$h_t(x) \sim \frac{1}{4\pi t} \left(1 + \frac{t}{6}K(x) + O(t^2)\right)$$

其中 $K(x)$ 是高斯曲率。

**尺度不变HKS**

对数采样时间：$t_j = t_{\min} \cdot \alpha^j$

归一化：
$$\tilde{h}_t(x) = \frac{h_t(x)}{\int_M h_t(y) \, dy}$$

**HKS的稳定性**

对于近似等距变形，HKS的变化有界：
$$|h_t(x) - h_t(f(x))| \leq C \cdot d_{\text{GH}}(M_1, M_2)$$

其中 $d_{\text{GH}}$ 是Gromov-Hausdorff距离。

### 22.2.4 Wave Kernel Signature

**波动方程与波核**

波动方程：
$$\frac{\partial^2 u}{\partial t^2} = -\Delta u$$

波核解：
$$u(x,t) = \int_M w_t(x,y) u_0(y) \, dy$$

谱表示：
$$w_t(x,y) = \sum_{i=0}^{\infty} \cos(\sqrt{\lambda_i} t) \phi_i(x) \phi_i(y)$$

**Wave Kernel Signature (WKS)**

能量概率分布：
$$\text{WKS}_e(x) = \sum_{i=0}^{\infty} \phi_i(x)^2 \psi_e(\lambda_i)$$

其中 $\psi_e$ 是能量滤波器：
$$\psi_e(\lambda) = C \exp\left(-\frac{(\log e - \log \lambda)^2}{2\sigma^2}\right)$$

**频率选择性**

WKS在频域的局部化优于HKS：
- HKS：低通滤波器 $e^{-\lambda t}$
- WKS：带通滤波器 $\psi_e(\lambda)$

**量子力学解释**

WKS对应量子粒子在能量 $e$ 处的位置概率密度：
$$|\langle x | E \rangle|^2 = \sum_{i: \lambda_i \approx E} |\phi_i(x)|^2$$

**尺度空间分析**

对数能量采样：$e_j = e_{\min} \cdot \gamma^j$

方差选择：$\sigma = k \cdot \Delta \log e$，其中 $k \approx 7$ 保证充分重叠。

### 22.2.5 谱距离与谱嵌入

**扩散距离**

基于热核的扩散距离：
$$d_t^2(x,y) = \sum_{z \in M} \frac{(k_t(x,z) - k_t(y,z))^2}{h_t(z)}$$

简化形式：
$$d_t^2(x,y) = \sum_{i=1}^{\infty} e^{-2\lambda_i t} (\phi_i(x) - \phi_i(y))^2$$

**谱嵌入**

将形状嵌入到特征空间：
$$\Psi: M \rightarrow \mathbb{R}^k$$
$$\Psi(x) = (\phi_1(x), \phi_2(x), \ldots, \phi_k(x))$$

嵌入保持扩散距离的近似：
$$\|Ψ(x) - Ψ(y)\|^2 \approx d_\infty^2(x,y)$$

**交换距离**

测量两个形状谱的差异：
$$d_{\text{CD}}(M_1, M_2) = \|\lambda(M_1) - \lambda(M_2)\|_2$$

其中 $\lambda(M) = (\lambda_1, \lambda_2, \ldots)$ 是特征值序列。

**Gromov-Wasserstein距离**

结合谱信息的形状距离：
$$d_{GW}(M_1, M_2) = \min_{\pi} \int\int |d_{M_1}(x,y) - d_{M_2}(x',y')|^2 \, d\pi(x,x') d\pi(y,y')$$

可通过谱近似加速计算。

**谱同步**

多个形状的联合谱分析：给定形状集合 $\{M_i\}$，寻找一致的谱基：
$$\min_{\{R_i \in SO(k)\}} \sum_{i,j} \|R_i \Phi_i - R_j \Phi_j\|_F^2$$

其中 $\Phi_i$ 是形状 $M_i$ 的前 $k$ 个特征函数矩阵。

## 22.3 最优传输与Wasserstein距离

最优传输理论提供了比较概率分布和形状的强大数学框架。在3D打印中，它用于形状插值、变形设计、材料分布优化等。Wasserstein距离考虑了几何结构，比简单的点对点距离更适合形状分析。

### 22.3.1 Monge-Kantorovich问题

**Monge问题（1781）**

给定源分布 $\mu$ 和目标分布 $\nu$，寻找传输映射 $T: X \rightarrow Y$ 最小化：
$$\min_{T_\#\mu = \nu} \int_X c(x, T(x)) \, d\mu(x)$$

其中 $T_\#\mu$ 是推前测度，$c(x,y)$ 是传输成本。

**Kantorovich松弛（1942）**

允许质量分裂，引入传输计划 $\pi \in \Pi(\mu, \nu)$：
$$\min_{\pi \in \Pi(\mu,\nu)} \int_{X \times Y} c(x,y) \, d\pi(x,y)$$

约束条件：
$$\int_Y d\pi(x,y) = d\mu(x), \quad \int_X d\pi(x,y) = d\nu(y)$$

**Wasserstein-p距离**

当 $c(x,y) = d(x,y)^p$ 时，定义 $p$-Wasserstein距离：
$$W_p(\mu,\nu) = \left(\inf_{\pi \in \Pi(\mu,\nu)} \int_{X \times Y} d(x,y)^p \, d\pi(x,y)\right)^{1/p}$$

最常用的是 $W_2$（二次Wasserstein距离）。

**对偶问题**

Kantorovich对偶：
$$W_1(\mu,\nu) = \sup_{f: \text{Lip}(f) \leq 1} \int f \, d(\mu - \nu)$$

一般对偶形式：
$$\int c \, d\pi^* = \sup_{\phi \oplus \psi \leq c} \int \phi \, d\mu + \int \psi \, d\nu$$

其中 $\phi \oplus \psi(x,y) = \phi(x) + \psi(y)$。

**Brenier定理**

对于 $\mathbb{R}^n$ 上的绝对连续测度，二次成本的最优传输映射是某凸函数的梯度：
$$T^*(x) = \nabla \phi(x)$$

其中 $\phi$ 满足Monge-Ampère方程：
$$\det(D^2\phi) = \frac{\rho_\mu(\nabla\phi^{-1})}{\rho_\nu}$$

### 22.3.2 离散最优传输

**离散设置**

源分布：$\mu = \sum_{i=1}^n a_i \delta_{x_i}$，目标分布：$\nu = \sum_{j=1}^m b_j \delta_{y_j}$

其中 $a \in \Delta_n$, $b \in \Delta_m$ 是概率单纯形。

**线性规划形式**

$$\min_{P \in \mathbb{R}_+^{n \times m}} \langle C, P \rangle_F$$

约束条件：
$$P\mathbf{1}_m = a, \quad P^T\mathbf{1}_n = b$$

其中 $C_{ij} = c(x_i, y_j)$ 是成本矩阵。

**计算复杂度**

- 精确解：$O(n^3 \log n)$（网络单纯形法）
- 近似解：$O(n^2/\epsilon)$（Sinkhorn算法）

**半离散最优传输**

源为离散，目标为连续：寻找Laguerre细胞分解
$$V_i = \{y : \|y - x_i\|^2 - \psi_i \leq \|y - x_j\|^2 - \psi_j, \forall j\}$$

使得 $\int_{V_i} d\nu(y) = a_i$。

**正则化传输**

熵正则化：
$$W_\epsilon(\mu,\nu) = \min_{P \in \Pi(a,b)} \langle C, P \rangle - \epsilon H(P)$$

其中 $H(P) = -\sum_{ij} P_{ij} \log P_{ij}$ 是熵。

### 22.3.3 Wasserstein重心

**重心问题**

给定测度 $\{\mu_k\}_{k=1}^K$ 和权重 $\{\lambda_k\}$，Wasserstein重心：
$$\bar{\mu} = \arg\min_\mu \sum_{k=1}^K \lambda_k W_2^2(\mu, \mu_k)$$

**固定支撑重心**

限制重心支撑在给定点集 $\{x_i\}_{i=1}^n$：
$$\min_{a \in \Delta_n} \sum_{k=1}^K \lambda_k W_2^2\left(\sum_i a_i \delta_{x_i}, \mu_k\right)$$

**自由支撑重心**

同时优化权重和位置：
$$\min_{a \in \Delta_n, X \in \mathbb{R}^{n \times d}} \sum_{k=1}^K \lambda_k W_2^2\left(\sum_i a_i \delta_{x_i}, \mu_k\right)$$

**迭代算法**

交替优化：
1. 固定支撑，更新权重（线性规划）
2. 固定权重，更新支撑（梯度下降）

收敛到局部最优。

**连续重心的特征**

一阶最优性条件：存在传输映射 $T_k: \text{supp}(\bar{\mu}) \rightarrow \text{supp}(\mu_k)$ 使得：
$$\sum_{k=1}^K \lambda_k T_k(x) = x, \quad \forall x \in \text{supp}(\bar{\mu})$$

### 22.3.4 Gromov-Wasserstein距离

**动机**

比较不同度量空间中的分布，无需嵌入到公共空间。

**定义**

$$GW_p(\mu,\nu) = \inf_{\pi \in \Pi(\mu,\nu)} \left(\int\int |d_X(x,x') - d_Y(y,y')|^p \, d\pi(x,y) d\pi(x',y')\right)^{1/p}$$

**四阶张量形式**

离散情况下，定义张量：
$$L_{ijkl} = |d_X(x_i, x_k) - d_Y(y_j, y_l)|^p$$

优化问题：
$$\min_{P \in \Pi(a,b)} \sum_{ijkl} L_{ijkl} P_{ij} P_{kl}$$

**下界与不变性**

GW距离的下界（通过谱）：
$$GW(\mu,\nu) \geq c \cdot |\lambda_1(\Delta_\mu) - \lambda_1(\Delta_\nu)|$$

不变性：对等距变换不变。

**Fused Gromov-Wasserstein**

结合特征和几何：
$$FGW_{\alpha}(\mu,\nu) = \min_{\pi} (1-\alpha) \int c_{feat} \, d\pi + \alpha \int\int c_{geom} \, d\pi^{\otimes 2}$$

**计算方法**

- 条件梯度法（Frank-Wolfe）
- 近端点算法
- 熵正则化加速

### 22.3.5 计算方法：Sinkhorn算法

**Sinkhorn-Knopp算法**

熵正则化问题的解具有形式：
$$P^* = \text{diag}(u) K \text{diag}(v)$$

其中 $K_{ij} = \exp(-C_{ij}/\epsilon)$ 是Gibbs核。

**Sinkhorn迭代**

交替更新缩放向量：
$$u^{(t+1)} = \frac{a}{Kv^{(t)}}, \quad v^{(t+1)} = \frac{b}{K^Tu^{(t+1)}}$$

收敛到唯一不动点。

**收敛分析**

线性收敛率：
$$\|P^{(t)} - P^*\|_1 \leq C \cdot \kappa^t$$

其中 $\kappa = \frac{\lambda_2(K)}{\lambda_1(K)} < 1$。

**Log域稳定化**

避免数值下溢：
$$\log u^{(t+1)} = \log a - \text{LSE}(\log K + \log v^{(t)})$$

其中 $\text{LSE}$ 是log-sum-exp操作。

**自适应熵退火**

$\epsilon$-scaling策略：
1. 从大 $\epsilon_0$ 开始
2. 逐步减小：$\epsilon_{k+1} = \gamma \epsilon_k$
3. 使用前一步解作为热启动

**多尺度Sinkhorn**

粗细网格策略：
1. 在粗网格上求解
2. 插值到细网格
3. 局部细化

复杂度从 $O(n^2)$ 降到 $O(n\log n)$。

**GPU加速**

矩阵运算并行化：
- 向量外积：$u \otimes v$
- 矩阵向量乘：$Kv$
- 逐元素运算：$\exp, \log, /$

批处理多个传输问题。

## 22.4 功能性分析与协同分割

功能性分析关注形状的语义理解和功能推断，这对3D打印的智能设计至关重要。协同分割通过分析多个相关形状来发现一致的部件结构，支持模块化设计和部件重用。

### 22.4.1 功能空间理论

**功能性建模框架**

形状功能定义为交互空间 $\mathcal{I}$ 到任务空间 $\mathcal{T}$ 的映射：
$$F: \mathcal{S} \times \mathcal{I} \rightarrow \mathcal{T}$$

其中 $\mathcal{S}$ 是形状空间。

**可供性（Affordance）理论**

Gibson的可供性概念的数学化：
$$A(S) = \{(p, n, a) : \text{contact}(p) \wedge \text{normal}(n) \wedge \text{action}(a)\}$$

可供性场：
$$\phi_a: \mathcal{S} \rightarrow \mathbb{R}^+$$
表示位置 $x$ 执行动作 $a$ 的适合度。

**功能基元分解**

将复杂功能分解为基元：
$$F = \sum_{i=1}^K w_i f_i$$

其中 $f_i$ 是功能基元（如支撑、包含、连接）。

**交互张量**

三阶张量编码人-物-环境交互：
$$\mathcal{T}_{ijk} = P(\text{human}_i, \text{object}_j, \text{context}_k)$$

通过张量分解提取潜在功能模式：
$$\mathcal{T} = \sum_{r=1}^R \lambda_r \cdot u_r \otimes v_r \otimes w_r$$

**功能一致性度量**

两个形状的功能相似度：
$$d_F(S_1, S_2) = \int_{\mathcal{I}} \|F(S_1, \xi) - F(S_2, \xi)\|^2 \, d\xi$$

### 22.4.2 关节运动分析

**运动学链建模**

关节形状表示为运动学链：
$$\mathcal{K} = (V, E, \Theta)$$

其中 $V$ 是部件集，$E$ 是关节连接，$\Theta$ 是关节参数。

**Denavit-Hartenberg参数**

标准化关节描述：
$$T_i^{i-1} = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}$$

**运动子空间**

部件 $i$ 的可达空间：
$$\mathcal{M}_i = \{T(\theta) \cdot p : \theta \in \Theta_{\text{valid}}, p \in P_i\}$$

**关节类型识别**

通过运动轨迹分析识别关节类型：
- 旋转关节：$\text{rank}(\text{cov}(\mathcal{M})) = 2$
- 平移关节：$\text{rank}(\text{cov}(\mathcal{M})) = 1$
- 球形关节：$\text{rank}(\text{cov}(\mathcal{M})) = 3$

**逆运动学优化**

给定目标配置，求解关节参数：
$$\theta^* = \arg\min_\theta \|f(\theta) - x_{\text{target}}\|^2 + \lambda R(\theta)$$

其中 $R(\theta)$ 是正则项（避免奇异配置）。

### 22.4.3 协同分割算法

**协同分割目标函数**

给定形状集合 $\{S_i\}_{i=1}^N$，寻找一致分割：
$$\min_{\{L_i\}} \sum_{i=1}^N E_{\text{unary}}(L_i, S_i) + \lambda \sum_{i,j} E_{\text{pair}}(L_i, L_j)$$

**一元项（Unary Term）**

形状内部的分割质量：
$$E_{\text{unary}}(L, S) = \sum_{f \in F} \sum_{l \in L} \delta(l_f \neq l) \cdot w_{ff'}$$

基于：
- 凹性：边界应在凹区域
- 紧致性：部件应紧凑
- 对称性：保持对称结构

**成对项（Pairwise Term）**

跨形状的一致性：
$$E_{\text{pair}}(L_i, L_j) = -\sum_{p,q} C_{ij}(p,q) \cdot \mathbb{1}[L_i(p) = L_j(q)]$$

其中 $C_{ij}$ 是对应矩阵。

**谱聚类协同分割**

构建超图 $H = (V, E)$：
- 节点：所有形状的所有面片
- 超边：连接相似面片

归一化割：
$$\text{NCut}(A, B) = \frac{\text{cut}(A,B)}{\text{vol}(A)} + \frac{\text{cut}(A,B)}{\text{vol}(B)}$$

**深度协同分割**

使用图神经网络：
$$h_v^{(k+1)} = \sigma\left(W_{\text{self}} h_v^{(k)} + \sum_{u \in N(v)} W_{\text{neigh}} h_u^{(k)}\right)$$

跨形状注意力：
$$\alpha_{ij} = \frac{\exp(h_i^T W_Q W_K^T h_j)}{\sum_k \exp(h_i^T W_Q W_K^T h_k)}$$

### 22.4.4 层次结构分析

**层次分解树**

形状的层次表示：
$$T = (N, E, \phi)$$

其中 $N$ 是节点（部件），$E$ 是父子关系，$\phi$ 是几何映射。

**自底向上聚类**

凝聚层次聚类：
$$d(C_i, C_j) = \min_{p \in C_i, q \in C_j} d_{\text{geom}}(p, q) + \lambda d_{\text{sem}}(C_i, C_j)$$

**自顶向下分割**

递归二分：
$$(L_{\text{left}}, L_{\text{right}}) = \arg\min_{L} \text{NCut}(L, \bar{L})$$

停止准则：
$$\frac{\text{NCut}(L, \bar{L})}{\text{NCut}_{\text{random}}} > \tau$$

**多分辨率分析**

不同层次的特征：
$$f^{(l)} = \text{pool}(f^{(l-1)})$$

池化操作保持几何和拓扑信息。

**语法规则学习**

从层次结构提取生成规则：
$$R: A \rightarrow BC \quad \text{with probability } p(B,C|A)$$

使用PCFG（概率上下文无关文法）建模。

### 22.4.5 部件关系图模型

**部件关系图**

$$G = (V, E, X, R)$$

其中：
- $V$：部件节点
- $E$：关系边
- $X$：节点特征（几何、外观）
- $R$：边特征（相对位置、连接类型）

**空间关系编码**

相对位置描述子：
$$r_{ij} = [\Delta p_{ij}, \Delta q_{ij}, \Delta s_{ij}, \theta_{ij}]$$

包含平移、旋转、缩放、方向。

**图匹配能量**

$$E(m) = \sum_{i} \psi_i(m_i) + \sum_{(i,j)} \psi_{ij}(m_i, m_j)$$

其中：
- $\psi_i$：节点匹配代价
- $\psi_{ij}$：边匹配代价

**谱松弛求解**

松弛为特征值问题：
$$M^* = \arg\max_{M^T M = I} \text{tr}(M^T A M)$$

其中 $A$ 是亲和矩阵。

**概率图模型**

条件随机场（CRF）：
$$P(L|S) = \frac{1}{Z} \exp\left(-\sum_i \phi_i(l_i, S) - \sum_{ij} \psi_{ij}(l_i, l_j, S)\right)$$

推断使用：
- 信念传播（BP）
- 图割（Graph Cuts）
- 平均场近似

**部件组合优化**

给定部件库 $\mathcal{P}$，生成新设计：
$$S^* = \arg\max_{S \in \text{Comp}(\mathcal{P})} F(S) - \lambda C(S)$$

其中 $F$ 是功能得分，$C$ 是组装成本。

## 22.5 图匹配与对应关系

形状对应是许多3D打印应用的核心，包括形状变形、纹理传输、统计形状分析等。现代方法结合了谱几何、优化理论和深度学习，实现了鲁棒的密集对应计算。

### 22.5.1 形状对应问题建模

**点对应表示**

对应映射 $\Pi: M \rightarrow N$ 满足：
- 双射性（理想情况）：$\Pi$ 是一对一映射
- 近等距性：$d_N(\Pi(x), \Pi(y)) \approx d_M(x, y)$

**函数对应表示**

线性算子 $T: L^2(N) \rightarrow L^2(M)$：
$$T(g)(x) = g(\Pi(x))$$

满足：
$$T(g \cdot h) = T(g) \cdot T(h)$$

**模糊对应**

软对应矩阵 $C \in \mathbb{R}^{n \times m}$：
$$C_{ij} = P(x_i \leftrightarrow y_j)$$

约束：
$$C\mathbf{1} = \mathbf{1}, \quad C^T\mathbf{1} = \mathbf{1}$$

**等距形变模型**

内在等距：
$$d_M^{geod}(x, y) = d_N^{geod}(\Pi(x), \Pi(y))$$

近似等距（$\epsilon$-等距）：
$$|d_M(x,y) - d_N(\Pi(x), \Pi(y))| \leq \epsilon$$

**对应质量度量**

测地误差：
$$e_{geod}(\Pi) = \frac{1}{|M|} \int_M d_N^{geod}(\Pi(x), \Pi^*(x)) \, dx$$

其中 $\Pi^*$ 是真实对应。

### 22.5.2 函数映射框架

**函数映射定义**

给定基函数 $\{\phi_i^M\}$ 和 $\{\phi_j^N\}$，函数映射矩阵：
$$C_{ij} = \langle T\phi_j^N, \phi_i^M \rangle$$

函数传输：
$$T(f) = \Phi^M C \Phi^{N\dagger} f$$

**约束构造**

1. **描述子保持**：
   $$C A = B$$
   其中 $A, B$ 是描述子矩阵

2. **算子交换性**：
   $$C \Delta^N = \Delta^M C$$

3. **向量场保持**：
   $$C D_v^N = D_u^M C$$
   其中 $D_v$ 是方向导数算子

**优化问题**

$$\min_C \|CA - B\|_F^2 + \mu_1 \|C\Delta^N - \Delta^M C\|_F^2 + \mu_2 \|C\|_{2,1}$$

其中 $\|C\|_{2,1} = \sum_j \|C_j\|_2$ 促进列稀疏性。

**从函数映射到点对应**

最近邻搜索：
$$\Pi(x) = \arg\min_{y \in N} \|\Phi^M(x) C - \Phi^N(y)\|^2$$

或使用ZoomOut细化：迭代增加基函数数量。

### 22.5.3 谱匹配方法

**谱嵌入匹配**

使用前 $k$ 个特征函数嵌入：
$$x \mapsto (\phi_1(x), \ldots, \phi_k(x))$$

匹配通过Procrustes分析：
$$R^* = \arg\min_{R \in O(k)} \|\Phi^M - \Phi^N P R\|_F^2$$

**符号翻转和特征值重复**

处理符号歧义：
$$\phi_i \sim \pm \phi_i$$

处理特征值简并：使用特征空间的任意正交变换。

**热核匹配**

基于热核签名的匹配能量：
$$E(\Pi) = \int_M \int_0^T |h_t^M(x) - h_t^N(\Pi(x))|^2 \, dt \, dx$$

**持续性匹配**

使用持续同调特征：
$$\Pi^* = \arg\min_\Pi d_{B}(PD(M), \Pi_*(PD(N)))$$

其中 $d_B$ 是瓶颈距离，$PD$ 是持续图。

### 22.5.4 深度函数映射

**无监督深度函数映射**

网络架构：
$$C = f_\theta(\Delta^M, \Delta^N, F^M, F^N)$$

损失函数：
$$\mathcal{L} = \mathcal{L}_{desc} + \lambda_1 \mathcal{L}_{comm} + \lambda_2 \mathcal{L}_{ortho}$$

其中：
- $\mathcal{L}_{desc}$：描述子保持
- $\mathcal{L}_{comm}$：算子交换
- $\mathcal{L}_{ortho} = \|C^TC - I\|_F^2$：正交性

**几何深度函数映射**

结合几何特征：
$$F = \text{concat}(HKS, WKS, SHOT, \text{normals})$$

特征提取网络：
$$h = \text{DiffusionNet}(M, F)$$

**注意力机制**

跨形状注意力：
$$\text{Attention}(Q^M, K^N, V^N) = \text{softmax}\left(\frac{Q^M (K^N)^T}{\sqrt{d}}\right) V^N$$

**端到端学习**

直接预测点对应：
$$\Pi = \text{SoftMax}(\Phi^M W \Phi^{N^T})$$

使用Gumbel-Softmax实现可微采样。

### 22.5.5 周期一致性约束

**双向一致性**

前向-后向一致性：
$$\Pi_{MN} \circ \Pi_{NM} \approx \text{id}_M$$

函数映射形式：
$$C_{MN} C_{NM} \approx I$$

**多路一致性**

给定形状集合 $\{S_i\}$，周期一致性：
$$C_{12} C_{23} \cdots C_{n1} \approx I$$

**一致性正则化**

优化问题：
$$\min_{\{C_{ij}\}} \sum_{i,j} E(C_{ij}) + \lambda \sum_{\text{cycle}} \|C_{\text{cycle}} - I\|_F^2$$

**同步优化**

联合优化所有对应：
$$\min_{\{C_{ij}\}} \sum_{i,j} \|C_{ij} - C_{ij}^{init}\|_F^2 \quad \text{s.t.} \quad C_{ij} C_{jk} = C_{ik}$$

使用交替方向乘子法（ADMM）求解。

**置换同步**

离散版本：寻找置换矩阵 $\{P_{ij}\}$ 满足：
$$P_{ij} P_{jk} = P_{ik}$$

谱松弛：
$$\min_X \|X - P \otimes I_d\|_F^2 \quad \text{s.t.} \quad X^TX = nI$$

## 本章小结

本章系统介绍了形状分析与检索的核心数学方法：

1. **形状描述子**：从D2、光场到球谐描述子，提供了多种编码三维几何的方法，强调了不变性和判别性的平衡。

2. **谱分析**：Laplace-Beltrami算子的特征分析揭示了形状的内在几何，HKS和WKS提供了多尺度的几何签名。

3. **最优传输**：Wasserstein距离为形状比较提供了几何感知的度量，Sinkhorn算法实现了高效计算。

4. **功能分析**：从可供性理论到协同分割，展示了语义理解在形状分析中的重要性。

5. **对应计算**：函数映射框架统一了连续和离散的对应表示，深度学习方法提升了鲁棒性。

关键数学工具：
- 谱理论：$\Delta\phi = \lambda\phi$
- 最优传输：$W_p(\mu,\nu) = \inf_\pi \int c \, d\pi$
- 函数映射：$T(f) = \Phi^M C \Phi^{N\dagger} f$
- 周期一致性：$C_{12}C_{23}C_{31} = I$

## 练习题

### 基础题

1. **球谐展开计算**
   给定单位球面上的函数 $f(\theta,\phi) = \cos^2\theta$，计算其前3阶球谐系数。
   
   *提示*：利用 $\cos\theta = Y_1^0$ 和球谐函数的正交性。

2. **离散Laplacian验证**
   对于正四面体网格，计算余切权重Laplacian矩阵，验证其特征值满足 $\lambda_0 = 0$。
   
   *提示*：利用对称性简化计算。

3. **Wasserstein距离计算**
   计算一维情况下 $\mu = \delta_0$ 和 $\nu = \frac{1}{2}(\delta_{-1} + \delta_1)$ 的2-Wasserstein距离。
   
   *提示*：直接构造最优传输计划。

4. **HKS稳定性**
   证明对于紧致流形，HKS在 $t \to \infty$ 时收敛到常数。
   
   *提示*：考虑特征值 $\lambda_0 = 0$ 的贡献。

### 挑战题

5. **谱同构问题**
   构造两个非同构的图，使其Laplacian特征值完全相同（同谱异构）。
   
   *提示*：考虑Sunada构造。

6. **函数映射的秩**
   证明：如果两个形状之间存在部分对应（非满射），则函数映射矩阵 $C$ 的秩小于 $\min(m,n)$。
   
   *提示*：分析零空间的维度。

7. **Gromov-Wasserstein下界**
   推导GW距离基于直径的下界：$GW(M,N) \geq \frac{1}{2}|\text{diam}(M) - \text{diam}(N)|$。
   
   *提示*：考虑最远点对的传输。

8. **周期一致性的谱特征**
   设 $C_{12}, C_{23}, C_{31}$ 是三个函数映射矩阵。证明：如果它们满足周期一致性，则乘积 $C_{12}C_{23}C_{31}$ 的特征值都是1。
   
   *提示*：利用特征值的乘性。

<details>
<summary>参考答案</summary>

1. $c_{00} = \sqrt{\pi/3}$, $c_{20} = 2\sqrt{\pi/45}$，其余为0。

2. 对于边长为 $a$ 的正四面体，非对角元素 $L_{ij} = -1/\sqrt{3}$（相邻顶点）。

3. $W_2^2 = 1$（质量均分传输到两点）。

4. 当 $t \to \infty$，$h_t(x) \to \phi_0^2 = 1/\text{vol}(M)$。

5. 例：两个16顶点的图，度序列相同但不同构。

6. 部分对应导致某些基函数无法被表示，降低矩阵秩。

7. 使用三角不等式和最优传输的性质。

8. 周期一致性意味着 $C_{cycle} = I$，故特征值为1。

</details>

## 常见陷阱与错误

### 1. 描述子设计陷阱
- **错误**：过度追求不变性，丧失判别能力
- **正确**：根据应用场景平衡不变性和判别性

### 2. 谱计算数值问题
- **错误**：直接计算大规模Laplacian的所有特征值
- **正确**：使用迭代方法（Lanczos）计算前 $k$ 个特征值

### 3. 最优传输正则化
- **错误**：熵正则化参数 $\epsilon$ 选择过小导致数值不稳定
- **正确**：使用 $\epsilon$-scaling策略，从大到小退火

### 4. 函数映射初始化
- **错误**：随机初始化函数映射矩阵
- **正确**：使用描述子匹配或ICP提供初始估计

### 5. 对应评估偏差
- **错误**：仅在训练集形状类别上评估
- **正确**：使用跨类别、部分匹配等多样化测试

## 最佳实践检查清单

### 形状描述子选择
- [ ] 分析目标应用的不变性需求
- [ ] 评估计算复杂度vs精度权衡
- [ ] 考虑多尺度/多分辨率描述子融合
- [ ] 验证对噪声和采样密度的鲁棒性

### 谱方法实施
- [ ] 选择合适的Laplacian离散化（余切权重、图Laplacian）
- [ ] 确定所需特征函数数量（通常50-200）
- [ ] 实现数值稳定的特征值求解器
- [ ] 处理特征值重复和符号歧义

### 最优传输计算
- [ ] 根据问题规模选择算法（精确vs近似）
- [ ] 合理设置熵正则化参数
- [ ] 实现GPU加速（如需要）
- [ ] 验证传输计划的合法性（行列和约束）

### 对应关系验证
- [ ] 检查双向一致性
- [ ] 评估测地误差分布
- [ ] 可视化对应结果
- [ ] 测试极端案例（大变形、部分匹配）

### 性能优化
- [ ] 使用空间数据结构加速邻域查询
- [ ] 实现多分辨率/分层策略
- [ ] 考虑并行化和向量化
- [ ] 缓存重复计算的中间结果