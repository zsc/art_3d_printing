# 第21章：深度生成模型

本章深入探讨深度学习在3D形状生成中的应用，涵盖从早期的体素GAN到最新的扩散模型。我们将详细推导各种生成模型的数学原理，分析其在3D打印中的实际应用，并讨论如何控制生成过程以满足设计约束。重点关注模型的表示能力、训练稳定性和生成质量之间的权衡。

## 21.1 3D-GAN与体素生成

### 21.1.1 体素表示的数学基础

体素（Voxel）是3D空间的规则网格离散化，可视为2D图像像素在三维的推广。对于分辨率为 $N^3$ 的体素网格，占用函数定义为：

$$O: \{0,1,...,N-1\}^3 \rightarrow \{0,1\}$$

其中 $O(i,j,k) = 1$ 表示体素被占用，$O(i,j,k) = 0$ 表示空体素。

连续形式可以表示为指示函数：
$$\chi(x,y,z) = \begin{cases} 
1 & \text{if } (x,y,z) \in \mathcal{S} \\
0 & \text{otherwise}
\end{cases}$$

体素化过程涉及采样定理。根据Nyquist准则，为准确表示具有最大频率 $f_{max}$ 的形状细节，体素分辨率需满足：

$$\Delta x \leq \frac{1}{2f_{max}}$$

**概率体素表示**：实际应用中常采用概率占用：
$$O: \{0,1,...,N-1\}^3 \rightarrow [0,1]$$

这允许表示不确定性和软边界。概率解释：
$$p(occupied|i,j,k) = O(i,j,k)$$

**有符号距离场（SDF）体素化**：
$$\Phi: \{0,1,...,N-1\}^3 \rightarrow \mathbb{R}$$

其中 $\Phi(i,j,k)$ 表示到最近表面的有符号距离：
- $\Phi < 0$：内部
- $\Phi = 0$：表面
- $\Phi > 0$：外部

SDF的梯度给出表面法向：
$$\mathbf{n} = \frac{\nabla\Phi}{||\nabla\Phi||}$$

**稀疏体素表示**：
大多数体素为空，可用稀疏数据结构：
- 哈希表：$\mathcal{H}: \mathbb{Z}^3 \rightarrow \{0,1\}$
- 八叉树：递归细分非空区域
- Run-length编码：压缩连续空体素

稀疏率定义：
$$\rho = \frac{|\{(i,j,k): O(i,j,k) = 1\}|}{N^3}$$

典型3D模型的稀疏率 $\rho < 0.1$。

### 21.1.2 生成对抗网络基础

GAN通过对抗训练学习数据分布。生成器 $G: \mathcal{Z} \rightarrow \mathcal{X}$ 从潜在空间映射到数据空间，判别器 $D: \mathcal{X} \rightarrow [0,1]$ 估计样本来自真实分布的概率。

原始GAN目标函数：
$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

在最优判别器 $D^*$ 下，生成器的损失等价于最小化JS散度：
$$L_G = 2JS(p_{data} || p_G) - 2\log 2$$

其中Jensen-Shannon散度定义为：
$$JS(P||Q) = \frac{1}{2}KL(P||M) + \frac{1}{2}KL(Q||M), \quad M = \frac{P+Q}{2}$$

### 21.1.3 3D-GAN架构

3D-GAN将2D卷积扩展到3D空间。3D卷积操作：
$$Y_{i,j,k,c'} = \sum_{di,dj,dk,c} W_{di,dj,dk,c,c'} \cdot X_{i+di,j+dj,k+dk,c} + b_{c'}$$

**生成器架构详解**：

生成器采用分数步长卷积（转置卷积）逐步上采样：
- 输入：$z \in \mathbb{R}^{200}$ (潜在向量)
- 全连接层：$z \rightarrow \mathbb{R}^{256 \times 4 \times 4 \times 4}$
- 3D转置卷积层序列：$4^3 \rightarrow 8^3 \rightarrow 16^3 \rightarrow 32^3 \rightarrow 64^3$

每层的详细配置：
```
Layer 1: ConvTranspose3d(256, 128, kernel=4, stride=2, padding=1)
Layer 2: ConvTranspose3d(128, 64, kernel=4, stride=2, padding=1)  
Layer 3: ConvTranspose3d(64, 32, kernel=4, stride=2, padding=1)
Layer 4: ConvTranspose3d(32, 16, kernel=4, stride=2, padding=1)
Layer 5: ConvTranspose3d(16, 1, kernel=4, stride=2, padding=1)
```

激活函数选择：
- 中间层：ReLU或LeakyReLU($\alpha = 0.2$)
- 输出层：Sigmoid（二值占用）或Tanh（SDF）

**判别器架构**：

判别器采用相反的架构，使用步长卷积下采样：
$$64^3 \rightarrow 32^3 \rightarrow 16^3 \rightarrow 8^3 \rightarrow 4^3 \rightarrow 256 \rightarrow 1$$

使用Batch Normalization稳定训练：
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

其中 $\mu_B, \sigma_B^2$ 是mini-batch的均值和方差。

**感受野计算**：
对于核大小 $k$，步长 $s$，层数 $L$ 的网络：
$$RF = 1 + \sum_{l=1}^L (k_l - 1) \prod_{j=1}^{l-1} s_j$$

例如，5层网络每层 $k=4, s=2$：
$$RF = 1 + 3 \cdot (1 + 2 + 4 + 8 + 16) = 94$$

### 21.1.4 训练稳定性与模式坍塌

**梯度惩罚（WGAN-GP）**：
$$L = \mathbb{E}_{\tilde{x} \sim p_g}[D(\tilde{x})] - \mathbb{E}_{x \sim p_r}[D(x)] + \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}[(||\nabla_{\hat{x}}D(\hat{x})||_2 - 1)^2]$$

其中 $\hat{x} = \epsilon x + (1-\epsilon)\tilde{x}$，$\epsilon \sim U[0,1]$。

**谱归一化**：
$$W_{SN} = \frac{W}{\sigma(W)}$$

其中 $\sigma(W)$ 是权重矩阵的最大奇异值，通过幂迭代法近似计算。

### 21.1.5 多分辨率体素生成

为克服内存限制，采用八叉树（Octree）表示：
$$\text{Memory} = O(N^2) \text{ vs } O(N^3)$$

自适应分辨率损失：
$$L_{adaptive} = \sum_{l=0}^{L} \alpha_l \cdot L_{GAN}^{(l)}$$

其中 $L_{GAN}^{(l)}$ 是第 $l$ 层分辨率的对抗损失。

## 21.2 点云VAE与图卷积

### 21.2.1 点云的概率建模

点云 $\mathcal{P} = \{p_i\}_{i=1}^N$，$p_i \in \mathbb{R}^3$ 可视为从底层曲面采样的点集。概率密度函数：

$$p(\mathcal{P}) = \prod_{i=1}^N p(p_i | \mathcal{S})$$

由于点云的置换不变性，需要设计置换等变的网络架构。

### 21.2.2 变分自编码器（VAE）原理

VAE通过变分推断学习潜在表示。证据下界（ELBO）：
$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - KL(q(z|x) || p(z))$$

重参数化技巧：
$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

对于多元高斯分布，KL散度的闭式解：
$$KL = \frac{1}{2}\sum_{j=1}^J (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)$$

### 21.2.3 PointNet编码器

PointNet通过对称函数实现置换不变性：
$$f(\{x_1, ..., x_n\}) = \gamma \circ g(MAX_{i=1,...,n}\{h(x_i)\})$$

其中 $h: \mathbb{R}^3 \rightarrow \mathbb{R}^K$ 是逐点MLP，MAX是逐通道最大池化。

**详细网络结构**：
1. **输入变换网络（T-Net 3×3）**：
   - 输入：$N \times 3$ 点云
   - MLP：$(3, 64, 128, 1024)$
   - 全局池化：$1024 \rightarrow 1024$
   - FC层：$(1024, 512, 256, 9)$
   - 输出：$3 \times 3$ 变换矩阵

2. **特征变换网络（T-Net 64×64）**：
   - 输入：$N \times 64$ 特征
   - 类似结构，输出 $64 \times 64$ 矩阵

3. **主网络**：
   ```
   Input(N×3) → T-Net → MLP(64,64) → T-Net → MLP(64,128,1024) → MaxPool → FC(512,256,K)
   ```

T-Net学习仿射变换矩阵，增强旋转不变性：
$$x'_i = T \cdot x_i$$

正则化损失确保变换接近正交：
$$L_{reg} = ||I - AA^T||_F^2$$

**PointNet++的分层采样**：
1. **采样层**：最远点采样（FPS）
   ```
   初始化：S = {random point}
   while |S| < M:
       p = argmax_{p∈P\S} min_{q∈S} ||p-q||
       S = S ∪ {p}
   ```

2. **分组层**：球查询（Ball Query）
   $$\mathcal{N}(p) = \{q : ||q - p|| < r\}$$

3. **特征聚合**：
   $$f'_i = MAX_{j \in \mathcal{N}(i)} MLP(f_j, p_j - p_i)$$

**关键点云特征**：
- 局部特征：$f_{local} = h(x_i)$
- 全局特征：$f_{global} = MAX_i(f_{local})$
- 混合特征：$f_{hybrid} = [f_{local}; f_{global}]$

### 21.2.4 图卷积网络（GCN）

点云的k-NN图表示：$\mathcal{G} = (\mathcal{V}, \mathcal{E})$

EdgeConv操作：
$$x'_i = \max_{j \in \mathcal{N}(i)} h_\Theta(x_i, x_j - x_i)$$

谱图卷积基于图拉普拉斯矩阵的特征分解：
$$L = I - D^{-1/2}AD^{-1/2} = U\Lambda U^T$$

谱域卷积：
$$g_\theta * x = U g_\theta(\Lambda) U^T x$$

ChebNet使用切比雪夫多项式近似：
$$g_\theta(\Lambda) \approx \sum_{k=0}^K \theta_k T_k(\tilde{\Lambda})$$

### 21.2.5 FoldingNet解码器

FoldingNet通过折叠2D网格生成3D点云：
$$\Phi: \mathbb{R}^2 \times \mathbb{R}^m \rightarrow \mathbb{R}^3$$

折叠操作：
$$p_i = \Phi(g_i, z) = MLP([g_i; z])$$

其中 $g_i$ 是2D网格点，$z$ 是潜在编码。

Chamfer距离用于训练：
$$d_{CD}(S_1, S_2) = \sum_{x \in S_1} \min_{y \in S_2} ||x-y||^2 + \sum_{y \in S_2} \min_{x \in S_1} ||x-y||^2$$

## 21.3 扩散模型：Point-E、Shap-E

### 21.3.1 扩散过程的数学基础

前向扩散过程定义为马尔可夫链：
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

累积形式：
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

反向过程：
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

### 21.3.2 去噪扩散概率模型（DDPM）

训练目标简化为去噪自编码器：
$$L_{simple} = \mathbb{E}_{t,x_0,\epsilon}[||\epsilon - \epsilon_\theta(x_t, t)||^2]$$

其中 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。

**变分下界推导**：
负对数似然的变分下界：
$$-\log p_\theta(x_0) \leq \mathbb{E}_q\left[\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right]$$

展开为：
$$L_{VLB} = L_T + \sum_{t=2}^T L_{t-1} + L_0$$

其中：
- $L_T = D_{KL}(q(x_T|x_0) || p(x_T))$（先验匹配）
- $L_{t-1} = D_{KL}(q(x_{t-1}|x_t,x_0) || p_\theta(x_{t-1}|x_t))$（去噪）
- $L_0 = -\log p_\theta(x_0|x_1)$（重构）

**后验分布**：
$$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t,x_0), \tilde{\beta}_t I)$$

其中：
$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$$

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

采样时的均值预测：
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t))$$

方差可以固定为 $\sigma_t^2 = \beta_t$ 或 $\sigma_t^2 = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$。

**噪声调度（Noise Schedule）**：
线性调度：
$$\beta_t = \beta_{min} + \frac{t-1}{T-1}(\beta_{max} - \beta_{min})$$

余弦调度（改进版）：
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

其中 $s = 0.008$ 是小偏移量。

### 21.3.3 Point-E架构

Point-E采用两阶段生成：
1. 文本到图像：使用GLIDE生成条件图像
2. 图像到点云：条件扩散模型

点云扩散的特殊考虑：
- 坐标归一化：$x \in [-1, 1]^3$
- 采样策略：FPS（最远点采样）保证均匀分布
- 条件编码：CLIP图像特征

损失函数：
$$L = \mathbb{E}_{t,x_0,\epsilon,c}[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t, c)||^2]$$

### 21.3.4 Shap-E：隐式函数扩散

Shap-E生成神经隐式表示而非显式几何：
$$f_\theta: \mathbb{R}^3 \times \mathbb{R}^d \rightarrow \mathbb{R}$$

隐式函数参数化为MLP权重，通过扩散模型生成：
$$\theta \sim p_\phi(\theta | c)$$

训练分两阶段：
1. 编码器训练：$E: \mathcal{X} \rightarrow \Theta$
2. 扩散模型训练：在潜在空间 $\Theta$ 上

渲染使用可微光线投射：
$$C(r) = \int_0^{\infty} T(t) \sigma(r(t)) c(r(t)) dt$$

其中 $T(t) = \exp(-\int_0^t \sigma(r(s))ds)$。

### 21.3.5 加速采样：DDIM与DPM-Solver

DDIM（去噪扩散隐式模型）提供确定性采样：
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\epsilon_\theta(x_t, t) + \sigma_t\epsilon$$

其中 $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$。

DPM-Solver使用指数积分器：
$$x_t = \bar{\alpha}_t x_0 + \bar{\sigma}_t \int_0^{\lambda_t} e^{\lambda} \epsilon_\theta(x_\lambda, \lambda) d\lambda$$

二阶求解器：
$$x_{t-1} \approx \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}}[\epsilon_\theta(x_t, t) + \frac{\sigma_{t-1} - \sigma_t}{2\sigma_t}(\epsilon_\theta(x_t, t) - \epsilon_\theta(x_s, s))]$$

## 21.4 自回归模型与Transformer

### 21.4.1 序列建模视角

将3D形状表示为token序列：
$$\mathcal{S} = (s_1, s_2, ..., s_n)$$

自回归分解：
$$p(\mathcal{S}) = \prod_{i=1}^n p(s_i | s_{<i})$$

对数似然：
$$\log p(\mathcal{S}) = \sum_{i=1}^n \log p(s_i | s_{<i})$$

### 21.4.2 Transformer架构回顾

自注意力机制：
$$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

**缩放因子的重要性**：
不使用缩放时，点积方差为 $d_k$，导致softmax饱和：
$$\text{Var}(q^Tk) = d_k \cdot \text{Var}(q_i) \cdot \text{Var}(k_i) = d_k$$

缩放后方差归一化为1。

多头注意力：
$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, ..., head_h)W^O$$

其中 $head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$。

**位置编码方法**：

1. **绝对位置编码**（原始Transformer）：
$$PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$$
$$PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})$$

2. **相对位置编码**（T5风格）：
$$\text{Attention}_{ij} = \frac{(q_i + r_{i-j})k_j^T}{\sqrt{d_k}}$$

3. **旋转位置编码（RoPE）**：
$$f_q(x_m, m) = R_m \cdot W_q x_m$$

其中旋转矩阵：
$$R_m = \begin{pmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{pmatrix}$$

**层归一化**：
Pre-LN（更稳定）：
$$x_{l+1} = x_l + \text{Attn}(\text{LN}(x_l))$$

Post-LN（原始）：
$$x_{l+1} = \text{LN}(x_l + \text{Attn}(x_l))$$

**前馈网络（FFN）**：
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

扩展因子通常为4：$d_{ff} = 4 \cdot d_{model}$

**GLU变体**（性能更好）：
$$\text{SwiGLU}(x) = (xW_1) \cdot \text{Swish}(xW_2) \cdot W_3$$

### 21.4.3 PolyGen：网格顶点序列生成

PolyGen将网格生成分解为两步：
1. 顶点生成：$p(V|I) = \prod_{i=1}^n p(v_i|v_{<i}, I)$
2. 面生成：$p(F|V,I) = \prod_{j=1}^m p(f_j|f_{<j}, V, I)$

顶点量化到8位整数：
$$v_{quantized} = \lfloor 255 \cdot \frac{v - v_{min}}{v_{max} - v_{min}} \rfloor$$

指针网络用于面索引：
$$p(f_{j,k} = i) = \text{softmax}(e_i^T \text{tanh}(W[h_j; v_i]))$$

### 21.4.4 Mesh-GPT与离散表示

VQ-VAE将连续几何量化为离散码本：
$$z_q = \arg\min_{z_k \in \mathcal{C}} ||z_e - z_k||^2$$

码本学习使用指数移动平均：
$$c_i^{(t+1)} = \gamma c_i^{(t)} + (1-\gamma) \frac{\sum_{z_q=c_i} z_e}{N_i}$$

GPT在离散潜在空间建模：
$$p(z) = \prod_{i=1}^L p(z_i | z_{<i})$$

### 21.4.5 GET3D：几何感知Transformer

GET3D引入几何归纳偏置：
- 局部参考框架
- 等变特征
- 分层采样

SE(3)等变注意力：
$$\alpha_{ij} = \text{softmax}(\frac{f_\theta(||x_i - x_j||, \langle v_i, x_j - x_i \rangle)}{\sqrt{d}})$$

分层token合并：
$$z_{l+1} = \text{Pool}(\{z_l^i\}_{i \in \mathcal{N}(j)})$$

## 21.5 条件生成与可控性

### 21.5.1 条件变分自编码器（CVAE）

CVAE的ELBO：
$$\mathcal{L}(x,c) = \mathbb{E}_{q(z|x,c)}[\log p(x|z,c)] - KL(q(z|x,c)||p(z|c))$$

后验分布参数化：
$$q(z|x,c) = \mathcal{N}(z; \mu_\phi(x,c), \sigma_\phi^2(x,c))$$

先验可以是条件高斯：
$$p(z|c) = \mathcal{N}(z; \mu_{prior}(c), \sigma_{prior}^2(c))$$

### 21.5.2 分类器引导

利用预训练分类器 $p(c|x)$ 引导生成：
$$\nabla_x \log p(x|c) = \nabla_x \log p(x) + \nabla_x \log p(c|x)$$

在扩散模型中：
$$\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t) - \sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log p(c|x_t)$$

引导强度 $w$ 控制条件强度：
$$\hat{\epsilon} = \epsilon_\theta(x_t, t) - w \sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log p(c|x_t)$$

### 21.5.3 无分类器引导（CFG）

同时训练条件和无条件模型：
$$\hat{\epsilon}_\theta(x_t, t, c) = (1+w) \epsilon_\theta(x_t, t, c) - w \epsilon_\theta(x_t, t, \emptyset)$$

训练时随机dropout条件：
$$L = \mathbb{E}_{t,x_0,\epsilon,c,p}[(1-p)||\epsilon - \epsilon_\theta(x_t, t, c)||^2 + p||\epsilon - \epsilon_\theta(x_t, t, \emptyset)||^2]$$

### 21.5.4 几何约束满足

硬约束通过投影实现：
$$x_{constrained} = \arg\min_{x \in \mathcal{C}} ||x - x_{generated}||^2$$

对于线性约束 $Ax = b$：
$$x_{proj} = x - A^T(AA^T)^{-1}(Ax - b)$$

**常见几何约束及实现**：

1. **体积约束**：
$$V(x) = \sum_{(i,j,k) \in x} \Delta x^3$$

软约束损失：
$$L_{vol} = (V(x) - V_{target})^2$$

硬约束投影（缩放）：
$$x_{scaled} = x \cdot \sqrt[3]{\frac{V_{target}}{V(x)}}$$

2. **对称约束**：
反射对称：
$$L_{sym} = ||x - \mathcal{R}(x)||^2$$

其中 $\mathcal{R}$ 是关于平面的反射算子。

旋转对称（$C_n$群）：
$$L_{rot} = \sum_{k=1}^{n-1} ||x - R_{2\pi k/n}(x)||^2$$

3. **连通性约束**：
使用持续同调的0维Betti数 $\beta_0$：
$$L_{conn} = (\beta_0(x) - 1)^2$$

连通分量通过并查集计算：
```
对每个体素v：
    if occupied(v):
        for neighbor n in 26-邻域:
            if occupied(n):
                union(v, n)
连通分量数 = 不相交集合数
```

4. **可打印性约束**：
悬垂角度约束：
$$L_{overhang} = \sum_{f \in faces} \max(0, \cos(\theta_f) - \cos(\theta_{max}))^2$$

其中 $\theta_f$ 是面法向与构建方向夹角。

5. **壁厚约束**：
最小壁厚通过距离场：
$$L_{thickness} = \sum_{p \in surface} \max(0, t_{min} - |SDF(p)|)^2$$

软约束通过损失函数：
$$L_{total} = L_{generation} + \lambda_1 L_{vol} + \lambda_2 L_{sym} + \lambda_3 L_{conn} + \lambda_4 L_{print}$$

**拉格朗日乘子法**：
对于等式约束 $g(x) = 0$：
$$\mathcal{L}(x, \lambda) = f(x) + \lambda^T g(x)$$

KKT条件：
- 平稳性：$\nabla_x \mathcal{L} = 0$
- 原始可行性：$g(x) = 0$
- 对偶可行性：$\lambda \geq 0$（不等式约束）
- 互补松弛：$\lambda_i g_i(x) = 0$

### 21.5.5 潜在空间操作

线性插值：
$$z_{interp} = (1-\alpha)z_1 + \alpha z_2$$

球面插值（保持范数）：
$$z_{slerp} = \frac{\sin((1-\alpha)\theta)}{\sin\theta}z_1 + \frac{\sin(\alpha\theta)}{\sin\theta}z_2$$

其中 $\cos\theta = \frac{z_1 \cdot z_2}{||z_1|| \cdot ||z_2||}$。

语义方向发现（SeFa）：
$$\mathbf{A} = [\mathbf{a}_1, ..., \mathbf{a}_k] = \text{PCA}(\{W^T W\})$$

编辑操作：
$$z_{edit} = z + \alpha \mathbf{a}_i$$

## 本章小结

本章系统介绍了深度生成模型在3D形状生成中的应用：

**核心概念**：
1. **体素GAN**：将2D GAN扩展到3D空间，通过对抗训练生成体素网格，面临内存限制和模式坍塌挑战
2. **点云VAE**：利用变分推断学习点云的潜在表示，需要处理置换不变性和不规则采样
3. **扩散模型**：通过逐步去噪过程生成高质量3D形状，Point-E和Shap-E展示了强大的生成能力
4. **自回归Transformer**：将3D生成视为序列预测问题，PolyGen和Mesh-GPT实现了高精度网格生成
5. **条件控制**：通过分类器引导、无分类器引导和约束投影实现可控生成

**关键公式**：
- GAN目标：$\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1-D(G(z)))]$
- VAE ELBO：$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - KL(q(z|x)||p(z))$
- 扩散前向过程：$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$
- 自注意力：$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- 无分类器引导：$\hat{\epsilon} = (1+w)\epsilon_\theta(x_t,t,c) - w\epsilon_\theta(x_t,t,\emptyset)$

**实际应用**：
- 快速原型设计：从文本或图像生成3D模型
- 设计空间探索：通过潜在空间插值生成变体
- 缺失部分补全：条件生成修复破损模型
- 风格迁移：学习并应用不同设计风格

## 练习题

### 基础题

**练习21.1** 推导3D卷积的参数量和计算复杂度
考虑输入尺寸 $N \times N \times N \times C_{in}$，卷积核大小 $K \times K \times K$，输出通道 $C_{out}$。

<details>
<summary>提示</summary>

考虑每个输出位置需要的乘加运算数，以及总的输出体素数。

</details>

<details>
<summary>答案</summary>

参数量：$K^3 \times C_{in} \times C_{out} + C_{out}$（包括偏置）

计算复杂度：$O(N^3 \times K^3 \times C_{in} \times C_{out})$

内存需求（激活）：$O(N^3 \times C_{out})$

对于典型的 $64^3$ 体素，$K=4$，$C_{in}=128$，$C_{out}=64$：
- 参数：$4^3 \times 128 \times 64 + 64 = 524,352$
- FLOPs：$64^3 \times 4^3 \times 128 \times 64 \approx 137 \text{ GFLOPs}$

</details>

**练习21.2** 证明PointNet的最大池化操作是置换不变的
给定点集 $\{x_1, ..., x_n\}$ 和置换 $\pi$，证明：
$$\max_i f(x_i) = \max_i f(x_{\pi(i)})$$

<details>
<summary>提示</summary>

利用最大值操作的交换律和结合律。

</details>

<details>
<summary>答案</summary>

设 $S = \{f(x_1), ..., f(x_n)\}$ 为变换后的特征集合。

对于任意置换 $\pi$，置换后的集合 $S' = \{f(x_{\pi(1)}), ..., f(x_{\pi(n)})\}$。

由于置换只是改变元素顺序，不改变集合本身：$S = S'$

因此：$\max(S) = \max(S')$

即：$\max_i f(x_i) = \max_i f(x_{\pi(i)})$

这保证了PointNet对点云顺序的不变性。

</details>

**练习21.3** 计算扩散模型的信噪比
给定 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，计算信噪比SNR(t)。

<details>
<summary>提示</summary>

信噪比定义为信号方差与噪声方差之比。

</details>

<details>
<summary>答案</summary>

信号部分：$\sqrt{\bar{\alpha}_t}x_0$，方差：$\bar{\alpha}_t \text{Var}(x_0) = \bar{\alpha}_t$（假设 $x_0$ 已归一化）

噪声部分：$\sqrt{1-\bar{\alpha}_t}\epsilon$，方差：$(1-\bar{\alpha}_t)\text{Var}(\epsilon) = 1-\bar{\alpha}_t$

信噪比：
$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$$

对数信噪比：
$$\log \text{SNR}(t) = \log \bar{\alpha}_t - \log(1-\bar{\alpha}_t)$$

当 $t \to 0$：$\text{SNR} \to \infty$（纯信号）
当 $t \to T$：$\text{SNR} \to 0$（纯噪声）

</details>

**练习21.4** 分析Transformer的感受野
对于 $L$ 层Transformer，每层有 $H$ 个注意力头，分析有效感受野的增长。

<details>
<summary>提示</summary>

考虑信息如何通过注意力机制传播。

</details>

<details>
<summary>答案</summary>

单层Transformer的感受野是全局的（每个token可以attend到所有其他token）。

但有效感受野取决于注意力权重的稀疏性：
- 第1层：直接连接，感受野 = 序列长度 $N$
- 第 $l$ 层：通过 $l$ 跳连接，理论感受野仍为 $N$

实际有效感受野受限于：
1. 注意力权重的熵：$H(\alpha) = -\sum_i \alpha_i \log \alpha_i$
2. 注意力头的多样性：不同头关注不同模式
3. 位置编码的衰减：远距离位置的区分度降低

经验观察：
- 浅层：局部模式（相邻token）
- 中层：语法结构（中等距离）
- 深层：语义关系（长距离）

有效感受野约为：$R_{eff} \approx \sqrt{L \cdot N}$

</details>

### 挑战题

**练习21.5** 设计一个结合GAN和扩散模型优点的混合架构
要求：利用GAN的快速采样和扩散模型的高质量生成。

<details>
<summary>提示</summary>

考虑在不同分辨率或不同阶段使用不同模型。

</details>

<details>
<summary>答案</summary>

**混合架构设计**：

1. **粗略生成阶段（GAN）**：
   - 使用StyleGAN生成低分辨率形状 $x_{coarse}$
   - 优点：单次前向传播，速度快
   - 输出：$16^3$ 体素或1024点的点云

2. **精细化阶段（扩散）**：
   - 条件扩散模型：$p(x_{fine}|x_{coarse})$
   - 使用较少的扩散步数（如50步而非1000步）
   - 输出：$64^3$ 体素或8192点的点云

3. **训练策略**：
   - 阶段1：独立训练GAN
   - 阶段2：固定GAN，训练条件扩散模型
   - 阶段3：联合微调（可选）

4. **损失函数**：
   $$L_{hybrid} = L_{GAN}(x_{coarse}) + \lambda L_{diffusion}(x_{fine}|x_{coarse}) + \mu L_{consistency}(x_{fine}, x_{coarse})$$

   其中一致性损失确保多尺度一致：
   $$L_{consistency} = ||\text{Downsample}(x_{fine}) - x_{coarse}||^2$$

5. **采样加速**：
   - 使用DDIM进行确定性采样
   - 知识蒸馏：用完整模型训练更少步数的学生模型

预期性能：
- 生成时间：GAN(10ms) + 扩散(500ms) = 510ms
- 质量：接近完整扩散模型
- 可控性：保留条件生成能力

</details>

**练习21.6** 推导并实现点云的最优传输距离
给定两个点云 $P, Q$，计算Wasserstein距离。

<details>
<summary>提示</summary>

将问题转化为线性规划，使用Sinkhorn算法近似求解。

</details>

<details>
<summary>答案</summary>

**Wasserstein距离定义**：
$$W_p(P,Q) = \left(\inf_{\gamma \in \Pi(P,Q)} \int_{X \times Y} d(x,y)^p d\gamma(x,y)\right)^{1/p}$$

**离散形式（地球移动距离EMD）**：
$$\text{EMD}(P,Q) = \min_{T \in \mathcal{T}} \sum_{i,j} T_{ij} \cdot d(p_i, q_j)$$

约束条件：
- $T_{ij} \geq 0$（非负性）
- $\sum_j T_{ij} = \frac{1}{n}$（行和约束）
- $\sum_i T_{ij} = \frac{1}{m}$（列和约束）

**Sinkhorn算法（熵正则化）**：
$$T^* = \arg\min_{T \in \mathcal{T}} \langle T, C \rangle - \epsilon H(T)$$

其中 $H(T) = -\sum_{ij} T_{ij} \log T_{ij}$ 是熵。

迭代更新：
```
初始化：u = ones(n), v = ones(m)
K = exp(-C/ε)  # 核矩阵
for iteration in range(max_iter):
    u = 1 / (K @ v)
    v = 1 / (K.T @ u)
T = diag(u) @ K @ diag(v)
```

**复杂度分析**：
- 精确EMD：$O(n^3 \log n)$（线性规划）
- Sinkhorn：$O(n^2 \cdot \text{iter})$
- 近似比：$(1 + \epsilon)$-近似

**几何性质**：
1. 度量性质：非负性、对称性、三角不等式
2. 对刚体变换不变（使用适当的地面距离）
3. 对异常值鲁棒（相比Chamfer距离）

</details>

**练习21.7** 分析条件扩散模型的模式覆盖问题
比较分类器引导和无分类器引导的多样性-保真度权衡。

<details>
<summary>提示</summary>

从梯度的方差和偏差角度分析。

</details>

<details>
<summary>答案</summary>

**分类器引导分析**：

梯度估计：
$$\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)$$

方差分析：
- $\text{Var}[\nabla_x \log p(y|x)]$ 依赖于分类器质量
- 噪声水平 $t$ 较大时，分类器梯度不可靠
- 导致方差累积：$\text{Var}_{total} = \text{Var}_{score} + w^2 \text{Var}_{classifier}$

**无分类器引导分析**：

有效得分函数：
$$s_{cfg}(x,y) = (1+w)s(x,y) - ws(x,\emptyset)$$

偏差-方差分解：
- 偏差：$\text{Bias} = w(\mathbb{E}[s(x,y)] - \mathbb{E}[s(x,\emptyset)])$
- 方差：$\text{Var} = (1+w)^2\text{Var}[s(x,y)] + w^2\text{Var}[s(x,\emptyset)]$

**模式覆盖比较**：

1. **多样性指标**（使用最近邻）：
   $$\text{Coverage} = \frac{|\{y \in Y_{real} : \exists x \in X_{gen}, d(x,y) < \epsilon\}|}{|Y_{real}|}$$

2. **保真度指标**（FID分数）：
   $$\text{FID} = ||\mu_{real} - \mu_{gen}||^2 + \text{Tr}(\Sigma_{real} + \Sigma_{gen} - 2\sqrt{\Sigma_{real}\Sigma_{gen}})$$

3. **权衡曲线**：
   - $w=0$：高多样性，低保真度
   - $w \uparrow$：保真度提升，多样性下降
   - 最优 $w^* \approx 1.5-3.0$（经验值）

**理论结果**：
CFG相当于采样自锐化分布：
$$p_{cfg}(x|y) \propto p(x|y)^{1+w} / p(x)^w$$

当 $w \to \infty$：收敛到模式（MAP估计）
当 $w = 0$：原始条件分布

</details>

**练习21.8** 设计一个用于3D打印的多尺度生成模型
要求支持局部细节控制和全局结构约束。

<details>
<summary>提示</summary>

考虑分层表示和条件独立性假设。

</details>

<details>
<summary>答案</summary>

**多尺度架构设计**：

1. **分层表示**：
   - Level 0：包围盒和主轴 $(B, A) \in \mathbb{R}^{12}$
   - Level 1：粗网格 $M_1 \in \mathbb{R}^{8^3}$
   - Level 2：中等网格 $M_2 \in \mathbb{R}^{32^3}$
   - Level 3：精细网格 $M_3 \in \mathbb{R}^{128^3}$

2. **条件生成链**：
   $$p(M_3, M_2, M_1, B, A) = p(A)p(B|A)p(M_1|B,A)p(M_2|M_1)p(M_3|M_2)$$

3. **局部细节控制**：
   用户可指定区域 $R$ 和期望特征 $F$：
   $$p(M_3|M_2, R, F) = p(M_3^R|F) \cdot p(M_3^{\bar{R}}|M_2)$$

4. **全局约束满足**：
   - 体积约束：$V(M) = \sum_{ijk} M_{ijk} \cdot \Delta x^3 = V_{target}$
   - 质心约束：$\text{COM}(M) = c_{target}$
   - 主轴约束：通过PCA对齐

5. **训练策略**：
   ```
   for level in [0, 1, 2, 3]:
       if level == 0:
           train_prior(p(A), p(B|A))
       else:
           train_super_resolution(p(M_level | M_{level-1}))
   ```

6. **损失函数**：
   $$L_{total} = \sum_{l=1}^3 \lambda_l L_{recon}^{(l)} + \mu L_{perceptual} + \nu L_{constraint}$$

   感知损失使用预训练的3D特征提取器：
   $$L_{perceptual} = \sum_{k} ||\phi_k(M_{real}) - \phi_k(M_{gen})||^2$$

7. **推理时编辑**：
   - 全局编辑：修改 $(B, A)$，重新生成所有级别
   - 局部编辑：固定 $M_1, M_2$，只重新生成 $M_3$ 的局部区域
   - 细节迁移：从另一个模型提取 $M_3^{local}$，融合到当前模型

**优势**：
- 内存效率：每个级别独立处理
- 可控性：不同尺度的独立控制
- 质量：粗到细的生成避免全局不一致

**实现考虑**：
- 使用U-Net作为超分辨率网络
- 注意力机制用于长程依赖
- 渐进式训练提高稳定性

</details>

## 常见陷阱与错误

### 1. 训练不稳定
- **问题**：GAN训练时生成器或判别器崩溃
- **原因**：学习率不平衡、梯度爆炸、模式坍塌
- **解决**：使用梯度惩罚、谱归一化、渐进式训练

### 2. 内存溢出
- **问题**：3D数据内存需求呈立方增长
- **原因**：$64^3 = 262,144$ vs $64^2 = 4,096$
- **解决**：使用稀疏表示、八叉树、混合精度训练

### 3. 模式坍塌
- **问题**：生成器只产生少数几种形状
- **原因**：判别器过强、训练数据不平衡
- **解决**：使用mini-batch discrimination、unrolled GAN

### 4. 几何artifacts
- **问题**：生成的形状有洞、自相交、不连通
- **原因**：缺乏几何约束、离散化误差
- **解决**：添加拓扑损失、使用隐式表示、后处理修复

### 5. 条件泄露
- **问题**：条件信息没有正确影响生成
- **原因**：条件编码不充分、模型容量不足
- **解决**：使用cross-attention、增强条件编码、辅助任务

### 6. 评估指标误导
- **问题**：FID/IS分数好但视觉质量差
- **原因**：指标的局限性、分布不匹配
- **解决**：使用多个指标、人工评估、任务特定指标

### 7. 过拟合
- **问题**：生成的形状过于接近训练集
- **原因**：模型容量过大、训练数据少、正则化不足
- **解决**：数据增强、dropout、早停、使用预训练模型

### 8. 采样速度慢
- **问题**：扩散模型需要数百步采样
- **原因**：马尔可夫链的序列性质
- **解决**：使用DDIM、DPM-Solver、知识蒸馏、并行采样

## 最佳实践检查清单

### 数据准备
- [ ] 数据归一化到 [-1, 1] 或 [0, 1]
- [ ] 检查并修复退化几何（自相交、非流形）
- [ ] 平衡数据集类别分布
- [ ] 预计算必要的特征（如SDF、曲率）
- [ ] 设置合理的训练/验证/测试分割

### 模型设计
- [ ] 选择适合任务的表示（体素/点云/网格/隐式）
- [ ] 考虑内存和计算约束
- [ ] 设计合适的网络容量
- [ ] 添加必要的归纳偏置（对称性、等变性）
- [ ] 实现条件机制（如需要）

### 训练配置
- [ ] 使用合适的优化器（Adam for GAN, AdamW for Transformer）
- [ ] 设置合理的学习率调度
- [ ] 实现梯度裁剪
- [ ] 监控训练指标（loss、梯度范数、生成样本）
- [ ] 定期保存检查点

### 评估方法
- [ ] 实现多个评估指标
- [ ] 可视化生成样本
- [ ] 测试插值和编辑能力
- [ ] 评估生成速度
- [ ] 检查失败案例

### 部署考虑
- [ ] 优化推理速度（量化、剪枝）
- [ ] 实现批处理
- [ ] 添加用户控制接口
- [ ] 处理边界情况
- [ ] 准备后处理流程
