# 第17章：可微渲染基础

## 本章概述

可微渲染是计算机图形学与机器学习的交叉前沿，通过使渲染过程可微分，实现了从2D图像反向推导3D几何、材质和光照的能力。本章深入探讨可微渲染的数学基础，包括渲染方程的可微形式、梯度计算策略、重参数化技巧，以及处理几何不连续性的边界积分方法。我们将从Monte Carlo积分的基本原理出发，逐步构建完整的可微渲染框架，为逆向渲染、神经渲染和3D重建等应用奠定理论基础。

## 学习目标

- 掌握渲染方程的数学表述及其Monte Carlo求解方法
- 理解可微渲染中的梯度计算挑战与解决方案
- 掌握重参数化梯度估计和REINFORCE算法
- 理解几何边界的可微处理方法
- 熟悉主流可微光栅化框架的数学原理

## 17.1 渲染方程与Monte Carlo积分

### 17.1.1 渲染方程的数学表述

渲染方程（Rendering Equation）由Kajiya于1986年提出，描述了光能在场景中的平衡：

$$L_o(\mathbf{x}, \omega_o) = L_e(\mathbf{x}, \omega_o) + \int_{\Omega} f_r(\mathbf{x}, \omega_i, \omega_o) L_i(\mathbf{x}, \omega_i) (\omega_i \cdot \mathbf{n}) d\omega_i$$

其中：
- $L_o(\mathbf{x}, \omega_o)$：点$\mathbf{x}$沿方向$\omega_o$的出射辐射度（radiance），单位：$W/(m^2 \cdot sr)$
- $L_e(\mathbf{x}, \omega_o)$：自发光项，描述表面的发射特性
- $f_r(\mathbf{x}, \omega_i, \omega_o)$：双向反射分布函数（BRDF），单位：$sr^{-1}$
- $L_i(\mathbf{x}, \omega_i)$：入射辐射度
- $\Omega$：半球立体角，$\Omega = 2\pi$
- $\mathbf{n}$：表面法向量
- $(\omega_i \cdot \mathbf{n})$：几何项，也写作$\cos\theta_i$

**BRDF的性质**：

1. **互易性（Helmholtz reciprocity）**：
$$f_r(\mathbf{x}, \omega_i, \omega_o) = f_r(\mathbf{x}, \omega_o, \omega_i)$$

2. **能量守恒**：
$$\int_{\Omega} f_r(\mathbf{x}, \omega_i, \omega_o) \cos\theta_o d\omega_o \leq 1$$

3. **非负性**：
$$f_r(\mathbf{x}, \omega_i, \omega_o) \geq 0$$

**光线追踪形式**：

将入射辐射度$L_i$表示为来自其他表面的出射辐射度：
$$L_i(\mathbf{x}, \omega_i) = L_o(\mathbf{h}(\mathbf{x}, \omega_i), -\omega_i)$$

其中$\mathbf{h}(\mathbf{x}, \omega)$是射线投射函数，返回从$\mathbf{x}$沿$\omega$方向的第一个交点。

**积分算子形式**：

定义线性算子$\mathcal{T}$：
$$\mathcal{T}L(\mathbf{x}, \omega_o) = \int_{\Omega} f_r(\mathbf{x}, \omega_i, \omega_o) L(\mathbf{h}(\mathbf{x}, \omega_i), -\omega_i) \cos\theta_i d\omega_i$$

渲染方程简化为：
$$L = L_e + \mathcal{T}L$$

形式解为：
$$L = (\mathcal{I} - \mathcal{T})^{-1}L_e = \sum_{k=0}^{\infty} \mathcal{T}^k L_e$$

### 17.1.2 递归展开与路径积分

渲染方程可递归展开为Neumann级数：

$$L_o = L_e + TL_e + T^2L_e + T^3L_e + \cdots = \sum_{k=0}^{\infty} T^k L_e$$

其中算子$T$定义为：
$$Tf(\mathbf{x}, \omega) = \int_{\Omega} f_r(\mathbf{x}, \omega', \omega) f(\mathbf{h}(\mathbf{x}, \omega'), -\omega') |\omega' \cdot \mathbf{n}| d\omega'$$

**路径空间表述**：

定义路径空间$\mathcal{P} = \bigcup_{k=1}^{\infty} \mathcal{M}^k$，其中$\mathcal{M}$是场景表面的集合。

一条长度为$k$的路径：
$$\bar{\mathbf{x}} = (\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_k)$$

路径的贡献函数：
$$f(\bar{\mathbf{x}}) = L_e(\mathbf{x}_k, \omega_{k,k-1}) \prod_{i=1}^{k} f_r(\mathbf{x}_i, \omega_{i,i+1}, \omega_{i,i-1}) G(\mathbf{x}_i, \mathbf{x}_{i-1})$$

几何因子：
$$G(\mathbf{x}, \mathbf{y}) = \frac{V(\mathbf{x}, \mathbf{y}) |\cos\theta_x| |\cos\theta_y|}{\|\mathbf{x} - \mathbf{y}\|^2}$$

其中$V(\mathbf{x}, \mathbf{y})$是可见性函数。

**测度理论视角**：

路径空间的测度：
$$d\mu(\bar{\mathbf{x}}) = d\mathcal{A}(\mathbf{x}_0) \prod_{i=1}^{k} d\mathcal{A}(\mathbf{x}_i)$$

其中$d\mathcal{A}$是表面面积测度。

像素值的积分：
$$I_j = \int_{\mathcal{A}_j} \int_{\mathcal{P}} W_e(\mathbf{x}_0) f(\bar{\mathbf{x}}) d\mu(\bar{\mathbf{x}}) d\mathcal{A}(\mathbf{x}_0)$$

$W_e$是像素重建滤波器，$\mathcal{A}_j$是像素$j$对应的图像平面区域。

**路径概率密度**：

路径采样的概率密度：
$$p(\bar{\mathbf{x}}) = p(\mathbf{x}_0) \prod_{i=0}^{k-1} p(\mathbf{x}_{i+1}|\mathbf{x}_i)$$

转移概率通常基于BRDF重要性采样：
$$p(\mathbf{x}_{i+1}|\mathbf{x}_i) \propto f_r(\mathbf{x}_i, \omega_{i,i+1}, \omega_{i,i-1}) |\cos\theta_{i+1}|$$

### 17.1.3 Monte Carlo估计

对于路径积分，使用Monte Carlo方法估计：

$$\hat{L}_o = \frac{1}{N} \sum_{i=1}^{N} \frac{f(\bar{\mathbf{x}}_i)}{p(\bar{\mathbf{x}}_i)}$$

其中$p(\bar{\mathbf{x}})$是路径的概率密度函数。

**估计器的性质**：

1. **无偏性**：
$$\mathbb{E}[\hat{L}_o] = \mathbb{E}\left[\frac{f(\bar{\mathbf{x}})}{p(\bar{\mathbf{x}})}\right] = \int_{\mathcal{P}} \frac{f(\bar{\mathbf{x}})}{p(\bar{\mathbf{x}})} p(\bar{\mathbf{x}}) d\mu(\bar{\mathbf{x}}) = L_o$$

2. **方差**：
$$\text{Var}[\hat{L}_o] = \frac{1}{N}\left(\int_{\mathcal{P}} \frac{f^2(\bar{\mathbf{x}})}{p(\bar{\mathbf{x}})} d\mu(\bar{\mathbf{x}}) - L_o^2\right)$$

3. **均方误差（MSE）**：
$$\text{MSE}[\hat{L}_o] = \text{Var}[\hat{L}_o] + \text{Bias}^2[\hat{L}_o] = \frac{\text{Var}[f/p]}{N}$$

**重要性采样**：选择合适的$p(\bar{\mathbf{x}})$以减小方差。

理想的重要性采样分布：
$$p^*(\bar{\mathbf{x}}) = \frac{|f(\bar{\mathbf{x}})|}{\int_{\mathcal{P}} |f(\bar{\mathbf{x}})| d\mu(\bar{\mathbf{x}})}$$

使用$p^*$时，方差为零（所有样本贡献相同）。

**实用的重要性采样策略**：

1. **BRDF采样**：根据材质属性采样反射方向
   - Lambert：$p(\omega) = \cos\theta/\pi$
   - Phong：$p(\omega) \propto \cos^n\alpha$
   - GGX：使用微表面分布采样

2. **光源采样**：直接采样光源表面
   - 面光源：均匀采样或根据立体角
   - 环境光：根据亮度分布采样

3. **BSDF采样**：结合反射和透射
   - Fresnel权重选择反射或折射
   - 分层采样不同波瓣

**方差减少技术**：

1. **分层采样（Stratified Sampling）**：
   将采样域分成$M$个子域：
   $$\hat{L}_o = \frac{1}{M} \sum_{j=1}^{M} \frac{1}{n_j} \sum_{i=1}^{n_j} \frac{f(\bar{\mathbf{x}}_{ji})}{p(\bar{\mathbf{x}}_{ji}|S_j)}$$

2. **拟蒙特卡洛（Quasi-Monte Carlo）**：
   使用低差异序列（Sobol、Halton）代替随机数

3. **控制变量（Control Variates）**：
   $$\hat{L}_o^{CV} = \hat{L}_o + c(\mathbb{E}[g] - \hat{g})$$
   其中$g$是已知期望的辅助函数

### 17.1.4 俄罗斯轮盘赌

为处理无限递归，引入俄罗斯轮盘赌终止策略：

$$\hat{L}_o = \begin{cases}
L_e + \frac{1}{p_{rr}} \int_{\Omega} f_r L_i (\omega_i \cdot \mathbf{n}) d\omega_i & \text{概率 } p_{rr} \\
L_e & \text{概率 } 1-p_{rr}
\end{cases}$$

**无偏性证明**：
$$\mathbb{E}[\hat{L}_o] = p_{rr} \cdot \left(L_e + \frac{1}{p_{rr}} \int_{\Omega} f_r L_i (\omega_i \cdot \mathbf{n}) d\omega_i\right) + (1-p_{rr}) \cdot L_e$$
$$= L_e + \int_{\Omega} f_r L_i (\omega_i \cdot \mathbf{n}) d\omega_i = L_o$$

**终止概率的选择**：

1. **固定概率**：$p_{rr} = 0.95$（简单但可能低效）

2. **自适应概率**：基于路径贡献
   $$p_{rr} = \min\left(1, \frac{\|f(\bar{\mathbf{x}}_k)\|}{\|f(\bar{\mathbf{x}}_0)\|}\right)$$

3. **基于深度的终止**：
   $$p_{rr}(k) = \begin{cases}
   1 & k < k_{min} \\
   \alpha^{k-k_{min}} & k \geq k_{min}
   \end{cases}$$
   其中$k$是路径长度，$\alpha \in (0,1)$是衰减系数。

**效率分析**：

路径长度的期望：
$$\mathbb{E}[k] = \sum_{i=0}^{\infty} i \cdot p_{rr}^i (1-p_{rr}) = \frac{p_{rr}}{1-p_{rr}}$$

计算复杂度：$O(\mathbb{E}[k])$

**分割俄罗斯轮盘赌（Splitting Russian Roulette）**：

在重要区域增加样本而非终止：
$$\hat{L}_o = \begin{cases}
n \cdot \frac{1}{n} \sum_{j=1}^{n} L_j & \text{重要区域} \\
\frac{1}{p_{rr}} L & \text{非重要区域，概率 } p_{rr}
\end{cases}$$

### 17.1.5 多重重要性采样（MIS）

当有多个采样策略时，使用多重重要性采样组合它们：

$$\hat{L}_o = \sum_{i=1}^{n_f} \frac{w_f(\mathbf{x}_{f,i}) f(\mathbf{x}_{f,i})}{p_f(\mathbf{x}_{f,i})} + \sum_{j=1}^{n_g} \frac{w_g(\mathbf{x}_{g,j}) f(\mathbf{x}_{g,j})}{p_g(\mathbf{x}_{g,j})}$$

**平衡启发式（Balance Heuristic）**：
$$w_s(\mathbf{x}) = \frac{n_s p_s(\mathbf{x})}{\sum_k n_k p_k(\mathbf{x})}$$

**功率启发式（Power Heuristic）**：
$$w_s(\mathbf{x}) = \frac{(n_s p_s(\mathbf{x}))^\beta}{\sum_k (n_k p_k(\mathbf{x}))^\beta}$$

常用$\beta = 2$，更好地处理高方差区域。

**MIS的性质**：

1. **无偏性**：如果$\sum_s w_s(\mathbf{x}) = 1$，则估计器无偏

2. **方差减少**：MIS的方差不超过任何单一策略：
   $$\text{Var}[\hat{L}_{MIS}] \leq \min_s \text{Var}[\hat{L}_s]$$

3. **最优性**：平衡启发式在二阶近似下最优

**典型应用场景**：

1. **直接光照**：
   - BRDF采样：处理高光反射
   - 光源采样：处理大面光源
   
2. **环境光照**：
   - BRDF采样：适合glossy材质
   - 环境贴图采样：适合漫反射

3. **双向路径追踪**：
   - 结合不同长度的光源和相机子路径

**实现细节**：

```pseudocode
function DirectLighting(x, wo):
    // BRDF采样
    wi_brdf = SampleBRDF(x, wo)
    L_brdf = Li(x, wi_brdf) * BRDF(x, wi_brdf, wo) * cos(wi_brdf) / pdf_brdf(wi_brdf)
    
    // 光源采样
    y = SampleLight()
    wi_light = normalize(y - x)
    if Visible(x, y):
        L_light = Le(y) * BRDF(x, wi_light, wo) * cos(wi_light) * G(x,y) / pdf_light(y)
    
    // MIS组合
    w_brdf = PowerHeuristic(pdf_brdf, pdf_light, beta=2)
    w_light = PowerHeuristic(pdf_light, pdf_brdf, beta=2)
    
    return w_brdf * L_brdf + w_light * L_light
```

## 17.2 梯度计算：有限差分vs自动微分

### 17.2.1 参数化渲染函数

考虑参数化的渲染过程：
$$I(\theta) = \int_{\mathcal{P}} f(\bar{\mathbf{x}}; \theta) d\mu(\bar{\mathbf{x}})$$

其中$\theta \in \mathbb{R}^n$可以是：
- **几何参数**：顶点位置、法线、UV坐标
- **材质参数**：反照率、粗糙度、折射率
- **光照参数**：光源位置、强度、颜色
- **相机参数**：视角、焦距、光圈

**梯度的存在性**：

渲染函数$I(\theta)$在以下情况下不可微：
1. 可见性不连续（遮挡边界）
2. 材质不连续（纹理边界）
3. 光照不连续（阴影边界）

**梯度的分解**：

使用Leibniz积分规则：
$$\frac{\partial I}{\partial \theta} = \int_{\mathcal{P}} \frac{\partial f}{\partial \theta}(\bar{\mathbf{x}}; \theta) d\mu(\bar{\mathbf{x}}) + \int_{\partial \mathcal{P}} f(\bar{\mathbf{x}}; \theta) \frac{\partial \mathcal{P}}{\partial \theta} \cdot \mathbf{n} ds$$

第一项：内部梯度（着色变化）  
第二项：边界梯度（几何变化）

### 17.2.2 有限差分方法

**前向差分**：
$$\nabla_{\theta} I \approx \frac{I(\theta + h) - I(\theta)}{h}$$

**中心差分**（更精确）：
$$\nabla_{\theta} I \approx \frac{I(\theta + h) - I(\theta - h)}{2h}$$

**Richardson外推**（高阶精度）：
$$\nabla_{\theta} I \approx \frac{4D(h/2) - D(h)}{3}$$
其中$D(h) = \frac{I(\theta+h) - I(\theta-h)}{2h}$

**误差分析**：

Taylor展开：
$$I(\theta + h) = I(\theta) + h\nabla I(\theta) + \frac{h^2}{2}\nabla^2 I(\theta) + O(h^3)$$

前向差分误差：
$$E_{forward} = \frac{h}{2}\nabla^2 I(\theta) + O(h^2)$$

中心差分误差：
$$E_{center} = \frac{h^2}{6}\nabla^3 I(\theta) + O(h^4)$$

**最优步长选择**：

总误差 = 截断误差 + 舍入误差

前向差分：
$$E_{total} = Ch + \frac{\epsilon}{h}$$
最小化：$h_{opt} = \sqrt{\epsilon/C} \approx \sqrt{\epsilon}$

中心差分：
$$E_{total} = Ch^2 + \frac{\epsilon}{h}$$
最小化：$h_{opt} = (\epsilon/2C)^{1/3} \approx \epsilon^{1/3}$

**复杂度分析**：
- 计算复杂度：$O(n)$次前向传播，$n$是参数维度
- 内存复杂度：$O(1)$，不需要存储计算图

**随机有限差分**：

为减少计算量，使用随机方向：
$$\nabla_{\theta} I \approx \frac{I(\theta + h\mathbf{v}) - I(\theta - h\mathbf{v})}{2h} \cdot \mathbf{v}$$
其中$\mathbf{v} \sim \mathcal{N}(0, I)$是随机方向。

### 17.2.3 自动微分

**前向模式（Forward Mode AD）**：

使用对偶数（dual numbers）：
$$x \rightarrow (x, \dot{x})$$

计算规则：
- 加法：$(a, \dot{a}) + (b, \dot{b}) = (a+b, \dot{a}+\dot{b})$
- 乘法：$(a, \dot{a}) \times (b, \dot{b}) = (ab, a\dot{b} + b\dot{a})$
- 链式法则：$\dot{y} = \frac{\partial f}{\partial x} \dot{x}$

复杂度：$O(n)$次前向传播，其中$n$是输入维度。

**反向模式（Reverse Mode AD）**：

构建计算图并反向传播：

1. **前向传播**：计算并记录中间值
   $$v_i = f_i(v_{j<i})$$

2. **反向传播**：计算伴随变量
   $$\bar{v}_j = \sum_{i: j \in \text{parents}(i)} \bar{v}_i \frac{\partial v_i}{\partial v_j}$$

复杂度：$O(m)$次反向传播，其中$m$是输出维度。

**混合模式**：

对于Hessian计算：
$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

使用forward-over-reverse：
1. 反向模式计算梯度
2. 前向模式对梯度求导

**梯度检查点（Gradient Checkpointing）**：

为减少内存使用：
1. 仅存储部分中间结果（检查点）
2. 反向传播时重新计算其他中间值

时空权衡：
- 内存：$O(\sqrt{n})$而非$O(n)$
- 计算：增加约33%

### 17.2.4 可微渲染的挑战

**1. 可见性不连续**

问题：当$\theta$变化时，可见性函数$V(\mathbf{x}, \mathbf{y}; \theta)$不连续。

数学表述：
$$V(\mathbf{x}, \mathbf{y}; \theta) = \begin{cases}
1 & \text{if no occlusion} \\
0 & \text{if occluded}
\end{cases}$$

梯度包含Dirac delta：
$$\frac{\partial V}{\partial \theta} = \delta(\text{boundary}) \cdot \text{normal\_velocity}$$

**2. 采样相关性**

问题：Monte Carlo采样引入高方差。

方差与采样数的关系：
$$\text{Var}[\nabla I] = \frac{\sigma^2}{N}$$

偏差-方差权衡：
- 增加采样：减少方差，增加计算
- 软化边界：减少方差，引入偏差

**3. 内存需求**

反向模式AD需要存储：
- 所有中间变量
- 计算图结构

内存复杂度：$O(\text{path\_length} \times \text{resolution})$

解决方案：
- 梯度检查点
- 局部反向传播
- 混合精度计算

**4. 数值稳定性**

问题：梯度爆炸或消失。

解决方案：
- 梯度裁剪：$\nabla = \text{clip}(\nabla, -c, c)$
- 梯度归一化：$\nabla = \nabla / \|\nabla\|$
- 自适应学习率

### 17.2.5 梯度估计的偏差与方差

**偏差来源**：

对于Monte Carlo估计：
$$I(\theta) = \mathbb{E}_{p(\mathbf{x};\theta)}[f(\mathbf{x};\theta)]$$

直接求导（错误！）：
$$\nabla_{\theta} I \stackrel{?}{=} \mathbb{E}_{p(\mathbf{x};\theta)}[\nabla_{\theta} f(\mathbf{x};\theta)]$$

正确的梯度：
$$\nabla_{\theta} I = \mathbb{E}_{p}[\nabla_{\theta} f] + \mathbb{E}_{p}[f \nabla_{\theta} \log p]$$

第二项经常被忽略，导致偏差。

**方差分析**：

梯度估计器的方差：
$$\text{Var}[\nabla_{\theta} \hat{I}] = \frac{1}{N} \text{Var}_{p}[\nabla_{\theta} (f/p)]$$

展开：
$$\text{Var}[\nabla_{\theta} \hat{I}] = \frac{1}{N} \left(\mathbb{E}_{p}\left[\left\|\nabla_{\theta} \frac{f}{p}\right\|^2\right] - \left\|\nabla_{\theta} I\right\|^2\right)$$

**信噪比（SNR）**：
$$\text{SNR} = \frac{\|\nabla_{\theta} I\|}{\sqrt{\text{Var}[\nabla_{\theta} \hat{I}]}} = \sqrt{N} \cdot \frac{\|\nabla_{\theta} I\|}{\sigma_{\nabla}}$$

要达到目标SNR，需要采样数：
$$N = \left(\frac{\sigma_{\nabla}}{\|\nabla_{\theta} I\|} \cdot \text{SNR}_{target}\right)^2$$

**控制偏差的方法**：

1. **重参数化**：消除采样分布对$\theta$的依赖

2. **似然比方法**：显式计算$\nabla_{\theta} \log p$

3. **双重采样**：
   $$\nabla_{\theta} I \approx \frac{1}{N} \sum_{i=1}^{N} \left(\nabla_{\theta} f(\mathbf{x}_i;\theta) + f(\mathbf{x}_i;\theta) \nabla_{\theta} \log p(\mathbf{x}_i;\theta)\right)$$

## 17.3 重参数化技巧与REINFORCE

### 17.3.1 重参数化梯度

**基本思想**：

将随机性从参数中分离：
$$\mathbf{z} \sim p(\mathbf{z}; \theta) \Rightarrow \mathbf{z} = g(\epsilon; \theta), \quad \epsilon \sim p(\epsilon)$$

其中$p(\epsilon)$不依赖于$\theta$。

**梯度计算**：

原始期望：
$$I(\theta) = \mathbb{E}_{p(\mathbf{z};\theta)}[f(\mathbf{z})]$$

重参数化后：
$$I(\theta) = \mathbb{E}_{p(\epsilon)}[f(g(\epsilon; \theta))]$$

梯度：
$$\nabla_{\theta} I = \mathbb{E}_{p(\epsilon)}[\nabla_{\theta} f(g(\epsilon; \theta))]$$

展开链式法则：
$$\nabla_{\theta} I = \mathbb{E}_{p(\epsilon)}\left[\frac{\partial f}{\partial \mathbf{z}}\bigg|_{\mathbf{z}=g(\epsilon;\theta)} \cdot \frac{\partial g}{\partial \theta}(\epsilon; \theta)\right]$$

**优点**：
1. 无偏估计
2. 低方差（相比REINFORCE）
3. 可使用自动微分

**局限性**：
1. 需要显式的重参数化公式
2. 不适用于离散分布
3. 不适用于复杂分布

### 17.3.2 常见分布的重参数化

**正态分布**：
$$\mathbf{z} \sim \mathcal{N}(\mu, \sigma^2)$$

重参数化：
$$\mathbf{z} = \mu + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

梯度：
$$\frac{\partial \mathbf{z}}{\partial \mu} = 1, \quad \frac{\partial \mathbf{z}}{\partial \sigma} = \epsilon$$

**均匀分布**：
$$\mathbf{z} \sim \mathcal{U}(a, b)$$

重参数化：
$$\mathbf{z} = a + (b-a)\epsilon, \quad \epsilon \sim \mathcal{U}(0, 1)$$

梯度：
$$\frac{\partial \mathbf{z}}{\partial a} = 1 - \epsilon, \quad \frac{\partial \mathbf{z}}{\partial b} = \epsilon$$

**指数分布**：
$$\mathbf{z} \sim \text{Exp}(\lambda)$$

重参数化：
$$\mathbf{z} = -\frac{\log(1-\epsilon)}{\lambda}, \quad \epsilon \sim \mathcal{U}(0, 1)$$

**Beta分布**：
$$\mathbf{z} \sim \text{Beta}(\alpha, \beta)$$

使用Kumaraswamy近似：
$$\mathbf{z} = (1-(1-\epsilon)^{1/\beta})^{1/\alpha}, \quad \epsilon \sim \mathcal{U}(0, 1)$$

**Gumbel-Softmax**（离散分布的连续松弛）：

离散分布：$\mathbf{z} \sim \text{Categorical}(\pi)$

连续松弛：
$$\mathbf{z}_i = \frac{\exp((\log \pi_i + g_i)/\tau)}{\sum_j \exp((\log \pi_j + g_j)/\tau)}$$

其中：
- $g_i = -\log(-\log(\epsilon_i))$，$\epsilon_i \sim \mathcal{U}(0, 1)$
- $\tau$是温度参数：$\tau \to 0$时趋近one-hot

梯度：
$$\frac{\partial \mathbf{z}_i}{\partial \log \pi_j} = \frac{1}{\tau}(\mathbf{z}_i(\delta_{ij} - \mathbf{z}_j))$$

### 17.3.3 REINFORCE算法

**似然比梯度定理**：

对于不可重参数化的分布：
$$\nabla_{\theta} \mathbb{E}_{p(\mathbf{z};\theta)}[f(\mathbf{z})] = \mathbb{E}_{p(\mathbf{z};\theta)}[f(\mathbf{z}) \nabla_{\theta} \log p(\mathbf{z}; \theta)]$$

**推导**：
$$\nabla_{\theta} \mathbb{E}_{p}[f] = \nabla_{\theta} \int f(\mathbf{z}) p(\mathbf{z};\theta) d\mathbf{z}$$
$$= \int f(\mathbf{z}) \nabla_{\theta} p(\mathbf{z};\theta) d\mathbf{z}$$
$$= \int f(\mathbf{z}) p(\mathbf{z};\theta) \frac{\nabla_{\theta} p(\mathbf{z};\theta)}{p(\mathbf{z};\theta)} d\mathbf{z}$$
$$= \mathbb{E}_{p}[f(\mathbf{z}) \nabla_{\theta} \log p(\mathbf{z}; \theta)]$$

**带基线的REINFORCE**：

为减小方差，引入基线$b$：
$$\nabla_{\theta} J = \mathbb{E}_{p}[(f(\mathbf{z}) - b) \nabla_{\theta} \log p(\mathbf{z}; \theta)]$$

基线不影响无偏性：
$$\mathbb{E}_{p}[b \cdot \nabla_{\theta} \log p] = b \cdot \nabla_{\theta} \mathbb{E}_{p}[1] = 0$$

**最优基线**：

最小化方差：
$$b^* = \arg\min_b \text{Var}[(f - b) \nabla_{\theta} \log p]$$

解：
$$b^* = \frac{\mathbb{E}[f(\mathbf{z}) \|\nabla_{\theta} \log p(\mathbf{z}; \theta)\|^2]}{\mathbb{E}[\|\nabla_{\theta} \log p(\mathbf{z}; \theta)\|^2]}$$

**实用的基线选择**：

1. **移动平均**：
   $$b_t = \alpha b_{t-1} + (1-\alpha) f(\mathbf{z}_t)$$

2. **神经网络基线**：
   $$b = V_{\phi}(\mathbf{s})$$
   其中$V_{\phi}$是值函数近似器

3. **自平均基线**：
   $$b = \frac{1}{N-1}\sum_{j \neq i} f(\mathbf{z}_j)$$

### 17.3.4 控制变量方法

**基本思想**：

将函数分解为可微和不可微部分：
$$f(\mathbf{z}; \theta) = f_{diff}(\mathbf{z}; \theta) + f_{disc}(\mathbf{z}; \theta)$$

梯度估计：
$$\nabla_{\theta} J = \underbrace{\mathbb{E}[\nabla_{\theta} f_{diff}]}_{\text{重参数化}} + \underbrace{\mathbb{E}[(f_{disc} - c) \nabla_{\theta} \log p]}_{\text{REINFORCE}}$$

**控制变量的选择**：

理想的控制变量应该：
1. 与$f_{disc}$高度相关
2. 可微分
3. 计算高效

常用选择：
- $c(\mathbf{z}) = f_{diff}(\mathbf{z}; \theta_{old})$
- $c(\mathbf{z}) = \mathbb{E}[f_{disc}|\mathbf{z}_{diff}]$

**Rao-Blackwellization**：

利用条件期望减少方差：
$$\text{Var}[\mathbb{E}[X|Y]] \leq \text{Var}[X]$$

应用：
$$c^*(\mathbf{z}) = \mathbb{E}[f_{disc}|\mathbf{z}_{observable}]$$

**实例：可微渲染中的应用**：

```pseudocode
function DifferentiableRender(scene, theta):
    // 可微部分：着色
    shading = ComputeShading(scene, theta)
    grad_shading = AutoDiff(shading, theta)
    
    // 不可微部分：可见性
    visibility = ComputeVisibility(scene, theta)
    
    // 控制变量：软可见性
    soft_visibility = SoftVisibility(scene, theta)
    
    // 组合梯度
    grad_total = grad_shading + 
                 (visibility - soft_visibility) * ScoreFunction(theta)
    
    return grad_total
```

## 17.4 边界采样与轮廓积分

### 17.4.1 可见性的数学表述

**可见性函数**：
$$V(\mathbf{x}, \mathbf{y}) = \begin{cases}
1 & \text{if } \mathbf{x} \text{ and } \mathbf{y} \text{ are mutually visible} \\
0 & \text{otherwise}
\end{cases}$$

**射线表示**：
$$V(\mathbf{x}, \mathbf{y}) = \prod_{t \in [0,1]} \mathbb{1}[\text{no intersection at } \mathbf{x} + t(\mathbf{y} - \mathbf{x})]$$

**分布意义下的梯度**：

可见性函数在边界处不连续，梯度包含Dirac delta：
$$\frac{\partial V}{\partial \theta} = \delta(d(\theta)) \cdot \frac{\partial d}{\partial \theta}$$

其中$d(\theta) = 0$定义了遮挡边界。

**边界的参数化**：

边界曲线：$\gamma(s; \theta)$，$s \in [0, L]$

边界速度：
$$\mathbf{v}(s) = \frac{\partial \gamma}{\partial \theta}(s; \theta)$$

边界法向：
$$\mathbf{n}(s) = \frac{\gamma'(s) \times \mathbf{e}_z}{\|\gamma'(s) \times \mathbf{e}_z\|}$$

### 17.4.2 边界积分公式

**Reynolds Transport定理**：

对于随参数变化的域$\Omega(\theta)$：
$$\frac{d}{d\theta} \int_{\Omega(\theta)} f(\mathbf{x}; \theta) d\mathbf{x} = \int_{\Omega(\theta)} \frac{\partial f}{\partial \theta} d\mathbf{x} + \int_{\partial \Omega(\theta)} f(\mathbf{x}; \theta) (\mathbf{v} \cdot \mathbf{n}) ds$$

物理意义：
- 第一项：内部变化（函数值的变化）
- 第二项：边界移动的贡献

**应用于渲染**：

像素值：
$$I(\theta) = \int_{\Omega_{visible}(\theta)} L(\mathbf{x}; \theta) d\mathbf{x}$$

梯度：
$$\frac{dI}{d\theta} = \int_{\Omega} \frac{\partial L}{\partial \theta} d\mathbf{x} + \int_{\partial \Omega} L (\mathbf{v} \cdot \mathbf{n}) ds$$

**边界速度的计算**：

对于三角形边缘：
$$\mathbf{v} = \frac{\partial \mathbf{p}_{edge}}{\partial \theta}$$

对于遮挡边界：
$$\mathbf{v} = \frac{\partial \mathbf{p}_{occluder}}{\partial \theta} - \frac{\partial \mathbf{p}_{occluded}}{\partial \theta}$$

### 17.4.3 轮廓积分的离散化

将轮廓离散为线段：
$$\nabla_{\theta} I = \sum_{e \in \text{edges}} \int_{e} f(\mathbf{x}) \frac{\partial \mathbf{x}}{\partial \theta} \cdot \mathbf{n}_e ds$$

使用Gauss-Legendre求积：
$$\int_{e} f(\mathbf{x}) ds \approx \sum_{i=1}^{n} w_i f(\mathbf{x}_i) |e|$$

### 17.4.4 边界采样策略

**均匀采样**：
$$p(\mathbf{x}) = \frac{1}{|\partial \Omega|}$$

**重要性采样**（基于辐射度）：
$$p(\mathbf{x}) \propto |f(\mathbf{x})|$$

**分层采样**：将边界分成$k$段，每段采样$n/k$个点。

### 17.4.5 软边界近似

使用sigmoid函数软化边界：
$$V_{soft}(\mathbf{x}, \mathbf{y}) = \sigma\left(\frac{d(\mathbf{x}, \mathbf{y})}{\epsilon}\right)$$

其中$d(\mathbf{x}, \mathbf{y})$是到遮挡边界的有符号距离，$\epsilon$控制软化程度。

## 17.5 可微光栅化：SoftRas、DIB-R

### 17.5.1 传统光栅化的不可微性

传统光栅化的深度测试：
$$z_{buffer}[i,j] = \min_{t \in \text{triangles}} z_t(i,j)$$

$\min$操作导致梯度几乎处处为零。

### 17.5.2 SoftRas方法

**软光栅化函数**：
$$C_{ij} = \frac{\sum_{t} w_{ijt} c_t}{\sum_{t} w_{ijt}}$$

其中权重函数：
$$w_{ijt} = \sigma(d_{ijt}/\gamma) \cdot \exp(-z_{ijt}^2/2\sigma_z^2)$$

- $d_{ijt}$：像素$(i,j)$到三角形$t$的有符号距离
- $\gamma$：控制边界模糊程度
- $\sigma_z$：深度方向的标准差

**概率解释**：每个三角形对像素的贡献是其可见概率。

### 17.5.3 DIB-R（Differentiable Interpolation-Based Renderer）

**前景/背景分解**：
$$C_{ij} = \alpha_{ij} C_{fg} + (1-\alpha_{ij}) C_{bg}$$

其中$\alpha_{ij}$是软前景掩码：
$$\alpha_{ij} = 1 - \prod_{t} (1 - \alpha_{ijt})$$

单个三角形的贡献：
$$\alpha_{ijt} = \text{softmax}(-d_{ijt}/\tau) \cdot \mathbb{1}[z_{ijt} < z_{max}]$$

### 17.5.4 梯度计算与反向传播

**对顶点位置的梯度**：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial C_{ij}} \frac{\partial C_{ij}}{\partial \mathbf{v}}$$

其中：
$$\frac{\partial C_{ij}}{\partial \mathbf{v}} = \sum_{t \ni \mathbf{v}} \frac{\partial C_{ij}}{\partial w_{ijt}} \frac{\partial w_{ijt}}{\partial d_{ijt}} \frac{\partial d_{ijt}}{\partial \mathbf{v}}$$

**有符号距离的梯度**：
$$\frac{\partial d_{ijt}}{\partial \mathbf{v}_k} = \frac{(\mathbf{p}_{ij} - \mathbf{v}_k) \cdot \mathbf{n}_t}{|\mathbf{n}_t|}$$

### 17.5.5 性能优化

**层次化光栅化**：
1. 粗光栅化：确定三角形覆盖的像素块
2. 细光栅化：仅处理相关像素

**早期剔除**：
- 视锥剔除：$\mathbf{v} \cdot \mathbf{n}_{frustum} > 0$
- 背面剔除：$(\mathbf{v}_1 - \mathbf{v}_0) \times (\mathbf{v}_2 - \mathbf{v}_0) \cdot \mathbf{view} < 0$

**梯度累积优化**：使用原子操作避免竞争条件。

## 本章小结

本章系统介绍了可微渲染的数学基础：

1. **渲染方程与Monte Carlo积分**：建立了光传输的数学模型，通过Monte Carlo方法实现数值求解，重要性采样和多重重要性采样技术降低方差。

2. **梯度计算策略**：比较了有限差分和自动微分方法，分析了各自的优缺点和适用场景。

3. **重参数化与REINFORCE**：解决了随机采样的可微性问题，重参数化适用于连续分布，REINFORCE处理离散和不可重参数化情况。

4. **边界积分理论**：通过Reynolds transport定理处理几何不连续性，轮廓积分提供了精确的梯度计算。

5. **可微光栅化框架**：SoftRas和DIB-R通过软化深度测试和边界，实现了端到端可微的光栅化。

关键公式：
- 渲染方程：$L_o = L_e + \int_{\Omega} f_r L_i (\omega_i \cdot \mathbf{n}) d\omega_i$
- 重参数化梯度：$\nabla_{\theta} \mathbb{E}[f] = \mathbb{E}[\nabla_{\theta} f(g(\epsilon; \theta))]$
- REINFORCE：$\nabla_{\theta} J = \mathbb{E}[f(\mathbf{z}) \nabla_{\theta} \log p(\mathbf{z}; \theta)]$
- 边界积分：$\frac{d}{d\theta} \int_{\Omega} f = \int_{\Omega} \frac{\partial f}{\partial \theta} + \int_{\partial \Omega} f (\mathbf{v} \cdot \mathbf{n})$

## 练习题

### 基础题

**练习17.1** 推导渲染方程的Neumann级数展开，并解释每一项的物理意义。

*提示*：考虑光线弹射次数与能量衰减的关系。

<details>
<summary>答案</summary>

渲染方程：$L = L_e + TL$

迭代展开：
- $L^{(0)} = L_e$（直接光照）
- $L^{(1)} = L_e + TL_e$（一次弹射）
- $L^{(2)} = L_e + TL_e + T^2L_e$（二次弹射）
- $L = \sum_{k=0}^{\infty} T^k L_e$

物理意义：$T^k L_e$表示经过$k$次反射/折射后的光照贡献。由于能量守恒，$\|T\| < 1$，级数收敛。

</details>

**练习17.2** 证明使用重要性采样时，当采样分布$p(x) \propto |f(x)|$时，Monte Carlo估计的方差最小。

*提示*：使用方差定义$\text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$。

<details>
<summary>答案</summary>

Monte Carlo估计：$\hat{I} = \frac{1}{N}\sum_{i=1}^{N} \frac{f(x_i)}{p(x_i)}$

方差：
$$\text{Var}[\hat{I}] = \frac{1}{N}\left(\int \frac{f^2(x)}{p(x)}dx - I^2\right)$$

最小化$\int \frac{f^2(x)}{p(x)}dx$，使用Cauchy-Schwarz不等式：
$$\int \frac{f^2(x)}{p(x)}dx \cdot \int p(x)dx \geq \left(\int |f(x)|dx\right)^2$$

等号成立条件：$p(x) \propto |f(x)|$

归一化：$p^*(x) = \frac{|f(x)|}{\int |f(x)|dx}$

</details>

**练习17.3** 推导正态分布$\mathcal{N}(\mu, \sigma^2)$的重参数化形式，并计算关于$\mu$和$\sigma$的梯度。

*提示*：使用标准正态分布的性质。

<details>
<summary>答案</summary>

重参数化：$z = \mu + \sigma \epsilon$，其中$\epsilon \sim \mathcal{N}(0, 1)$

对于函数$f(z)$的期望：
$$\mathbb{E}_{z \sim \mathcal{N}(\mu, \sigma^2)}[f(z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[f(\mu + \sigma \epsilon)]$$

梯度：
$$\frac{\partial}{\partial \mu} \mathbb{E}[f(z)] = \mathbb{E}\left[\frac{\partial f}{\partial z}(\mu + \sigma \epsilon)\right]$$

$$\frac{\partial}{\partial \sigma} \mathbb{E}[f(z)] = \mathbb{E}\left[\epsilon \frac{\partial f}{\partial z}(\mu + \sigma \epsilon)\right]$$

</details>

### 挑战题

**练习17.4** 设计一个结合重参数化和REINFORCE的混合梯度估计器，用于处理部分可微的渲染函数。分析其偏差和方差特性。

*提示*：考虑控制变量方法和Rao-Blackwellization。

<details>
<summary>答案</summary>

混合估计器：
$$\nabla_{\theta} J = \nabla_{\theta} \mathbb{E}[f_{diff}(\mathbf{z}; \theta)] + \mathbb{E}[(f_{disc}(\mathbf{z}) - c(\mathbf{z})) \nabla_{\theta} \log p(\mathbf{z}; \theta)]$$

其中：
- $f = f_{diff} + f_{disc}$
- $f_{diff}$：可微部分（使用重参数化）
- $f_{disc}$：不连续部分（使用REINFORCE）
- $c(\mathbf{z})$：控制变量

最优控制变量（最小化方差）：
$$c^*(\mathbf{z}) = \mathbb{E}[f_{disc}|\mathbf{z}_{diff}]$$

偏差分析：估计器无偏，因为两部分都是无偏估计。

方差分析：
$$\text{Var}[\nabla] = \text{Var}[\nabla f_{diff}] + \text{Var}[(f_{disc} - c) \nabla \log p]$$

使用控制变量降低第二项方差。

</details>

**练习17.5** 推导软光栅化中有符号距离函数对三角形顶点的梯度，考虑三角形退化的情况。

*提示*：使用重心坐标和点到平面的距离公式。

<details>
<summary>答案</summary>

点$\mathbf{p}$到三角形的有符号距离：

1. 投影到三角形平面：
$$\mathbf{p}_{proj} = \mathbf{p} - ((\mathbf{p} - \mathbf{v}_0) \cdot \mathbf{n})\mathbf{n}$$

2. 检查重心坐标：
$$(\lambda_0, \lambda_1, \lambda_2) = \text{barycentric}(\mathbf{p}_{proj})$$

3. 距离计算：
- 内部：$d = (\mathbf{p} - \mathbf{v}_0) \cdot \mathbf{n}$
- 边缘：$d = \min_{e} \text{dist}(\mathbf{p}, e)$
- 顶点：$d = \min_{v} \|\mathbf{p} - \mathbf{v}\|$

梯度（内部情况）：
$$\frac{\partial d}{\partial \mathbf{v}_0} = -\mathbf{n} + \frac{\partial \mathbf{n}}{\partial \mathbf{v}_0} \cdot (\mathbf{p} - \mathbf{v}_0)$$

其中：
$$\frac{\partial \mathbf{n}}{\partial \mathbf{v}_0} = \frac{1}{\|\mathbf{e}_1 \times \mathbf{e}_2\|}(I - \mathbf{n}\mathbf{n}^T)(\mathbf{e}_2 \times - \mathbf{e}_1 \times)$$

退化处理：当$\|\mathbf{e}_1 \times \mathbf{e}_2\| < \epsilon$时，使用数值稳定的替代公式。

</details>

**练习17.6** 分析多重重要性采样（MIS）在双向路径追踪中的应用，推导最优权重函数。

*提示*：考虑平衡启发式和功率启发式。

<details>
<summary>答案</summary>

双向路径追踪的MIS：

路径贡献：
$$C = \sum_{s=0}^{k} C_{s,t}$$

其中$C_{s,t}$是$s$个光源子路径顶点和$t$个相机子路径顶点的贡献。

MIS权重（平衡启发式）：
$$w_{s,t}(\mathbf{x}) = \frac{p_{s,t}(\mathbf{x})}{\sum_{s',t'} p_{s',t'}(\mathbf{x})}$$

功率启发式（$\beta=2$）：
$$w_{s,t}(\mathbf{x}) = \frac{p_{s,t}^2(\mathbf{x})}{\sum_{s',t'} p_{s',t'}^2(\mathbf{x})}$$

最优性分析：平衡启发式在所有无偏权重中具有最小方差的二阶近似。

实际应用：
- 直接光照：结合BSDF采样和光源采样
- 焦散：优先使用光源路径
- 间接漫反射：优先使用相机路径

</details>

**练习17.7** 设计一个自适应采样策略，根据梯度的局部方差动态调整采样密度。

*提示*：使用梯度的空间相关性和时间相关性。

<details>
<summary>答案</summary>

自适应采样框架：

1. **方差估计**：
$$\hat{\sigma}^2_{ij} = \frac{1}{N-1}\sum_{k=1}^{N} (g_k - \bar{g})^2$$

2. **采样密度分配**：
$$N_{ij} = N_{total} \frac{\hat{\sigma}_{ij}}{\sum_{i',j'} \hat{\sigma}_{i',j'}}$$

3. **空间相关性**：使用Gaussian filter平滑方差图：
$$\tilde{\sigma}^2_{ij} = \sum_{(i',j') \in \mathcal{N}(i,j)} w_{i',j'} \hat{\sigma}^2_{i',j'}$$

4. **时间相关性**：指数移动平均：
$$\sigma^2_{ij}(t) = \alpha \tilde{\sigma}^2_{ij}(t) + (1-\alpha) \sigma^2_{ij}(t-1)$$

5. **分层采样**：将图像分块，每块内部均匀采样。

收敛性分析：
- 总体方差：$\text{Var}_{total} \propto \sum_{ij} \frac{\sigma^2_{ij}}{N_{ij}}$
- 最优分配最小化总体方差

</details>

**练习17.8** 推导带有运动模糊的可微渲染公式，考虑时间维度的积分。

*提示*：将时间作为额外的积分维度。

<details>
<summary>答案</summary>

运动模糊渲染方程：
$$I(\mathbf{x}) = \int_0^T \int_{\Omega} L(\mathbf{x}, \omega, t) dt d\omega$$

其中物体位置随时间变化：
$$\mathbf{v}(t) = \mathbf{v}_0 + t\mathbf{v}_{velocity}$$

可微形式：
$$\frac{\partial I}{\partial \mathbf{v}_0} = \int_0^T \int_{\Omega} \frac{\partial L}{\partial \mathbf{v}} \frac{\partial \mathbf{v}(t)}{\partial \mathbf{v}_0} dt d\omega$$

离散化（分层采样）：
$$I \approx \frac{1}{N_t} \sum_{i=1}^{N_t} L(\mathbf{x}, \omega, t_i)$$

梯度计算考虑：
1. 可见性随时间变化
2. 着色随时间变化
3. 时间采样的重要性

优化：使用时间相干性减少采样。

</details>

## 常见陷阱与错误

### 1. 梯度偏差问题

**陷阱**：直接对Monte Carlo估计求导可能产生有偏梯度。

**正确做法**：
```
# 错误：先采样后求导
samples = sample(p(theta))
loss = mean(f(samples))
grad = autograd(loss, theta)  # 有偏！

# 正确：使用重参数化或REINFORCE
epsilon = sample(p_base)
samples = reparameterize(epsilon, theta)
loss = mean(f(samples))
grad = autograd(loss, theta)  # 无偏
```

### 2. 数值稳定性

**陷阱**：软光栅化中过小的$\sigma$导致梯度消失或爆炸。

**调试技巧**：
- 监控梯度范数：$\|\nabla\|_2$
- 使用梯度裁剪：$\nabla = \text{clip}(\nabla, -c, c)$
- 自适应调整软化参数

### 3. 采样效率

**陷阱**：均匀采样在高维空间中效率极低。

**优化策略**：
- 使用重要性采样
- 分层采样减少方差
- 准蒙特卡洛方法（如Sobol序列）

### 4. 内存管理

**陷阱**：存储完整计算图导致内存溢出。

**解决方案**：
- 梯度检查点（gradient checkpointing）
- 分块处理大场景
- 使用低精度（如float16）计算

### 5. 边界处理

**陷阱**：硬边界导致梯度为零，优化停滞。

**改进方法**：
- 使用软边界近似
- 边界积分提供准确梯度
- 结合多种梯度估计方法

## 最佳实践检查清单

### 算法设计审查

- [ ] 是否正确处理了可见性不连续？
- [ ] 梯度估计是否无偏？
- [ ] 是否使用了适当的重要性采样？
- [ ] 内存使用是否在可接受范围内？
- [ ] 是否考虑了数值稳定性？

### 实现优化审查

- [ ] 是否利用了GPU并行计算？
- [ ] 是否实现了早期剔除优化？
- [ ] 是否使用了空间数据结构加速？
- [ ] 是否实现了自适应采样？
- [ ] 是否缓存了可重用的中间结果？

### 验证测试审查

- [ ] 是否通过了梯度检查（有限差分验证）？
- [ ] 是否在简单场景下验证了正确性？
- [ ] 是否测试了极端情况（退化三角形等）？
- [ ] 是否评估了收敛速度？
- [ ] 是否比较了不同方法的性能？

### 参数调优审查

- [ ] 软化参数$\sigma$是否合适？
- [ ] 采样数量是否足够？
- [ ] 学习率是否适应梯度尺度？
- [ ] 是否需要预热（warm-up）？
- [ ] 是否需要渐进式训练？