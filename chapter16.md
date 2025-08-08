# 第16章：3D Gaussian Splatting

3D Gaussian Splatting是2023年兴起的革命性三维场景表示方法，通过显式的高斯原语实现了实时的高质量新视角合成。与神经隐式表示相比，它提供了更快的渲染速度、更好的可编辑性和更直观的几何解释。本章深入探讨其数学原理、优化算法和工程实现细节，重点关注可微光栅化、自适应密度控制和动态场景建模等核心技术。

## 16.1 高斯混合模型与球谐函数

### 16.1.1 三维高斯基础

三维高斯分布作为3D Gaussian Splatting的核心原语，其数学形式和几何意义决定了整个系统的表达能力。每个高斯可以看作一个"软椭球"，通过叠加实现复杂几何的表示。

#### 概率密度与几何解释

三维高斯分布的概率密度函数为：

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{3/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

其中 $\boldsymbol{\mu} \in \mathbb{R}^3$ 是均值（中心位置），$\Sigma \in \mathbb{R}^{3 \times 3}$ 是协方差矩阵（决定形状和方向）。

等概率密度面满足：
$$(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) = c^2$$

这定义了一个椭球，其主轴方向由$\Sigma$的特征向量决定，半轴长度正比于特征值的平方根。

#### 协方差矩阵的参数化

为了保证协方差矩阵的对称正定性，采用分解形式：
$$\Sigma = RSS^TR^T$$

其中：
- $R \in SO(3)$ 是旋转矩阵，定义椭球的方向
- $S = \text{diag}(s_x, s_y, s_z)$ 是对角缩放矩阵，$s_x, s_y, s_z > 0$ 定义椭球的半轴长度

这种参数化的优势：
1. **保证正定性**：只要$s_i > 0$，$\Sigma$必然正定
2. **直观控制**：旋转和缩放分离，便于优化
3. **数值稳定**：避免直接优化对称矩阵的6个独立元素

#### 四元数旋转表示

使用单位四元数 $\mathbf{q} = (q_w, q_x, q_y, q_z)$ 表示旋转，满足$||\mathbf{q}|| = 1$：

$$R(\mathbf{q}) = \begin{bmatrix}
1-2(q_y^2+q_z^2) & 2(q_xq_y-q_wq_z) & 2(q_xq_z+q_wq_y) \\
2(q_xq_y+q_wq_z) & 1-2(q_x^2+q_z^2) & 2(q_yq_z-q_wq_x) \\
2(q_xq_z-q_wq_y) & 2(q_yq_z+q_wq_x) & 1-2(q_x^2+q_y^2)
\end{bmatrix}$$

四元数的优势：
- **无奇异性**：避免欧拉角的万向锁问题
- **插值平滑**：球面线性插值(SLERP)自然
- **梯度计算**：$\frac{\partial R}{\partial \mathbf{q}}$形式简洁

#### 高斯的有效支撑域

虽然高斯分布理论上有无限支撑，实际计算中使用截断：
$$\text{support} = \{\mathbf{x} : (\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) \leq k^2\}$$

通常取$k=3$（覆盖99.7%的概率质量），对应的包围球半径：
$$r_{bound} = k \cdot \sqrt{\lambda_{max}}$$
其中$\lambda_{max}$是$\Sigma$的最大特征值。

### 16.1.2 投影到2D图像平面

将3D高斯投影到2D图像平面是渲染的关键步骤，需要处理透视投影的非线性变换。这个过程保持了高斯分布的性质（在一阶近似下），但改变了其参数。

#### 相机模型与坐标变换

采用针孔相机模型，完整的投影流程：
1. **世界到相机坐标**：$\mathbf{x}_c = W(\mathbf{x}_w - \mathbf{t})$
2. **透视投影**：$\mathbf{x}_n = [x_c/z_c, y_c/z_c]^T$
3. **图像坐标**：$\mathbf{x}_i = K\mathbf{x}_n$

其中$K$是内参矩阵：
$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

#### 雅可比矩阵推导

透视投影是非线性变换，使用一阶泰勒展开近似：
$$\mathbf{x}' \approx \pi(\boldsymbol{\mu}) + J(\mathbf{x} - \boldsymbol{\mu})$$

其中雅可比矩阵$J = \frac{\partial \pi}{\partial \mathbf{x}}|_{\mathbf{x}=\boldsymbol{\mu}}$：

$$J = \begin{bmatrix}
\frac{f_x}{z} & 0 & -\frac{f_x x}{z^2} \\
0 & \frac{f_y}{z} & -\frac{f_y y}{z^2}
\end{bmatrix}$$

这里$(x,y,z)$是高斯中心在相机坐标系下的位置。

#### 2D协方差的计算

通过线性变换的协方差传播公式：
$$\Sigma_{2D} = J \Sigma_{3D} J^T$$

展开计算：
$$\Sigma_{2D} = \begin{bmatrix} \sigma_{xx}' & \sigma_{xy}' \\ \sigma_{xy}' & \sigma_{yy}' \end{bmatrix}$$

其中各元素为$\Sigma_{3D}$元素的加权组合，权重由深度$z$决定。

#### EWA (Elliptical Weighted Average) 滤波

为了反走样，需要考虑像素的有限大小，添加低通滤波器：
$$\Sigma_{final} = \Sigma_{2D} + \epsilon I$$

其中$\epsilon \approx 0.3$像素，防止高斯过于尖锐导致的走样。

实际实现中，还需考虑：
- **近平面裁剪**：$z < z_{near}$的高斯需特殊处理
- **数值稳定性**：当$z$很小时，雅可比元素可能爆炸
- **保守包围盒**：考虑投影非线性，适当扩大2D包围盒

### 16.1.3 球谐函数表示颜色

球谐函数(Spherical Harmonics)提供了一种紧凑而高效的方式来表示视角相关的外观，特别适合低频的光照变化。

#### 球谐函数基础

球谐函数是拉普拉斯方程在球面上的解，形成完备正交基：
$$c(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} Y_l^m(\mathbf{d})$$

其中：
- $\mathbf{d} = (\theta, \phi)$ 是观察方向的球坐标
- $l$ 是阶数（degree），$m$ 是序数（order）
- $Y_l^m$ 是球谐基函数，$c_{lm}$ 是对应系数

#### 实球谐函数

实际使用实球谐函数（Real SH）避免复数运算：

**0阶（常数项）**：
$$Y_0^0 = \frac{1}{2}\sqrt{\frac{1}{\pi}}$$

**1阶（线性项）**：
$$Y_1^{-1} = \frac{1}{2}\sqrt{\frac{3}{\pi}} y, \quad Y_1^0 = \frac{1}{2}\sqrt{\frac{3}{\pi}} z, \quad Y_1^1 = \frac{1}{2}\sqrt{\frac{3}{\pi}} x$$

**2阶（二次项）**：
$$Y_2^{-2} = \frac{1}{2}\sqrt{\frac{15}{\pi}} xy, \quad Y_2^{-1} = \frac{1}{2}\sqrt{\frac{15}{\pi}} yz$$
$$Y_2^0 = \frac{1}{4}\sqrt{\frac{5}{\pi}} (3z^2-1), \quad Y_2^1 = \frac{1}{2}\sqrt{\frac{15}{\pi}} xz$$
$$Y_2^2 = \frac{1}{4}\sqrt{\frac{15}{\pi}} (x^2-y^2)$$

#### 存储与计算优化

对于$L$阶SH，需要$(L+1)^2$个系数：
- 0阶：1个系数（全向量颜色）
- 1阶：4个系数（添加线性变化）
- 2阶：9个系数（添加二次变化）
- 3阶：16个系数（通常足够）

RGB颜色需要3倍存储：$3 \times (L+1)^2$个浮点数。

计算优化技巧：
1. **递推计算**：使用递推关系避免重复计算三角函数
2. **向量化**：将SH系数组织为矩阵形式批量计算
3. **预计算查表**：对离散视角预计算SH值

#### 频谱分析与带宽限制

球谐展开类似于傅里叶级数，不同阶数对应不同频率：
- 低阶（$l \leq 2$）：捕获漫反射和环境光照
- 中阶（$3 \leq l \leq 5$）：捕获光泽反射
- 高阶（$l > 5$）：捕获镜面高光和细节

Nyquist采样定理的球面版本：需要至少$(2L+1)^2$个采样点完美重建$L$阶信号。

### 16.1.4 不透明度与体积渲染

不透明度控制是实现正确混合和遮挡关系的关键，需要结合几何衰减和材质属性。

#### 不透明度参数化

每个高斯维护一个可学习的不透明度参数$\alpha_{raw} \in \mathbb{R}$，通过sigmoid函数映射到[0,1]：
$$\alpha_{base} = \sigma(\alpha_{raw}) = \frac{1}{1 + e^{-\alpha_{raw}}}$$

这种参数化的优势：
- 无约束优化：梯度下降可以自由更新$\alpha_{raw}$
- 自然初始化：$\alpha_{raw} = 0$对应$\alpha = 0.5$
- 梯度流畅：sigmoid在中间区域梯度最大

#### 空间衰减函数

高斯的空间影响通过概率密度的归一化版本表示：
$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

注意这里省略了归一化常数$(2\pi)^{3/2}|\Sigma|^{1/2}$，因为我们关心的是相对权重。

最终的不透明度：
$$\alpha(\mathbf{x}) = \alpha_{base} \cdot G(\mathbf{x})$$

#### 体积渲染积分

沿射线$\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$的颜色积分：
$$C = \int_0^{\infty} T(t) \sigma(t) c(t) dt$$

其中透射率：
$$T(t) = \exp\left(-\int_0^t \sigma(s) ds\right)$$

对于高斯splatting，使用alpha合成近似：
$$C \approx \sum_{i=1}^{N} T_i \alpha_i c_i, \quad T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$$

这种近似在高斯密度足够时收敛到真实积分。

## 16.2 可微光栅化与排序

可微光栅化是3D Gaussian Splatting实现实时渲染和端到端优化的核心技术。通过精心设计的前向渲染和反向传播算法，实现了高效的梯度计算。

### 16.2.1 瓦片化光栅化

瓦片化(Tiling)策略将图像空间划分为固定大小的区块，显著减少了每个像素需要处理的高斯数量，是实现实时性能的关键。

#### 瓦片划分策略

将图像分割成$T_w \times T_h$的瓦片网格，通常使用$16 \times 16$像素的瓦片：
$$N_{tiles} = \lceil \frac{W}{16} \rceil \times \lceil \frac{H}{16} \rceil$$

每个瓦片维护一个高斯列表，存储与该瓦片相交的所有高斯索引。

#### 2D包围盒计算

对于投影后的2D高斯，计算其轴对齐包围盒(AABB)：

1. **特征值分解**：
$$\Sigma_{2D} = V \Lambda V^T, \quad \Lambda = \text{diag}(\lambda_1, \lambda_2)$$

2. **置信椭圆半径**：
$$r = k\sqrt{\max(\lambda_1, \lambda_2)}$$
其中$k=3$对应99.7%置信度。

3. **保守包围盒**：
$$\text{AABB} = [\mu_x - r, \mu_x + r] \times [\mu_y - r, \mu_y + r]$$

实际中可以使用更紧的包围盒：
$$r_x = k\sqrt{\sigma_{xx}}, \quad r_y = k\sqrt{\sigma_{yy}}$$

#### 瓦片分配算法

```
for each gaussian g:
    bbox = compute_2d_bbox(g)
    tile_min = floor(bbox.min / 16)
    tile_max = ceil(bbox.max / 16)
    
    for tx in [tile_min.x, tile_max.x]:
        for ty in [tile_min.y, tile_max.y]:
            tile[tx][ty].add(g.id)
```

优化技巧：
- **分层瓦片**：使用多级瓦片加速大高斯的处理
- **动态瓦片大小**：根据高斯密度自适应调整
- **早期剔除**：利用深度缓冲区提前剔除被遮挡的高斯

### 16.2.2 深度排序与alpha混合

正确的深度排序是实现透明度混合的前提，需要在效率和准确性之间平衡。

#### 排序策略

1. **全局排序**：
对所有高斯按深度排序，复杂度$O(N\log N)$，内存访问不连续。

2. **瓦片内排序**：
每个瓦片独立排序，复杂度$O(N_t\log N_t)$，其中$N_t \ll N$。

3. **基数排序优化**：
利用深度的有限精度，使用基数排序达到$O(N)$复杂度：
$$z_{quantized} = \lfloor z \cdot 2^{16} \rfloor$$

#### Alpha混合计算

前向渲染的alpha混合公式：
$$C = \sum_{i=1}^{N} T_i \alpha_i c_i$$

累积透射率的递推计算：
$$T_1 = 1, \quad T_{i+1} = T_i \cdot (1 - \alpha_i)$$

#### 饱和停止优化

当累积不透明度接近1时停止：
$$\sum_{j=1}^{i} \alpha_j T_j > 1 - \epsilon$$

或等价地：
$$T_i < \epsilon_{stop}$$

典型取$\epsilon_{stop} = 0.001$。

实际实现考虑：
- **数值稳定性**：防止$T_i$下溢，使用对数空间计算
- **向量化**：批量处理多个像素的alpha混合
- **内存局部性**：优化数据布局减少cache miss

### 16.2.3 反向传播梯度计算

可微光栅化的关键是高效准确地计算损失对所有高斯参数的梯度。这需要仔细处理alpha混合的链式法则。

#### 颜色梯度

对于渲染颜色$C = \sum_{i} T_i \alpha_i c_i$，各参数的梯度：

**颜色系数梯度**：
$$\frac{\partial \mathcal{L}}{\partial c_i} = \frac{\partial \mathcal{L}}{\partial C} \cdot T_i \alpha_i$$

这是最简单的情况，梯度正比于高斯对最终颜色的贡献。

**不透明度梯度**：
$$\frac{\partial \mathcal{L}}{\partial \alpha_i} = \frac{\partial \mathcal{L}}{\partial C} \cdot \left(T_i c_i - \sum_{j=i+1}^{N} \frac{T_j \alpha_j c_j}{1-\alpha_i}\right)$$

第二项来自$\alpha_i$对后续高斯透射率的影响。

#### 几何参数梯度

**位置梯度**通过多条路径传播：
$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}} = \frac{\partial \mathcal{L}}{\partial \alpha} \frac{\partial \alpha}{\partial G} \frac{\partial G}{\partial \boldsymbol{\mu}} + \frac{\partial \mathcal{L}}{\partial \mathbf{x}'} \frac{\partial \mathbf{x}'}{\partial \boldsymbol{\mu}}$$

其中：
$$\frac{\partial G}{\partial \boldsymbol{\mu}} = G \cdot \Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu})$$

**协方差梯度**：
$$\frac{\partial G}{\partial \Sigma} = \frac{1}{2} G \left(\Sigma^{-1}\mathbf{d}\mathbf{d}^T\Sigma^{-1} - \Sigma^{-1}\right)$$
其中$\mathbf{d} = \mathbf{x} - \boldsymbol{\mu}$。

#### 参数化梯度

从$\Sigma = RSS^TR^T$到原始参数的梯度：

**缩放梯度**：
$$\frac{\partial \mathcal{L}}{\partial s_k} = 2s_k \sum_{i,j} \frac{\partial \mathcal{L}}{\partial \Sigma_{ij}} R_{ik}R_{jk}$$

**四元数梯度**：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{q}} = \frac{\partial \mathcal{L}}{\partial R} \frac{\partial R}{\partial \mathbf{q}}$$

需要投影到单位球面保持归一化：
$$\nabla_{\mathbf{q}} \leftarrow \nabla_{\mathbf{q}} - (\mathbf{q}^T\nabla_{\mathbf{q}})\mathbf{q}$$

#### 梯度累积与原子操作

由于多个像素可能同时更新同一个高斯的梯度，需要原子操作：
```
atomicAdd(&grad_mu[g_id], local_grad_mu)
atomicAdd(&grad_sigma[g_id], local_grad_sigma)
```

优化策略：
- **局部累积**：先在共享内存累积，减少原子操作
- **梯度分块**：将梯度分配到不同内存区域，后续归约
- **混合精度**：使用fp16累积，定期转换到fp32

### 16.2.4 数值稳定性考虑

数值稳定性对训练的收敛性和最终质量至关重要。

#### 协方差正定性维护

1. **正则化项**：
$$\Sigma_{reg} = RSS^TR^T + \epsilon I$$
其中$\epsilon \approx 10^{-7}$防止奇异。

2. **缩放下界**：
$$s_i = \max(s_i, s_{min})$$
典型$s_{min} = 10^{-7}$。

3. **条件数控制**：
$$\kappa(\Sigma) = \frac{\lambda_{max}}{\lambda_{min}} < \kappa_{max}$$
当条件数过大时，重新初始化或分裂高斯。

#### 梯度爆炸控制

1. **梯度裁剪**：
$$\nabla_{clipped} = \nabla \cdot \min\left(1, \frac{C_{max}}{||\nabla||_2}\right)$$
其中$C_{max}$是最大梯度范数。

2. **自适应裁剪**：
基于历史统计动态调整：
$$C_{max} = \mu_{grad} + k \cdot \sigma_{grad}$$

3. **分层裁剪**：
对不同参数使用不同阈值：
- 位置：较大阈值允许快速移动
- 颜色：中等阈值保持稳定
- 不透明度：小阈值防止突变

#### 学习率调度

1. **尺度相关学习率**：
$$lr_{\boldsymbol{\mu}} = lr_{base} \cdot \frac{1}{\text{scale}(\mathcal{G})}$$
其中$\text{scale}(\mathcal{G}) = \sqrt[3]{||\Sigma||}$。

2. **指数衰减**：
$$lr(t) = lr_0 \cdot \gamma^{t/T}$$

3. **Warmup策略**：
$$lr(t) = \begin{cases}
lr_0 \cdot t/T_{warm} & t < T_{warm} \\
lr_0 & t \geq T_{warm}
\end{cases}$$

#### 浮点精度管理

1. **混合精度训练**：
- 前向传播：fp16减少内存带宽
- 梯度累积：fp32保持精度
- 参数更新：fp32避免舍入误差

2. **Kahan求和**：
对于大量高斯的累积，使用补偿求和算法：
```
sum = 0, c = 0
for value in values:
    y = value - c
    t = sum + y
    c = (t - sum) - y
    sum = t
```

## 16.3 自适应密度控制

### 16.3.1 密度自适应算法

基于梯度和重建误差的自适应策略：

1. **过度重建区域检测**：
$$\nabla_{\boldsymbol{\mu}} = ||\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}}||$$

当 $\nabla_{\boldsymbol{\mu}} > \tau_{grad}$ 时，需要增加高斯密度。

2. **尺度判断**：
大高斯分裂条件：
$$\max(s_x, s_y, s_z) > \tau_{scale}$$

3. **不透明度剪枝**：
$$\alpha < \tau_{\alpha} \Rightarrow \text{remove}$$

### 16.3.2 高斯分裂策略

分裂操作创建两个子高斯：
$$\boldsymbol{\mu}_{new} = \boldsymbol{\mu} \pm \delta \cdot \mathbf{n}$$

其中 $\mathbf{n}$ 是沿最大特征值方向：
$$\Sigma = V \Lambda V^T, \quad \mathbf{n} = V[:,\arg\max(\Lambda)]$$

新的缩放因子：
$$S_{new} = \phi \cdot S, \quad \phi \approx 0.6$$

### 16.3.3 高斯克隆

对于欠重建区域，克隆小高斯：
$$\text{if } \max(s_x, s_y, s_z) < \tau_{small} \text{ and } \nabla_{\boldsymbol{\mu}} > \tau_{grad}$$

克隆位置沿梯度方向：
$$\boldsymbol{\mu}_{clone} = \boldsymbol{\mu} + \epsilon \cdot \frac{\nabla_{\boldsymbol{\mu}}}{||\nabla_{\boldsymbol{\mu}}||}$$

### 16.3.4 定期重置与合并

1. **不透明度重置**：
每$N$次迭代重置接近透明的高斯：
$$\alpha < 0.01 \Rightarrow \alpha_{reset} = 0.01$$

2. **高斯合并**：
距离小于阈值的相似高斯合并：
$$d(\mathcal{G}_i, \mathcal{G}_j) = ||\boldsymbol{\mu}_i - \boldsymbol{\mu}_j|| + \beta \cdot ||\text{vec}(\Sigma_i) - \text{vec}(\Sigma_j)||$$

合并后的参数：
$$\boldsymbol{\mu}_{merged} = \frac{\alpha_i\boldsymbol{\mu}_i + \alpha_j\boldsymbol{\mu}_j}{\alpha_i + \alpha_j}$$

## 16.4 压缩与量化技术

### 16.4.1 向量量化

使用码本量化高斯属性：

1. **位置量化**：
将场景包围盒离散化：
$$\boldsymbol{\mu}_{quantized} = \text{round}\left(\frac{\boldsymbol{\mu} - \mathbf{b}_{min}}{\mathbf{b}_{max} - \mathbf{b}_{min}} \cdot 2^{b}\right)$$

其中 $b$ 是量化位数（通常16位）。

2. **旋转量化**：
四元数归一化后量化：
$$q_{quantized} = \text{round}(q \cdot 2^{10})$$

利用单位四元数约束，只存储3个分量。

3. **尺度对数量化**：
$$s_{log} = \log(s), \quad s_{quantized} = \text{round}\left(\frac{s_{log} - s_{log,min}}{s_{log,max} - s_{log,min}} \cdot 2^{8}\right)$$

### 16.4.2 球谐系数压缩

1. **DCT变换**：
对SH系数应用离散余弦变换：
$$C_{DCT} = DCT(c_{lm})$$

保留前$k$个系数（能量集中）。

2. **主成分分析**：
收集训练数据构建PCA基：
$$\mathbf{c} = \bar{\mathbf{c}} + \sum_{i=1}^{k} a_i \mathbf{v}_i$$

其中 $\mathbf{v}_i$ 是主成分。

3. **向量码本**：
使用k-means聚类构建码本：
$$\mathcal{C} = \{c_1, c_2, ..., c_K\}$$

每个高斯存储最近码字索引。

### 16.4.3 熵编码与存储优化

1. **霍夫曼编码**：
统计属性分布，构建霍夫曼树：
$$H(X) = -\sum_{i} p_i \log_2 p_i$$

2. **空间哈希**：
稀疏体素网格存储：
$$hash(x,y,z) = (x \cdot p_1 \oplus y \cdot p_2 \oplus z \cdot p_3) \mod M$$

其中 $p_1, p_2, p_3$ 是大质数。

3. **八叉树层次结构**：
递归细分空间：
$$\text{LOD}_i = \text{merge}(\text{children at level } i+1)$$

### 16.4.4 流式传输与渐进加载

1. **重要性排序**：
$$importance = \alpha \cdot volume \cdot view\_frequency$$

其中 $volume = \sqrt{||\Sigma||}$

2. **视锥体剔除**：
$$\text{visible} = \text{frustum\_test}(\boldsymbol{\mu}, radius)$$

3. **细节层次(LOD)**：
根据距离选择简化版本：
$$LOD = \max(0, \log_2(\frac{d}{d_{ref}}))$$

## 16.5 动态场景与4D表示

### 16.5.1 时间维度建模

4D高斯表示：
$$\mathcal{G}(t) = \{\boldsymbol{\mu}(t), \Sigma(t), \alpha(t), \mathbf{c}(t)\}$$

1. **轨迹参数化**：
使用多项式基：
$$\boldsymbol{\mu}(t) = \sum_{i=0}^{n} \mathbf{a}_i t^i$$

或Fourier基：
$$\boldsymbol{\mu}(t) = \boldsymbol{\mu}_0 + \sum_{k=1}^{K} \mathbf{A}_k \sin(k\omega t) + \mathbf{B}_k \cos(k\omega t)$$

2. **神经轨迹**：
$$\boldsymbol{\mu}(t) = \boldsymbol{\mu}_0 + MLP_{\theta}(t, \mathbf{z})$$

其中 $\mathbf{z}$ 是潜在编码。

### 16.5.2 变形场方法

1. **显式变形场**：
$$\mathbf{x}(t) = \mathbf{x}_0 + \mathbf{d}(\mathbf{x}_0, t)$$

变形场正则化：
$$\mathcal{L}_{reg} = \lambda_1 ||\nabla \mathbf{d}||^2 + \lambda_2 ||\nabla^2 \mathbf{d}||^2$$

2. **光流约束**：
$$\frac{\partial I}{\partial t} + \nabla I \cdot \mathbf{v} = 0$$

其中 $\mathbf{v} = \frac{\partial \mathbf{x}}{\partial t}$

3. **刚性分解**：
$$\mathbf{x}(t) = R(t)\mathbf{x}_0 + \mathbf{t}(t)$$

使用Procrustes分析估计 $R(t), \mathbf{t}(t)$。

### 16.5.3 时间一致性优化

1. **时序平滑**：
$$\mathcal{L}_{smooth} = \sum_{t} ||\mathcal{G}(t+\Delta t) - \mathcal{G}(t)||^2$$

2. **循环一致性**：
$$\mathcal{L}_{cycle} = ||\mathcal{T}_{t \to t+T} \circ \mathcal{T}_{0 \to t}(\mathbf{x}) - \mathcal{T}_{0 \to t+T}(\mathbf{x})||$$

3. **长程依赖**：
使用LSTM或Transformer建模时序关系：
$$\mathbf{h}_t = LSTM(\mathcal{G}(t), \mathbf{h}_{t-1})$$

### 16.5.4 增量更新与在线学习

1. **关键帧策略**：
$$\text{is\_keyframe} = \Delta_{pose} > \tau_{rot} \text{ or } ||\Delta_{trans}|| > \tau_{trans}$$

2. **局部更新**：
只优化视锥体内的高斯：
$$\mathcal{G}_{active} = \{\mathcal{G}_i | \text{in\_frustum}(\boldsymbol{\mu}_i)\}$$

3. **时间窗口优化**：
$$\mathcal{L} = \sum_{t \in [T-w, T]} \mathcal{L}_{render}(t) + \lambda \mathcal{L}_{consistency}$$

### 16.5.5 多视图-多时刻联合优化

优化目标：
$$\mathcal{L}_{total} = \sum_{v,t} \mathcal{L}_{photo}(I_{v,t}, \hat{I}_{v,t}) + \lambda_1 \mathcal{L}_{temporal} + \lambda_2 \mathcal{L}_{spatial}$$

其中：
- $\mathcal{L}_{photo}$：光度一致性损失
- $\mathcal{L}_{temporal}$：时间平滑项  
- $\mathcal{L}_{spatial}$：空间正则化

## 本章小结

3D Gaussian Splatting通过显式的高斯原语表示实现了实时高质量渲染，主要创新包括：

1. **高效表示**：使用各向异性3D高斯配合球谐函数，紧凑表示几何和外观
2. **可微光栅化**：基于瓦片的排序算法实现快速前向渲染和梯度反传
3. **自适应优化**：通过分裂、克隆和剪枝动态调整高斯密度
4. **压缩技术**：向量量化、熵编码等方法大幅减少存储需求
5. **动态扩展**：4D表示和变形场支持动态场景建模

关键数学工具：
- 多元高斯分布：$p(\mathbf{x}) \propto \exp(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}))$
- 球谐展开：$c(\mathbf{d}) = \sum_{l,m} c_{lm} Y_l^m(\mathbf{d})$
- Alpha混合：$C = \sum_i T_i \alpha_i c_i$，$T_i = \prod_{j<i}(1-\alpha_j)$
- 梯度下降：$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$

## 练习题

### 基础题

**练习16.1** 证明3D高斯投影到2D平面后仍是高斯分布。设3D高斯中心为$\boldsymbol{\mu}$，协方差为$\Sigma$，投影矩阵为$P$。

<details>
<summary>提示</summary>
使用线性变换的性质和特征函数方法。
</details>

<details>
<summary>答案</summary>
线性变换下高斯分布保持高斯性。设投影$\mathbf{y} = P\mathbf{x}$，则：
$$\mathbf{y} \sim \mathcal{N}(P\boldsymbol{\mu}, P\Sigma P^T)$$
证明通过特征函数：
$$\phi_Y(t) = E[e^{it^TY}] = e^{it^TP\mu - \frac{1}{2}t^TP\Sigma P^Tt}$$
这正是均值为$P\boldsymbol{\mu}$、协方差为$P\Sigma P^T$的高斯分布特征函数。
</details>

**练习16.2** 推导2D高斯在像素$(u,v)$处的值，给定中心$(u_0,v_0)$和2×2协方差矩阵$\Sigma'$。

<details>
<summary>提示</summary>
直接应用2D高斯公式并简化。
</details>

<details>
<summary>答案</summary>
$$G(u,v) = \exp\left(-\frac{1}{2}\begin{bmatrix}u-u_0\\v-v_0\end{bmatrix}^T (\Sigma')^{-1} \begin{bmatrix}u-u_0\\v-v_0\end{bmatrix}\right)$$
设$\Sigma' = \begin{bmatrix}a & b\\b & c\end{bmatrix}$，则：
$$G(u,v) = \exp\left(-\frac{1}{2(ac-b^2)}[c(u-u_0)^2 - 2b(u-u_0)(v-v_0) + a(v-v_0)^2]\right)$$
</details>

**练习16.3** 计算$N$个高斯alpha混合的计算复杂度，并分析早停策略的影响。

<details>
<summary>提示</summary>
考虑排序、累积透射率计算和早停条件。
</details>

<details>
<summary>答案</summary>
不带早停：$O(N\log N)$排序 + $O(N)$混合 = $O(N\log N)$
带早停（平均停在$k$个高斯）：$O(N\log N)$排序 + $O(k)$混合
当$k \ll N$时，混合成本大幅降低，但排序仍是瓶颈。
瓦片化减少每个瓦片的$N$，使得排序成本降为$O(N_t\log N_t)$，$N_t$是瓦片内高斯数。
</details>

**练习16.4** 设计一个判断高斯是否需要分裂的度量，考虑梯度magnitude和空间extent。

<details>
<summary>提示</summary>
结合位置梯度和尺度信息。
</details>

<details>
<summary>答案</summary>
分裂度量：
$$S = ||\nabla_{\boldsymbol{\mu}}\mathcal{L}|| \cdot \max(s_x, s_y, s_z)$$
当$S > \tau$时触发分裂。这确保大的欠拟合高斯被分裂。
另一种方法使用Hessian：
$$S = \text{tr}(H_{\boldsymbol{\mu}}) \cdot \text{vol}(\Sigma)^{1/3}$$
</details>

### 挑战题

**练习16.5** 推导球谐函数旋转公式。给定旋转矩阵$R$，如何变换SH系数使得$c'(\mathbf{d}) = c(R^T\mathbf{d})$？

<details>
<summary>提示</summary>
使用Wigner D-矩阵和球谐函数的旋转性质。
</details>

<details>
<summary>答案</summary>
SH系数变换通过Wigner D-矩阵：
$$c'_{lm} = \sum_{m'=-l}^{l} D^l_{mm'}(R) c_{lm'}$$
其中$D^l_{mm'}(R)$是Wigner D-矩阵元素。
对于低阶（$l \leq 2$），可以预计算旋转矩阵。
$l=1$时直接是3×3旋转：$\mathbf{c}'_1 = R\mathbf{c}_1$
</details>

**练习16.6** 分析3D Gaussian Splatting的内存带宽需求。假设场景有100万个高斯，每个高斯48字节属性，渲染1920×1080图像。

<details>
<summary>提示</summary>
考虑排序、光栅化和混合各阶段的内存访问。
</details>

<details>
<summary>答案</summary>
内存带宽分析：
1. 高斯数据加载：$10^6 \times 48B = 48MB$
2. 投影后2D高斯：$10^6 \times 24B = 24MB$（位置+协方差）
3. 深度排序索引：$10^6 \times 8B = 8MB$
4. 瓦片分配（假设平均10瓦片/高斯）：$10^7 \times 4B = 40MB$
5. 像素渲染（每像素访问~30高斯）：$1920 \times 1080 \times 30 \times 16B \approx 1GB$
总带宽需求：~1.1GB/帧，30fps需要33GB/s带宽。
</details>

**练习16.7** 设计一个基于八叉树的高斯组织结构，支持视锥体剔除和LOD选择。给出数据结构和查询算法。

<details>
<summary>提示</summary>
每个节点存储包围盒和高斯列表，递归遍历。
</details>

<details>
<summary>答案</summary>
```
struct OctreeNode {
    AABB bounds;
    vector<GaussianID> gaussians;
    array<OctreeNode*, 8> children;
    float avgSize;  // 平均高斯尺寸
    int level;
}

QueryVisible(node, frustum, viewpoint):
    if not frustum.intersects(node.bounds):
        return []
    
    distance = ||viewpoint - node.center||
    if node.isLeaf or distance > LOD_threshold * node.avgSize:
        return node.gaussians
    
    visible = []
    for child in node.children:
        visible += QueryVisible(child, frustum, viewpoint)
    return visible
```
</details>

**练习16.8** （开放题）提出一种新的高斯压缩方案，结合神经网络和传统压缩技术。分析压缩率和质量权衡。

<details>
<summary>提示</summary>
考虑：潜在空间编码、可学习量化、上下文熵编码。
</details>

<details>
<summary>答案</summary>
混合压缩方案：
1. **神经编码器**：将局部高斯群编码到潜在向量
   $$\mathbf{z} = \text{Encoder}(\{\mathcal{G}_i\}_{i \in \text{local}})$$
2. **向量量化**：使用可学习码本
   $$\mathbf{z}_q = \text{VQ}(\mathbf{z})$$
3. **上下文模型**：基于空间邻域预测
   $$p(\mathbf{z}_q | \mathbf{z}_{neighbors})$$
4. **算术编码**：利用预测分布压缩

预期压缩率：10-50×，取决于质量要求
关键是平衡压缩率、解码速度和重建质量。
</details>

## 常见陷阱与错误 (Gotchas)

1. **数值不稳定**
   - 协方差矩阵可能退化，需添加正则项$\epsilon I$
   - 梯度爆炸，使用梯度裁剪和自适应学习率
   - 四元数需保持归一化

2. **排序瓶颈**
   - 全局排序开销大，使用瓦片局部排序
   - 考虑近似排序算法（如基数排序）
   - 利用帧间相干性避免重复排序

3. **内存管理**
   - 动态分配导致碎片化，使用内存池
   - GPU内存有限，实现分块渲染
   - 注意瓦片列表的动态增长

4. **训练不稳定**
   - 学习率过大导致高斯"爆炸"
   - 密度控制参数敏感，需仔细调节
   - 早期过度剪枝影响最终质量

5. **压缩伪影**
   - 过度量化产生块状伪影
   - SH截断导致高光丢失
   - 空间哈希冲突造成混叠

## 最佳实践检查清单

### 实现优化
- [ ] 使用共享内存优化瓦片内排序
- [ ] 实现视锥体剔除减少处理高斯数
- [ ] 采用分层数据结构加速空间查询
- [ ] 利用时间相干性复用计算结果
- [ ] 实现自适应精度（fp16/fp32混合）

### 质量控制
- [ ] 监控梯度范数防止训练崩溃
- [ ] 定期验证高斯分布合理性（无极端值）
- [ ] 检查alpha混合饱和度避免过度累积
- [ ] 验证视角一致性（多视角渲染对比）
- [ ] 评估压缩后的PSNR/SSIM指标

### 内存效率
- [ ] 实现增量式场景加载
- [ ] 使用LOD减少远处细节
- [ ] 定期整理内存避免碎片化
- [ ] 监控GPU内存使用防止溢出
- [ ] 实现高斯回收池减少分配开销

### 鲁棒性设计
- [ ] 处理退化情况（共线高斯、零尺度）
- [ ] 实现优雅降级（内存不足时降低质量）
- [ ] 添加断言检查关键不变量
- [ ] 实现checkpoint机制支持训练恢复
- [ ] 设计fallback路径处理硬件限制