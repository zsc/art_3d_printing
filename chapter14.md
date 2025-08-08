# 第14章：多视图几何

多视图几何是三维重建的数学基础，它研究如何从多个二维图像恢复三维场景结构。本章将深入探讨从两视图对极几何到大规模三维重建的完整数学框架，包括经典的几何方法和现代深度学习方法。我们将重点关注数学原理的推导、优化算法的设计以及实际系统的构建策略。

## 14.1 对极几何与基础矩阵

### 14.1.1 针孔相机模型

针孔相机模型是多视图几何的基础。三维点 $\mathbf{X} = [X, Y, Z]^T$ 投影到图像平面的过程可以表示为：

$$\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K[R|\mathbf{t}]\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

其中 $\lambda$ 是深度标量，内参矩阵 $K$ 定义为：

$$K = \begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}$$

这里 $f_x, f_y$ 是焦距（单位：像素），$(c_x, c_y)$ 是主点坐标，$s$ 是倾斜参数（通常为0，表示像素坐标轴正交）。

**径向畸变模型**：实际相机存在畸变，最常见的是径向畸变：

$$\begin{aligned}
x_d &= x_u(1 + k_1r^2 + k_2r^4 + k_3r^6) \\
y_d &= y_u(1 + k_1r^2 + k_2r^4 + k_3r^6)
\end{aligned}$$

其中 $(x_u, y_u)$ 是未畸变的归一化坐标，$r^2 = x_u^2 + y_u^2$，$k_1, k_2, k_3$ 是径向畸变系数。

**切向畸变**：由镜头和成像平面不平行引起：

$$\begin{aligned}
x_d &= x_u + [2p_1x_uy_u + p_2(r^2 + 2x_u^2)] \\
y_d &= y_u + [p_1(r^2 + 2y_u^2) + 2p_2x_uy_u]
\end{aligned}$$

完整的投影过程为：
1. 世界坐标到相机坐标：$\mathbf{X}_c = R\mathbf{X}_w + \mathbf{t}$
2. 归一化平面投影：$[x_u, y_u]^T = [X_c/Z_c, Y_c/Z_c]^T$
3. 畸变校正：应用径向和切向畸变
4. 像素坐标：$[u, v]^T = [f_xx_d + c_x, f_yy_d + c_y]^T$

### 14.1.2 对极约束

考虑两个相机观察同一个三维点 $\mathbf{X}$，在两个图像中的投影分别为 $\mathbf{x}_1$ 和 $\mathbf{x}_2$。对极约束表达为：

$$\mathbf{x}_2^T F \mathbf{x}_1 = 0$$

其中 $F$ 是 $3 \times 3$ 基础矩阵（Fundamental Matrix）。

**对极几何的几何解释**：
- **对极平面**：由两个相机中心 $\mathbf{C}_1, \mathbf{C}_2$ 和三维点 $\mathbf{X}$ 确定的平面
- **对极线**：对极平面与图像平面的交线，$\mathbf{l}_2 = F\mathbf{x}_1$
- **对极点**：另一相机中心在当前图像中的投影，$\mathbf{e}_1 = P_1\mathbf{C}_2$

**基础矩阵的性质**：
1. **秩约束**：$\text{rank}(F) = 2$（奇异矩阵）
2. **对极点满足**：$F\mathbf{e}_1 = 0$，$F^T\mathbf{e}_2 = 0$（对极点是 $F$ 的左右零空间）
3. **自由度**：9个元素 - 1个尺度 - 1个秩约束 = 7个自由度
4. **与本质矩阵的关系**：$E = K_2^T F K_1$

**基础矩阵的代数推导**：

设点 $\mathbf{X}$ 在两个相机坐标系下分别为 $\mathbf{X}_1$ 和 $\mathbf{X}_2$，有：
$$\mathbf{X}_2 = R\mathbf{X}_1 + \mathbf{t}$$

两边同时与 $\mathbf{t}$ 做叉乘：
$$\mathbf{t} \times \mathbf{X}_2 = \mathbf{t} \times R\mathbf{X}_1$$

再与 $\mathbf{X}_2$ 做点乘：
$$\mathbf{X}_2^T [\mathbf{t}]_\times \mathbf{X}_2 = \mathbf{X}_2^T [\mathbf{t}]_\times R \mathbf{X}_1 = 0$$

由于 $\mathbf{x} = K^{-1}\tilde{\mathbf{x}}$（$\tilde{\mathbf{x}}$ 是齐次图像坐标），代入得：
$$\tilde{\mathbf{x}}_2^T K_2^{-T} [\mathbf{t}]_\times R K_1^{-1} \tilde{\mathbf{x}}_1 = 0$$

因此：$F = K_2^{-T} [\mathbf{t}]_\times R K_1^{-1} = K_2^{-T} E K_1^{-1}$

### 14.1.3 本质矩阵分解

本质矩阵 $E$ 编码了相机间的相对位姿，可以分解为旋转和平移：

$$E = [\mathbf{t}]_\times R$$

其中 $[\mathbf{t}]_\times$ 是平移向量的反对称矩阵（叉积矩阵）：

$$[\mathbf{t}]_\times = \begin{bmatrix}
0 & -t_z & t_y \\
t_z & 0 & -t_x \\
-t_y & t_x & 0
\end{bmatrix}$$

满足性质：$[\mathbf{t}]_\times \mathbf{v} = \mathbf{t} \times \mathbf{v}$

**本质矩阵的性质**：
1. **内在约束**：$2E^T E E - \text{tr}(E^T E)E = 0$（Demazure约束）
2. **奇异值**：理想情况下为 $(s, s, 0)$，实际中需要强制此约束
3. **尺度不确定性**：只能恢复平移方向，不能恢复尺度

**SVD分解恢复位姿**：

对 $E$ 进行SVD分解：$E = U\Sigma V^T$

理想情况下 $\Sigma = \text{diag}(s, s, 0)$。强制此约束：
$$E' = U\text{diag}(1, 1, 0)V^T$$

四组可能的解：

$$\begin{aligned}
R_1 &= UWV^T, & \mathbf{t}_1 &= U(:,3) \\
R_2 &= UWV^T, & \mathbf{t}_2 &= -U(:,3) \\
R_3 &= UW^TV^T, & \mathbf{t}_3 &= U(:,3) \\
R_4 &= UW^TV^T, & \mathbf{t}_4 &= -U(:,3)
\end{aligned}$$

其中 $W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ 是旋转90度的矩阵。

**消歧方法**：通过三角化测试点，正确的解应满足：
1. 点在两个相机前方（正深度）
2. 重建点的视差角合理（不接近0或180度）

**Chirality检查**：对于三角化的点 $\mathbf{X}$：
- 相机1中的深度：$Z_1 = \mathbf{r}_3^{(1)} \cdot \mathbf{X} + t_z^{(1)} > 0$
- 相机2中的深度：$Z_2 = \mathbf{r}_3^{(2)} \cdot \mathbf{X} + t_z^{(2)} > 0$

其中 $\mathbf{r}_3$ 是旋转矩阵的第三行。

### 14.1.4 八点算法

八点算法是估计基础矩阵的经典方法。给定至少8对匹配点 $\{\mathbf{x}_i \leftrightarrow \mathbf{x}_i'\}$，利用对极约束：

$$\mathbf{x}_i'^T F \mathbf{x}_i = 0$$

展开为线性形式：
$$[x_i'x_i, x_i'y_i, x_i', y_i'x_i, y_i'y_i, y_i', x_i, y_i, 1] \cdot \text{vec}(F) = 0$$

构造约束矩阵 $A$：

$$A = \begin{bmatrix}
x_1'x_1 & x_1'y_1 & x_1' & y_1'x_1 & y_1'y_1 & y_1' & x_1 & y_1 & 1 \\
x_2'x_2 & x_2'y_2 & x_2' & y_2'x_2 & y_2'y_2 & y_2' & x_2 & y_2 & 1 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
x_n'x_n & x_n'y_n & x_n' & y_n'x_n & y_n'y_n & y_n' & x_n & y_n & 1
\end{bmatrix}$$

**求解步骤**：

1. **最小二乘求解**：求解 $\min_{\mathbf{f}} \|A\mathbf{f}\|^2$ subject to $\|\mathbf{f}\| = 1$
   - 解是 $A^T A$ 的最小特征值对应的特征向量
   - 或等价地，$A$ 的最小奇异值对应的右奇异向量

2. **秩约束强制**：得到的 $F$ 通常满秩，需要强制秩为2
   - SVD分解：$F = U\Sigma V^T$，其中 $\Sigma = \text{diag}(\sigma_1, \sigma_2, \sigma_3)$
   - 强制秩2：$F' = U\text{diag}(\sigma_1, \sigma_2, 0)V^T$

**算法的问题**：
- 对噪声敏感（条件数高）
- 数值不稳定（坐标范围大）
- 未考虑统计最优性

### 14.1.5 归一化八点算法

Hartley归一化是八点算法的关键改进，将条件数从 $10^8$ 降至 $10^2$ 量级。

**归一化变换**：

1. **平移到质心**：
   $$\bar{\mathbf{x}} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i = [\bar{x}, \bar{y}, 1]^T$$

2. **各向同性缩放**：
   $$s = \frac{\sqrt{2}}{\frac{1}{n}\sum_{i=1}^n \sqrt{(x_i-\bar{x})^2 + (y_i-\bar{y})^2}}$$
   
   使得平均距离为 $\sqrt{2}$（对角线长度）

3. **归一化矩阵**：
   $$T = \begin{bmatrix} 
   s & 0 & -s\bar{x} \\ 
   0 & s & -s\bar{y} \\ 
   0 & 0 & 1 
   \end{bmatrix}$$

**完整算法**：

1. 归一化点：$\hat{\mathbf{x}}_i = T_1\mathbf{x}_i$，$\hat{\mathbf{x}}_i' = T_2\mathbf{x}_i'$
2. 用八点算法计算 $\hat{F}$（归一化空间）
3. 反归一化：$F = T_2^{-T}\hat{F}T_1^{-1}$

**理论分析**：

归一化的效果可以通过条件数分析理解：
$$\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

归一化后，$A$ 的各列具有相似的范数，显著降低条件数。

**RANSAC鲁棒估计**：

实际应用中结合RANSAC处理外点：
1. 随机采样8对点
2. 计算基础矩阵 $F$
3. 计算内点：$d(\mathbf{x}_i', F\mathbf{x}_i) + d(\mathbf{x}_i, F^T\mathbf{x}_i') < \tau$
4. 重复并选择内点最多的模型
5. 用所有内点重新估计

其中 $d(\mathbf{x}, \mathbf{l})$ 是点到直线的距离：
$$d(\mathbf{x}, \mathbf{l}) = \frac{|l_1x + l_2y + l_3|}{\sqrt{l_1^2 + l_2^2}}$$

## 14.2 光束法平差与稀疏优化

### 14.2.1 重投影误差

光束法平差(Bundle Adjustment, BA)是多视图几何中的核心优化问题，同时优化相机参数和三维点坐标。目标是最小化重投影误差：

$$\min_{\{P_i\}, \{\mathbf{X}_j\}} \sum_{(i,j) \in \mathcal{V}} \rho(\|\mathbf{x}_{ij} - \pi(P_i, \mathbf{X}_j)\|^2_{\Sigma_{ij}})$$

其中：
- $P_i = \{R_i, \mathbf{t}_i, K_i\}$ 是第 $i$ 个相机的参数
- $\mathbf{X}_j$ 是第 $j$ 个三维点
- $\mathcal{V}$ 是可见性集合（哪些点在哪些相机中可见）
- $\pi$ 是投影函数
- $\Sigma_{ij}$ 是测量不确定性协方差
- $\rho$ 是鲁棒核函数

**常用鲁棒核函数**：

1. **Huber核**（对小误差二次，大误差线性）：
$$\rho_{\text{Huber}}(r) = \begin{cases}
\frac{1}{2}r^2 & \text{if } |r| \leq \delta \\
\delta(|r| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$$

2. **Cauchy核**（长尾分布）：
$$\rho_{\text{Cauchy}}(r) = \frac{\delta^2}{2}\log\left(1 + \frac{r^2}{\delta^2}\right)$$

3. **Tukey双权核**（完全拒绝大误差）：
$$\rho_{\text{Tukey}}(r) = \begin{cases}
\frac{\delta^2}{6}\left[1 - \left(1 - \frac{r^2}{\delta^2}\right)^3\right] & \text{if } |r| \leq \delta \\
\frac{\delta^2}{6} & \text{otherwise}
\end{cases}$$

**参数化选择**：

1. **旋转表示**：
   - 四元数：4参数，需要归一化约束
   - 轴角/Rodriguez：3参数，奇异性在 $\pi$ 
   - 李代数 $\mathfrak{so}(3)$：3参数，局部线性化好

2. **相机中心 vs 平移向量**：
   - 相机中心：$\mathbf{C} = -R^T\mathbf{t}$，数值稳定性更好
   - 平移向量：$\mathbf{t}$，与本质矩阵分解一致

**不确定性建模**：

考虑各向异性噪声，马氏距离：
$$d_M^2 = (\mathbf{x} - \hat{\mathbf{x}})^T\Sigma^{-1}(\mathbf{x} - \hat{\mathbf{x}})$$

其中 $\Sigma$ 可以从特征检测的Hessian矩阵估计。

### 14.2.2 雅可比矩阵结构

BA的雅可比矩阵具有特殊的稀疏块结构，这是高效求解的关键：

$$J = \begin{bmatrix}
J_{c_1} & J_{p_1} \\
J_{c_2} & J_{p_2} \\
\vdots & \vdots \\
J_{c_m} & J_{p_m}
\end{bmatrix} = \begin{bmatrix}
\frac{\partial \mathbf{r}_{11}}{\partial P_1} & 0 & \cdots & \frac{\partial \mathbf{r}_{11}}{\partial \mathbf{X}_1} & 0 & \cdots \\
0 & \frac{\partial \mathbf{r}_{22}}{\partial P_2} & \cdots & 0 & \frac{\partial \mathbf{r}_{22}}{\partial \mathbf{X}_2} & \cdots \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots
\end{bmatrix}$$

**李代数参数化的雅可比**：

使用李代数 $\mathfrak{se}(3)$ 参数化避免过参数化：

$$\boldsymbol{\xi} = \begin{bmatrix} \boldsymbol{\phi} \\ \boldsymbol{\rho} \end{bmatrix} \in \mathbb{R}^6$$

其中 $\boldsymbol{\phi} \in \mathfrak{so}(3)$ 是旋转，$\boldsymbol{\rho}$ 是平移。

扰动模型：$T(\boldsymbol{\xi} + \delta\boldsymbol{\xi}) = T(\boldsymbol{\xi})\exp(\delta\boldsymbol{\xi}^{\wedge})$

**链式法则计算**：

$$\frac{\partial \mathbf{r}}{\partial \boldsymbol{\xi}} = \frac{\partial \mathbf{r}}{\partial \mathbf{u}} \frac{\partial \mathbf{u}}{\partial \mathbf{p}} \frac{\partial \mathbf{p}}{\partial \boldsymbol{\xi}}$$

其中：
1. 残差对像素：$\frac{\partial \mathbf{r}}{\partial \mathbf{u}} = -I_{2\times 2}$
2. 像素对3D点（相机坐标）：
   $$\frac{\partial \mathbf{u}}{\partial \mathbf{p}} = \begin{bmatrix}
   \frac{f_x}{Z} & 0 & -\frac{f_xX}{Z^2} \\
   0 & \frac{f_y}{Z} & -\frac{f_yY}{Z^2}
   \end{bmatrix}$$
3. 3D点对李代数：
   $$\frac{\partial \mathbf{p}}{\partial \boldsymbol{\xi}} = \begin{bmatrix}
   I_{3\times 3} & -[\mathbf{p}]_\times
   \end{bmatrix}$$

**稀疏性分析**：

设有 $n$ 个相机，$m$ 个3D点，平均每个点被 $k$ 个相机观察：
- 非零元素数：$O(km)$
- 总元素数：$O(nm)$
- 稀疏度：$\frac{k}{n}$（通常 $k \ll n$）

### 14.2.3 Schur补优化

Schur补是BA加速的核心技术，利用 $H_{pp}$ 的块对角结构将大规模问题分解。

**正规方程的块结构**：

高斯-牛顿法的正规方程：$H\Delta \mathbf{x} = -\mathbf{g}$

$$\begin{bmatrix}
H_{cc} & H_{cp} \\
H_{pc} & H_{pp}
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{c} \\
\Delta \mathbf{p}
\end{bmatrix}
= 
-\begin{bmatrix}
\mathbf{g}_c \\
\mathbf{g}_p
\end{bmatrix}$$

其中：
- $H_{cc}$：相机-相机块，维度 $6n \times 6n$（稠密）
- $H_{pp}$：点-点块，维度 $3m \times 3m$（块对角）
- $H_{cp} = H_{pc}^T$：相机-点块（稀疏）

**Schur补消元**：

从第二行解出：$\Delta \mathbf{p} = -H_{pp}^{-1}(\mathbf{g}_p + H_{pc}\Delta \mathbf{c})$

代入第一行：
$$(H_{cc} - H_{cp}H_{pp}^{-1}H_{pc})\Delta \mathbf{c} = -(\mathbf{g}_c - H_{cp}H_{pp}^{-1}\mathbf{g}_p)$$

定义减少的系统：
$$S\Delta \mathbf{c} = \mathbf{v}$$

其中 $S = H_{cc} - H_{cp}H_{pp}^{-1}H_{pc}$ 是Schur补。

**计算复杂度分析**：

- 直接求解：$O((6n + 3m)^3)$
- Schur补：
  - 计算 $S$：$O(n^2m)$（利用块对角性）
  - 求解 $S\Delta \mathbf{c}$：$O(n^3)$
  - 回代求 $\Delta \mathbf{p}$：$O(nm)$
- 总复杂度：$O(n^2m + n^3)$，当 $m \gg n$ 时显著优于直接法

**实现细节**：

1. $H_{pp}^{-1}$ 不需要显式计算，只需逐块求逆：
   $$H_{pp} = \text{diag}(H_{pp}^{(1)}, ..., H_{pp}^{(m)})$$
   每块 $H_{pp}^{(j)}$ 是 $3\times 3$ 矩阵

2. 利用对称性只计算上三角部分

3. 预条件子选择：块Jacobi或不完全Cholesky

### 14.2.4 Levenberg-Marquardt算法

LM算法是BA的标准求解器，通过阻尼因子在高斯-牛顿法和梯度下降之间自适应切换。

**阻尼策略**：

修改的正规方程：
$$(H + \lambda D)\Delta \mathbf{x} = -\mathbf{g}$$

其中 $D$ 是阻尼矩阵：
- $D = I$：标准LM
- $D = \text{diag}(H)$：Marquardt变体（尺度不变）

**信赖域解释**：

LM等价于信赖域子问题：
$$\min_{\Delta \mathbf{x}} \frac{1}{2}\|\mathbf{r} + J\Delta \mathbf{x}\|^2 \quad \text{s.t.} \quad \|\Delta \mathbf{x}\| \leq \Delta$$

其中 $\lambda$ 是信赖域半径 $\Delta$ 的拉格朗日乘子。

**自适应更新**：

增益比：
$$\rho = \frac{f(\mathbf{x}) - f(\mathbf{x} + \Delta \mathbf{x})}{L(0) - L(\Delta \mathbf{x})} = \frac{\text{实际下降}}{\text{预测下降}}$$

其中 $L(\Delta \mathbf{x}) = f(\mathbf{x}) + \mathbf{g}^T\Delta \mathbf{x} + \frac{1}{2}\Delta \mathbf{x}^T H \Delta \mathbf{x}$

更新规则：
- 若 $\rho > 0.75$：$\lambda \leftarrow \lambda/3$（扩大信赖域）
- 若 $0.25 < \rho < 0.75$：$\lambda$ 不变
- 若 $\rho < 0.25$：$\lambda \leftarrow 2\lambda$（缩小信赖域）
- 若 $\rho < 0$：拒绝步长，$\lambda \leftarrow 4\lambda$

**收敛准则**：

1. 梯度范数：$\|\mathbf{g}\| < \epsilon_g$
2. 步长：$\|\Delta \mathbf{x}\| < \epsilon_x(\|\mathbf{x}\| + \epsilon_x)$
3. 函数值变化：$|f_{k+1} - f_k| < \epsilon_f(|f_k| + \epsilon_f)$

### 14.2.5 边缘化与滑动窗口

在SLAM等实时应用中，需要维持有界计算复杂度。边缘化技术保留历史信息的同时控制优化变量数。

**信息矩阵形式**：

贝叶斯框架下，信息矩阵 $\Lambda = \Sigma^{-1}$：

$$\begin{bmatrix}
\Lambda_{mm} & \Lambda_{mr} \\
\Lambda_{rm} & \Lambda_{rr}
\end{bmatrix}
\begin{bmatrix}
\mathbf{x}_m \\
\mathbf{x}_r
\end{bmatrix}
= 
\begin{bmatrix}
\boldsymbol{\eta}_m \\
\boldsymbol{\eta}_r
\end{bmatrix}$$

其中 $\mathbf{x}_m$ 是要边缘化的变量，$\mathbf{x}_r$ 是保留的变量。

**边缘化过程**：

1. **Schur补计算**：
   $$\Lambda' = \Lambda_{rr} - \Lambda_{rm}\Lambda_{mm}^{-1}\Lambda_{mr}$$
   $$\boldsymbol{\eta}' = \boldsymbol{\eta}_r - \Lambda_{rm}\Lambda_{mm}^{-1}\boldsymbol{\eta}_m$$

2. **先验因子**：边缘化产生的 $(\Lambda', \boldsymbol{\eta}')$ 作为保留变量的先验

3. **线性化点**：需要固定边缘化时的线性化点，避免不一致性

**滑动窗口策略**：

1. **关键帧窗口**：保持最近 $N$ 个关键帧
2. **边缘化策略**：
   - 边缘化最老的关键帧及其观测的仅被该帧观测的地图点
   - 保留共视地图点，更新其先验

**FEJ (First-Estimate Jacobian)**：

为保持一致性，在边缘化点固定雅可比：
$$J(\mathbf{x}_0) \text{ instead of } J(\mathbf{x}_k)$$

这避免了由于线性化点变化导致的信息矩阵秩增加。

## 14.3 SLAM：ORB-SLAM、LSD-SLAM

### 14.3.1 特征点SLAM：ORB-SLAM框架

ORB-SLAM使用ORB特征进行跟踪和建图。特征描述子的汉明距离：

$$d_H(\mathbf{d}_1, \mathbf{d}_2) = \sum_{i=1}^{256} \mathbf{d}_1[i] \oplus \mathbf{d}_2[i]$$

**词袋模型加速**：使用DBoW2进行快速位置识别，通过TF-IDF权重计算相似度：

$$s(\mathbf{v}_1, \mathbf{v}_2) = 1 - \frac{1}{2}\left|\frac{\mathbf{v}_1}{|\mathbf{v}_1|} - \frac{\mathbf{v}_2}{|\mathbf{v}_2|}\right|$$

**本质图优化**：保留关键帧之间的共视关系、生成树和回环边，构成本质图进行全局优化。

### 14.3.2 直接法SLAM：LSD-SLAM

LSD-SLAM直接最小化光度误差：

$$E(\boldsymbol{\xi}) = \sum_{\mathbf{p} \in \Omega} \|I_{\text{ref}}(\mathbf{p}) - I_{\text{cur}}(\pi(\mathbf{p}, d(\mathbf{p}), \boldsymbol{\xi}))\|_\delta$$

其中 $\|\cdot\|_\delta$ 是Huber范数，$d(\mathbf{p})$ 是逆深度。

**逆深度参数化**：使用逆深度 $\rho = 1/Z$ 改善数值稳定性，特别是对远处点：

$$\mathbf{x}' = K\exp(\boldsymbol{\xi}^{\wedge})(K^{-1}\mathbf{x}/\rho + \mathbf{t}_0)$$

**光度误差雅可比**：
$$\frac{\partial r}{\partial \boldsymbol{\xi}} = -\nabla I_{\text{cur}} \frac{\partial \pi}{\partial \boldsymbol{\xi}}$$

其中图像梯度 $\nabla I$ 使用Sobel算子计算。

### 14.3.3 半稠密建图

LSD-SLAM维护半稠密深度图，只在梯度较大的像素估计深度。深度滤波使用贝叶斯框架：

$$p(d|\mathcal{Z}) \propto p(\mathcal{Z}|d)p(d)$$

假设深度服从逆深度的高斯分布：
$$p(\rho) = \mathcal{N}(\mu_\rho, \sigma_\rho^2)$$

深度融合更新：
$$\sigma_\rho^{-2} = \sigma_{\rho,\text{old}}^{-2} + \sigma_{\rho,\text{obs}}^{-2}$$
$$\mu_\rho = \sigma_\rho^2(\sigma_{\rho,\text{old}}^{-2}\mu_{\rho,\text{old}} + \sigma_{\rho,\text{obs}}^{-2}\mu_{\rho,\text{obs}})$$

### 14.3.4 回环检测与位姿图优化

回环检测后的位姿图优化：

$$\min_{\{\mathbf{T}_i\}} \sum_{(i,j) \in \mathcal{E}} \|\log(\mathbf{T}_{ij}^{-1}\mathbf{T}_i^{-1}\mathbf{T}_j)\|_{\Sigma_{ij}}^2$$

其中 $\|\cdot\|_{\Sigma}^2$ 是马氏距离，$\log$ 是SE(3)的对数映射。

使用g2o或Ceres求解，利用流形优化避免过参数化。

## 14.4 稠密重建：PMVS、COLMAP

### 14.4.1 PMVS：基于面片的多视图立体

PMVS将场景表示为定向面片集合。每个面片 $p$ 包含：
- 中心位置 $\mathbf{c}(p)$
- 法向量 $\mathbf{n}(p)$  
- 参考图像集 $\mathcal{V}(p)$

**光度一致性度量**：

$$\text{NCC}(p, I_i, I_j) = \frac{\sum_{\mathbf{q} \in \mathcal{N}(p)} (I_i(\mathbf{q}) - \bar{I}_i)(I_j(\pi_{ij}(\mathbf{q})) - \bar{I}_j)}{\sqrt{\sum(I_i(\mathbf{q}) - \bar{I}_i)^2 \sum(I_j(\pi_{ij}(\mathbf{q})) - \bar{I}_j)^2}}$$

**面片优化**：最大化多视图一致性：

$$\mathbf{c}^*, \mathbf{n}^* = \arg\max_{\mathbf{c}, \mathbf{n}} \frac{1}{|\mathcal{V}|}\sum_{i,j \in \mathcal{V}} \text{NCC}(p(\mathbf{c}, \mathbf{n}), I_i, I_j)$$

使用共轭梯度法优化，法向量在单位球面上参数化。

### 14.4.2 COLMAP：鲁棒的稠密重建

COLMAP使用PatchMatch立体算法进行稠密匹配。

**随机初始化**：为每个像素随机采样深度和法向：
$$d \sim \mathcal{U}[d_{\min}, d_{\max}]$$
$$\mathbf{n} \sim \mathcal{S}^2$$

**传播**：测试邻域的深度/法向假设：
$$\mathcal{C}(p, d_p, \mathbf{n}_p) < \mathcal{C}(p, d_q, \mathbf{n}_q) \Rightarrow (d_p, \mathbf{n}_p) \leftarrow (d_q, \mathbf{n}_q)$$

**随机搜索**：在当前估计附近随机扰动：
$$d' = d + \Delta d \cdot \epsilon^i, \quad i = 0, 1, 2, ...$$

其中 $\epsilon < 1$ 控制搜索范围的指数衰减。

### 14.4.3 深度图融合

**几何一致性检查**：

$$|d_i - d_j(\mathbf{x}_j)| < \tau_d \cdot d_i$$
$$\mathbf{n}_i^T \mathbf{n}_j > \tau_n$$

其中 $\mathbf{x}_j$ 是 $\mathbf{x}_i$ 在视图 $j$ 中的对应点。

**截断符号距离函数(TSDF)融合**：

$$\text{TSDF}(\mathbf{x}) = \min\left(\max\left(\frac{d_{\text{proj}} - d_{\text{ray}}}{\tau}, -1\right), 1\right)$$

权重更新：
$$W(\mathbf{x}) \leftarrow W(\mathbf{x}) + w_i$$
$$D(\mathbf{x}) \leftarrow \frac{W_{\text{old}} \cdot D_{\text{old}} + w_i \cdot d_i}{W_{\text{old}} + w_i}$$

### 14.4.4 网格提取

使用Marching Cubes从TSDF提取等值面：

$$\text{TSDF}(\mathbf{x}) = 0$$

顶点位置通过线性插值确定：
$$\mathbf{v} = \mathbf{v}_1 + \frac{|\text{TSDF}(\mathbf{v}_1)|}{|\text{TSDF}(\mathbf{v}_1)| + |\text{TSDF}(\mathbf{v}_2)|} (\mathbf{v}_2 - \mathbf{v}_1)$$

## 14.5 深度学习MVS：MVSNet系列

### 14.5.1 MVSNet：端到端深度推断

MVSNet构建代价体进行深度推断。

**可微单应变换**：将参考图像特征扭曲到其他视图：

$$H_i(d) = K_i \cdot (R_i - \frac{\mathbf{t}_i \mathbf{n}_{\text{ref}}^T}{d}) \cdot K_{\text{ref}}^{-1}$$

**代价体构建**：
$$\mathcal{V}(d) = \text{Var}(\{F_i(H_i(d))\}_{i=1}^N)$$

其中 $F_i$ 是从图像 $i$ 提取的特征。

**3D CNN正则化**：使用U-Net结构的3D卷积网络：

$$\mathcal{P} = \text{SoftMax}(\text{3DCNN}(\mathcal{V}))$$

**深度回归**：
$$d = \sum_{i=1}^D d_i \cdot \mathcal{P}(d_i)$$

### 14.5.2 R-MVSNet：循环优化

R-MVSNet使用GRU进行迭代优化：

$$h_t = \text{GRU}(h_{t-1}, \mathcal{V}_t)$$
$$\Delta d_t = \text{MLP}(h_t)$$
$$d_t = d_{t-1} + \Delta d_t$$

这允许在内存受限情况下处理高分辨率图像。

### 14.5.3 Cascade-MVSNet：级联细化

多尺度级联结构：

$$d^{(l+1)} = d^{(l)} + \Delta d^{(l+1)}$$

其中 $\Delta d^{(l+1)}$ 在尺度 $l+1$ 的局部范围内搜索：

$$d \in [d^{(l)} - \sigma^{(l)}, d^{(l)} + \sigma^{(l)}]$$

不确定性估计用于自适应调整搜索范围：

$$\sigma^2 = \sum_{i=1}^D (d_i - \bar{d})^2 \cdot \mathcal{P}(d_i)$$

### 14.5.4 Vis-MVSNet：可见性感知

引入可见性权重处理遮挡：

$$w_i = \exp\left(-\alpha \cdot \max_{j \neq i} \text{NCC}(I_{\text{ref}}, I_j)\right)$$

加权代价聚合：
$$\mathcal{C} = \sum_{i=1}^N w_i \cdot \|F_{\text{ref}} - F_i\|^2$$

### 14.5.5 训练策略与损失函数

**深度损失**：
$$\mathcal{L}_d = \sum_{\mathbf{p} \in \mathcal{M}} |d(\mathbf{p}) - d^*(\mathbf{p})|$$

其中 $\mathcal{M}$ 是有效深度掩码。

**多尺度监督**：
$$\mathcal{L} = \sum_{l=1}^L \lambda_l \mathcal{L}_d^{(l)}$$

**难例挖掘**：
$$\mathcal{L}_{\text{focal}} = -(1 - p_t)^\gamma \log(p_t)$$

其中 $p_t$ 是正确深度假设的概率。

## 本章小结

本章系统介绍了多视图几何的核心理论和实践方法：

1. **对极几何基础**：推导了基础矩阵和本质矩阵的数学性质，掌握了八点算法及其归一化改进
2. **光束法平差**：深入理解了大规模非线性优化的稀疏结构利用，包括Schur补和LM算法
3. **SLAM系统**：对比了特征点方法(ORB-SLAM)和直接法(LSD-SLAM)的数学框架
4. **稠密重建**：学习了从面片扩展(PMVS)到PatchMatch(COLMAP)的稠密匹配算法
5. **深度学习方法**：掌握了MVSNet系列的可微几何变换和代价体学习

关键数学工具：
- 李群/李代数用于旋转表示
- 鲁棒估计理论处理外点
- 贝叶斯滤波进行深度融合
- 3D卷积网络进行代价正则化

## 练习题

### 基础题

**练习14.1** 证明基础矩阵的秩为2，并说明为什么这个约束在实际计算中很重要。

<details>
<summary>提示</summary>

考虑基础矩阵与本质矩阵的关系 $E = K_2^T F K_1$，以及本质矩阵的构造 $E = [\mathbf{t}]_\times R$。

</details>

<details>
<summary>答案</summary>

由于 $E = [\mathbf{t}]_\times R$，其中 $[\mathbf{t}]_\times$ 是反对称矩阵，其秩为2（一个特征值为0）。旋转矩阵 $R$ 是满秩的，因此 $\text{rank}(E) = \text{rank}([\mathbf{t}]_\times) = 2$。

由于 $F = K_2^{-T} E K_1^{-1}$，且 $K_1, K_2$ 都是可逆的，因此 $\text{rank}(F) = \text{rank}(E) = 2$。

这个约束重要因为：
1. 减少了自由度从9到7
2. 保证了对极线束的存在
3. 在八点算法后需要强制秩约束以提高精度

</details>

**练习14.2** 推导针孔相机模型下，图像点 $\mathbf{x} = [u, v, 1]^T$ 对相机位姿参数 $\boldsymbol{\xi} \in \mathfrak{se}(3)$ 的雅可比矩阵。

<details>
<summary>提示</summary>

使用链式法则：$\frac{\partial \mathbf{x}}{\partial \boldsymbol{\xi}} = \frac{\partial \mathbf{x}}{\partial \mathbf{X}_c} \frac{\partial \mathbf{X}_c}{\partial \boldsymbol{\xi}}$，其中 $\mathbf{X}_c$ 是相机坐标系下的3D点。

</details>

<details>
<summary>答案</summary>

设 $\mathbf{X}_c = [X_c, Y_c, Z_c]^T = \exp(\boldsymbol{\xi}^{\wedge})\mathbf{X}_w$。

投影函数：$u = f_x \frac{X_c}{Z_c} + c_x$，$v = f_y \frac{Y_c}{Z_c} + c_y$

第一部分：
$$\frac{\partial \mathbf{x}}{\partial \mathbf{X}_c} = \begin{bmatrix}
\frac{f_x}{Z_c} & 0 & -\frac{f_x X_c}{Z_c^2} \\
0 & \frac{f_y}{Z_c} & -\frac{f_y Y_c}{Z_c^2}
\end{bmatrix}$$

第二部分（使用左扰动模型）：
$$\frac{\partial \mathbf{X}_c}{\partial \boldsymbol{\xi}} = [-[\mathbf{X}_c]_\times, \mathbf{I}_{3\times 3}]$$

最终雅可比：
$$\frac{\partial \mathbf{x}}{\partial \boldsymbol{\xi}} = \begin{bmatrix}
\frac{f_x}{Z_c} & 0 & -\frac{f_x X_c}{Z_c^2} \\
0 & \frac{f_y}{Z_c} & -\frac{f_y Y_c}{Z_c^2}
\end{bmatrix} \begin{bmatrix}
0 & Z_c & -Y_c & 1 & 0 & 0 \\
-Z_c & 0 & X_c & 0 & 1 & 0 \\
Y_c & -X_c & 0 & 0 & 0 & 1
\end{bmatrix}$$

</details>

**练习14.3** 给定5个相机和100个3D点的BA问题，计算正规方程矩阵的稀疏结构和非零元素数量。假设每个点平均被3个相机观察到。

<details>
<summary>提示</summary>

考虑相机参数块(6×5=30维)和点参数块(3×100=300维)的结构。

</details>

<details>
<summary>答案</summary>

雅可比矩阵维度：(2×3×100) × (30+300) = 600 × 330

正规方程 $H = J^TJ$ 维度：330 × 330

块结构：
- $H_{cc}$：30×30（相机-相机块）
- $H_{cp}$：30×300（相机-点块）  
- $H_{pc}$：300×30（点-相机块）
- $H_{pp}$：300×300（点-点块，块对角）

非零元素：
- $H_{cc}$：全部非零 = 900
- $H_{pp}$：100个3×3块 = 900
- $H_{cp}$ 和 $H_{pc}$：每个点连接3个相机，约 30×3×3 + 3×100×6 = 2070

总计约3870个非零元素，稀疏度 = 3870/(330×330) ≈ 3.6%

</details>

### 挑战题

**练习14.4** 设计一个算法，从单应性矩阵 $H$ 分解出旋转 $R$、平移 $\mathbf{t}$ 和平面法向量 $\mathbf{n}$。处理多解问题。

<details>
<summary>提示</summary>

单应性分解：$H = K_2(R + \mathbf{t}\mathbf{n}^T/d)K_1^{-1}$。使用SVD分解并利用物理约束消歧。

</details>

<details>
<summary>答案</summary>

算法步骤：

1. 计算 $H' = K_2^{-1}HK_1 = R + \mathbf{t}\mathbf{n}^T/d$

2. SVD分解：$H' = U\Sigma V^T$

3. 构造中间矩阵：
   $$S = UU^T H' V^TV^T = U\Sigma V^T$$

4. 令 $\lambda_1, \lambda_2, \lambda_3$ 为 $S$ 的奇异值，计算：
   $$d_1 = \sqrt{\lambda_1^2 - \lambda_3^2}$$
   $$d_2 = \sqrt{\lambda_2^2 - \lambda_3^2}$$

5. 四组可能解：
   - $R = U\text{diag}(1,1,1)V^T \pm \frac{2d_1d_2}{\lambda_1+\lambda_2}U\mathbf{w}\mathbf{v}^T$
   - $\mathbf{t} = \pm(U\mathbf{u}_3)$
   - $\mathbf{n} = \pm(V\mathbf{v}_3)$

6. 消歧：使用重建点的正深度约束，即 $Z > 0$ 在两个相机中。

</details>

**练习14.5** 推导并实现逆深度参数化的EKF-SLAM更新方程，分析其相比于XYZ参数化的优势。

<details>
<summary>提示</summary>

逆深度参数：$\mathbf{y} = [u_0, v_0, \rho, \theta, \phi]^T$，其中 $\rho = 1/d$。

</details>

<details>
<summary>答案</summary>

状态向量：$\mathbf{x} = [\mathbf{x}_c, \mathbf{y}_1, ..., \mathbf{y}_n]$

3D点表示：
$$\mathbf{X} = \mathbf{r}_0 + \frac{1}{\rho}\mathbf{m}(\theta, \phi)$$

其中 $\mathbf{m} = [\cos\phi\sin\theta, \sin\phi, \cos\phi\cos\theta]^T$

观测模型雅可比：
$$H = \frac{\partial h}{\partial \mathbf{y}} = \frac{\partial \pi}{\partial \mathbf{X}} \frac{\partial \mathbf{X}}{\partial \mathbf{y}}$$

其中：
$$\frac{\partial \mathbf{X}}{\partial \rho} = -\frac{1}{\rho^2}\mathbf{m}$$
$$\frac{\partial \mathbf{X}}{\partial \theta} = \frac{1}{\rho}\frac{\partial \mathbf{m}}{\partial \theta}$$

EKF更新：
$$K = P H^T(H P H^T + R)^{-1}$$
$$\mathbf{x}_{k+1} = \mathbf{x}_k + K(\mathbf{z} - h(\mathbf{x}_k))$$
$$P_{k+1} = (I - KH)P_k$$

优势：
1. 无穷远点可表示（$\rho \rightarrow 0$）
2. 线性化误差更小
3. 初始化不确定性更好建模（高斯分布在逆深度空间）

</details>

**练习14.6** 分析MVSNet中代价体的内存复杂度，提出一种内存高效的变体。

<details>
<summary>提示</summary>

考虑分块处理、深度假设采样策略、特征通道压缩等方法。

</details>

<details>
<summary>答案</summary>

标准MVSNet内存分析：
- 输入：$N$ 视图，分辨率 $H \times W$，$D$ 个深度假设，$F$ 维特征
- 代价体：$O(H \times W \times D \times F)$
- 典型值：$640 \times 480 \times 128 \times 32 \times 4\text{bytes} = 5GB$

内存优化方案：

1. **深度采样优化**：
   - 使用反比例采样：$d_i = \frac{1}{\frac{1}{d_{near}} + i\Delta}$
   - 级联策略：粗到细，每级减少深度范围

2. **特征压缩**：
   - PCA降维：$F' = 8$ 而非 32
   - 二进制特征：1-bit量化

3. **分块处理**：
   ```
   将图像分成 $K \times K$ 块
   for each block:
       构建局部代价体
       3D CNN正则化
       深度推断
   合并结果
   ```

4. **循环计算**（如R-MVSNet）：
   - 不存储完整代价体
   - 使用GRU迭代更新
   - 内存：$O(H \times W \times F)$

5. **混合精度**：
   - 特征：FP16
   - 代价体：INT8
   - 梯度：FP32

综合方案可将内存降至原来的1/20。

</details>

**练习14.7** 设计一个融合IMU的视觉SLAM系统，推导预积分公式和误差传播方程。

<details>
<summary>提示</summary>

IMU预积分避免重复积分，考虑bias的影响。

</details>

<details>
<summary>答案</summary>

IMU运动模型：
$$\dot{\mathbf{p}} = \mathbf{v}, \quad \dot{\mathbf{v}} = R(\mathbf{a}_m - \mathbf{b}_a - \mathbf{n}_a) - \mathbf{g}, \quad \dot{R} = R[\boldsymbol{\omega}_m - \mathbf{b}_g - \mathbf{n}_g]_\times$$

预积分量定义：
$$\Delta \tilde{R}_{ij} = \prod_{k=i}^{j-1} \exp((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)\Delta t)$$
$$\Delta \tilde{\mathbf{v}}_{ij} = \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t$$
$$\Delta \tilde{\mathbf{p}}_{ij} = \sum_{k=i}^{j-1} [\Delta \tilde{\mathbf{v}}_{ik}\Delta t + \frac{1}{2}\Delta \tilde{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t^2]$$

一阶近似（bias变化）：
$$\Delta R_{ij} \approx \Delta \tilde{R}_{ij} \exp(\frac{\partial \Delta R_{ij}}{\partial \mathbf{b}_g}\delta \mathbf{b}_g)$$
$$\Delta \mathbf{v}_{ij} \approx \Delta \tilde{\mathbf{v}}_{ij} + \frac{\partial \Delta \mathbf{v}_{ij}}{\partial \mathbf{b}_g}\delta \mathbf{b}_g + \frac{\partial \Delta \mathbf{v}_{ij}}{\partial \mathbf{b}_a}\delta \mathbf{b}_a$$

协方差传播：
$$P_{k+1} = F_k P_k F_k^T + G_k Q G_k^T$$

其中 $F$ 是状态转移矩阵，$G$ 是噪声矩阵，$Q$ 是过程噪声。

残差定义：
$$\mathbf{r}_{\Delta R} = \log(\Delta \tilde{R}_{ij}^T R_i^T R_j)$$
$$\mathbf{r}_{\Delta \mathbf{v}} = R_i^T(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) - \Delta \tilde{\mathbf{v}}_{ij}$$
$$\mathbf{r}_{\Delta \mathbf{p}} = R_i^T(\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) - \Delta \tilde{\mathbf{p}}_{ij}$$

优化目标：
$$\min \sum (\|\mathbf{r}_{\text{visual}}\|^2 + \|\mathbf{r}_{\text{IMU}}\|^2_{P^{-1}})$$

</details>

## 常见陷阱与错误

1. **数值稳定性问题**
   - 未归一化导致八点算法失败
   - 直接参数化深度而非逆深度
   - 忽略旋转的流形结构

2. **退化配置**
   - 纯旋转运动无法恢复平移尺度
   - 共面点导致单应性退化
   - 基线过小导致三角化不稳定

3. **优化陷阱**
   - BA未使用鲁棒核函数
   - Schur补实现错误
   - 过参数化旋转

4. **实时性问题**
   - 未利用稀疏性
   - 关键帧选择不当
   - 地图点过多未剔除

5. **深度学习特有**
   - 代价体分辨率过高OOM
   - 训练/测试分辨率不匹配
   - 未处理深度范围差异

## 最佳实践检查清单

### 系统设计
- [ ] 选择合适的参数化（四元数/李代数/轴角）
- [ ] 设计鲁棒的初始化流程
- [ ] 实现关键帧和地图点管理策略
- [ ] 考虑多传感器融合架构

### 数值优化
- [ ] 使用归一化和预处理
- [ ] 实现自适应鲁棒核函数
- [ ] 利用问题的稀疏结构
- [ ] 监控优化收敛性

### 深度估计
- [ ] 验证相机标定精度
- [ ] 处理遮挡和边界
- [ ] 多尺度一致性检查
- [ ] 置信度估计和滤波

### 工程实现
- [ ] 内存和计算复杂度分析
- [ ] 并行化关键算法
- [ ] 实现在线和离线模式
- [ ] 完整的评估pipeline
