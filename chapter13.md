# 第13章：多相流与传热

本章深入探讨3D打印过程中的多相流动与传热现象的数学建模。我们将从熔融沉积过程的连续介质力学描述开始，逐步深入到相变、表面张力效应、粉末床熔融等复杂物理过程的数学模型。这些理论不仅对理解打印过程至关重要，也为过程优化和质量控制提供了理论基础。

## 13.1 熔融沉积过程建模

### 13.1.1 流变学基础

熔融聚合物的流动特性通常表现为非牛顿流体行为。剪切速率依赖的粘度可用幂律模型描述：

$$\eta(\dot{\gamma}) = K\dot{\gamma}^{n-1}$$

其中$K$是稠度系数，$n$是流动指数。对于大多数热塑性聚合物，$n < 1$（剪切变稀）。

更精确的Cross-WLF模型同时考虑剪切速率和温度的影响：

$$\eta(\dot{\gamma}, T) = \frac{\eta_0(T)}{1 + (\eta_0(T)\dot{\gamma}/\tau^*)^{1-n}}$$

其中温度依赖性通过Williams-Landel-Ferry方程描述：

$$\log_{10}\left(\frac{\eta_0(T)}{\eta_0(T_r)}\right) = -\frac{C_1(T-T_r)}{C_2 + (T-T_r)}$$

### 13.1.2 挤出过程的流动方程

在喷嘴内的流动可简化为轴对称Poiseuille流。对于幂律流体，速度分布为：

$$v_z(r) = \frac{n}{n+1}\left(\frac{\Delta P}{2KL}\right)^{1/n}R^{(n+1)/n}\left[1-\left(\frac{r}{R}\right)^{(n+1)/n}\right]$$

体积流率：

$$Q = \frac{\pi n}{3n+1}\left(\frac{\Delta P R}{2KL}\right)^{1/n}R^3$$

### 13.1.3 自由表面流动

材料从喷嘴挤出后的自由膨胀（die swell）现象可通过Tanner-Lodge方程预测：

$$\frac{D_e}{D_0} = 1 + \frac{1}{8}\left(\frac{\tau_{11} - \tau_{22}}{\tau_{12}}\right)^2_{exit}$$

其中$D_e$是挤出后直径，$D_0$是喷嘴直径，$\tau_{ij}$是应力张量分量。

对于粘弹性流体，需要考虑Weissenberg数和Deborah数的影响：

$$Wi = \frac{\lambda \dot{\gamma}}{1} = \lambda \frac{4Q}{\pi R^3}$$

$$De = \frac{\lambda}{t_{res}} = \frac{\lambda v}{L}$$

其中$\lambda$是松弛时间，$t_{res}$是停留时间。当$Wi > 1$时，弹性效应显著，膨胀比可达2-3。

挤出物的表面不稳定性（鲨鱼皮效应）发生的临界条件：

$$\tau_w > \tau_{critical} \approx 0.1 \text{ MPa}$$

这种不稳定性的波长满足：

$$\lambda_{instability} \sim D_0 \left(\frac{\eta}{\rho v^2}\right)^{1/3}$$

### 13.1.4 沉积层形态

单条挤出线的横截面形状由表面张力和重力平衡决定。Young-Laplace方程给出：

$$\gamma\kappa = \Delta P$$

对于椭圆形截面，曲率$\kappa$的计算涉及两个主曲率半径：

$$\kappa = \kappa_1 + \kappa_2 = \frac{1}{R_1} + \frac{1}{R_2}$$

稳态下的截面形状可通过最小化总能量得到：

$$E = \int_{\Omega} \rho g z \, dA + \gamma \oint_{\partial\Omega} ds$$

对于典型的FDM工艺参数，截面形状可近似为：

$$\frac{(x/a)^{2+\epsilon}}{1} + \frac{(y/b)^{2-\epsilon}}{1} = 1$$

其中$\epsilon = Bo^{1/3}$，Bond数$Bo = \rho g h^2/\gamma$表征重力与表面张力的相对重要性。

层间结合强度依赖于温度历史，可通过爬行理论（reptation theory）建模：

$$\sigma_{weld}(t) = \sigma_{\infty} \left[1 - \exp\left(-\left(\frac{t}{\tau_r(T)}\right)^{1/4}\right)\right]$$

其中爬行时间$\tau_r \propto M_w^{3.4}\exp(E_a/RT)$，$M_w$是分子量。

层间空隙率的预测模型：

$$\phi_{void} = 1 - \frac{\pi}{4}\frac{w \cdot h}{(w + g)(h + \Delta z)}$$

其中$w$是线宽，$h$是层高，$g$是线间距，$\Delta z$是层间距。

## 13.2 Stefan问题与相变

### 13.2.1 经典Stefan问题

一维Stefan问题描述固液相界面的移动：

$$\frac{\partial T_s}{\partial t} = \alpha_s\frac{\partial^2 T_s}{\partial x^2}, \quad x < s(t)$$

$$\frac{\partial T_l}{\partial t} = \alpha_l\frac{\partial^2 T_l}{\partial x^2}, \quad x > s(t)$$

界面条件（Stefan条件）：

$$\rho L_f\frac{ds}{dt} = k_s\frac{\partial T_s}{\partial x}\bigg|_{x=s^-} - k_l\frac{\partial T_l}{\partial x}\bigg|_{x=s^+}$$

### 13.2.2 焓方法

为避免显式追踪相界面，采用焓-孔隙度方法。定义总焓：

$$H = h + \Delta H$$

其中显焓$h = \int_0^T c_p dT$，潜热项$\Delta H = \beta L_f$，液相分数$\beta$满足：

$$\beta = \begin{cases}
0 & T < T_s \\
\frac{T - T_s}{T_l - T_s} & T_s \leq T \leq T_l \\
1 & T > T_l
\end{cases}$$

能量方程变为：

$$\frac{\partial H}{\partial t} + \nabla \cdot (vH) = \nabla \cdot (k\nabla T) + S$$

动量方程中的糊状区阻力采用Carman-Kozeny模型：

$$S_{momentum} = -\frac{C(1-\beta)^2}{\beta^3 + \epsilon} v$$

其中$C = 1.6 \times 10^6 \text{ kg/(m}^3\text{s)}$是糊状区常数，$\epsilon = 10^{-3}$防止除零。

迭代求解算法：
1. 由$H^n$计算$T^{n+1}$：采用Newton-Raphson迭代
2. 更新液相分数$\beta^{n+1}(T^{n+1})$
3. 求解动量方程得到$v^{n+1}$
4. 求解能量方程得到$H^{n+1}$

收敛准则：
$$\|H^{n+1} - H^n\|_2 < \epsilon_H \quad \text{且} \quad \|\beta^{n+1} - \beta^n\|_\infty < \epsilon_\beta$$

### 13.2.3 相场方法

相场模型通过序参数$\phi$光滑地描述相变：

$$\frac{\partial \phi}{\partial t} = M\nabla^2\left(\frac{\partial f}{\partial \phi} - \epsilon^2\nabla^2\phi\right)$$

其中$f(\phi, T)$是自由能密度，$\epsilon$控制界面厚度。耦合的温度场方程：

$$c_p\frac{\partial T}{\partial t} = \nabla \cdot (k\nabla T) + L_f\frac{\partial \phi}{\partial t}$$

### 13.2.4 各向异性凝固

晶体生长的各向异性通过修正的Gibbs-Thomson条件描述：

$$T_{interface} = T_m - \Gamma(\theta)\kappa - v_n/\mu_k$$

其中$\Gamma(\theta) = \gamma(\theta) + \gamma''(\theta)$是各向异性毛细长度，$\mu_k$是动力学系数。

表面能的各向异性可表示为：

$$\gamma(\theta) = \gamma_0[1 + \epsilon_4\cos(4\theta) + \epsilon_6\cos(6\theta)]$$

其中$\theta$是界面法向与晶体主轴的夹角，$\epsilon_4$、$\epsilon_6$是各向异性强度。

枝晶生长的选择准则（边缘稳定性理论）：

$$\sigma^* = \frac{2d_0 D}{R^2 v} = \frac{1}{4\pi^2}$$

其中$d_0 = \Gamma T_m c_p/L_f^2$是毛细长度，$R$是枝晶尖端半径。

枝晶间距的一次臂间距$\lambda_1$遵循：

$$\lambda_1 = A \cdot G^{-0.5} \cdot v^{-0.25}$$

其中$G$是温度梯度，$v$是凝固速度，$A$是与合金成分相关的常数。

二次臂间距的粗化遵循LSW理论：

$$\lambda_2^3 - \lambda_{2,0}^3 = \frac{8\gamma D \Gamma}{9(1-k)^2 m c_0} t$$

其中$k$是平衡分配系数，$m$是液相线斜率。

## 13.3 Marangoni效应与表面张力

### 13.3.1 热毛细流动

温度梯度引起的表面张力梯度驱动Marangoni流动：

$$\tau_s = \frac{\partial \gamma}{\partial T}\nabla_s T$$

其中$\nabla_s = (I - nn^T)\nabla$是表面梯度算子。Marangoni数：

$$Ma = \frac{|\partial \gamma/\partial T|\Delta T L}{\mu \alpha}$$

### 13.3.2 熔池动力学

激光或电子束加工中的熔池表面变形由以下方程组控制：

$$\rho\left(\frac{\partial v}{\partial t} + v \cdot \nabla v\right) = -\nabla p + \mu\nabla^2 v + \rho g + F_{buoy}$$

自由表面边界条件：

$$p - p_0 = \gamma\kappa + p_{recoil}$$

$$\mu\frac{\partial v_t}{\partial n} = \frac{\partial \gamma}{\partial T}\frac{\partial T}{\partial t}$$

浮力项采用Boussinesq近似：

$$F_{buoy} = \rho g \beta_T (T - T_{ref})$$

熔池特征速度的尺度分析：

$$v_{Marangoni} \sim \frac{|\partial\gamma/\partial T| \Delta T}{\mu}$$

$$v_{buoyancy} \sim \sqrt{g\beta_T \Delta T L}$$

转变Reynolds数：

$$Re_{transition} = \frac{\rho g \beta_T L^2}{|\partial\gamma/\partial T|}$$

熔池振荡频率的预测（基于线性稳定性分析）：

$$f_{oscillation} = \frac{1}{2\pi}\sqrt{\frac{\gamma}{\rho L^3}(n^2 - 1)}$$

其中$n$是振荡模式数。

### 13.3.3 液滴聚并

两个液滴聚并的初期动力学由毛细-粘性平衡控制：

$$r_{neck}(t) \sim \begin{cases}
(\gamma R/\mu)^{1/2}t^{1/2} & \text{粘性主导} \\
(\gamma t^2/\rho R^3)^{1/4} & \text{惯性主导}
\end{cases}$$

### 13.3.4 润湿动力学

动态接触角通过Hoffman-Voinov-Tanner定律描述：

$$\theta_d^3 - \theta_0^3 = 9Ca\ln(L_{macro}/L_{micro})$$

其中毛细数$Ca = \mu v/\gamma$，$L_{macro}/L_{micro}$是宏观到微观长度尺度比。

## 13.4 粉末床熔融仿真

### 13.4.1 激光-物质相互作用

激光能量吸收遵循Beer-Lambert定律：

$$I(z) = I_0(1-R)e^{-\alpha z}$$

其中$R$是反射率，$\alpha$是吸收系数。对于粉末床，有效吸收率通过射线追踪计算：

$$A_{eff} = 1 - \sum_{n=1}^{N} R^n P_n$$

其中$P_n$是经过$n$次反射后逃逸的概率。

Fresnel反射率的角度依赖性：

$$R(\theta) = \frac{1}{2}\left[\frac{(n-\cos\theta)^2 + k^2}{(n+\cos\theta)^2 + k^2} + \frac{(n-1/\cos\theta)^2 + k^2}{(n+1/\cos\theta)^2 + k^2}\right]$$

其中$n$是折射率，$k$是消光系数。

粉末床的有效吸收深度：

$$\delta_{eff} = \frac{d_p}{3(1-\psi)\sqrt{1-g}}$$

其中$d_p$是粉末直径，$\psi$是孔隙率，$g$是各向异性因子。

多重散射的辐射传输方程：

$$\frac{1}{c}\frac{\partial I}{\partial t} + \hat{s} \cdot \nabla I = -(\kappa + \sigma_s)I + \kappa I_b + \frac{\sigma_s}{4\pi}\int_{4\pi} p(\hat{s}', \hat{s})I(\hat{s}')d\Omega'$$

其中$\kappa$是吸收系数，$\sigma_s$是散射系数，$p$是相函数。

### 13.4.2 热源模型

Goldak双椭球热源模型广泛用于描述移动热源：

$$q(x,y,z,t) = \frac{6\sqrt{3}Q}{\pi^{3/2}abc}\exp\left(-3\frac{x^2}{a^2} - 3\frac{y^2}{b^2} - 3\frac{(z-vt)^2}{c^2}\right)$$

对于粉末床，需考虑孔隙率$\psi$对有效热导率的影响：

$$k_{eff} = k_{solid}(1-\psi) + k_{gas}\psi$$

### 13.4.3 熔池失稳机制

Plateau-Rayleigh不稳定性导致球化：

$$\lambda_{critical} = 2\pi\sqrt{2}r$$

扰动增长率的色散关系：

$$\omega^2 = \frac{\gamma k}{\rho r^2}(1 - k^2r^2)(kr)I_1(kr)/I_0(kr)$$

其中$I_0$、$I_1$是修正Bessel函数。

Kelvin-Helmholtz不稳定性的增长率：

$$\omega = \frac{k\Delta v}{2} - \sqrt{\left(\frac{k\Delta v}{2}\right)^2 - \frac{\gamma k^3}{\rho}}$$

临界速度差：

$$\Delta v_{critical} = 2\sqrt{\frac{\gamma k}{\rho}}$$

热毛细不稳定性的临界Marangoni数：

$$Ma_{critical} = \frac{8(1 + Bi)}{Pr}$$

其中Biot数$Bi = hL/k$表征表面散热。

Benard-Marangoni对流的胞元尺度：

$$\lambda_{cell} = 2\pi\sqrt{\frac{2h}{|d\gamma/dT|\Delta T/\mu\alpha}}$$

### 13.4.4 孔隙形成机理

锁孔（keyhole）模式下的蒸汽反冲压力：

$$p_{recoil} = 0.54p_{sat}\exp\left(\frac{L_v(T-T_b)}{RT_bT}\right)$$

锁孔深度的预测模型（基于能量平衡）：

$$d_{keyhole} = \frac{P(1-R)}{2\pi k_t T_b}\ln\left(\frac{T_b}{T_0}\right)$$

其中$P$是激光功率，$k_t$是热导率。

孔隙捕获的临界条件涉及熔池流动的Peclet数：

$$Pe = \frac{vL}{\alpha} > Pe_{critical}$$

气泡在熔池中的运动由修正的Rayleigh-Plesset方程描述：

$$R\ddot{R} + \frac{3}{2}\dot{R}^2 = \frac{1}{\rho}\left(p_g - p_\infty - \frac{2\gamma}{R} - 4\mu\frac{\dot{R}}{R}\right)$$

气泡上浮速度（Stokes-Hadamard公式）：

$$v_{bubble} = \frac{2\rho g R^2}{3\mu}\frac{1 + \mu_g/\mu}{2 + 3\mu_g/\mu}$$

孔隙率预测的统计模型：

$$P_{porosity} = \exp\left(-\frac{(E_v - E_{v,critical})^2}{2\sigma_E^2}\right)$$

其中能量密度$E_v = P/(v \cdot h \cdot t)$，$h$是舱口间距，$t$是层厚。

## 13.5 残余应力预测

### 13.5.1 热弹塑性本构模型

增量形式的本构关系：

$$d\sigma = C^{ep}:(d\varepsilon - d\varepsilon^{th})$$

其中弹塑性刚度张量：

$$C^{ep} = C^e - \frac{C^e:n \otimes n:C^e}{n:C^e:n + H}$$

热应变率：

$$\dot{\varepsilon}^{th} = \alpha(T)\dot{T}I$$

### 13.5.2 固有应变方法

固有应变$\varepsilon^*$包含塑性应变、相变应变和热应变的不可恢复部分：

$$\varepsilon^* = \varepsilon^p + \varepsilon^{tr} + \int_{T_{ref}}^{T_{max}}\alpha dT$$

残余应力通过弹性响应计算：

$$\sigma^{res} = C^e:(\varepsilon^{total} - \varepsilon^*)$$

### 13.5.3 层间应力演化

逐层沉积过程的应力演化可通过递归关系描述：

$$\sigma_n = f(\sigma_{n-1}, \Delta T_n, t_{cool})$$

其中$f$是考虑热循环和应力松弛的演化算子。

应力松弛的Maxwell模型：

$$\frac{d\sigma}{dt} = -\frac{\sigma}{\tau_{relax}(T)}$$

松弛时间的Arrhenius关系：

$$\tau_{relax} = \tau_0 \exp\left(\frac{Q}{RT}\right)$$

层间热循环引起的应力增量：

$$\Delta\sigma_{thermal} = \int_{T_0}^{T_{max}} E(T)\alpha(T)dT$$

累积塑性应变的演化：

$$\varepsilon_p^{(n)} = \varepsilon_p^{(n-1)} + \Delta\varepsilon_p^{(n)}$$

其中增量塑性应变由屈服准则决定：

$$\Delta\varepsilon_p = \begin{cases}
0 & \text{if } f < 0 \\
\lambda \frac{\partial f}{\partial \sigma} & \text{if } f = 0
\end{cases}$$

$f = \|\sigma_{dev}\| - \sigma_y(T, \dot{\varepsilon})$是von Mises屈服函数。

### 13.5.4 变形预测与补偿

基于敏感度分析的变形补偿：

$$\Delta u_{comp} = -\left(\frac{\partial u}{\partial p}\right)^{-1}u_{target}$$

其中$p$是设计参数，$u$是位移场。

敏感度矩阵的计算（伴随方法）：

$$\frac{\partial u}{\partial p} = -K^{-1}\frac{\partial K}{\partial p}u + K^{-1}\frac{\partial f}{\partial p}$$

迭代补偿策略：

$$p_{n+1} = p_n - \alpha J^{-1}(u_{measured} - u_{target})$$

Levenberg-Marquardt优化：

$$p_{n+1} = p_n - (J^TJ + \lambda I)^{-1}J^T r$$

其中$r = u_{measured} - u_{target}$是残差向量。

变形的主成分分析（PCA）降阶：

$$u \approx \bar{u} + \sum_{i=1}^{m} a_i \phi_i$$

其中$\phi_i$是主模态，通过求解特征值问题得到：

$$C\phi_i = \lambda_i\phi_i$$

协方差矩阵$C = \frac{1}{n-1}\sum_{j=1}^n (u_j - \bar{u})(u_j - \bar{u})^T$。

补偿效果的置信区间估计：

$$P(|u_{final} - u_{target}| < \epsilon) = 1 - \alpha$$

基于Bootstrap方法或贝叶斯推断确定$\epsilon$。

## 本章小结

本章系统介绍了3D打印过程中多相流与传热现象的数学建模方法：

1. **熔融沉积建模**：从流变学基础出发，建立了非牛顿流体的挤出流动模型
2. **Stefan问题**：详细讨论了相变过程的数学描述，包括焓方法和相场方法
3. **Marangoni效应**：分析了表面张力驱动的流动及其在熔池动力学中的作用
4. **粉末床熔融**：建立了激光-物质相互作用和熔池失稳的理论框架
5. **残余应力**：介绍了热弹塑性本构模型和固有应变方法

关键公式总结：
- 幂律流体粘度：$\eta = K\dot{\gamma}^{n-1}$
- Stefan条件：$\rho L_f\frac{ds}{dt} = k_s\nabla T_s - k_l\nabla T_l$
- Marangoni应力：$\tau_s = \frac{\partial \gamma}{\partial T}\nabla_s T$
- 蒸汽反冲压力：$p_{recoil} = 0.54p_{sat}\exp\left(\frac{L_v(T-T_b)}{RT_bT}\right)$
- 固有应变：$\varepsilon^* = \varepsilon^p + \varepsilon^{tr} + \varepsilon^{th}$

## 练习题

### 基础题

**练习13.1** 推导幂律流体在圆管中的流量-压降关系。
<details>
<summary>提示</summary>
从广义牛顿流体的运动方程出发，利用轴对称假设简化。
</details>

<details>
<summary>答案</summary>
对于幂律流体$\tau = K\dot{\gamma}^n$，在圆管中：
剪切应力分布：$\tau_{rz} = \frac{r\Delta P}{2L}$
速度梯度：$\frac{dv_z}{dr} = -\left(\frac{r\Delta P}{2KL}\right)^{1/n}$
积分得速度分布，再积分得流量：
$$Q = \frac{\pi n}{3n+1}\left(\frac{\Delta P R}{2KL}\right)^{1/n}R^3$$
</details>

**练习13.2** 证明一维Stefan问题的相似解存在条件。
<details>
<summary>提示</summary>
假设$s(t) = 2\lambda\sqrt{\alpha t}$，寻找使边界条件满足的$\lambda$值。
</details>

<details>
<summary>答案</summary>
设相似变量$\eta = x/(2\sqrt{\alpha t})$，温度分布：
$$T_s(\eta) = A + B\text{erf}(\eta)$$
$$T_l(\eta) = C + D\text{erfc}(\eta)$$
Stefan条件给出超越方程：
$$\lambda e^{\lambda^2}\text{erf}(\lambda) = \frac{c_s(T_m - T_0)}{L_f\sqrt{\pi}}$$
</details>

**练习13.3** 计算球形液滴聚并的能量耗散率。
<details>
<summary>提示</summary>
使用Rayleigh耗散函数$\Phi = \int_V \tau_{ij}\dot{\gamma}_{ij}dV$。
</details>

<details>
<summary>答案</summary>
对于粘性主导的聚并，颈部半径$r \sim (\gamma R t/\mu)^{1/2}$
耗散功率：
$$P = \frac{d}{dt}(4\pi R^2\gamma - \pi r^2\gamma) \approx \frac{\pi\gamma^2 R}{\mu}$$
总耗散能量约为初始表面能的一半。
</details>

### 挑战题

**练习13.4** 分析激光功率密度对熔池深宽比的影响，考虑Marangoni对流和浮力的竞争。
<details>
<summary>提示</summary>
建立无量纲分析，比较$Ma/Pr$和$Ra$的相对大小。
</details>

<details>
<summary>答案</summary>
深宽比$D/W$主要由Peclet数和无量纲功率决定：
$$\frac{D}{W} \sim Pe^{1/2}\left(\frac{Q}{\rho c_p v L^2 T_m}\right)^{1/3}$$
当$Ma/Pr > Ra$时，Marangoni对流主导，形成浅而宽的熔池；
当$Ra > Ma/Pr$时，浮力对流主导，可能形成更深的熔池。
转变发生在：$|\partial\gamma/\partial T|\Delta T \approx \rho g\beta\Delta T L$
</details>

**练习13.5** 推导多层沉积过程中的应力叠加原理，考虑材料的粘弹性行为。
<details>
<summary>提示</summary>
使用Boltzmann叠加原理和遗传积分。
</details>

<details>
<summary>答案</summary>
对于线性粘弹性材料：
$$\sigma(t) = \int_0^t G(t-\tau)\frac{\partial\varepsilon}{\partial\tau}d\tau$$
第$n$层引起的应力增量：
$$\Delta\sigma_n(t) = \int_{t_n}^t G(t-\tau)\frac{\partial\varepsilon_n}{\partial\tau}d\tau$$
总应力：
$$\sigma_{total}(t) = \sum_{i=1}^n \Delta\sigma_i(t)H(t-t_i)$$
其中$H$是Heaviside函数。考虑温度依赖性时需要时温等效原理。
</details>

**练习13.6** 设计一个自适应网格细化策略用于追踪相变界面，给出误差估计。
<details>
<summary>提示</summary>
基于温度梯度和相场参数的二阶导数设计指示子。
</details>

<details>
<summary>答案</summary>
细化指示子：
$$\eta_K = h_K^2\|\nabla^2 T\|_{L^2(K)} + h_K|\nabla\phi|_{L^\infty(K)}$$
后验误差估计：
$$\|e\|_{H^1} \leq C\left(\sum_K \eta_K^2\right)^{1/2}$$
自适应策略：
1. 标记$\eta_K > \theta_{ref}\max_K\eta_K$的单元细化
2. 标记$\eta_K < \theta_{coar}\max_K\eta_K$的单元粗化
3. 保持细化级别差不超过1
</details>

## 常见陷阱与错误

1. **数值稳定性问题**
   - Stefan问题的显式格式需要极小时间步长：$\Delta t < \frac{\Delta x^2}{2\alpha}$
   - 相场方法的界面厚度必须充分解析：至少4-5个网格点
   - 强对流问题需要迎风格式或SUPG稳定化

2. **物理模型选择**
   - 忽略Marangoni效应可能严重低估熔池流速
   - 简化的点热源模型不适用于大功率激光加工
   - 常物性假设在大温度梯度下失效

3. **边界条件处理**
   - 自由表面的法向应力平衡容易遗漏表面张力项
   - 移动热源需要考虑参考系变换
   - 辐射边界条件的线性化仅在小温差下有效

4. **多尺度问题**
   - 熔池尺度（mm）vs 粉末颗粒（μm）需要均质化处理
   - 快速凝固的时间尺度（μs）vs 热传导（s）需要自适应时间步
   - 残余应力的局部（层间）vs 全局（构件）效应需要多尺度方法

## 最佳实践检查清单

### 建模阶段
- [ ] 是否进行了无量纲分析确定主导物理机制？
- [ ] 材料属性的温度依赖性是否充分考虑？
- [ ] 相变潜热是否正确处理（避免数值振荡）？
- [ ] 边界条件是否物理合理且数学适定？

### 数值方法
- [ ] 网格是否足够细化以捕捉温度梯度和相界面？
- [ ] 时间步长是否满足CFL条件和扩散稳定性？
- [ ] 非线性求解器是否收敛（检查残差历史）？
- [ ] 是否进行了网格无关性验证？

### 验证与确认
- [ ] 是否与解析解或基准解对比？
- [ ] 质量和能量是否守恒（检查全局平衡）？
- [ ] 参数敏感性分析是否完成？
- [ ] 实验验证是否涵盖关键物理量？

### 工程应用
- [ ] 计算成本是否可接受（考虑降阶模型）？
- [ ] 结果是否包含不确定性量化？
- [ ] 是否提供了工艺窗口和优化建议？
- [ ] 文档是否清晰记录了假设和限制？