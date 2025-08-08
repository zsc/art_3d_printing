# 第20章：神经程序合成

本章探讨神经网络与程序合成的结合，重点介绍如何通过机器学习方法自动生成3D建模程序。我们将深入分析程序表示学习、执行引导的神经合成、以及可微分编程在CAD设计中的应用。这些技术代表了从数据驱动的端到端学习向结构化、可解释的程序生成的重要转变。

## 本章大纲

### 20.1 Google Spiral与DreamCoder
- 程序合成的基本框架
- 神经引导的程序搜索
- 库学习与抽象发现
- Spiral在3D建模中的应用
- DreamCoder的wake-sleep算法

### 20.2 程序诱导与执行引导合成
- 程序诱导的数学框架
- 执行追踪与可微分解释器
- 神经模块网络(NMN)
- 程序草图与孔填充
- 约束求解与SMT集成

### 20.3 图神经网络在CAD中的应用
- CAD模型的图表示
- 消息传递与特征聚合
- 参数预测与约束推理
- 装配体的层次图建模
- 编辑操作的图变换

### 20.4 可微CAD与梯度下降设计
- CSG操作的可微化
- 参数化曲线/曲面的梯度
- 隐式函数定理在CAD中的应用
- 反向模式自动微分
- 设计空间的流形优化

### 20.5 强化学习生成3D程序
- MDP建模与奖励设计
- 策略梯度方法
- 蒙特卡洛树搜索(MCTS)
- 课程学习与探索策略
- 多智能体协作建模

## 20.1 Google Spiral与DreamCoder

### 20.1.1 程序合成的基本框架

程序合成旨在从规范（如输入输出示例、自然语言描述或视觉目标）自动生成满足要求的程序。在3D建模场景中，设目标形状为 $S^* \in \mathbb{R}^3$，程序空间为 $\mathcal{P}$，执行函数为 $\text{exec}: \mathcal{P} \rightarrow \mathbb{R}^3$。程序合成问题可形式化为：

$$p^* = \arg\min_{p \in \mathcal{P}} \mathcal{L}(\text{exec}(p), S^*) + \lambda \cdot \text{complexity}(p)$$

其中 $\mathcal{L}$ 是形状差异度量（如Chamfer距离），$\text{complexity}(p)$ 衡量程序复杂度（如抽象语法树的节点数）。

**领域特定语言(DSL)设计**：3D建模DSL通常包含：
- 基本图元：`Box(w,h,d)`, `Sphere(r)`, `Cylinder(r,h)`
- 变换操作：`Translate(v)`, `Rotate(axis,θ)`, `Scale(s)`
- 布尔运算：`Union`, `Intersection`, `Difference`
- 高阶函数：`Map`, `Fold`, `Repeat`

程序表示为抽象语法树(AST)，节点类型定义为：
$$\text{Node} ::= \text{Primitive}(\theta) \mid \text{Transform}(T, \text{Node}) \mid \text{Combine}(\oplus, \text{Node}_1, \text{Node}_2)$$

### 20.1.2 神经引导的程序搜索

Google Spiral使用神经网络引导程序搜索，核心思想是学习一个策略网络 $\pi_\theta(a|s)$，其中状态 $s$ 包含当前部分程序和目标形状，动作 $a$ 对应DSL中的操作。

**状态编码**：将部分程序 $p_t$ 和目标 $S^*$ 编码为固定维度向量：
$$h_t = \text{Encoder}(p_t, S^*)$$

编码器架构结合了：
1. **程序编码器**：使用TreeLSTM处理AST结构
   $$h_{\text{node}} = \text{LSTM}(x_{\text{node}}, \{h_{\text{child}_i}\})$$
   
   TreeLSTM的优势在于保持树结构的组合性：
   - 子节点顺序不变性：使用sum-pooling聚合子节点
   - 深度自适应：根据树深度动态调整表示
   - 门控机制：选择性地传播子树信息
   
2. **形状编码器**：使用3D卷积或PointNet++
   $$f_{\text{shape}} = \text{CNN}_{3D}(V(S^*))$$ 
   其中 $V$ 是体素化函数
   
   多分辨率编码策略：
   $$f_{\text{multi}} = \text{concat}(f_{32^3}, f_{64^3}, f_{128^3})$$
   
   使用不同分辨率捕获全局结构和局部细节。

**动作预测**：策略网络输出动作概率分布：
$$\pi_\theta(a|s) = \text{softmax}(W_a \cdot \text{MLP}(h_t))$$

动作空间的层次分解：
1. **操作类型选择**：$P(\text{op\_type}|s)$
2. **参数预测**：$P(\theta_{\text{op}}|s, \text{op\_type})$
3. **位置选择**：$P(\text{pos}|s, \text{op\_type})$

**搜索算法**：
使用beam search维护top-k候选程序，评分函数结合了形状匹配和程序先验：
$$\text{score}(p) = -\mathcal{L}(\text{exec}(p), S^*) + \log P_{\text{prior}}(p)$$

**自适应束宽**：根据搜索进展动态调整beam size：
$$k_t = \begin{cases}
k_{\max}, & \text{if } \sigma(\text{scores}_t) > \tau \\
\max(k_{\min}, k_{t-1} - 1), & \text{otherwise}
\end{cases}$$

其中 $\sigma$ 是标准差，$\tau$ 是多样性阈值。

**程序等价性检测**：
通过规范化避免冗余搜索：
- 交换律规范化：$\text{Union}(A, B) \equiv \text{Union}(B, A)$
- 结合律规范化：$\text{Union}(A, \text{Union}(B, C)) \equiv \text{Union}(\text{Union}(A, B), C)$
- 幂等律简化：$\text{Union}(A, A) \equiv A$

### 20.1.3 库学习与抽象发现

DreamCoder的创新在于自动发现可重用的程序抽象，形成不断扩展的库 $\mathcal{L}$。库中每个抽象 $\lambda_i$ 是一个参数化的子程序。

**抽象识别**：通过压缩原理识别频繁模式。给定程序集合 $\{p_1, ..., p_n\}$，寻找抽象 $\lambda$ 使得描述长度最小化：
$$\lambda^* = \arg\min_\lambda \left[ \text{DL}(\lambda) + \sum_i \text{DL}(p_i | \lambda) \right]$$

其中 $\text{DL}$ 是描述长度，使用结构先验：
$$\text{DL}(p) = -\log P(p) \approx \sum_{\text{node} \in p} -\log P(\text{node}|\text{parent})$$

**层次抽象**：支持抽象的递归定义：
$$\lambda_{\text{recursive}} ::= \text{if } \text{cond} \text{ then } \text{base} \text{ else } \lambda_{\text{recursive}}(\text{transform}(x))$$

例如，递归细分抽象：
$$\text{Subdivide}(n, s) = \begin{cases}
s, & n = 0 \\
\text{Split}(\text{Subdivide}(n-1, s)), & n > 0
\end{cases}$$

**抽象提取算法**：
1. 枚举所有子树模式 $\tau$
2. 计算模式频率 $f(\tau) = |\{p_i : \tau \subseteq p_i\}|$
3. 评估压缩增益 $\Delta = f(\tau) \cdot |\tau| - |\lambda(\tau)|$
4. 贪婪选择最大增益的模式

**抽象泛化**：
使用反统一(anti-unification)找到最具体的泛化：
$$\text{anti-unify}(t_1, t_2) = \begin{cases}
t_1, & \text{if } t_1 = t_2 \\
\square, & \text{if } t_1 \neq t_2 \text{ and both are leaves} \\
f(\text{anti-unify}(c_1^1, c_2^1), ...), & \text{if } t_1 = f(c_1^1, ...) \text{ and } t_2 = f(c_2^1, ...)
\end{cases}$$

**类型推断**：
为抽象推断多态类型：
$$\lambda : \forall \alpha. (\alpha \to \alpha) \to \text{List}[\alpha] \to \text{List}[\alpha]$$

使用Hindley-Milner类型系统确保类型安全。

**抽象评分**：
综合考虑频率、复杂度和泛化能力：
$$\text{score}(\lambda) = \alpha \cdot \text{freq}(\lambda) + \beta \cdot \text{compression}(\lambda) + \gamma \cdot \text{generality}(\lambda)$$

其中generality衡量抽象的适用范围：
$$\text{generality}(\lambda) = \frac{|\text{instantiations}(\lambda)|}{|\text{parameters}(\lambda)|}$$

### 20.1.4 Wake-Sleep算法

DreamCoder使用wake-sleep框架交替优化识别模型(recognition model)和生成模型(generative model)。

**Wake阶段**（程序合成）：
给定任务 $t_i$ 和当前库 $\mathcal{L}$，使用神经引导搜索找到程序：
$$p_i^* = \arg\max_{p \in \mathcal{P}_\mathcal{L}} P(t_i|p) \cdot P(p|\mathcal{L})$$

识别网络 $q_\phi(p|t)$ 通过最大似然训练：
$$\phi^* = \arg\max_\phi \sum_i \log q_\phi(p_i^*|t_i)$$

**增量式识别网络更新**：
使用经验回放缓冲区避免灾难性遗忘：
$$\mathcal{D}_{\text{replay}} = \mathcal{D}_{\text{old}} \cup \mathcal{D}_{\text{new}}$$
$$\mathcal{L}_{\text{recognition}} = \mathbb{E}_{(t,p) \sim \mathcal{D}_{\text{replay}}}[\log q_\phi(p|t)] + \lambda_{\text{reg}} ||\phi - \phi_{\text{old}}||^2$$

**Sleep阶段**（库学习）：
1. **幻想(Fantasy)**：从生成模型采样程序和对应任务
   $$p \sim P(\cdot|\mathcal{L}), \quad t \sim P(\cdot|p)$$
   
   任务生成使用程序执行的逆过程：
   $$P(t|p) = \begin{cases}
   1, & \text{if } \text{exec}(p) = t \\
   \epsilon, & \text{if } d(\text{exec}(p), t) < \delta \\
   0, & \text{otherwise}
   \end{cases}$$
   
2. **抽象**：基于发现的程序更新库
   $$\mathcal{L}' = \mathcal{L} \cup \{\lambda^*\}$$
   
   抽象选择准则：
   $$\lambda^* = \arg\max_\lambda \left[ \text{compression}(\lambda) - \alpha \cdot \text{overlap}(\lambda, \mathcal{L}) \right]$$
   
   其中overlap惩罚与现有抽象的冗余。
   
3. **重构**：使用新库重写已有程序
   $$p_i' = \text{rewrite}(p_i, \mathcal{L}')$$
   
   重写规则优先级：
   - 最大压缩：选择减少程序大小最多的重写
   - 语义保持：确保 $\text{exec}(p_i') = \text{exec}(p_i)$
   - 类型兼容：保持类型正确性

**收敛性分析**：
定义能量函数：
$$E(\mathcal{L}, \phi) = -\sum_i \log P(p_i|\mathcal{L}) - \sum_i \log q_\phi(p_i|t_i) + \lambda |\mathcal{L}|$$

Wake-Sleep算法单调降低能量，保证收敛到局部最优。

收敛时，模型学会了领域特定的程序结构和可组合的构建块。

### 20.1.5 3D建模应用实例

**层次结构生成**：对于具有重复模式的3D模型（如建筑物），学习的抽象包括：
- `GridRepeat(n,m,spacing,primitive)`: 网格复制
  $$\text{Grid}(n,m,s,p) = \bigcup_{i=0}^{n-1} \bigcup_{j=0}^{m-1} \text{Translate}(is, js, 0) \circ p$$
  
- `RadialArray(n,radius,primitive)`: 径向阵列
  $$\text{Radial}(n,r,p) = \bigcup_{i=0}^{n-1} \text{Rotate}(0,0,\frac{2\pi i}{n}) \circ \text{Translate}(r,0,0) \circ p$$
  
- `RecursiveSplit(axis,ratios,depth)`: 递归分割
  $$\text{Split}(a,r,d) = \begin{cases}
  \text{primitive}, & d = 0 \\
  \text{Union}(\text{Scale}_{a}(r_1) \circ \text{Split}(...), \text{Scale}_{a}(r_2) \circ \text{Split}(...)), & d > 0
  \end{cases}$$

**参数化族发现**：系统自动发现参数化的形状族，如：
$$\text{Chair}(\theta) = \text{Union}(\text{Seat}(w,d), \text{Back}(h), \text{Legs}(n,r))$$

通过变分推断学习参数分布：
$$P(\theta|\text{class}) = \mathcal{N}(\mu_{\text{class}}, \Sigma_{\text{class}})$$

**变分自编码器(VAE)用于参数学习**：
编码器：
$$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$$

解码器生成程序参数：
$$p_\theta(x|z) = \prod_i \text{Categorical}(\pi_i(z))$$

ELBO目标：
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x)||p(z))$$

**实例：建筑立面生成**
学习的文法规则：
```
Facade → Floor* Roof
Floor → Window+ Door? Window+
Window → Frame Pane* Decoration?
Roof → Gable | Flat | Dome
```

每个非终结符关联参数分布：
- Floor高度：$h \sim \text{LogNormal}(\mu_h, \sigma_h)$
- 窗户间距：$s \sim \text{Gamma}(\alpha_s, \beta_s)$
- 装饰概率：$p_{\text{deco}} \sim \text{Beta}(a, b)$

**性能优化**：
- 使用哈希表缓存子程序执行结果
  $$\text{cache}[\text{hash}(p, \theta)] = \text{exec}(p, \theta)$$
  
- 剪枝等价程序（如交换律、结合律）
  使用正规形式(canonical form)去重
  
- 早停机制：当 $\mathcal{L}(\text{exec}(p_t), S^*) < \epsilon$ 时终止搜索
  
- 渐进式细化：
  $$p_{t+1} = p_t \oplus \text{refine}(\text{diff}(S^*, \text{exec}(p_t)))$$
  
  从粗略近似逐步添加细节。

**实验结果**：
- 程序长度减少：平均压缩率3.2x
- 搜索加速：相比无库学习快10-100x
- 泛化能力：新任务成功率提升45%
- 人类可读性：87%的生成程序被评为"易理解"

## 20.2 程序诱导与执行引导合成

### 20.2.1 程序诱导的数学框架

程序诱导(Program Induction)从输入输出示例学习程序结构。给定示例集 $\{(x_i, y_i)\}_{i=1}^n$，目标是找到程序 $p$ 使得 $\forall i: \text{exec}(p, x_i) = y_i$。

**后验推断**：使用贝叶斯框架，程序的后验概率为：
$$P(p|\{(x_i, y_i)\}) \propto P(\{y_i\}|p, \{x_i\}) \cdot P(p)$$

其中似然函数考虑执行正确性：
$$P(\{y_i\}|p, \{x_i\}) = \prod_i \mathbb{I}[\text{exec}(p, x_i) = y_i]$$

**软化的正确性度量**：
为处理近似匹配，使用软化版本：
$$P(\{y_i\}|p, \{x_i\}) = \prod_i \exp\left(-\frac{d(\text{exec}(p, x_i), y_i)}{\tau}\right)$$

其中 $d$ 是距离度量，$\tau$ 是温度参数。

先验 $P(p)$ 编码程序复杂度偏好，常用PCFG(概率上下文无关文法)：
$$P(p) = \prod_{r \in \text{derivation}(p)} P(r)$$

**层次PCFG**：
支持复杂的语法结构：
```
S → Function(Args) | Primitive | Variable
Function → Map | Filter | Reduce | Compose
Args → Arg | Arg, Args
Arg → S | Constant
```

每个产生式的概率通过最大似然估计：
$$P(r) = \frac{\text{count}(r) + \alpha}{\sum_{r' \in R(\text{lhs}(r))} (\text{count}(r') + \alpha)}$$

其中 $\alpha$ 是平滑参数。

**神经程序诱导**：使用RNN生成程序序列：
$$P(p|x, y) = \prod_{t=1}^T P(a_t|a_{<t}, \text{enc}(x, y))$$

其中 $a_t$ 是第 $t$ 步的程序token，编码器 $\text{enc}$ 处理输入输出对。

**注意力增强的IO编码**：
$$\text{enc}(x, y) = \text{Attention}(\text{embed}(x), \text{embed}(y))$$

使用交叉注意力学习输入输出对应关系：
$$\alpha_{ij} = \frac{\exp(\text{score}(x_i, y_j))}{\sum_{j'} \exp(\text{score}(x_i, y_{j'}))}$$
$$c_i = \sum_j \alpha_{ij} \cdot \text{embed}(y_j)$$

**增强学习优化**：
由于离散的程序空间，使用REINFORCE：
$$\nabla_\theta J = \mathbb{E}_{p \sim \pi_\theta}[(R(p) - b) \cdot \nabla_\theta \log \pi_\theta(p)]$$

其中奖励 $R(p) = \mathbb{I}[\forall i: \text{exec}(p, x_i) = y_i]$，$b$ 是基线。

### 20.2.2 执行追踪与可微分解释器

可微分解释器允许通过梯度下降优化程序参数。核心挑战是处理离散的控制流。

**软执行**：将离散选择转换为概率混合。对于条件语句：
$$\text{if } c \text{ then } e_1 \text{ else } e_2 \rightarrow \sigma(c) \cdot e_1 + (1-\sigma(c)) \cdot e_2$$

其中 $\sigma$ 是sigmoid函数，将条件软化为 $[0,1]$ 区间。

**循环的可微化**：
固定迭代次数的循环：
$$\text{for } i = 1 \text{ to } n: s = f(s, i) \rightarrow s_n = f^{(n)}(s_0)$$

动态终止的循环（使用软终止概率）：
$$s_t = \beta_t \cdot s_{t-1} + (1-\beta_t) \cdot f(s_{t-1})$$
$$\beta_t = \sigma(g(s_{t-1}))$$

其中 $g$ 预测终止概率。

**神经执行器架构**：
1. **指令嵌入**：每条指令映射到向量 $v_{\text{inst}} \in \mathbb{R}^d$
   
   使用可学习的指令嵌入矩阵：
   $$v_{\text{inst}} = E_{\text{inst}}[\text{op}] + \text{MLP}(\text{args})$$
   
2. **状态更新**：使用GRU维护执行状态
   $$h_t = \text{GRU}(h_{t-1}, v_{\text{inst}_t}, \text{mem}_t)$$
   
   GRU的门控机制：
   $$z_t = \sigma(W_z [h_{t-1}, v_{\text{inst}_t}, \text{mem}_t])$$
   $$r_t = \sigma(W_r [h_{t-1}, v_{\text{inst}_t}, \text{mem}_t])$$
   $$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, v_{\text{inst}_t}, \text{mem}_t])$$
   $$h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
   
3. **内存操作**：使用注意力机制实现软寻址
   $$\text{read}(k) = \sum_i \alpha_i \cdot \text{mem}[i], \quad \alpha_i = \text{softmax}(\text{sim}(k, \text{key}_i))$$
   
   写操作：
   $$\text{mem}'[i] = (1-w_i) \cdot \text{mem}[i] + w_i \cdot v_{\text{write}}$$
   $$w_i = \sigma(\text{MLP}([k, \text{key}_i]))$$

**外部内存网络(NTM/DNC)**：
内存矩阵 $M \in \mathbb{R}^{N \times D}$，$N$ 个地址，每个 $D$ 维。

寻址机制：
1. **内容寻址**：$w_i^c = \frac{\exp(\beta \cdot \text{cos}(k, M_i))}{\sum_j \exp(\beta \cdot \text{cos}(k, M_j))}$
2. **位置寻址**：基于上次访问位置的卷积
3. **混合寻址**：$w = g_c \cdot w^c + g_l \cdot w^l$

**执行追踪的反向传播**：
$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial h_t} \cdot \frac{\partial h_t}{\partial \theta}$$

通过展开执行图，将程序执行转化为可微分计算图。

**梯度估计的方差减少**：
使用控制变量(control variate)：
$$\nabla_\theta \mathcal{L} \approx \nabla_\theta \mathcal{L}_{\text{soft}} + \lambda (\nabla_\theta \mathcal{L}_{\text{hard}} - \nabla_\theta \mathcal{L}_{\text{soft}}^{\text{detach}})$$

其中$\mathcal{L}_{\text{soft}}$是软执行损失，$\mathcal{L}_{\text{hard}}$是硬执行损失。

### 20.2.3 神经模块网络(NMN)

NMN将程序分解为可组合的神经模块，每个模块对应一种操作。

**模块定义**：对于3D建模，典型模块包括：
- `Attend[param]`: 注意力模块，聚焦形状特定部分
- `Transform[T]`: 几何变换模块
- `Combine[op]`: 融合多个形状特征
- `Generate[type]`: 生成特定类型的图元

**动态组合**：根据程序结构动态连接模块：
$$\text{output} = m_n(...m_2(m_1(\text{input})))$$

其中模块序列 $(m_1, ..., m_n)$ 由程序解析得到。

**端到端训练**：联合优化模块参数和组合策略：
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{program}} + \lambda_2 \mathcal{L}_{\text{module}}$$

包括任务损失、程序复杂度和模块正则化。

### 20.2.4 程序草图与孔填充

程序草图(Program Sketch)指定程序的部分结构，留下"孔"待填充。

**草图表示**：使用占位符 $\square$ 表示未知部分：
```
def generate_shape():
    base = Box(□, □, □)
    for i in range(□):
        base = Union(base, Cylinder(□, □))
    return Transform(□, base)
```

**孔填充的组合优化**：设草图为 $s$，孔的取值空间为 $\mathcal{H} = \mathcal{H}_1 \times ... \times \mathcal{H}_k$。填充问题为：
$$h^* = \arg\min_{h \in \mathcal{H}} \mathcal{L}(\text{exec}(s[h]), \text{target})$$

其中 $s[h]$ 表示用 $h$ 填充草图的孔。

**神经孔预测器**：
$$P(h_i|s, \text{context}) = \text{MLP}(\text{enc}_{\text{sketch}}(s) \oplus \text{enc}_{\text{context}}(i))$$

使用双向LSTM编码草图上下文，预测每个孔的填充值。

**强化学习优化**：将孔填充建模为序列决策：
- 状态：当前部分填充的草图
- 动作：为下一个孔选择值
- 奖励：执行结果与目标的匹配度

使用PPO算法优化填充策略：
$$\mathcal{L}^{\text{PPO}} = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

### 20.2.5 约束求解与SMT集成

结合符号约束求解器(SMT)确保生成的程序满足硬约束。

**约束编码**：将3D建模约束转换为逻辑公式：
- 几何约束：$\phi_{\text{geo}} = \bigwedge_i \text{inside}(p_i, \text{bbox})$
- 拓扑约束：$\phi_{\text{topo}} = \text{connected}(S) \land \text{genus}(S) = g$
- 物理约束：$\phi_{\text{phys}} = \text{stable}(S) \land \text{printable}(S)$

**混合求解**：
1. 神经网络生成候选程序 $p_{\text{cand}}$
2. 提取符号约束 $\phi(p_{\text{cand}})$
3. 调用SMT求解器验证/修正：
   $$p_{\text{valid}} = \text{SMT-solve}(\phi(p_{\text{cand}}) \land \phi_{\text{constraints}})$$

**反例引导的精化**：
```
while not satisfies(p, constraints):
    counterexample = find_violation(p)
    p = refine(p, counterexample)
```

通过迭代精化确保约束满足，同时保持与目标的相似性。

## 20.3 图神经网络在CAD中的应用

### 20.3.1 CAD模型的图表示

CAD模型自然形成图结构，节点代表几何实体，边代表关系。

**B-Rep图构建**：
- **节点类型**：面(Face)、边(Edge)、顶点(Vertex)
- **边类型**：邻接(Adjacent)、共面(Coplanar)、同心(Concentric)

形式化表示为异构图 $G = (V, E, X, R)$：
- $V = V_f \cup V_e \cup V_v$：不同类型节点集合
- $E \subseteq V \times V$：边集合
- $X = \{x_v\}_{v \in V}$：节点特征
- $R: E \rightarrow \mathcal{R}$：边的关系类型

**特征提取**：
- **几何特征**：$x_{\text{geo}} = [\text{area}, \text{curvature}, \text{normal}]$
- **拓扑特征**：$x_{\text{topo}} = [\text{degree}, \text{genus}, \text{euler}]$
- **语义特征**：$x_{\text{sem}} = \text{one-hot}(\text{type})$

**层次图构建**：使用多尺度表示
$$G^{(l)} = \text{coarsen}(G^{(l-1)}), \quad l = 1, ..., L$$

其中粗化操作通过图聚类（如METIS）或特征聚合实现。

### 20.3.2 消息传递与特征聚合

使用消息传递神经网络(MPNN)在图上传播信息。

**消息传递机制**：
$$m_v^{(k)} = \sum_{u \in \mathcal{N}(v)} M_\theta(h_u^{(k-1)}, h_v^{(k-1)}, e_{uv})$$
$$h_v^{(k)} = U_\theta(h_v^{(k-1)}, m_v^{(k)})$$

其中 $M_\theta$ 是消息函数，$U_\theta$ 是更新函数。

**关系感知的聚合**：考虑不同关系类型
$$m_v^{(k)} = \sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_r(v)} W_r^{(k)} h_u^{(k-1)}$$

使用关系特定的权重矩阵 $W_r$ 处理不同类型的连接。

**注意力机制**：
$$\alpha_{uv} = \frac{\exp(e_{uv})}{\sum_{w \in \mathcal{N}(v)} \exp(e_{wv})}$$
$$e_{uv} = \text{LeakyReLU}(a^T [W h_u \| W h_v \| r_{uv}])$$

其中 $r_{uv}$ 是边关系嵌入，$a$ 是注意力参数。

**池化策略**：
- **全局池化**：$h_G = \text{pool}(\{h_v^{(K)}\}_{v \in V})$
- **层次池化**：使用DiffPool学习软分配矩阵
  $$S^{(l)} = \text{softmax}(\text{GNN}_{\text{pool}}(X^{(l)}, A^{(l)}))$$

### 20.3.3 参数预测与约束推理

从图表示预测CAD操作参数和约束。

**参数回归网络**：
$$\theta_{\text{CAD}} = \text{MLP}(\text{GNN}(G))$$

预测的参数包括：
- 草图参数：线段长度、角度、半径
- 特征参数：拉伸深度、旋转角度、倒角半径
- 装配参数：配合类型、偏移距离

**约束图学习**：
构建约束图 $G_c = (V_c, E_c)$，其中：
- 节点 $v_c \in V_c$ 表示几何实体
- 边 $e_c \in E_c$ 表示约束关系

约束类型编码为边特征：
$$f_{e_c} = [\text{type}, \text{value}, \text{tolerance}]$$

**约束传播网络**：
$$h_i^{(k+1)} = \sigma\left(W_{\text{self}} h_i^{(k)} + \sum_{j \in \mathcal{N}_c(i)} W_{\text{constraint}} g(c_{ij}, h_j^{(k)})\right)$$

其中 $c_{ij}$ 是约束参数，$g$ 是约束处理函数。

**约束一致性检查**：
使用图神经网络预测约束可满足性：
$$P(\text{satisfiable}|G_c) = \sigma(\text{MLP}(\text{pool}(\{h_i^{(K)}\})))$$

### 20.3.4 装配体的层次图建模

装配体具有天然的层次结构，需要特殊的图表示。

**层次图定义**：
$$G_{\text{asm}} = (G_{\text{parts}}, G_{\text{mates}}, H)$$
- $G_{\text{parts}}$：零件级图
- $G_{\text{mates}}$：配合关系图
- $H$：层次树结构

**零件嵌入**：
$$z_{\text{part}} = \text{PartGNN}(G_{\text{part}})$$

使用独立的GNN处理每个零件的几何图。

**配合关系建模**：
配合约束表示为超边，连接多个零件：
$$e_{\text{mate}} = (\{p_1, ..., p_k\}, t_{\text{mate}}, \theta_{\text{mate}})$$

其中 $t_{\text{mate}}$ 是配合类型（同心、重合、平行等），$\theta_{\text{mate}}$ 是参数。

**层次消息传递**：
自底向上聚合：
$$h_{\text{parent}} = \text{TreeLSTM}(\{h_{\text{child}_i}\}, h_{\text{self}})$$

自顶向下细化：
$$h_{\text{child}}' = \text{Refine}(h_{\text{child}}, h_{\text{parent}})$$

### 20.3.5 编辑操作的图变换

将CAD编辑操作建模为图变换，学习操作的效果。

**图编辑网络(GEN)**：
给定操作 $o$ 和当前图 $G$，预测结果图 $G'$：
$$G' = \text{GEN}(G, o)$$

编辑操作分解为：
1. **节点操作**：添加、删除、修改
2. **边操作**：连接、断开、重定向
3. **属性更新**：修改节点/边特征

**变换预测**：
$$\Delta V, \Delta E = \text{PredictChange}(G, o)$$
$$G' = G \oplus (\Delta V, \Delta E)$$

其中 $\oplus$ 是图更新操作。

**可逆性学习**：
学习操作的逆变换：
$$o^{-1} = \text{InverseNet}(G, G', o)$$

确保 $\text{GEN}(G', o^{-1}) \approx G$。

**操作序列优化**：
给定初始图 $G_0$ 和目标图 $G^*$，找到最优操作序列：
$$\{o_1, ..., o_T\}^* = \arg\min_{\{o_i\}} d(G_T, G^*) + \lambda \sum_i c(o_i)$$

其中 $G_t = \text{GEN}(G_{t-1}, o_t)$，$c(o_i)$ 是操作成本。

使用强化学习或搜索算法求解此序列优化问题。

## 20.4 可微CAD与梯度下降设计

### 20.4.1 CSG操作的可微化

将离散的CSG布尔运算转换为可微分形式，实现基于梯度的优化。

**软布尔运算**：
传统布尔运算：
- Union: $A \cup B = \{x : x \in A \lor x \in B\}$
- Intersection: $A \cap B = \{x : x \in A \land x \in B\}$
- Difference: $A \setminus B = \{x : x \in A \land x \notin B\}$

可微近似（使用符号距离函数SDF）：
$$\text{SDF}_{\cup}(x) = \min(\text{SDF}_A(x), \text{SDF}_B(x))$$
$$\text{SDF}_{\cap}(x) = \max(\text{SDF}_A(x), \text{SDF}_B(x))$$
$$\text{SDF}_{\setminus}(x) = \max(\text{SDF}_A(x), -\text{SDF}_B(x))$$

**平滑最小/最大函数**：
使用LogSumExp近似：
$$\text{smooth-min}(a, b; k) = -\frac{1}{k}\log(e^{-ka} + e^{-kb})$$
$$\text{smooth-max}(a, b; k) = \frac{1}{k}\log(e^{ka} + e^{kb})$$

参数 $k$ 控制平滑程度，$k \to \infty$ 时收敛到真实min/max。

**梯度计算**：
对参数 $\theta$ 的梯度：
$$\frac{\partial \text{SDF}_{\cup}}{\partial \theta} = \begin{cases}
\frac{\partial \text{SDF}_A}{\partial \theta}, & \text{if } \text{SDF}_A < \text{SDF}_B \\
\frac{\partial \text{SDF}_B}{\partial \theta}, & \text{otherwise}
\end{cases}$$

平滑版本的梯度：
$$\frac{\partial \text{smooth-min}}{\partial a} = \frac{e^{-ka}}{e^{-ka} + e^{-kb}}$$

### 20.4.2 参数化曲线/曲面的梯度

计算参数化几何的导数，支持形状优化。

**Bézier曲线的梯度**：
Bézier曲线定义：
$$C(t) = \sum_{i=0}^n B_i^n(t) P_i$$

其中 $B_i^n(t) = \binom{n}{i}t^i(1-t)^{n-i}$ 是Bernstein基函数。

对控制点的梯度：
$$\frac{\partial C(t)}{\partial P_i} = B_i^n(t)$$

对参数的二阶导数（曲率相关）：
$$\frac{\partial^2 C}{\partial t^2} = n(n-1)\sum_{i=0}^{n-2} B_i^{n-2}(t)(P_{i+2} - 2P_{i+1} + P_i)$$

**NURBS曲面的梯度**：
NURBS曲面：
$$S(u,v) = \frac{\sum_{i,j} N_i^p(u)N_j^q(v)w_{ij}P_{ij}}{\sum_{i,j} N_i^p(u)N_j^q(v)w_{ij}}$$

对控制点的梯度（使用商法则）：
$$\frac{\partial S}{\partial P_{ij}} = \frac{N_i^p(u)N_j^q(v)w_{ij}}{W(u,v)}$$

其中 $W(u,v) = \sum_{i,j} N_i^p(u)N_j^q(v)w_{ij}$。

### 20.4.3 隐式函数定理在CAD中的应用

利用隐式函数定理计算约束系统的敏感度。

**约束系统**：
设约束方程组：$F(x, p) = 0$
- $x \in \mathbb{R}^n$：设计变量
- $p \in \mathbb{R}^m$：参数
- $F: \mathbb{R}^{n+m} \rightarrow \mathbb{R}^k$：约束函数

**隐式函数定理**：
若 $\det(\frac{\partial F}{\partial x}) \neq 0$，则存在隐函数 $x = g(p)$，且：
$$\frac{\partial x}{\partial p} = -\left(\frac{\partial F}{\partial x}\right)^{-1} \frac{\partial F}{\partial p}$$

**应用示例**：
草图约束求解：
$$F = \begin{bmatrix}
d(P_1, P_2) - L_1 \\
\angle(L_1, L_2) - \theta \\
P_3 \in L_1
\end{bmatrix} = 0$$

计算点位置对长度参数的敏感度：
$$\frac{\partial P}{\partial L_1} = -J_x^{-1} \frac{\partial F}{\partial L_1}$$

### 20.4.4 反向模式自动微分

实现CAD操作的高效梯度计算。

**计算图构建**：
将CAD程序转换为计算图：
```
x → [Primitive] → y₁ → [Transform] → y₂ → [Boolean] → z
```

**前向传播**：
$$y_1 = f_1(x, \theta_1)$$
$$y_2 = f_2(y_1, \theta_2)$$
$$z = f_3(y_2, \theta_3)$$

**反向传播**：
$$\bar{\theta}_3 = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \theta_3}$$
$$\bar{y}_2 = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial y_2}$$
$$\bar{\theta}_2 = \bar{y}_2 \cdot \frac{\partial y_2}{\partial \theta_2}$$

**矢量-雅可比积(VJP)**：
对于函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$，VJP计算：
$$v^T J_f = v^T \frac{\partial f}{\partial x}$$

避免显式构造雅可比矩阵，提高效率。

**检查点技术**：
对于深层CAD操作序列，使用梯度检查点减少内存：
```
checkpoint_1 → segment_1 → checkpoint_2 → segment_2 → ...
```

仅存储检查点状态，需要时重新计算中间值。

### 20.4.5 设计空间的流形优化

CAD设计空间常具有流形结构，需要特殊的优化方法。

**约束流形**：
设计空间定义为：
$$\mathcal{M} = \{x \in \mathbb{R}^n : g(x) = 0\}$$

其中 $g: \mathbb{R}^n \rightarrow \mathbb{R}^m$ 是约束函数。

**切空间与梯度投影**：
点 $x \in \mathcal{M}$ 处的切空间：
$$T_x\mathcal{M} = \{v : Dg(x) \cdot v = 0\}$$

欧氏梯度投影到切空间：
$$\text{grad}_\mathcal{M} f = \Pi_{T_x\mathcal{M}}(\nabla f) = \nabla f - Dg^T(Dg \cdot Dg^T)^{-1}Dg \cdot \nabla f$$

**测地线与回缩**：
更新步骤需要回缩(retraction)映射 $R_x: T_x\mathcal{M} \rightarrow \mathcal{M}$：
$$x_{k+1} = R_{x_k}(-\alpha_k \cdot \text{grad}_\mathcal{M} f(x_k))$$

对于简单约束，使用投影：
$$R_x(v) = \text{proj}_\mathcal{M}(x + v)$$

**Riemannian优化算法**：
- **Riemannian梯度下降**：
  $$x_{k+1} = R_{x_k}(-\alpha_k \cdot \text{grad}_\mathcal{M} f(x_k))$$
  
- **Riemannian共轭梯度**：
  $$\eta_{k+1} = -\text{grad}_\mathcal{M} f(x_{k+1}) + \beta_k \mathcal{T}_{x_k \to x_{k+1}}(\eta_k)$$
  
  其中 $\mathcal{T}$ 是向量传输算子。

**应用：形状空间优化**：
在形状空间 $\mathcal{S}$ 上定义度量：
$$d_\mathcal{S}(S_1, S_2) = \inf_{\phi} \|\phi\|_{H^1}$$

其中 $\phi: S_1 \rightarrow S_2$ 是微分同胚。使用此度量进行形状插值和优化。

## 20.5 强化学习生成3D程序

### 20.5.1 MDP建模与奖励设计

将3D程序生成建模为马尔可夫决策过程(MDP)。

**MDP定义**：
$$\mathcal{M} = (S, A, T, R, \gamma)$$
- $S$：状态空间（部分程序 + 当前形状）
- $A$：动作空间（DSL操作）
- $T: S \times A \rightarrow \Delta(S)$：状态转移概率
- $R: S \times A \rightarrow \mathbb{R}$：奖励函数
- $\gamma \in [0,1]$：折扣因子

**状态表示**：
$$s_t = (\text{AST}_t, \text{render}(\text{AST}_t), \text{target}, t)$$

包含当前程序树、渲染结果、目标形状和时间步。

**奖励设计**：
多目标奖励函数：
$$R(s, a) = \alpha_1 R_{\text{shape}} + \alpha_2 R_{\text{complexity}} + \alpha_3 R_{\text{constraint}} + \alpha_4 R_{\text{progress}}$$

其中：
- $R_{\text{shape}} = -d(\text{render}(s), \text{target})$：形状匹配奖励
- $R_{\text{complexity}} = -\lambda \cdot |\text{AST}|$：简洁性奖励
- $R_{\text{constraint}} = \mathbb{I}[\text{valid}(s)]$：约束满足奖励
- $R_{\text{progress}} = d(\text{render}(s_{t-1}), \text{target}) - d(\text{render}(s_t), \text{target})$：进步奖励

**稀疏奖励处理**：
使用潜在函数塑形(potential-based shaping)：
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

其中 $\Phi(s)$ 是状态潜在函数，如到目标的距离。

### 20.5.2 策略梯度方法

使用策略梯度直接优化动作选择策略。

**REINFORCE算法**：
策略梯度估计：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) G_t]$$

其中 $G_t = \sum_{k=t}^T \gamma^{k-t} r_k$ 是回报。

**方差减少**：
使用基线函数 $b(s)$：
$$\nabla_\theta J(\theta) = \mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b(s_t))]$$

常用价值函数 $V(s)$ 作为基线。

**PPO(Proximal Policy Optimization)**：
限制策略更新幅度：
$$\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_t[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是重要性采样比。

**A3C(Asynchronous Advantage Actor-Critic)**：
并行训练多个智能体：
```
for each worker:
    collect trajectory τ using π_θ
    compute advantages A_t
    update θ using ∇_θ L^{A3C}
    update global parameters
```

### 20.5.3 蒙特卡洛树搜索(MCTS)

结合MCTS进行程序空间的有效搜索。

**UCT(Upper Confidence Tree)**：
选择动作使用UCB1公式：
$$a^* = \arg\max_a \left[ Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}} \right]$$

其中 $Q(s,a)$ 是动作价值估计，$N$ 是访问次数。

**MCTS步骤**：
1. **选择(Selection)**：从根节点使用UCB选择路径
2. **扩展(Expansion)**：添加新的子节点
3. **模拟(Simulation)**：执行rollout到终止状态
4. **回传(Backpropagation)**：更新路径上的统计信息

**神经网络引导的MCTS**：
使用神经网络提供先验和价值估计：
$$p(a|s), V(s) = f_\theta(s)$$

修改UCB公式：
$$a^* = \arg\max_a \left[ Q(s,a) + c \cdot p(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)} \right]$$

**程序化MCTS**：
- 节点表示部分程序
- 边表示DSL操作
- 使用程序等价性剪枝重复分支
- 缓存子程序执行结果

### 20.5.4 课程学习与探索策略

设计课程逐步增加任务难度。

**课程设计**：
定义任务序列 $\{T_1, ..., T_n\}$，难度递增：
- $T_1$：单个图元
- $T_2$：简单组合（2-3个操作）
- $T_3$：带重复的结构
- $T_n$：复杂层次结构

**自动课程学习**：
根据学习进度调整难度：
$$P(T_i) \propto \exp(-\beta \cdot |R_i - R_{\text{target}}|)$$

其中 $R_i$ 是任务 $T_i$ 的平均奖励。

**探索策略**：
1. **ε-贪婪**：
   $$a = \begin{cases}
   \arg\max_a Q(s,a), & \text{with prob } 1-\epsilon \\
   \text{random}, & \text{with prob } \epsilon
   \end{cases}$$

2. **Boltzmann探索**：
   $$P(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}$$

3. **好奇心驱动**：
   添加内在奖励：
   $$R_{\text{intrinsic}} = \eta \cdot \|\hat{s}_{t+1} - s_{t+1}\|^2$$
   
   其中 $\hat{s}_{t+1}$ 是预测的下一状态。

### 20.5.5 多智能体协作建模

使用多个专门化的智能体协作生成复杂3D模型。

**角色分工**：
- **结构智能体**：负责整体布局和层次结构
- **细节智能体**：添加局部细节和装饰
- **约束智能体**：确保物理和几何约束

**通信机制**：
智能体间消息传递：
$$m_{i \to j} = f_{\text{msg}}(h_i, \text{intent}_i)$$
$$h_j' = g_{\text{update}}(h_j, \{m_{k \to j}\}_{k \neq j})$$

**协调策略**：
使用中央协调器或去中心化协商：
$$\pi_{\text{joint}}(a_1, ..., a_n|s) = \prod_i \pi_i(a_i|s, \{m_{j \to i}\})$$

**信用分配**：
使用反事实推理分配奖励：
$$A_i = R(\tau) - R(\tau_{-i})$$

其中 $\tau_{-i}$ 是移除智能体 $i$ 动作的轨迹。

**层次强化学习**：
- **高层策略**：选择子目标
  $$g \sim \pi_{\text{high}}(g|s)$$
  
- **低层策略**：执行子目标
  $$a \sim \pi_{\text{low}}(a|s, g)$$
  
- **选项框架**：
  $$o = (I, \pi, \beta)$$
  
  其中 $I$ 是初始集，$\pi$ 是策略，$\beta$ 是终止条件。

## 本章小结

本章深入探讨了神经程序合成在3D建模中的应用，涵盖了从基础的程序搜索到高级的强化学习方法。关键概念包括：

1. **程序合成框架**：通过神经引导搜索和库学习自动生成3D建模程序
2. **可微分编程**：将离散的CAD操作转换为可微分形式，实现基于梯度的优化
3. **图神经网络**：利用CAD模型的图结构进行参数预测和约束推理
4. **强化学习**：将程序生成建模为序列决策问题，使用策略梯度和树搜索方法
5. **多智能体协作**：通过角色分工和通信机制生成复杂的3D模型

核心数学工具：
- 贝叶斯推断：$P(p|data) \propto P(data|p) \cdot P(p)$
- 隐式函数定理：$\frac{\partial x}{\partial p} = -(\frac{\partial F}{\partial x})^{-1} \frac{\partial F}{\partial p}$
- 策略梯度：$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) A(s,a)]$
- 消息传递：$h_v^{(k)} = U(h_v^{(k-1)}, \sum_{u \in \mathcal{N}(v)} M(h_u^{(k-1)}, e_{uv}))$

## 练习题

### 基础题

**练习20.1** 证明LogSumExp函数 $f(x_1, ..., x_n) = \log(\sum_i e^{x_i})$ 是凸函数。
<details>
<summary>提示</summary>
计算Hessian矩阵并证明其半正定性。利用 $\frac{\partial^2 f}{\partial x_i \partial x_j} = \delta_{ij}p_i - p_i p_j$，其中 $p_i = \frac{e^{x_i}}{\sum_k e^{x_k}}$。
</details>

<details>
<summary>答案</summary>
Hessian矩阵 $H = \text{diag}(p) - pp^T$，其中 $p$ 是概率向量。对任意 $v$，有 $v^T H v = \sum_i p_i v_i^2 - (\sum_i p_i v_i)^2 = \text{Var}_p(v) \geq 0$，因此 $H \succeq 0$，函数凸。
</details>

**练习20.2** 给定CSG树，推导其体积对叶节点参数的梯度公式。假设叶节点是参数化的图元（如Box(w,h,d)）。
<details>
<summary>提示</summary>
使用链式法则和布尔运算的导数规则。考虑体积积分 $V = \int_{\Omega} \mathbb{I}[x \in S] dx$。
</details>

<details>
<summary>答案</summary>
对于Union操作：$\frac{\partial V_{A \cup B}}{\partial \theta} = \frac{\partial V_A}{\partial \theta} + \frac{\partial V_B}{\partial \theta} - \frac{\partial V_{A \cap B}}{\partial \theta}$。使用包含-排斥原理递归计算。
</details>

**练习20.3** 设计一个简单的DSL用于生成2D多边形，包括基本操作和组合规则。计算生成n边正多边形所需的最少操作数。
<details>
<summary>提示</summary>
考虑旋转对称性和递归结构。DSL可包含Line、Rotate、Repeat等操作。
</details>

<details>
<summary>答案</summary>
DSL: `Polygon(n) = Repeat(n, λi. Line(1) + Rotate(2π/n))`。最少操作数：O(log n)，使用分治策略。
</details>

### 挑战题

**练习20.4** 考虑程序合成中的组合爆炸问题。给定DSL有k种操作，每个操作最多有m个参数，每个参数有n个可能值。证明深度为d的程序空间大小的上界，并提出一种剪枝策略。
<details>
<summary>提示</summary>
使用树的计数方法。考虑语义等价性和交换律等对称性。
</details>

<details>
<summary>答案</summary>
程序空间大小上界：$O((k \cdot n^m)^{2^d - 1})$。剪枝策略：1) 规范化表示（如交换操作排序）；2) 记忆化等价程序；3) 使用抽象解释预测程序效果。
</details>

**练习20.5** 推导图神经网络在CAD约束传播中的不动点条件。假设约束图是强连通的，证明消息传递算法的收敛性。
<details>
<summary>提示</summary>
使用Banach不动点定理。定义适当的度量空间和压缩映射。
</details>

<details>
<summary>答案</summary>
定义映射 $T: h \mapsto \sigma(W_1 h + W_2 \sum_{j \in \mathcal{N}} h_j)$。若 $\|W_1\| + |\mathcal{N}| \cdot \|W_2\| < 1$（谱范数），则T是压缩映射，存在唯一不动点。
</details>

**练习20.6** 分析强化学习生成3D程序时的样本复杂度。给定ε-δ PAC学习框架，推导需要多少样本才能学到ε-最优策略。
<details>
<summary>提示</summary>
使用Hoeffding不等式和union bound。考虑状态-动作空间的覆盖。
</details>

<details>
<summary>答案</summary>
样本复杂度：$O(\frac{|S| \cdot |A| \cdot H^3}{\epsilon^2} \log \frac{|S| \cdot |A|}{\delta})$，其中H是horizon。对于连续空间，使用覆盖数替代。
</details>

**练习20.7**（开放问题）设计一个神经-符号混合系统，结合程序合成和端到端学习的优势。系统应能：1) 从少量示例学习；2) 生成可解释的程序；3) 满足硬约束。讨论架构设计和训练策略。
<details>
<summary>提示</summary>
考虑分层架构：底层神经网络提取特征，中层符号推理，顶层程序生成。使用可微分的符号执行器。
</details>

**练习20.8**（研究方向）探讨如何将因果推理引入程序合成。当修改3D模型的某个部分时，如何预测对其他部分的影响？设计一个因果图模型并讨论干预和反事实推理。
<details>
<summary>提示</summary>
构建结构因果模型(SCM)，节点表示程序组件，边表示因果关系。使用do-calculus进行干预分析。
</details>

## 常见陷阱与错误 (Gotchas)

### 1. 梯度消失/爆炸
- **问题**：深层程序树的梯度传播不稳定
- **解决**：使用梯度裁剪、层归一化、残差连接

### 2. 离散-连续鸿沟
- **问题**：离散的程序操作难以优化
- **解决**：使用Gumbel-Softmax、REINFORCE with baseline、连续松弛

### 3. 搜索空间爆炸
- **问题**：程序空间指数增长
- **解决**：使用启发式剪枝、beam search、学习的先验

### 4. 约束违反
- **问题**：神经网络生成的程序可能违反硬约束
- **解决**：投影到可行域、拉格朗日松弛、约束层设计

### 5. 样本效率低
- **问题**：强化学习需要大量交互
- **解决**：使用专家演示、模型基强化学习、迁移学习

### 6. 局部最优
- **问题**：非凸优化容易陷入局部最优
- **解决**：多起点优化、模拟退火、进化策略

## 最佳实践检查清单

### 系统设计
- [ ] DSL设计是否覆盖目标域的关键操作？
- [ ] 是否考虑了程序的可组合性和模块化？
- [ ] 约束系统是否完备且一致？
- [ ] 是否设计了合适的抽象层次？

### 神经架构
- [ ] 编码器是否捕获了程序和形状的关键特征？
- [ ] 是否使用了适合的归纳偏置（如等变性）？
- [ ] 网络容量是否与任务复杂度匹配？
- [ ] 是否考虑了计算效率和内存占用？

### 优化策略
- [ ] 损失函数是否平衡了多个目标？
- [ ] 是否使用了合适的正则化技术？
- [ ] 学习率调度是否合理？
- [ ] 是否监控了梯度流和收敛性？

### 搜索算法
- [ ] 搜索策略是否平衡了探索和利用？
- [ ] 是否利用了问题结构进行剪枝？
- [ ] 是否缓存了中间结果？
- [ ] 终止条件是否合理？

### 评估指标
- [ ] 是否评估了程序的正确性？
- [ ] 是否测量了生成效率？
- [ ] 是否考虑了程序的可解释性？
- [ ] 是否进行了消融实验？

### 工程实践
- [ ] 是否实现了增量式计算？
- [ ] 是否处理了数值稳定性问题？
- [ ] 是否设计了容错机制？
- [ ] 是否优化了关键路径的性能？
