# 对MILP的进一步修改

# 考虑Queue size

## Queue size越大，并发度越大

$$
\begin{aligned}
\max \quad & \frac{\tau}{\tau^{ref}} + \beta \cdot \frac{\sum_{i=1}^{n} d_i}{D^{ref}} \\[2ex]
\text{s.t.} \quad & \tau + \frac{D_o}{D_i} \cdot \frac{d_i}{T} \leq \frac{D_o}{D_i} \cdot p_i \cdot UT_i, && \forall i \\[1ex]
& 0 \leq d_i \leq \max(0, Q_i - Q_i^{target}), && \forall i \\[1ex]
& \sum_{i=1}^{n} u_i \cdot p_i \leq N_{cpu} \\[1ex]
& \sum_{i=1}^{n} m_i \cdot p_i \leq M_{mem} \\[1ex]
& \sum_{i=1}^{n} g_i \cdot p_i \leq N_{gpu} \\[1ex]
& p_i \in \mathbb{Z}^+, \quad d_i \geq 0, \quad \tau \geq 0
\end{aligned}
$$

归一化参考值的选取

$$
\tau^{ref} = \min_{i} \left( \frac{D_o}{D_i} \cdot UT_i \cdot p_i^{max} \right)
$$

$$
D^{ref} = \sum_{i=1}^{n} \max(0, Q_i - Q_i^{target})
$$

其中 $p_i^{max}$ 是算子 $i$ 在资源约束下的理论最大并行度。

## β 的含义

| β\betaβ | 行为 |
| --- | --- |
| 0 | 只优化吞吐量，忽略队列消化 |
| 0.5 | 吞吐量权重是队列消化的 2 倍 |
| 1 | 两者同等重要 |
| 2 | 队列消化权重是吞吐量的 2 倍 |

## 让Q_{target} 为0

### 简化后的模型

$$
\begin{aligned}
\max \quad & \frac{\tau}{\tau^{ref}} + \beta \cdot \frac{\sum_{i=1}^{n} d_i}{D^{ref}} \\[2ex]
\text{s.t.} \quad & \tau + \frac{D_o}{D_i} \cdot \frac{d_i}{T} \leq \frac{D_o}{D_i} \cdot p_i \cdot UT_i, && \forall i \\[1ex]
& 0 \leq d_i \leq Q_i, && \forall i \\[1ex]
& \sum_{i=1}^{n} u_i \cdot p_i \leq N_{cpu} \\[1ex]
& \sum_{i=1}^{n} m_i \cdot p_i \leq M_{mem} \\[1ex]
& \sum_{i=1}^{n} g_i \cdot p_i \leq N_{gpu} \\[1ex]
& p_i \in \mathbb{Z}^+, \quad d_i \geq 0, \quad \tau \geq 0
\end{aligned}
$$

### 归一化参考值

$\tau^{ref} = \min_{i} \left( \frac{D_o}{D_i} \cdot UT_i \cdot p_i^{max} \right)$

$D^{ref} = \sum_{i=1}^{n}$

简化后目标就是尽可能把队列清空到 0，约束也更简洁了。

## 维持queue size在固定范围

### 决策变量

- $p_i \in \mathbb{Z}^+$：算子 i 的并行度
- $\tau \geq 0$：系统吞吐量
- $\delta_i^+, \delta_i^- \geq 0$：缓冲区水位偏离目标的正负偏差

### 完整 MILP 公式

$$
\begin{aligned}
\max \quad & \tau - \alpha \sum_{i=2}^{n} (\delta_i^+ + \delta_i^-) \\[2ex]
\text{s.t.} \quad & \tau \leq \frac{D_o}{D_i} \cdot p_i \cdot UT_i, && \forall i = 1, \ldots, n \\[1ex]
& \sum_{i=1}^{n} u_i \cdot p_i \leq N_{cpu} \\[1ex]
& \sum_{i=1}^{n} m_i \cdot p_i \leq M_{cpu} \\[1ex]
& \sum_{i=1}^{n} n_i \cdot p_i \leq N_{gpu} \\[1ex]
& B_i^{current} + T \cdot \left( \frac{D_i}{D_{i-1}} \cdot p_{i-1} \cdot UT_{i-1} - p_i \cdot UT_i \right) - B_i^{target} = \delta_i^+ - \delta_i^-, && \forall i = 2, \ldots, n \\[1ex]
& p_i \in \mathbb{Z}^+, && \forall i = 1, \ldots, n \\[1ex]
& \delta_i^+, \delta_i^- \geq 0, && \forall i = 2, \ldots, n \\[1ex]
& \tau \geq 0
\end{aligned}
$$

### 模型行为

当缓冲区水位高于目标时（$B_i^{current} > B_i^{target}$），模型会倾向于增加 $p_i$（加速消费）或减少 $p_{i-1}$（减缓生产），使 $\delta_i^+$减小。

当缓冲区水位低于目标时（ $B_i^{current} < B_i^{target}$ ），模型会倾向于减少 $p_i$ 或增加 $p_{i-1}$，使 $\delta_i^-$减小。

$\alpha$ 的大小决定了系统在吞吐量和缓冲区稳定性之间的权衡。

### 多目标归一化

1. 相对偏差+加权

$$
\begin{aligned}
\max \quad & \frac{\tau}{\color{red}\tau^{ref}} - \alpha \sum_{i=2}^{n} {\color{red}w_i} \cdot \frac{\delta_i^+ + \delta_i^-}{\color{red}B_i^{target}} \\[2ex]
\text{s.t.} \quad & \tau \leq \frac{D_o}{D_i} \cdot p_i \cdot UT_i, && \forall i = 1, \ldots, n \\[1ex]
& \sum_{i=1}^{n} u_i \cdot p_i \leq N_{cpu} \\[1ex]
& \sum_{i=1}^{n} m_i \cdot p_i \leq M_{cpu} \\[1ex]
& \sum_{i=1}^{n} n_i \cdot p_i \leq N_{gpu} \\[1ex]
& B_i^{current} + T \cdot \left( \frac{D_i}{D_{i-1}} \cdot p_{i-1} \cdot UT_{i-1} - p_i \cdot UT_i \right) - B_i^{target} = \delta_i^+ - \delta_i^-, && \forall i = 2, \ldots, n \\[1ex]
& p_i \in \mathbb{Z}^+, && \forall i = 1, \ldots, n \\[1ex]
& \delta_i^+, \delta_i^- \geq 0, && \forall i = 2, \ldots, n \\[1ex]
& \tau \geq 0
\end{aligned}
$$

对于 $w_i$的加权，我认为应该按照真实运行时间的比例来分配，真实运行时间越大，说明该算子运行成本越高，越应该要尽可能充分利用，少处于空闲状态。

其中 $\tau^{ref}$ 是参考吞吐量（比如不考虑缓冲区约束时的最优吞吐量，或者历史平均吞吐量）

1. 时间尺度统一

$$
\begin{aligned}
\max \quad & \tau - \alpha \sum_{i=2}^{n}  \frac{\delta_i^+ + \delta_i^-}{\color{red}T} \\[2ex]
\text{s.t.} \quad & \tau \leq \frac{D_o}{D_i} \cdot p_i \cdot UT_i, && \forall i = 1, \ldots, n \\[1ex]
& \sum_{i=1}^{n} u_i \cdot p_i \leq N_{cpu} \\[1ex]
& \sum_{i=1}^{n} m_i \cdot p_i \leq M_{cpu} \\[1ex]
& \sum_{i=1}^{n} n_i \cdot p_i \leq N_{gpu} \\[1ex]
& B_i^{current} + T \cdot \left( \frac{D_i}{D_{i-1}} \cdot p_{i-1} \cdot UT_{i-1} - p_i \cdot UT_i \right) - B_i^{target} = \delta_i^+ - \delta_i^-, && \forall i = 2, \ldots, n \\[1ex]
& p_i \in \mathbb{Z}^+, && \forall i = 1, \ldots, n \\[1ex]
& \delta_i^+, \delta_i^- \geq 0, && \forall i = 2, \ldots, n \\[1ex]
& \tau \geq 0
\end{aligned}
$$

此时 $\frac{\delta_i}{T}$ 的单位变成"记录数/秒"，与 $\tau$ 同量纲。 $\alpha$ 的含义是：每单位缓冲区偏离速率对应多少吞吐量损失。