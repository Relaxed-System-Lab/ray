# Adaptation Layer 设计文档（vLLM）

## 目标与范围
- 目标：在 vLLM 算子上实现论文中的 adaptation layer，通过 workload 特征与观测吞吐，在线聚类 + 配置调优 + 策略切换。
- 范围：仅对算子名包含 "vllm"（大小写不敏感）启用。
- 非目标：不实现 placement layer；不实现非 vLLM 算子的适配逻辑。

## 数据流与集成点
### 总体流程（每次 DS2 autoscaler 迭代）
1) 计算 raw throughput（delta_rows / delta_wall_time）。
2) vLLM 算子进入 observation layer，得到过滤后的可持续吞吐（observed throughput）。
3) 从 metrics 或输出队列提取 workload features（vLLM 输入/输出 token 的均值/方差）。
4) 将 features + observed throughput 提交给 adaptation layer：
   - 聚类匹配/更新；
   - 判断是否触发调优；
   - 判断是否要切换到已调优的配置。
5) 若有可用配置，调用 op.apply_adaptive_config(config) 进行热切换；成功后重置 observation layer。

### 接口与依赖
- 在 `ds2_autoscaler.py`：
  - `self._adaptation_layer = VLLMAdaptationLayer()`
  - `if self._is_vllm_op(op): ...` 插入特征提取与适配逻辑。
- 在 `map_operator.py`：
  - 采集 vLLM 输出 token 统计（均值/方差）。
- 在 `actor_pool_map_operator.py`：
  - 通过 `_extra_metrics` 暴露 vLLM token 统计给 autoscaler。

## 架构与时序
### 组件关系（概览）
```
Ray Data Pipeline
  └─ ActorPoolMapOperator (VideoCaptionVLLM)
      ├─ MapOperator: token stats 采集
      ├─ _extra_metrics: vLLM token stats 输出
      └─ DS2 Autoscaler
           ├─ Observation Layer (吞吐滤波)
           └─ Adaptation Layer
                ├─ Feature Extractor
                ├─ Online Clustering
                ├─ BO Tuning (可选)
                └─ Config Switch (apply_adaptive_config)
```

### 时序（每次 autoscaling 迭代）
```
DS2 Autoscaler
  ├─ compute raw throughput
  ├─ Observation.observe(raw_throughput, queue_size, util, ...)
  ├─ FeatureExtractor.extract(op, op_state)
  ├─ Adaptation.observe(features, observed_throughput)
  │    ├─ assign/update cluster
  │    ├─ maybe trigger tuning
  │    └─ maybe emit switch decision
  └─ if decision:
       ├─ op.apply_adaptive_config(config)
       └─ Observation.reset()
```

## Workload Feature 设计
### 特征定义
- mean_input_tokens
- var_input_tokens
- mean_output_tokens
- var_output_tokens

### 提取方式（优先级）
1) **metrics.extra_metrics**
   - `vllm_input_tokens_mean`
   - `vllm_input_tokens_var`
   - `vllm_output_tokens_mean`
   - `vllm_output_tokens_var`
2) **输出队列采样（fallback）**
   - 从 operator output queue / op_state output queue 抽样 block
   - 解析 payload 的 `num_input_tokens` / `num_generated_tokens`

### 统计方式
- 采用在线 Welford 算法维护均值/方差。
- 特征向量在聚类前做 `log1p` 变换，降低极值影响。

## 聚类与调优逻辑
### 聚类（在线）
- 使用欧氏距离在特征空间匹配或新建 cluster。
- 关键参数：
  - max_clusters=6
  - distance_threshold=1.2
  - decay_gamma=0.995
  - decay_interval_s=300
  - centroid_drift_threshold=0.4

### 调优触发
- 触发条件：
  - cluster.count >= min_samples (默认 5)
  - cluster 占比 >= min_cluster_fraction (默认 0.1)
  - 未处于 tuning 或 tuning 冷却期
  - tuned 后记录 config 与最后 tuned centroid。

### 调优器（BO）
- 优先使用 sklearn GP (Matern kernel) 进行 EI 采样。
- sklearn 不可用时回退随机搜索。
- 参数空间（参考 SCOOT）：
  - max_num_seqs
  - max_num_batched_tokens
  - block_size
  - scheduler_delay_factor
  - enable_chunked_prefill
  - enable_prefix_caching
  - disable_custom_all_reduce
  - use_v2_block_manager

### 配置切换
- 使用 match history 统计 dominant cluster。
- 满足一致性阈值后，若该 cluster 已 tuned 且 config 存在，尝试 apply。
- 成功后调用 observation layer reset，避免旧配置影响新吞吐估计。

## 关键日志
- `vLLM features (metrics)`：特征来源/数值
- `vLLM adaptation: created initial cluster` / `created new cluster`：聚类变化
- `vLLM adaptation: tuning triggered` / `Tuned config`：调优状态
- `vLLM adaptation applied`：配置切换成功

## 可配置项（常用）
- `min_samples`：触发 tuning 的样本阈值（默认 5）。
- `decay_interval_s / decay_gamma`：样本衰减周期与系数。
- `distance_threshold`：新建 cluster 的距离阈值。
- `cooldown_s / tuning_cooldown_s`：切换与调优冷却时间。
- `prune_threshold`：衰减后保留 cluster 的最小 count。

## 调参建议（面向“可触发 tuning”）
假设 autoscaler 约 60s 触发一次观察：
- **min_samples**：建议 3~5（20 会导致很难触发）。
- **decay_interval_s**：建议 ≥ 300（避免每分钟衰减并清空）。
- **decay_gamma**：建议 0.995~1.0（采样稀疏时更稳定）。
- **prune 门槛**：建议从 `<1` 放宽到 `<0.1`（避免新 cluster 立刻被删）。
- **cooldown_s**：若需要更快切换，可降到 30~60；不需要频繁切换则保持默认。

> 触发 tuning 的关键不是“单次特征质量”，而是“样本数量能累积到门槛”。

## 当前采用的配置（本次实验）
- min_samples=5
- distance_threshold=1.2
- decay_interval_s=300
- decay_gamma=0.995
- prune_threshold=0.1
- tuning_cooldown_s=60
- cooldown_s=60

## 失败与容错
- 当 metrics 不含 token 统计时，fallback 到输出队列采样。
- 输出队列为空时，记录 `vLLM feature snapshot missing`。
- apply_adaptive_config 不存在或失败时，仅记录日志，不中断。

## 性能与安全
- 采样限制：max_blocks=2, max_rows=256（避免阻塞与开销过大）。
- 仅对 vLLM op 生效，避免影响其他算子。
- 不改变 DS2 solver 逻辑与 placement 行为。
