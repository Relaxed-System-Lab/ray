# 未触发 Tuning 的问题分析与总结

## 现象
- adaptation layer 日志中持续出现 “created initial cluster”。
- 没有出现 “tuning triggered / Tuned config / adaptation applied”。

## 复现步骤（本次 run）
1) 确保 GPU 上只保留 `/usr/lib/xorg/Xorg`，其余进程全部 kill（nvidia-smi + kill -9）。
2) 运行 pipeline：  
   `python -u -m autoscaler_experiments.pipelines.pipeline_v4_multi_gpu_stages -c /home/panding/Code/ray_for_opensora/file_version/src/autoscaler_experiments/configs/config_v4_splitter_7.yaml -a ds2`
3) 观察日志：  
   `/tmp/ray/session_2026-02-03_14-30-50_346209_1144731/logs/ray-data/ray-data.log`

## 根因分析
1) **采样频率过低**
   - adaptation 只在 DS2 autoscaler 迭代时触发，一般约 60s 一次。
   - 每次只有 1 个样本进入 cluster。

2) **decay + prune 使 cluster 无法累积样本**
   - 每 60s 触发一次 decay：`count *= 0.98`。
   - 接着 prune：`count < 1` 的 cluster 会被删除。
   - 结果：新建 cluster count=1 → decay 后变 0.98 → 立即被清除。
   - 下次迭代又重新创建初始 cluster。

3) **tuning 门槛过高**
   - tuning 需要 `count >= min_samples (默认 20)` 且占比 >= 0.1。
   - cluster 每次都被 prune，永远无法达到 20。

## 日志摘录（关键证据）
来自：`/tmp/ray/session_2026-02-03_14-30-50_346209_1144731/logs/ray-data/ray-data.log`
```
2026-02-03 14:33:01,393 INFO adaptation_layer.py:333 -- vLLM features (metrics) for MapBatches(VideoCaptionVLLM): mean_in=1544.58 var_in=3738.78 mean_out=132.31 var_out=1244.80
2026-02-03 14:33:01,394 INFO adaptation_layer.py:675 -- vLLM adaptation: created initial cluster 0 for MapBatches(VideoCaptionVLLM)
2026-02-03 14:34:01,402 INFO adaptation_layer.py:333 -- vLLM features (metrics) for MapBatches(VideoCaptionVLLM): mean_in=1545.53 var_in=3459.97 mean_out=134.79 var_out=1599.89
2026-02-03 14:34:01,403 INFO adaptation_layer.py:675 -- vLLM adaptation: created initial cluster 1 for MapBatches(VideoCaptionVLLM)
```
同一 run 内未出现 `tuning triggered / Tuned config / vLLM adaptation applied` 日志。

## 证据摘要（本次 run）
- 每分钟输出一次 “vLLM features (metrics)” 与 “created initial cluster”。
- 未出现 tuning / switch 相关日志。

## 本次实验结果（新配置）
使用配置：`min_samples=5, decay_interval_s=300, decay_gamma=0.995, prune_threshold=0.1, cooldown_s=60, tuning_cooldown_s=60`。

关键日志（新 run）：
```
2026-02-04 17:16:57,227 INFO adaptation_layer.py:333 -- vLLM features (metrics) for MapBatches(VideoCaptionVLLM): mean_in=1545.11 var_in=3584.14 mean_out=137.07 var_out=1223.37
2026-02-04 17:16:57,228 INFO adaptation_layer.py:677 -- vLLM adaptation: created initial cluster 0 for MapBatches(VideoCaptionVLLM)
2026-02-04 17:18:57,400 INFO adaptation_layer.py:695 -- vLLM adaptation: created new cluster 1 for MapBatches(VideoCaptionVLLM) (distance=0.699)
2026-02-04 17:23:57,745 INFO adaptation_layer.py:695 -- vLLM adaptation: created new cluster 2 for MapBatches(VideoCaptionVLLM) (distance=1.078)
```
该 run 仍未出现 `tuning triggered / Tuned config / vLLM adaptation applied`。

初步判断：虽然降低了 min_samples 并放宽 decay，但 **distance_threshold=0.6** 导致特征漂移时频繁新建 cluster，使得单个 cluster 的样本数仍未达到 5。

## 结论
- tuning 没触发不是逻辑错误，而是 **采样频率 + decay 策略** 组合导致 cluster 无法积累样本。

## 可选修复方向（任选其一）
1) **降低 tuning 触发门槛**
   - `min_samples` 调小（例如 3 或 5）。

2) **放宽/关闭 decay 与 prune**
   - 增大 `decay_interval_s`（例如 300s）。
   - 将 `decay_gamma` 提高到 1.0。
   - 将 prune 门槛从 `<1` 改为 `<0.1`。

3) **提高采样频率**
   - 缩短 autoscaler 迭代周期（更多 samples）。

## 参数影响速查表
| 参数 | 作用 | 对 tuning 触发的影响 | 推荐（20 分钟内可触发） |
| --- | --- | --- | --- |
| min_samples | cluster 需要的最小样本数 | 过大则永远触发不了 | 3~5 |
| decay_interval_s | 计数衰减周期 | 太短会频繁清空 cluster | ≥ 300 |
| decay_gamma | 衰减系数 | 过小导致 count 快速下降 | 0.995~1.0 |
| prune 门槛 | 删除小 count cluster | `<1` 会立刻删新 cluster | `<0.1` |
| autoscaler 周期 | 采样频率 | 周期越短样本越快累积 | 30~60s |

## 推荐参数组合（按目标）
1) **快速触发 tuning（开发验证用）**
   - min_samples=3
   - decay_interval_s=300
   - decay_gamma=1.0
   - prune < 0.1

2) **稳定聚类 + 可触发 tuning（默认推荐）**
   - min_samples=5
   - decay_interval_s=300
   - decay_gamma=0.995
   - prune < 0.1

3) **保守模式（减少误触发）**
   - min_samples=10
   - decay_interval_s=600
   - decay_gamma=0.99
   - prune < 0.1

## 实验对比模板（建议记录）
```
Run ID:
Date:
Pipeline:
Autoscaler period:

Params:
  min_samples:
  decay_interval_s:
  decay_gamma:
  prune threshold:

Result:
  tuning_triggered: (Y/N)
  tuned_config_count:
  adaptation_applied: (Y/N)
  final_count:
  notes:
```

## 评估建议
- 若目标是“在一次 20 分钟 pipeline 内触发 tuning”，优先选：
  - `min_samples=3~5` + `decay_interval_s>=300`。
- 变更后观察日志中是否出现：
  - `tuning triggered` / `Tuned config` / `vLLM adaptation applied`。
