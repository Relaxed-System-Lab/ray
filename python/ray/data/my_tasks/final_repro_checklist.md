# 最终复现与验收清单（2026-02-08）

## 1) 环境与目录
- 激活环境：`conda activate opensora_paperversion`
- 进入目录：`cd /home/panding/Code/ray_for_opensora_paper_version/file_version/src`
- 说明：`opensora_paperversion` 里的 `ray.data` 已软链接到当前 workspace。

## 2) 运行命令（建议带日志落盘）
```bash
RAY_DATA_DISABLE_PROGRESS_BARS=1 \
python -u -m autoscaler_experiments.pipelines.pipeline_v4_multi_gpu_stages \
  -c /home/panding/Code/ray_for_opensora/file_version/src/autoscaler_experiments/configs/config_v4_splitter_7.yaml \
  -a ds2 \
  2>&1 | tee /home/panding/Code/ray_2_48/python/ray/data/my_tasks/logs/pipeline_$(date +%Y%m%d_%H%M%S)_final_repro.log
```

## 3) 验收关键点（日志 grep）
```bash
LOG=/home/panding/Code/ray_2_48/python/ray/data/my_tasks/logs/<your_log>.log

# actor 粒度 observe + 调参/rollout流程
rg -n "probe result\\(actor-attributed\\)|tuning completed|rollout start|scope=rollout" "$LOG"

# 安全约束生效（忽略高风险 key + 裁剪 max_num_seqs）
rg -n "Adaptive config keys ignored|Adaptive config adjusted" "$LOG"

# 成功结束信号
rg -n "execution finished|Final count|Stats:" "$LOG"

# 失败信号（应为空）
rg -n "ActorDiedError|Engine core initialization failed|No common block size|No available memory for the cache blocks|Dataset dataset_11_0 execution failed" "$LOG"
```

## 4) 本次已验证成功日志
- `my_tasks/logs/pipeline_20260208_191215_after_key_whitelist.log`

关键证据（可直接点开对应行）：
- `my_tasks/logs/pipeline_20260208_191215_after_key_whitelist.log:411`
- `my_tasks/logs/pipeline_20260208_191215_after_key_whitelist.log:469`
- `my_tasks/logs/pipeline_20260208_191215_after_key_whitelist.log:528`
- `my_tasks/logs/pipeline_20260208_191215_after_key_whitelist.log:529`
- `my_tasks/logs/pipeline_20260208_191215_after_key_whitelist.log:536`
- `my_tasks/logs/pipeline_20260208_191215_after_key_whitelist.log:912`
- `my_tasks/logs/pipeline_20260208_191215_after_key_whitelist.log:914`

## 5) 相关代码提交
- `6a93c411a1`：`[data] harden vLLM adaptive config with baseline safety gates`
- `794e0ef105`：`[data] enforce vLLM max_num_batched_tokens constraint`
- `9ed86ab34f`：`[data] add actor-attributed vLLM adaptation flow`
