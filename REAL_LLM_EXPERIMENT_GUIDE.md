# Real LLM Experiment Guide

本指南详细说明如何使用真实 LLM API（如阿里云 DashScope Qwen3-32B）运行火箭发射排程实验。

---

## 1. 环境变量设置

### 1.1 API 密钥配置

**推荐方式**：使用环境变量存储 API 密钥，避免硬编码。

```powershell
# Windows PowerShell（临时）
$env:DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# Windows PowerShell（永久，当前用户）
[Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxx", "User")

# Windows CMD
set DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Linux / macOS
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

**验证密钥已设置**：

```powershell
# PowerShell
echo $env:DASHSCOPE_API_KEY

# CMD
echo %DASHSCOPE_API_KEY%

# Linux/macOS
echo $DASHSCOPE_API_KEY
```

### 1.2 其他环境变量（可选）

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API 密钥 | 无（必须设置） |
| `OPENAI_API_KEY` | OpenAI 兼容 API 密钥 | 无 |
| `LLM_CACHE_DIR` | 缓存目录路径 | `.llm_cache` |

---

## 2. Base URL 与 Model 配置

### 2.1 阿里云 DashScope（Qwen 系列）

```bash
# 默认配置（已内置）
--llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1
--llm-model qwen3-32b
--llm-key-env DASHSCOPE_API_KEY
```

### 2.2 OpenAI 官方

```bash
--llm-base-url https://api.openai.com/v1
--llm-model gpt-4o
--llm-key-env OPENAI_API_KEY
```

### 2.3 本地部署（Ollama / vLLM）

```bash
# Ollama
--llm-base-url http://localhost:11434/v1
--llm-model qwen2.5:32b
--llm-key-env OLLAMA_API_KEY  # 可设为任意值

# vLLM
--llm-base-url http://localhost:8000/v1
--llm-model Qwen/Qwen2.5-32B-Instruct
--llm-key-env VLLM_API_KEY
```

### 2.4 其他 OpenAI 兼容服务

| 服务商 | Base URL | 说明 |
|--------|----------|------|
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/deployments/<deployment>` | 需额外配置 |
| DeepSeek | `https://api.deepseek.com/v1` | 兼容 OpenAI 格式 |
| Moonshot | `https://api.moonshot.cn/v1` | 兼容 OpenAI 格式 |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | 兼容 OpenAI 格式 |

---

## 3. 缓存配置

### 3.1 启用缓存（默认）

缓存可确保相同输入返回相同输出，保证实验可复现。

```bash
# 默认启用，缓存目录为 .llm_cache
python run_experiments.py --policy llm_real ...

# 指定缓存目录
python run_experiments.py --policy llm_real --llm-cache-dir my_cache ...
```

### 3.2 关闭缓存

关闭缓存会导致每次调用都请求 API，增加成本且结果不可复现。

```bash
python run_experiments.py --policy llm_real --llm-no-cache ...
```

### 3.3 缓存机制说明

| 特性 | 说明 |
|------|------|
| 缓存键 | `hash(system_prompt + user_prompt + model + temperature)` |
| 缓存格式 | JSON 文件，按哈希值分目录存储 |
| 缓存位置 | `.llm_cache/<hash[:2]>/<hash>.json` |
| 缓存命中 | 跳过 API 调用，直接返回缓存结果 |
| 缓存失效 | 手动删除缓存文件或目录 |

### 3.4 查看缓存统计

```bash
# 统计缓存文件数量
Get-ChildItem -Recurse .llm_cache -Filter *.json | Measure-Object

# 查看缓存大小
(Get-ChildItem -Recurse .llm_cache | Measure-Object -Sum Length).Sum / 1MB
```

---

## 4. 运行实验

### 4.1 单次测试（验证配置）

```bash
# 最小化测试：1 个 seed，1 个扰动级别
python run_experiments.py \
    --policy llm_real \
    --seeds 42 \
    --disturbance-levels medium \
    --output-dir logs/test_llm_real \
    --llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --llm-model qwen3-32b \
    --llm-key-env DASHSCOPE_API_KEY \
    --llm-temperature 0.0 \
    --llm-timeout 60
```

**预期输出**：
- `logs/test_llm_real/results_per_episode.csv`
- `logs/test_llm_real/episode_42_medium_llm_real/` 目录

### 4.2 快速验证（3 seeds）

```bash
python run_experiments.py \
    --policy llm_real \
    --seeds 42 123 456 \
    --disturbance-levels light medium heavy \
    --output-dir logs/quick_test \
    --llm-model qwen3-32b \
    --llm-key-env DASHSCOPE_API_KEY
```

### 4.3 全套实验（与 baseline 对比）

```bash
# 步骤 1：运行所有 baseline 策略（可并行）
python run_experiments.py \
    --policies fixed greedy full_unlock mockllm \
    --seeds 1 2 3 4 5 6 7 8 9 10 \
    --disturbance-levels light medium heavy \
    --output-dir results/full_experiment \
    --workers 4

# 步骤 2：运行 llm_real（强制串行）
python run_experiments.py \
    --policy llm_real \
    --seeds 1 2 3 4 5 6 7 8 9 10 \
    --disturbance-levels light medium heavy \
    --output-dir results/full_experiment \
    --llm-model qwen3-32b \
    --llm-key-env DASHSCOPE_API_KEY \
    --llm-temperature 0.0

# 步骤 3：分析结果
python analyze.py \
    --input results/full_experiment \
    --output results/full_experiment/figures
```

### 4.4 单 Episode 调试

```bash
python run_one_episode.py \
    --seed 42 \
    --disturbance medium \
    --policy llm_real \
    --output-dir logs/debug_episode \
    --llm-model qwen3-32b \
    --llm-key-env DASHSCOPE_API_KEY \
    --verbose
```

---

## 5. 常见错误与处理

### 5.1 HTTP 429 - Rate Limit Exceeded

**错误信息**：
```
openai.RateLimitError: Error code: 429 - Rate limit exceeded
```

**原因**：API 请求频率超过限制（通常 TPM/RPM 限制）

**解决方案**：

```bash
# 方案 1：增加重试间隔（内置指数退避）
--llm-timeout 120  # 增加超时时间，给重试留空间

# 方案 2：降低并发（已自动强制 workers=1）
# llm_real 策略自动串行执行

# 方案 3：使用缓存减少重复请求
# 确保未使用 --llm-no-cache

# 方案 4：联系供应商提升配额
```

**DashScope 限流参考**（2024 年）：

| 模型 | RPM | TPM |
|------|-----|-----|
| qwen-turbo | 500 | 100K |
| qwen-plus | 200 | 50K |
| qwen-max | 100 | 30K |
| qwen3-32b | 60 | 20K |

### 5.2 Timeout Error

**错误信息**：
```
openai.APITimeoutError: Request timed out
httpx.ReadTimeout: timed out
```

**原因**：API 响应时间超过设定的超时阈值

**解决方案**：

```bash
# 增加超时时间
--llm-timeout 120  # 默认 60 秒，可增至 120 秒

# 检查网络连接
ping dashscope.aliyuncs.com

# 使用代理（如需）
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
```

### 5.3 JSON 解析失败

**错误信息**：
```
json.JSONDecodeError: Expecting property name enclosed in double quotes
LLM returned invalid JSON, falling back to default
```

**原因**：LLM 返回的内容不是有效 JSON 格式

**处理机制**：
1. 系统自动使用 `fix_json()` 尝试修复常见错误
2. 修复失败则触发 fallback 策略（使用默认参数）
3. fallback 事件记录在 `llm_fallback_count` 指标中

**预防措施**：

```bash
# 使用低 temperature 提高输出稳定性
--llm-temperature 0.0

# 使用更强大的模型
--llm-model qwen-max  # 比 qwen-turbo 更稳定
```

**手动检查 LLM 输出**：

```bash
# 查看 LLM 决策日志
type logs\episode_42_llm_real\llm_decisions.jsonl

# 检查 fallback 次数
python -c "import json; data=open('logs/episode_42_llm_real/episode_summary.json').read(); print(json.loads(data).get('llm_fallback_count', 0))"
```

### 5.4 API Key 无效

**错误信息**：
```
openai.AuthenticationError: Error code: 401 - Invalid API key
```

**解决方案**：

```bash
# 检查环境变量是否设置
echo $env:DASHSCOPE_API_KEY

# 检查密钥格式（DashScope 以 sk- 开头）
# 确保没有多余空格或换行符

# 重新设置
$env:DASHSCOPE_API_KEY = "sk-xxxxxxxx"
```

### 5.5 模型不存在

**错误信息**：
```
openai.NotFoundError: Error code: 404 - Model not found
```

**解决方案**：

```bash
# 检查模型名称拼写
# DashScope 可用模型：qwen-turbo, qwen-plus, qwen-max, qwen3-32b

# 列出可用模型（如 API 支持）
curl -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
     https://dashscope.aliyuncs.com/compatible-mode/v1/models
```

### 5.6 网络连接错误

**错误信息**：
```
httpx.ConnectError: [Errno 11001] getaddrinfo failed
requests.exceptions.ConnectionError: Connection refused
```

**解决方案**：

```bash
# 检查网络连接
ping dashscope.aliyuncs.com

# 检查 DNS
nslookup dashscope.aliyuncs.com

# 使用代理
$env:HTTPS_PROXY = "http://proxy:port"

# 检查防火墙设置
```

---

## 6. 性能优化建议

### 6.1 减少 API 调用

```bash
# 1. 启用缓存（默认）
# 相同场景只调用一次 API

# 2. 使用 mockllm 进行开发测试
--policy mockllm  # 不调用真实 API

# 3. 先用少量 seeds 验证
--seeds 42  # 单 seed 测试
```

### 6.2 降低成本

```bash
# 1. 使用更便宜的模型进行初步实验
--llm-model qwen-turbo  # 比 qwen-max 便宜 10x

# 2. 缓存复用
# 多次运行相同实验自动复用缓存

# 3. 监控 token 使用
# 查看 results_per_episode.csv 中的 llm_total_tokens
```

### 6.3 提高稳定性

```bash
# 1. 使用低 temperature
--llm-temperature 0.0

# 2. 增加超时时间
--llm-timeout 120

# 3. 选择稳定的模型
--llm-model qwen-max  # 输出质量更高
```

---

## 7. 实验检查清单

### 运行前检查

- [ ] API 密钥已设置并验证
- [ ] 网络连接正常
- [ ] 缓存目录可写入
- [ ] 磁盘空间充足（每 episode ~1MB 日志）

### 运行中监控

- [ ] 观察 console 输出是否有错误
- [ ] 检查 API 调用是否正常（非 429/timeout）
- [ ] 确认缓存正在写入

### 运行后验证

- [ ] `results_per_episode.csv` 包含所有 seed
- [ ] `llm_calls` 列非零（表示确实调用了 LLM）
- [ ] `llm_fallback_count` 检查 fallback 率
- [ ] 运行 `analyze.py` 生成报告

---

## 8. 快速参考

### 最小可运行命令

```bash
# 设置密钥
$env:DASHSCOPE_API_KEY = "sk-xxx"

# 运行单次测试
python run_experiments.py --policy llm_real --seeds 42 --disturbance-levels medium --output-dir logs/test
```

### 完整参数参考

```bash
python run_experiments.py \
    --policy llm_real \
    --seeds 1 2 3 4 5 \
    --disturbance-levels light medium heavy \
    --output-dir results/experiment \
    --llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --llm-model qwen3-32b \
    --llm-key-env DASHSCOPE_API_KEY \
    --llm-temperature 0.0 \
    --llm-timeout 60 \
    --llm-cache-dir .llm_cache
```

---

*文档版本: 1.0 | 更新日期: 2026-01-17*
