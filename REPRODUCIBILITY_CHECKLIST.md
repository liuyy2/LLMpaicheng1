# 复现实验清单 (Reproducibility Checklist)

> 按此清单操作可完整复现本项目的实验结果

---

## 1. 环境配置

### 1.1 Python 版本

```
要求: Python 3.10 或 3.11
验证命令: python --version

已测试版本:
✓ Python 3.11.0 (推荐)
✓ Python 3.10.12
✗ Python 3.9.x (未测试)
✗ Python 3.12.x (OR-Tools 兼容性待验证)
```

### 1.2 依赖安装

```bash
# 方式 1: pip 直接安装
pip install ortools>=9.9 numpy matplotlib

# 方式 2: conda 环境 (推荐)
conda create -n llmpaicheng python=3.11
conda activate llmpaicheng
pip install ortools>=9.9 numpy matplotlib

# 验证安装
python -c "import ortools; print(f'OR-Tools: {ortools.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
```

### 1.3 依赖版本要求

| 包 | 最低版本 | 推荐版本 | 说明 |
|----|----------|----------|------|
| ortools | 9.9.3963 | 9.10.4067 | CP-SAT 求解器 |
| numpy | 1.24.0 | 1.26.x | 数值计算 |
| matplotlib | 3.7.0 | 3.8.x | 绘图 |

**注意**: OR-Tools 9.10 使用 `NewFixedSizeIntervalVar`（无 "d"）

---

## 2. 代码文件校验

### 2.1 必需文件清单

```
LLMpaicheng1/
├── config.py              # 全局配置
├── scenario.py            # 场景生成
├── disturbance.py         # 扰动模型
├── solver_cpsat.py        # CP-SAT 求解器
├── simulator.py           # 仿真主循环
├── metrics.py             # 指标计算
├── features.py            # 特征提取
├── run_experiments.py     # 批量实验
├── analyze.py             # 结果分析
├── run_one_episode.py     # 单次运行
└── policies/
    ├── __init__.py        # 策略工厂
    ├── base.py            # 基类
    ├── policy_fixed.py    # 固定权重策略
    ├── policy_nofreeze.py # 无冻结策略
    ├── policy_greedy.py   # 贪心策略
    └── policy_llm_meta.py # MockLLM 策略
```

### 2.2 文件完整性校验

```bash
# 检查所有必需文件存在
python -c "
import os
required = [
    'config.py', 'scenario.py', 'disturbance.py', 
    'solver_cpsat.py', 'simulator.py', 'metrics.py',
    'features.py', 'run_experiments.py', 'analyze.py',
    'policies/__init__.py', 'policies/base.py',
    'policies/policy_fixed.py', 'policies/policy_nofreeze.py',
    'policies/policy_greedy.py', 'policies/policy_llm_meta.py'
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print(f'FAIL: Missing files: {missing}')
else:
    print('PASS: All required files present')
"
```

---

## 3. 快速验证 (< 5 分钟)

### 3.1 单 Episode 测试

```bash
# 命令
python run_one_episode.py --seed 42

# 预期输出校验点
# ✓ Scenario: 20 tasks, 3 pads
# ✓ On-time rate: 100.00%
# ✓ Episode drift: < 0.1
# ✓ Replans: 40-50 次
# ✓ Runtime: < 2 秒
```

### 3.2 求解器测试

```bash
# 命令
python test_solver.py

# 预期输出
# ✓ 所有测试通过
# ✓ 无 INFEASIBLE 状态
```

### 3.3 快速实验测试

```bash
# 命令 (约 3-5 分钟)
python run_experiments.py --quick --output results_quick/

# 预期输出文件
# ✓ results_quick/results_per_episode.csv (90 行: 18 episodes × 5 policies)
# ✓ results_quick/summary.csv (10 行: 2 datasets × 5 policies)
# ✓ results_quick/tuning_results.csv (8 行: 2×2×2 参数组合)
# ✓ results_quick/best_params.json
```

---

## 4. 完整实验复现

### 4.1 批量实验命令

```bash
# 完整实验 (约 30-60 分钟)
python run_experiments.py \
    --train-seeds 60 \
    --test-seeds 60 \
    --solver-timeout 10.0 \
    --lambda 5.0 \
    --output results/

# 参数说明
# --train-seeds 60    : 训练集 60 个 episodes
# --test-seeds 60     : 测试集 60 个 episodes  
# --solver-timeout 10 : 每次求解限时 10 秒
# --lambda 5.0        : 综合目标 delay + 5.0 × drift
```

### 4.2 种子分配规则

```
训练集: seed 0-59
  - light:  seed 0, 3, 6, ..., 57  (20 个)
  - medium: seed 1, 4, 7, ..., 58  (20 个)
  - heavy:  seed 2, 5, 8, ..., 59  (20 个)

测试集: seed 60-119
  - light:  seed 60, 63, 66, ..., 117 (20 个)
  - medium: seed 61, 64, 67, ..., 118 (20 个)
  - heavy:  seed 62, 65, 68, ..., 119 (20 个)
```

### 4.3 网格搜索参数空间

```python
# 完整参数网格 (256 组合)
freeze_horizon_hours = [0, 2, 6, 12]     # 4 值
w_delay_values = [5.0, 10.0, 20.0, 50.0] # 4 值
w_shift_values = [0.0, 0.2, 1.0, 2.0]    # 4 值
w_switch_values = [0, 60, 180, 600]      # 4 值

# 快速模式网格 (16 组合)
freeze_horizon_hours = [0, 6]            # 2 值
w_delay_values = [5.0, 20.0]             # 2 值
w_shift_values = [0.0, 1.0]              # 2 值
w_switch_values = [0, 180]               # 2 值
```

---

## 5. 分析与绘图

### 5.1 分析命令

```bash
# 生成图表和统计
python analyze.py --input results/ --output figures/

# 显示图表（可选）
python analyze.py --input results/ --output figures/ --show
```

### 5.2 预期输出文件

```
figures/
├── delay_vs_drift_scatter.png     # Delay-Drift 散点图
├── replans_distribution.png       # 重排次数箱线图
├── switches_distribution.png      # 切换次数箱线图
├── policy_comparison_combined.png # 综合得分对比
├── policy_comparison_delay.png    # Delay 对比
├── policy_comparison_drift.png    # Drift 对比
├── delay_by_disturbance.png       # 按扰动级别 Delay
├── drift_by_disturbance.png       # 按扰动级别 Drift
├── solve_time_comparison.png      # 求解耗时对比
├── tuning_heatmap.png            # 调参热力图
└── summary_with_tests.csv        # 含 t 检验的汇总
```

---

## 6. 输出文件校验

### 6.1 CSV 格式校验

```bash
# 校验 results_per_episode.csv
python -c "
import csv
with open('results/results_per_episode.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    
expected_cols = ['seed', 'disturbance_level', 'policy_name', 'dataset',
                 'completed', 'total', 'on_time_rate', 'avg_delay', 
                 'max_delay', 'episode_drift', 'total_shifts', 'total_switches',
                 'num_replans', 'num_forced_replans', 'avg_solve_time_ms', 'total_runtime_s']

missing = set(expected_cols) - set(rows[0].keys())
if missing:
    print(f'FAIL: Missing columns: {missing}')
else:
    print(f'PASS: {len(rows)} records, all columns present')
"
```

### 6.2 数值范围校验

```bash
# 校验指标合理性
python -c "
import csv
with open('results/results_per_episode.csv') as f:
    reader = csv.DictReader(f)
    errors = []
    for i, row in enumerate(reader):
        if float(row['on_time_rate']) < 0 or float(row['on_time_rate']) > 1:
            errors.append(f'Row {i}: on_time_rate out of [0,1]')
        if float(row['episode_drift']) < 0:
            errors.append(f'Row {i}: negative drift')
        if float(row['avg_delay']) < 0:
            errors.append(f'Row {i}: negative delay')
            
if errors:
    for e in errors[:5]: print(f'FAIL: {e}')
else:
    print('PASS: All metrics in valid range')
"
```

### 6.3 统计一致性校验

```bash
# 校验 summary.csv 与 results_per_episode.csv 一致
python -c "
import csv
from collections import defaultdict

# 加载原始数据
records = defaultdict(list)
with open('results/results_per_episode.csv') as f:
    for row in csv.DictReader(f):
        key = (row['dataset'], row['policy_name'])
        records[key].append(float(row['avg_delay']))

# 加载汇总
with open('results/summary.csv') as f:
    for row in csv.DictReader(f):
        key = (row['dataset'], row['policy_name'])
        expected_mean = sum(records[key]) / len(records[key])
        actual_mean = float(row['avg_delay_mean'])
        if abs(expected_mean - actual_mean) > 0.01:
            print(f'FAIL: {key} mean mismatch')
            break
    else:
        print('PASS: Summary statistics consistent')
"
```

---

## 7. 已知问题与解决方案

### 7.1 OR-Tools API 兼容性

```
问题: AttributeError: 'CpModel' object has no attribute 'NewFixedSizedIntervalVar'
原因: OR-Tools 9.10 移除了 "d"
解决: 使用 NewFixedSizeIntervalVar (无 "d")
```

### 7.2 中文编码问题

```
问题: UnicodeEncodeError: 'gbk' codec can't encode character
原因: Windows 默认 GBK 编码
解决: 
  - 设置环境变量: set PYTHONIOENCODING=utf-8
  - 或使用 PowerShell: $env:PYTHONIOENCODING="utf-8"
```

### 7.3 matplotlib 中文显示

```
问题: 图表中文显示为方块
解决: 代码中已配置 SimHei 字体
  matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
```

---

## 8. 复现验证签名

完成以下所有检查后，实验视为成功复现：

```
□ Python 版本 3.10/3.11
□ OR-Tools >= 9.9 安装成功
□ run_one_episode.py --seed 42 成功运行
□ run_experiments.py --quick 生成 4 个输出文件
□ analyze.py 生成 11 个图表文件
□ summary.csv 包含 10 行数据 (5 策略 × 2 数据集)
□ 所有 on_time_rate >= 0.9 (90%+ 任务按时完成)
□ 无 INFEASIBLE 或 ERROR 状态

签名: ________________  日期: ________________
```

---

## 附录: 一键复现脚本

```bash
#!/bin/bash
# reproduce.sh - 一键复现脚本

set -e  # 遇错停止

echo "=== Step 1: Environment Check ==="
python --version
python -c "import ortools; print(f'OR-Tools: {ortools.__version__}')"

echo "=== Step 2: Quick Validation ==="
python run_one_episode.py --seed 42

echo "=== Step 3: Quick Experiment ==="
python run_experiments.py --quick --output results_quick/

echo "=== Step 4: Analysis ==="
python analyze.py --input results_quick/ --output figures_quick/

echo "=== Step 5: Verify Outputs ==="
ls -la results_quick/
ls -la figures_quick/

echo "=== DONE: Quick reproduction successful ==="
```

Windows PowerShell 版本:

```powershell
# reproduce.ps1 - 一键复现脚本 (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "=== Step 1: Environment Check ===" -ForegroundColor Cyan
python --version
python -c "import ortools; print(f'OR-Tools: {ortools.__version__}')"

Write-Host "=== Step 2: Quick Validation ===" -ForegroundColor Cyan
python run_one_episode.py --seed 42

Write-Host "=== Step 3: Quick Experiment ===" -ForegroundColor Cyan
python run_experiments.py --quick --output results_quick/

Write-Host "=== Step 4: Analysis ===" -ForegroundColor Cyan
python analyze.py --input results_quick/ --output figures_quick/

Write-Host "=== Step 5: Verify Outputs ===" -ForegroundColor Cyan
Get-ChildItem results_quick/
Get-ChildItem figures_quick/

Write-Host "=== DONE: Quick reproduction successful ===" -ForegroundColor Green
```
