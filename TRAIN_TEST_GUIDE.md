# 训练集与测试集分离执行指南

## 📋 核心概念

### 为什么要分离？

1. **灵活性**：可以在不同时间、不同机器上执行
2. **效率**：训练完成后可以多次使用同一参数测试不同场景
3. **资源管理**：训练阶段耗时较长（~15,360次仿真），可在高性能服务器执行；测试阶段较快（~300次），可在本地执行

---

## 🚀 三种运行模式

### 模式 1：完整流程 (默认)

```bash
python run_experiments.py --train-seeds 60 --test-seeds 60 --output results/
```

**特点**：
- ✅ 一次性完成所有步骤
- ⏱️ 耗时：训练调参 + 测试评估 + 训练集完整评估
- 💾 输出：完整的所有文件

**适用场景**：
- 小规模实验（--quick 模式）
- 首次运行，不确定参数
- 有充足时间和资源

---

### 模式 2：仅训练调参 ⭐ 推荐

```bash
python run_experiments.py --mode train-only --train-seeds 60 --output results/
```

**执行内容**：
1. ✅ 在训练集上网格搜索 256 组参数
2. ✅ 选出最优参数（综合目标最小）
3. ✅ 在训练集上评估所有策略
4. ✅ 保存结果

**输出文件**：
```
results/
├── best_params.json          ⭐ 最优参数（供测试阶段使用）
├── tuning_results.csv        所有 256 组参数的结果
├── results_per_episode.csv   训练集每个 episode 的详细指标
└── summary.csv               训练集的汇总统计
```

**最优参数示例** (best_params.json):
```json
{
  "w_delay": 20.0,
  "w_shift": 1.0,
  "w_switch": 180,
  "freeze_horizon": 12
}
```

**适用场景**：
- 大规模实验的第一阶段
- 需要先确定最优参数
- 在高性能服务器上运行

---

### 模式 3：仅测试评估

```bash
python run_experiments.py --mode test-only --test-seeds 60 --output results/
```

**前置条件**：
- ✅ 必须存在 `results/best_params.json`（由训练阶段生成）
- 或通过 `--load-params` 指定参数文件路径

**执行内容**：
1. ✅ 加载最优参数
2. ✅ 在测试集上评估所有策略（fixed_tuned, fixed_default, nofreeze, greedy, mockllm）
3. ✅ 合并训练集和测试集数据
4. ✅ 更新结果文件

**输出文件**：
```
results/
├── results_per_episode.csv   追加测试集数据，包含 train + test
└── summary.csv               更新为完整统计
```

**适用场景**：
- 已完成训练阶段
- 使用已有的最优参数
- 在本地机器上快速评估

---

## 📝 完整工作流示例

### 示例 1：标准分离流程

```bash
# 步骤 1：训练阶段（在服务器上，使用 8 个并行进程）
python run_experiments.py --mode train-only \
                          --train-seeds 60 \
                          --workers 8 \
                          --output results/

# 查看最优参数
cat results/best_params.json
# 输出：{"w_delay": 20.0, "w_shift": 1.0, "w_switch": 180, "freeze_horizon": 12}

# 步骤 2：测试阶段（在本地机器，或稍后执行）
python run_experiments.py --mode test-only \
                          --test-seeds 60 \
                          --workers 4 \
                          --output results/

# 步骤 3：生成分析图表
python analyze.py --input results/ --output figures/
```

---

### 示例 2：使用不同测试集大小

```bash
# 训练一次（60 episodes）
python run_experiments.py --mode train-only --train-seeds 60 --output results/

# 小规模验证测试（30 episodes）
python run_experiments.py --mode test-only --test-seeds 30 --output results_test30/
python analyze.py --input results_test30/ --output figures_test30/

# 完整测试（60 episodes）
python run_experiments.py --mode test-only --test-seeds 60 --output results_test60/
python analyze.py --input results_test60/ --output figures_test60/

# 大规模测试（120 episodes）
python run_experiments.py --mode test-only --test-seeds 120 --output results_test120/
python analyze.py --input results_test120/ --output figures_test120/
```

---

### 示例 3：使用自定义参数（跳过调参）

```bash
# 创建自定义参数文件
cat > custom_params.json << EOF
{
  "w_delay": 15.0,
  "w_shift": 0.5,
  "w_switch": 100,
  "freeze_horizon": 18
}
EOF

# 直接使用自定义参数在测试集上评估
python run_experiments.py --mode test-only \
                          --test-seeds 60 \
                          --load-params custom_params.json \
                          --output results_custom/

# 分析结果
python analyze.py --input results_custom/ --output figures_custom/
```

---

## 🔍 重要说明

### 1. 种子分配规则

**训练集**：seeds = 0 到 (N_train - 1)  
**测试集**：seeds = N_train 到 (N_train + N_test - 1)

示例：
```python
--train-seeds 60 --test-seeds 60
# 训练集: seeds 0-59
# 测试集: seeds 60-119
```

⚠️ **注意**：测试阶段仍需指定 `--train-seeds`，以确保测试集种子从正确的位置开始！

```bash
# ❌ 错误：测试集种子会从 0 开始
python run_experiments.py --mode test-only --test-seeds 60

# ✅ 正确：测试集种子从 60 开始
python run_experiments.py --mode test-only --train-seeds 60 --test-seeds 60
```

---

### 2. 扰动强度分布

每个数据集自动均匀分配三种扰动强度：

| 扰动级别 | 占比 | 训练集 (60) | 测试集 (60) |
|---------|------|-------------|-------------|
| light   | 1/3  | 20 episodes | 20 episodes |
| medium  | 1/3  | 20 episodes | 20 episodes |
| heavy   | 1/3  | 20 episodes | 20 episodes |

---

### 3. 并行处理

**训练阶段**（耗时较长）：
```bash
--workers 8   # 推荐使用多进程加速
```

**测试阶段**（相对较快）：
```bash
--workers 4   # 适度并行即可
```

---

### 4. 文件覆盖行为

- **train-only 模式**：会覆盖输出目录的所有文件
- **test-only 模式**：
  - 如果存在 `results_per_episode.csv`，会**合并**训练集和测试集数据
  - 如果不存在，则只保存测试集数据

---

## ⚡ 快速测试模式

对于所有模式，都支持 `--quick` 快速测试：

```bash
# 完整流程快速测试 (9 train + 9 test)
python run_experiments.py --quick --output results_quick/

# 仅训练快速测试 (9 train)
python run_experiments.py --mode train-only --quick --output results_quick/

# 仅测试快速测试 (9 test)
python run_experiments.py --mode test-only --quick --output results_quick/
```

快速模式调整：
- 训练集：9 episodes (每种扰动 3 个)
- 测试集：9 episodes (每种扰动 3 个)
- 调参网格：2×2×2×2 = 16 组合（而非 256）

---

## 🎯 最佳实践

### 推荐工作流

1. **开发阶段**：使用 `--quick` 模式快速验证代码
   ```bash
   python run_experiments.py --quick --output debug/
   ```

2. **正式实验**：分阶段执行
   ```bash
   # 训练（服务器，8核）
   python run_experiments.py --mode train-only --train-seeds 60 --workers 8 --output results/
   
   # 测试（本地，4核）
   python run_experiments.py --mode test-only --train-seeds 60 --test-seeds 60 --workers 4 --output results/
   
   # 分析
   python analyze.py --input results/ --output figures/
   ```

3. **参数敏感性分析**：固定训练参数，多次测试
   ```bash
   # 训练一次
   python run_experiments.py --mode train-only --train-seeds 60 --output results/
   
   # 测试不同 lambda 权重
   python run_experiments.py --mode test-only --train-seeds 60 --test-seeds 60 --lambda 3.0 --output results_lambda3/
   python run_experiments.py --mode test-only --train-seeds 60 --test-seeds 60 --lambda 7.0 --output results_lambda7/
   ```

---

## ❓ 常见问题

### Q1: 测试阶段找不到 best_params.json？

**A**: 确保指定正确的输出目录：
```bash
# 训练时使用的输出目录
python run_experiments.py --mode train-only --output results/

# 测试时必须使用相同的输出目录
python run_experiments.py --mode test-only --output results/

# 或手动指定参数文件
python run_experiments.py --mode test-only --load-params results/best_params.json --output results/
```

---

### Q2: 如何只运行测试集，不运行训练集？

**A**: 使用 `test-only` 模式，但仍需指定 `--train-seeds` 以确定测试集起始种子：
```bash
python run_experiments.py --mode test-only \
                          --train-seeds 60 \
                          --test-seeds 60 \
                          --output results/
```

---

### Q3: 可以修改测试集大小吗？

**A**: 可以！测试阶段可以使用任意大小：
```bash
# 训练固定 60
python run_experiments.py --mode train-only --train-seeds 60 --output results/

# 测试可变
python run_experiments.py --mode test-only --train-seeds 60 --test-seeds 30 --output results_small/
python run_experiments.py --mode test-only --train-seeds 60 --test-seeds 120 --output results_large/
```

---

### Q4: 如何验证分阶段和完整流程结果一致？

**A**: 对比测试：
```bash
# 完整流程
python run_experiments.py --train-seeds 9 --test-seeds 9 --output full/

# 分阶段
python run_experiments.py --mode train-only --train-seeds 9 --output staged/
python run_experiments.py --mode test-only --train-seeds 9 --test-seeds 9 --output staged/

# 对比结果
diff full/summary.csv staged/summary.csv
```

---

## 📊 输出文件对比

| 文件 | full 模式 | train-only | test-only |
|------|-----------|------------|-----------|
| best_params.json | ✅ | ✅ | ❌ (需已存在) |
| tuning_results.csv | ✅ | ✅ | ❌ |
| results_per_episode.csv | ✅ (train+test) | ✅ (仅train) | ✅ (合并train+test) |
| summary.csv | ✅ (train+test) | ✅ (仅train) | ✅ (合并train+test) |

---

## 🎓 总结

- **train 和 test 可以分开进行**
- 使用 `--mode train-only` 和 `--mode test-only` 参数
- 测试阶段需要 `best_params.json` 文件
- 测试阶段仍需指定 `--train-seeds` 以确定种子范围
- 适用于大规模实验、资源受限、或需要多次测试的场景
