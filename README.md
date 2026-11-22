# DDA5001 — 2026 Fall Final Project

Course code for DDA5001 at CUHK-Shenzhen. See the handbook and Blackboard for deadlines and submission requirements.

| Part | Topic |
| --- | --- |
| [Part I](p1/README.md) | GPT pre-training: optimizers and learning rates |
| Part II | Coming later |
| Part III | Coming later |

Based on [nanoGPT](https://github.com/karpathy/nanoGPT) and the [2025 course code](https://github.com/CUHKSZ-Course/DDA5001-25Fall). See [LICENSE](LICENSE).

## Part I 机房运行步骤

### 1. 把 TODO 改动带到工作站

在 Ubuntu 上打开学校预装的 `~/work/project1`。这个目录应直接包含 `train.py`，课程数据已准备好。

从本 Fork [下载只含七处 TODO 改动的补丁](https://github.com/Charlie0768/DDA5001-26Fall/commit/e0532406731d8907ff254ce0e904dd920e975689.patch)，保存为 `part1-todos.patch`，放到 `~/work/project1`。在该目录执行：

```bash
cd ~/work/project1
source /opt/dda5001/venv/bin/activate
nvidia-smi

patch --batch --forward --dry-run --fuzz=0 -p2 < part1-todos.patch &&
patch --batch --forward --fuzz=0 -p2 < part1-todos.patch &&
python check_solution.py &&
python train.py config/smoke.py --device=cuda --out_dir=checks/gpu
```

补丁只需应用一次。预检查报错时，先确认三份源文件是否已经含有改动或模板版本是否不同。小测试重跑时请选择新的 `out_dir`。如果你已经从这个 Fork 克隆了完整仓库并进入 `p1`，七处改动已经包含在源文件中，可以直接运行检查和 GPU 小测试。

### 2. 顺序跑六组正式实验

用原始配置启动默认 AdamW：

```bash
python train.py config/train_shakespeare_char.py
```

为 Muon 和 SGDM 各复制一份 `config/train_shakespeare_char.py`，按下表设置 `optimizer_name` 与 `out_dir`。默认学习率已经写在原配置里。每次用 `python train.py config/对应配置文件.py` 启动一组。

| 组别 | 配置文件 | `optimizer_name` | 默认学习率 | `out_dir` |
| --- | --- | --- | --- | --- |
| AdamW | `train_shakespeare_char.py` | `adamw` | `learning_rate = 0.0003` | `runs/adamw-default` |
| Muon | `muon-default.py` | `muon` | `muon_learning_rate = 0.01` | `runs/muon-default` |
| SGDM | `sgdm-default.py` | `sgdm` | `sgdm_learning_rate = 0.1` | `runs/sgdm-default` |

跑完三个默认组后查看 loss 曲线，为每个优化器选一个不同的学习率。每组复制对应的默认配置，只改该优化器的学习率和 `out_dir`，例如 `runs/adamw-other`。每次只开一个训练任务，六次运行用独立输出目录。课程在测试硬件上估计六组约需三小时，请给备份留出时间。

### 3. 画图、采样和备份

对六组分别运行 `python plot.py --run-dir=runs/组名`，再画总对比图。从一组已完成的 `last.pt` 生成样例，下面以 AdamW 默认组为例：

```bash
python plot.py --runs-dir=runs
python sample.py --checkpoint=runs/adamw-default/last.pt --tokens=400
python export_results.py --runs-dir=runs --output=part1-results.zip
```

把报告 PDF 加进提交 ZIP。离开前把 `runs/`、实验配置和导出的 ZIP 复制到自己的存储并确认能打开。工作站重启会清掉本地文件，`last.pt` 等 checkpoint 随运行目录另外备份。
