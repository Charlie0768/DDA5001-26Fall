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

在机房终端复制以下命令，把七处 TODO 的补丁保存为 `part1-todos.patch`。在浏览器打开补丁链接时，GitHub 可能直接显示文本。

```bash
cd ~/work/project1 &&
curl --fail --location --output part1-todos.patch \
  'https://github.com/Charlie0768/DDA5001-26Fall/commit/e0532406731d8907ff254ce0e904dd920e975689.patch'
```

也可以提前把补丁文件带到机房，放到 `~/work/project1` 并命名为 `part1-todos.patch`。文件到位后执行：

```bash
cd ~/work/project1
source /opt/dda5001/venv/bin/activate
nvidia-smi

patch --batch --forward --dry-run --fuzz=0 -p2 < part1-todos.patch &&
patch --batch --forward --fuzz=0 -p2 < part1-todos.patch &&
python check_solution.py &&
python train.py config/smoke.py --device=cuda --out_dir=checks/gpu
```

补丁只需应用一次。预检查报错时，先确认三份源文件是否已经含有改动或模板版本是否不同。小测试重跑时请选择新的 `out_dir`。如果你已克隆本 Fork，或者点击仓库页面的 **Code → Download ZIP** 下载完整代码并解压，进入其中的 `p1` 后可以直接运行检查和 GPU 小测试，源文件已包含七处改动。

### 2. 顺序跑六组正式实验

#### 默认 AdamW、Muon 和 SGDM

以下命令都在包含 `train.py` 的项目目录执行。如果你新开了机房终端，先执行 `cd ~/work/project1` 和 `source /opt/dda5001/venv/bin/activate`。

默认 AdamW 的学习率是 `0.0003`，结果在 `runs/adamw-default`。这组已经在跑或已完成时，直接接着做下一组。

```bash
python train.py config/train_shakespeare_char.py
```

**AdamW 完成后，整段复制以下命令跑默认 Muon 和 SGDM。** 命令会从原配置生成两份配置。Muon 成功结束后才会启动 SGDM，所以可以在同一个终端一次粘贴。

```bash
sed -e "s/^optimizer_name = .*/optimizer_name = 'muon'/" \
    -e "s|^out_dir = .*|out_dir = 'runs/muon-default'|" \
    config/train_shakespeare_char.py > config/muon-default.py &&
sed -e "s/^optimizer_name = .*/optimizer_name = 'sgdm'/" \
    -e "s|^out_dir = .*|out_dir = 'runs/sgdm-default'|" \
    config/train_shakespeare_char.py > config/sgdm-default.py &&
python train.py config/muon-default.py &&
python train.py config/sgdm-default.py
```

Muon 使用 `muon_learning_rate = 0.01`，SGDM 使用 `sgdm_learning_rate = 0.1`。每组正式训练应跑到 **5,000 步、92,160,000 个预测 token**。看到 `Completed; endpoint saved as last.pt and summary.json.` 就完成了一组。若其中一组已完成，可单独执行另一组的 `python train.py config/对应文件.py`。同一时间只跑一组。

#### 看默认组曲线，再选第二个学习率

三个默认组完成后，生成各自的训练和验证 loss 图：

```bash
python plot.py --run-dir=runs/adamw-default &&
python plot.py --run-dir=runs/muon-default &&
python plot.py --run-dir=runs/sgdm-default
```

打开每组目录的 `loss_vs_tokens.png`。默认组下降平缓但慢，可以试更大的学习率。曲线反复大幅波动或上升，可以试更小的学习率。下面是可选的起点，最终根据你实际看到的曲线为每个优化器选一个值。

| 优化器 | 默认 | 可试较小值 | 可试较大值 |
| --- | ---: | ---: | ---: |
| AdamW | 0.0003 | 0.00015 | 0.0006 |
| Muon | 0.01 | 0.005 | 0.02 |
| SGDM | 0.1 | 0.05 | 0.2 |

把下面**前三行**改成你选的值再执行。示例中的三个数值都是默认值的一半。命令从默认配置生成各自的第二组配置，只改对应学习率和输出目录，随后依次训练。

```bash
adamw_lr=0.00015
muon_lr=0.005
sgdm_lr=0.05

sed -e "s/^learning_rate = .*/learning_rate = ${adamw_lr}/" \
    -e "s|^out_dir = .*|out_dir = 'runs/adamw-other'|" \
    config/train_shakespeare_char.py > config/adamw-other.py &&
sed -e "s/^muon_learning_rate = .*/muon_learning_rate = ${muon_lr}/" \
    -e "s|^out_dir = .*|out_dir = 'runs/muon-other'|" \
    config/muon-default.py > config/muon-other.py &&
sed -e "s/^sgdm_learning_rate = .*/sgdm_learning_rate = ${sgdm_lr}/" \
    -e "s|^out_dir = .*|out_dir = 'runs/sgdm-other'|" \
    config/sgdm-default.py > config/sgdm-other.py &&
python train.py config/adamw-other.py &&
python train.py config/muon-other.py &&
python train.py config/sgdm-other.py
```

训练目录名必须唯一。如果某组失败，先看它的 `run.log`，保留失败目录供排查。需要重新从头跑同一组时，把失败目录移到 `runs/` 之外，再用原名称重跑。汇总脚本会检查 `runs/` 下的实验是否全部完成。课程测试硬件上六组约需三小时，按现场进度留出备份时间。

### 3. 六组跑完后核对、画图和导出

下面的命令检查六个目录的完成状态和预算，并打印报告需要的学习率、终点验证 loss、训练秒数。

```bash
python - <<'PY'
import json
from pathlib import Path

lr_fields = {"adamw": "learning_rate", "muon": "muon_learning_rate", "sgdm": "sgdm_learning_rate"}
print(f"{'run':<16} {'peak LR':>10} {'val loss':>10} {'train s':>10}")
for method, field in lr_fields.items():
    for setting in ("default", "other"):
        run = Path("runs") / f"{method}-{setting}"
        config = json.loads((run / "config.json").read_text())
        summary = json.loads((run / "summary.json").read_text())
        if (summary["status"], summary["step"], summary["tokens"]) != ("completed", 5000, 92160000):
            raise SystemExit(f"{run} has not completed the required training budget")
        peak_lr = config[field] * config["lr_multiplier"]
        print(f"{run.name:<16} {peak_lr:>10g} {summary['endpoint_scores']['loss']:>10.4f} {summary['train_seconds']:>10.1f}")
PY
```

核对通过后，生成六组各自的训练和验证 loss 图，再画跨组验证 loss 对比图。从一个终点 checkpoint 生成文本，下面用默认 AdamW 的 `last.pt` 做例子。导出时终端应显示 `"trials": 6`。

```bash
(
  set -e
  for run in adamw-default muon-default sgdm-default adamw-other muon-other sgdm-other; do
    python plot.py --run-dir="runs/$run"
  done
  python plot.py --runs-dir=runs
  python sample.py --checkpoint=runs/adamw-default/last.pt --tokens=400
  python export_results.py --runs-dir=runs --output=part1-results.zip
)
```

- 各组的训练与验证曲线在 `runs/组名/loss_vs_tokens.png`，总验证对比图在 `runs/comparison_tokens_vs_tokens.png`。
- 生成文本在 `runs/adamw-default/samples.txt`。报告摘一小段，标明优化器和学习率。
- 先把 `part1-results.zip`、`config/` 和 `runs/` 复制到自己的存储并确认能打开。导出的 ZIP 已包含源代码和轻量结果。工作站重启会清掉本地文件，`last.pt` 等 checkpoint 要随运行目录单独备份。

### 4. 写报告并交 Blackboard

报告正文最多 1 页，参考文献和附录合计最多额外 2 页。正文应放六组训练、验证 loss 曲线和终点结果，简要说明学习率影响以及实际观察到的曲线交叉，再放一小段生成文本、主要困难和解决办法、每位组员的贡献。比较以训练预算终点为准，使用上表的终点验证 loss 和训练时间。

把报告命名为 `PartI_Report.pdf`，和导出的 ZIP 放在同一目录，然后执行：

```bash
zip -j part1-results.zip PartI_Report.pdf &&
unzip -t part1-results.zip
```

打开 ZIP 确认有报告、源代码、六组配置和结果、图和生成文本。提交包应排除课程数据集和 checkpoint。每组在 Blackboard 的 Part I 入口交一个 ZIP。
