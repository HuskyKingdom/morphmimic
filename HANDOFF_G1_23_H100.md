# 任务交接：在离线 H100 集群上训练 23 自由度 G1 的 MorphMimic 策略

> 本文档写给另一台电脑上的 Claude Code 执行。请先通读全文，判断自己处在下面哪个环境，再执行对应阶段。遇到文中标注【问用户】的地方，必须先询问用户，不要自己猜。

## 0. 背景（必读）

- **代码仓库**：https://github.com/HuskyKingdom/morphmimic ，分支 `g1-23dof-offline`（提交 `2c2991e`）。这是 lzyang2000/morphmimic 的 fork，本分支加了：
  - 新机器人 `g1_23`（23 自由度 Unitree G1）。在 29 自由度模型里把缺失的 6 个关节锁在 0 位：`waist_roll`、`waist_pitch`、左右 `wrist_pitch`、左右 `wrist_yaw`。
  - 离线容器与集群脚本，都在 `docker/` 目录下。
  - 中文说明：`README_G1_23DOF_zh.md`。
- **目标**：用 BONES-SEED 数据训练一个 `g1_23` 单机器人策略（4 万次迭代），导出 `policy.onnx` 和 `g1_23.json`，用于真机部署。
- **约束**：H100 集群**没有网络**。镜像和数据必须在联网的 Linux 机器上准备好，再拷进集群。
- **当前状态**：代码已推送，但 **Docker 镜像从未构建过，GPU 训练从未跑过**。第一次构建和运行都可能出错，按第 4 节排查。
- **关键文件**：
  - `docker/Dockerfile`：镜像定义（uv 锁定环境，torch cu128，mjlab 1.6.0，Python 3.11）
  - `docker/build_offline.sh`：构建镜像，导出成 tar.gz 和 .sif
  - `docker/fetch_seed.sh`：下载 BONES-SEED 的 G1 数据
  - `docker/run.sh`：宿主机上的启动器（Apptainer 或 Docker，断网运行，挂载仓库、数据和缓存）
  - `docker/pipeline_g1_23.sh`：容器内的流水线，包含 data / check / smoke / train / export 五个阶段
  - `docker/slurm_g1_23.sbatch`：Slurm 作业模板
  - `docker/selfcheck.py`：检查显卡、驱动、依赖和 g1_23 模型，可选做 GPU 实测

## 1. 先判断环境

运行以下命令，再决定执行哪个阶段：

```bash
uname -m; cat /etc/os-release | head -2
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo "no GPU"
command -v docker apptainer singularity sbatch
curl -sS -m 5 -o /dev/null -w "%{http_code}\n" https://pypi.org/simple/ || echo "offline"
```

- **能联网 + 有 Docker + x86_64**：执行阶段 A（准备镜像和数据）。
- **断网 + 有 `sbatch` + 有 H100**：执行阶段 B（集群训练）。
- 两个条件都不满足：停下来，把检测结果告诉用户。

## 阶段 A：联网的 Linux 构建机

### A1. 获取代码

```bash
git clone -b g1-23dof-offline --recursive https://github.com/HuskyKingdom/morphmimic.git
cd morphmimic && git log --oneline -1
```

`--recursive` 会拉取 GMR 子模块（约 1.2 GB）。Dockerfile 只用到其中的 `third_party/GMR/pyproject.toml`，但这个文件必须存在。

### A2. 构建镜像

磁盘至少需要 30 GB 空闲。镜像预计十几 GB，主要是 torch cu128。

```bash
df -h . && bash docker/build_offline.sh 2>&1 | tee build.log
```

成功的标志：

- 输出 `offline import ok`（这一步是在 `--network none` 下导入所有依赖做的检查）；
- 生成 `dist/morphmimic_g1-23.tar.gz`，如果构建机装了 Apptainer，还会生成 `dist/morphmimic_g1-23.sif`；
- 生成 `dist/SHA256SUMS`。

失败时按第 4 节排查，修好后重新构建。**如果修改了仓库文件，要提交并推送回分支 `g1-23dof-offline`**，提交信息写清楚改了什么、为什么改。

### A3. 下载数据

BONES-SEED 是门控数据集，需要用户本人在 https://huggingface.co/datasets/bones-studio/seed 上点同意许可，然后登录：`hf auth login`，或者设置 `HF_TOKEN`。【问用户】数据放在哪个目录（下文记为 `DATA_DIR`）；如果需要 HF token，让用户自己输入，**不要把 token 写进任何文件或提交**。

```bash
bash docker/fetch_seed.sh "$DATA_DIR"
```

这一步下载 `g1.tar.gz`（23.5 GB），并解压到 `$DATA_DIR/seed/`。最后会打印 CSV 根目录和动作片段数量。

### A4. 交给集群

需要拷进集群的三样东西：

1. `dist/` 里的 `.sif`（或 `.tar.gz`）和 `SHA256SUMS`；
2. `$DATA_DIR/seed/`（解压后可以删掉 `g1.tar.gz`）；
3. 整个 `morphmimic` 仓库目录（含 `.git`，方便集群上查看改动）。

【问用户】怎样把文件传进集群（rsync 中转、移动硬盘等）和集群上的目标路径。传完之后在集群上用 `sha256sum -c SHA256SUMS` 校验。

## 阶段 B：离线 H100 集群

### B0. 收集信息

【问用户】下面几项，然后改写 `docker/slurm_g1_23.sbatch`：

| 项目 | 改哪里 | 说明 |
|---|---|---|
| 分区 / 账号 / QOS | `#SBATCH` 行，例如 `--partition`、`--account` | 模板里没写，需要按集群补上 |
| 单任务最长时间 | `#SBATCH --time` | 默认 24 小时，训练约 5 小时，加上首次编译 |
| 内存 | `#SBATCH --mem` | 默认 200G。全量数据单机器人的内存占用没有官方数字，多机器人时每个进程 120–130 GB |
| `SIF` | 脚本里的变量 | `.sif` 镜像路径 |
| `DATA` | 脚本里的变量 | 放 `seed/` 的那个目录，转换后的 npz（约 90 GB）也会写在这里 |
| `CACHE` | 脚本里的变量 | 可写、持久的目录，存 Warp / Inductor / Triton 的编译缓存 |

如果只带了 `.tar.gz`，先在集群上转换：

```bash
apptainer build ~/images/morphmimic_g1-23.sif docker-archive://morphmimic_g1-23.tar.gz
```

如果集群只能用 Docker，不能用 Apptainer：用 `docker load -i` 导入镜像，调用 `run.sh` 时把 `SIF=` 换成 `IMAGE=morphmimic:g1-23`。

### B1–B5. 按顺序执行，每一步验收通过后再进行下一步

所有命令都在仓库根目录执行，日志写在 `logs/slurm_*.out`。

| 步骤 | 命令 | 验收标准 |
|---|---|---|
| B1 数据 | `sbatch --export=ALL,STAGE=data docker/slurm_g1_23.sbatch`（这一步只用 CPU；集群允许的话可以去掉 GPU 申请） | 打印 `after feasibility filter: N (H h)`，N 应接近 12.9 万段；生成 `$DATA/seed_g1_23_npz50/subsets/{tiny,1h,7h,40h,all}.yaml` |
| B2 自检 | `…STAGE=check …` | 驱动版本 ≥ 575，没有 WARNING；H100 显示为 `sm_90`；6 个锁定关节的 `range` 是 ±0.002、`armature` 是 1.000；总质量约 32.11 kg；GPU 实测中 6 个锁定关节的 `max abs q` 应在 10 mrad 以内 |
| B3 冒烟 | `…STAGE=smoke …` | 60 次迭代跑完，生成 `logs/smoke_g1_23/`，没有 OOM 或编译报错。首次编译要几分钟 |
| B4 训练 | `…STAGE=train …` | 训练日志正常输出，每 4000 次迭代保存一次 `logs/g1_23/ckpt_*.pt`。任务超时的话，重新提交同一条命令即可，会从 `latest.pt` 自动续训 |
| B5 导出 | `…STAGE=export …` | 生成 `out/g1_23/policy.onnx`、`out/g1_23/g1_23.json`、`MANIFEST.txt`，以及仿真视频 `out/g1_23/walk_ff_loop_315_R_001__A257.mp4`；导出时 ONNX 与 PyTorch 的一致性检查通过 |

可以调整的变量，通过 `--export=ALL,STAGE=…,变量=值` 传入：

- `GPUS`：默认 `0`；写成 `0,1,2,3` 是数据并行，学习率会按卡数的平方根放大，同时要把 `--gres` 改成对应卡数；
- `ITERS`：默认 `40000`；
- `SUBSET`：默认 `all`；
- `NUM_ENVS`：默认 `4096`；
- `IT`：导出哪一次迭代的 checkpoint；
- `CLIP`：仿真视频用哪段动作。

**建议先用单卡复现论文设置**，有了基线再试多卡。

交互调试（在分到的计算节点上）：

```bash
SIF=... DATA=... CACHE=... bash docker/run.sh bash docker/pipeline_g1_23.sh check
```

### B6. 交付给用户

把下面这些交回给用户，用来上真机：

- `out/g1_23/` 整个目录；
- `assets/` 目录（sim2sim 资源）；
- `deploy/sim2sim/robots/g1_23.json`：如果 export 阶段的 `--bake` 改动了这个文件，把 diff 也一起报告。

真机步骤见 `README_G1_23DOF_zh.md` 第 3 节，**不在集群上执行**。

## 4. 已知风险与排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 构建时 uv 安装脚本 404，或读 `uv.lock` 报 revision 不支持 | Dockerfile 固定了 `UV_VERSION=0.9.5` | `docker build --build-arg UV_VERSION=<较新版本> …`，或改 Dockerfile 里的默认值 |
| `uv sync` 报找不到 GMR 的源码或构建文件 | `.dockerignore` 只放进了 GMR 的 `pyproject.toml` | 删掉 `.dockerignore` 里关于 `third_party/GMR` 的两行，改为拷贝完整的 GMR |
| 磁盘不足 | 镜像很大 | 清理 `docker system prune`，或换到空间更大的目录构建 |
| selfcheck 报 `torch sees no GPU` | Apptainer 没加 `--nv`，或 `CUDA_VISIBLE_DEVICES` 被设成空字符串 | 用 `docker/run.sh` 启动（会自动处理这两点） |
| 驱动版本 < 575 | warp-lang < 1.18 需要 CUDA 12.9 驱动 | 让用户联系集群管理员，不要自己降级依赖 |
| `--gpu` 报 index 越界 | `select_gpus` 现在按 Slurm 分配的卡编号，从 0 开始 | 用 `GPUS=0`（单卡），或 `0..n-1` |
| 首次运行卡在编译 | 正在编译 Warp kernel 和 Inductor 的 CUDA graph | 正常现象，等几分钟。`CACHE` 目录要保留，后续任务会复用 |
| 视频渲染时报 EGL 错误 | 计算节点上 EGL 不可用 | 设置 `MUJOCO_GL=osmesa` 再跑 export（镜像里装了 libosmesa6） |
| check 阶段锁定关节偏移明显大于 10 mrad | GPU 仿真下锁定没有本地测试那么硬 | **停下来报告用户**，不要直接开始训练。可选的修复方向：在 `src/morphmimic/robots/g1_23.py` 里增大 `LOCK_ARMATURE`，或调小 `LOCK_SOLREF`（时间常数不能小于 0.01） |
| 训练 OOM | 显存或内存不够 | 先试 `SUBSET=40h`，或减小 `NUM_ENVS`。注意这会偏离论文设置，要报告用户 |

## 5. 执行规则

- 只在分支 `g1-23dof-offline` 上提交，**不推送到上游 lzyang2000**。提交前先运行 `git status` 看一眼。
- 不要把 HF token、集群账号、密码写进文件或提交。
- 不要修改 `g1` / `x2` / `t2` 原有的行为。修改 `g1_23` 相关参数之前，先说明理由并征得用户同意。
- 每个阶段完成后，向用户简要汇报：做了什么、验收结果的关键数字、下一步计划。失败时附上报错的关键几行。
