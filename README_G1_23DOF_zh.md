# MorphMimic · G1 23 自由度适配与离线集群训练

本分支 `g1-23dof-offline` 在原仓库基础上做了两件事：

1. 新增机器人 `g1_23`（23 自由度 Unitree G1），从数据转换、训练、导出一直到真机部署全链路打通；
2. 把训练环境打包成离线 Docker / Apptainer 镜像，附带断网 H100 集群的 Slurm 脚本。

原有的 `g1` / `x2` / `t2` 三个机器人的行为保持不变。

---

## 1. 23 自由度是怎么实现的

23 自由度 G1 比 29 自由度少 6 个关节：腰部横滚和俯仰（`waist_roll` / `waist_pitch`），以及左右手腕的俯仰和偏航（`wrist_pitch` / `wrist_yaw`）。

**做法是"锁关节"，不是删关节。** Unitree 官方的 `g1_23dof_rev_1_0.urdf` 和 `g1_29dof_rev_1_0.urdf` 关节原点完全一致，所以 29 自由度模型把这 6 个关节锁在 0 位后，运动学就和 23 自由度完全相同（本地正运动学对比，误差为 0）。这样一来：

- 29 槽的规范关节布局、映射（恒等）、执行器、动作缩放、跟踪的身体部位都不用改；
- 网络结构、观测和动作维度都不变，世界模型和导出流程也不用动。

| 环节 | 处理方式 | 文件 |
|---|---|---|
| 仿真模型 | 锁定关节的范围设为 ±2 mrad，限位 solref 设为 0.01，armature 设为 1.0 | `src/morphmimic/robots/g1_23.py` |
| 质量分布 | 按 23 自由度 URDF 修正：去掉腰部两个连杆共 0.3 kg；手腕滚转连杆换成 0.357 kg 的橡胶手（29 自由度版手腕总重 0.82 kg）。修正后总质量 32.11 kg，URDF 为 32.107 kg | 同上 |
| 动作 | 策略在锁定关节上的输出被丢弃，PD 目标固定为 0，PD 反而帮着保持锁定（真机上这些关节本来就没有电机） | `src/morphmimic/env/actions.py`、`deploy/sim2sim/obs.py` |
| 参考动作 | 直接复用 BONES-SEED 的 G1 CSV，转换时把 6 个关节置零，再用 g1_23 模型重新做正运动学，保证参考轨迹自洽 | `src/morphmimic/data/seed_convert.py` |
| 形态向量 | 缩放范围仍固定为论文里的三台机器人，加入 g1_23 后 g1/x2/t2 的向量不变 | `src/morphmimic/robots/morphology.py` |
| 真机 | 锁定关节读数置 0，下发 kp=kd=0；支持两种电机编号布局（见第 4 节） | `deploy/hardware/g1_node.py`、`deploy/hardware/probe_motors.py` |
| 部署常量 | `g1_23.json` 和站立姿态（锁定关节置零后用正运动学重算）已生成；在集群上执行 `export --bake` 会重新生成 | `deploy/sim2sim/robots/g1_23.json`、`deploy/vr_teleop/gmr_configs/stand.json`、`tools/g1_23/derive_g1_23.py` |
| 遥操作 | GMR 仍然解算 29 自由度 G1，结果中的锁定关节置零 | `deploy/vr_teleop/*` |
| 集群选卡 | `--gpu 0` 改为在已有的 `CUDA_VISIBLE_DEVICES`（即 Slurm 分配的卡）里按序号选取，不再覆盖成物理卡号 | `src/morphmimic/scripts/gpu_guard.py` |

**为什么 armature 要设成 1.0：** 软限位的抗扭刚度和关节惯量成正比。手腕锁定关节后面的连杆几乎没有质量，如果只用电机原本的 armature（0.004），手部发生自碰撞时手腕锁会被推开约 0.15 rad。armature 改为 1.0 后，在纯 MuJoCo、mjlab 求解器设置下，让其余所有关节每 0.2 秒跳到一个随机全行程目标（比任何跟踪动作都剧烈），6 个锁定关节最大偏移都不超过 6 mrad（约 0.35°）。armature 只作用于锁定关节自身的坐标，而这个坐标不运动，所以不影响其他关节的动力学。

---

## 2. 离线训练全流程

### A. 在有网络的 Linux 机器上（x86_64，装有 Docker）

```bash
git clone -b g1-23dof-offline <你的仓库地址> morphmimic && cd morphmimic   # 或者用 git bundle 拷过去
bash docker/build_offline.sh        # 生成 dist/morphmimic_g1-23.tar.gz，有 apptainer 时再生成 .sif
bash docker/fetch_seed.sh /path/to/data   # 需先在 HF 页面同意 BONES-SEED 许可并登录
```

需要拷贝到集群的东西：`dist/`（镜像）、`/path/to/data/seed`（数据，`g1.tar.gz` 有 23.5 GB，解压后可以删掉压缩包），以及整个仓库。

### B. 在集群上（断网）

```bash
# 如果没有带 .sif 过去，在集群上从 tar 构建：
apptainer build ~/images/morphmimic_g1-23.sif docker-archive://morphmimic_g1-23.tar.gz
# 先按集群情况修改 docker/slurm_g1_23.sbatch 里的 #SBATCH 行和 SIF / DATA / CACHE 路径
sbatch --export=ALL,STAGE=data   docker/slurm_g1_23.sbatch   # CPU 任务：CSV -> npz，生成子集
sbatch --export=ALL,STAGE=check  docker/slurm_g1_23.sbatch   # 自检 + GPU 实测锁定关节漂移
sbatch --export=ALL,STAGE=smoke  docker/slurm_g1_23.sbatch   # 跑 60 次迭代，确认编译和显存正常
sbatch --export=ALL,STAGE=train  docker/slurm_g1_23.sbatch   # 正式训练：40k 次迭代
sbatch --export=ALL,STAGE=export docker/slurm_g1_23.sbatch   # 导出 ONNX，并生成一段 sim2sim 视频
```

交互式调试也可以直接调用：`SIF=... DATA=... bash docker/run.sh bash docker/pipeline_g1_23.sh check`

**资源与耗时：**

| 项目 | 数值 | 来源 |
|---|---|---|
| 单机器人 4 万次迭代 | 约 4.6 GPU 小时（H100 NVL） | 论文：三个专用模型合计 13.9 GPU 小时 |
| 显存（4096 个环境） | 全量数据 51–58 GB；1h 子集约 28 GB | README |
| 内存 | 多机器人训练时每个进程 120–130 GB（全量数据）；单机器人没有官方数字，sbatch 里先申请 200G | README / 估计 |
| 首次运行 | 编译 Warp kernel 和 CUDA graph 需要几分钟，缓存写在 `$CACHE`，后续任务复用 | README |
| 多卡 | `GPUS=0,1,2,3` 为数据并行，学习率按 √卡数 放大。这和论文的单卡设置不同，建议先用单卡复现 | train.py |

集群驱动必须 ≥ 575（CUDA 12.9），`selfcheck` 会检查并报告。

---

## 3. 真机部署（23 自由度 G1）

1. 把 `out/g1_23/`（`policy.onnx`、`g1_23.json`）和仓库拷到连接机器人的 Linux 电脑上。按原 README 执行 `uv sync --group export`，再运行 `bash deploy/install_unitree_sdk.sh`。
2. **先确认电机编号布局**（只读取状态，不会下发任何命令）：让机器人进入阻尼模式或吊起来，运行
   `uv run --group export python -m deploy.hardware.probe_motors --net <网卡>`，然后用手转动左手腕：
   - 如果第 19 槽变化、第 20 和 21 槽不动，说明是 `sdk` 布局（Unitree SDK2 的 `G1JointIndex`，默认值）；
   - 如果第 17 槽变化，说明是 `compact` 布局，后面的命令要加 `--motor-layout compact`。
3. 先在仿真里看效果：`python -m deploy.run --deploy out/g1_23 --assets assets --robot g1_23 --clip <clip> --interface motion --from-stand 2 --viewer native`
4. 上真机（吊架保护，先用 `--kp-scale 0.5`）：
   `python -m deploy.run --deploy out/g1_23 --robot g1_23 --hardware g1_23 --net <网卡> --interface motion --max-steps 1500 --kp-scale 0.5`
   遥控器操作和原版相同：START 站立，A 启动策略，Y 播放片段，B 阻尼，SELECT 急停。

需要低层控制权限（`LowCmd`），也就是 EDU 版。

---

## 4. 已验证与未验证

**已在本地验证（Windows，纯 MuJoCo 3.13，mjlab 和 torch 用桩代替）：**
- g1_23 模型能编译，质量和 URDF 一致，与 29 自由度模型的正运动学误差为 0，锁定关节在极端负载下的偏移 ≤ 6 mrad；
- 执行器替换只作用于腰部和 4010 两组执行器；
- 电机映射在 `sdk` 和 `compact` 两种布局下读写都正确，锁定关节的动作目标为 0；
- 原 G1 站立状态能被精确复现（误差 < 1e-4），g1_23 站立姿态由此推导；
- 所有改动的 Python 文件都能编译，shell 脚本语法检查通过；`select_gpus` 在 Slurm 环境变量下的行为已测试。

**尚未验证（需要 Linux + GPU，第一次在集群上请按顺序跑 `check` 和 `smoke`）：**
- Docker 镜像构建（本机没有 Docker），以及在没有网络的情况下运行；
- mjlab / MuJoCo Warp 中 g1_23 环境的实际训练效果，以及 GPU 仿真下锁定关节的漂移（`check` 阶段会输出）；
- 真机的电机编号布局：SDK2 示例代码写的是 `sdk` 布局，而 unitree_mujoco 的文档表格是 `compact` 布局，以 `probe_motors` 实测为准；
- 五点遥操作时手部目标仍来自 29 自由度 IK，会有几厘米偏差；全身接口不受影响。
