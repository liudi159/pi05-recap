# Stiff 单臂、Xense 与双相机接入

本适配对应现有 Stiff 的 `rm65_xense` 数据合同：一台 RM65、一个 Xense 平行夹爪、
一个腕部相机和一个外部相机。相机槽位中的 `left_wrist` 是模型槽名，不代表新增左臂。

| 来源 | 模型输入 / 含义 |
|---|---|
| `observation.images.external` | `base_0_rgb`，外部第三人称 RGB |
| `observation.images.wrist` | `left_wrist_0_rgb`，腕部 RGB |
| 不存在的第二腕部相机 | `right_wrist_0_rgb` 填零，mask=false |
| `observation.state[0:6]` | `joint_1` 到 `joint_6`，弧度 |
| `observation.state[6]` | `slider_left` 单侧夹指闭合量，米 |
| `action[0:6]` | 原始绝对关节参考目标；适配器减去动作块起始观测关节角 |
| `action[6]` | 绝对单侧夹指闭合量，保持米制，不做关节增量转换 |

输出先恢复绝对关节参考，再取前 7 维。不是 LIBERO 的末端位姿增量，也不把夹指位移
当成夹爪总开口、归一化开合量或电机底层命令。32 维中的其余通道仅用于模型补齐。
此适配不连接机械臂、夹爪或相机驱动；从关节参考到实际硬件命令仍属于原控制器。

## 已匹配的服务器记录

- Stiff 源码：`/home/andy/projects/IRMV_R2S2R_DIZHOU/stiff_lab`。
- 场景：`stiff_lab/assets/chemistry_cleaning_rear_candidate_08/scene.json`。
- 场景 SHA-256：`d9f24ef81a58b9ffe7f7feb6da94436e07a57af6b4f71ecdf8b82ccc1dc39585`。
- LeRobot 数据：该源项目的 `run_artifacts/cleaning_dual_lerobot_01`。
- Episode：该源项目的 `run_artifacts/cleaning_dual_rgb_import_01/episode.json`。
- 完整本地绑定和相机参数：本项目 `.cache/stiff_rm65_alignment/binding.json`。
- 验证报告：本项目 `.cache/stiff_rm65_alignment/report.json`。

该样本为当前装配的 **3 帧诊断记录**，50 Hz 仿真控制时钟，两路 640×480 RGB。
RGB 是已有离线渲染记录，50 Hz 不是实际硬件相机采集频率。
原始光学坐标为 x 右 / y 下 / z 前，保存每帧 `K` 与 `T_world_camera_optical`。
不把 OpenGL 相机矩阵直接当成 optical 矩阵，不独立旋转或改变已有相机安装。
模型图像缩放不改写原始标定；这不是重新完成真实硬件手眼标定的声明。

当前样本 `diagnostic=true`、`full_success=false`、`training_ready=false`。
旧装配另有长轨迹，但相机安装和场景身份不同，不能仅因同为 7 维就混合训练。
七滑块 coupon 的七个米制通道也不能当成 RM65 的六弧度加一米通道。

## 离线验证

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 \
.venv/bin/python scripts/align_stiff_data.py \
  --dataset /home/andy/projects/IRMV_R2S2R_DIZHOU/run_artifacts/cleaning_dual_lerobot_01 \
  --episode /home/andy/projects/IRMV_R2S2R_DIZHOU/run_artifacts/cleaning_dual_rgb_import_01/episode.json \
  --scene /home/andy/projects/IRMV_R2S2R_DIZHOU/stiff_lab/assets/chemistry_cleaning_rear_candidate_08/scene.json \
  --output .cache/stiff_rm65_alignment_new
```

输出目录必须是新目录。读取原数据、核对哈希和每帧图像、动作、时间戳、内外参及
50 步动作窗口尾部补齐；不重新运行仿真，不加载策略权重。LeRobot 读取时的 float32
时间量化按相同表示精确比较；原始相机与状态时钟仍按 float64 核对。

## 后续训练配置

- `pi05_stiff_rm65`：普通 π0.5 适配，可作为冻结特征提取的配置。
- `pi05_stiff_rm65_recap_lora`：RECAP LoRA 配置。
- 原始初始化权重：`/home/andy/models/openpi/pi05_base/params`。
- 数据根目录通过 `--dataset-root` 传给统计、标签对齐及阶段训练脚本；离线 manifest
  的每个 iteration 可填写 `dataset_root`。底层训练配置字段为 `data.local_root`。

训练加载器要求显式本地根目录、匹配的 RM65/Xense 数据合同和来源哈希，且拒绝
未取得 `training_ready` 资格、包含诊断轨迹或非 train split 的输入。不要手工改标记绕过验收。
当前样本用于数据接口验证，不用于计算正式归一化统计或训练。
归一化必须来自该装配独立训练集，不能沿用 `pi05_libero` 的动作统计。
冻结特征提取可指定 `--config pi05_stiff_rm65 --norm-stats <训练统计目录>`。
RECAP 还需要独立评估轨迹、真实成功/失败/人工干预标签及图像特征，不能由诊断样本臆造。

服务器 Tezoi 副本可共用相同的只读数据路径与模型权重；原始图像、轨迹和完整标定
保留在服务器，GitHub 仅包含适配代码、使用说明和测试。
