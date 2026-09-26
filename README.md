# 基于 π0.5 的 RECAP 复现部署

这是以 openpi 的 π0.5 为基础、参考 π*0.6 / RECAP 论文和 hzm8341/pi0.6 的研究实现。
安装状态见 [DEPLOYMENT.md](DEPLOYMENT.md)，代码来源见 [PROVENANCE.md](PROVENANCE.md)。

## 复现范围

| 部分 | 本仓库 |
|---|---|
| 策略骨干 | π0.5 / PaliGemma，兼容官方 π0.5 初始化权重 |
| 优势条件 | positive/negative tokens、条件 dropout、人类干预强制正标签 |
| 策略训练 | JAX flow matching；新增 LIBERO 全参数/LoRA 配置 |
| 价值学习 | 201 bins 交叉熵线性分类头；输入为冻结 π0.5 的视觉/语言/状态特征 |
| 优势估计 | n-step、终止状态不 bootstrap、按任务分位数阈值 |
| 数据对齐 | 显式 episode ID 映射 + frame ID 校验，拒绝静默错位 |
| 迭代 | 已采集数据上的离线分阶段流程；每轮策略从同一固定基础权重初始化 |
| 论文完整模型/结果 | 未实现 Gemma 3 4B + 860M action expert、完整 KI 目标、在线 HITL 和论文效果验证 |

冻结特征分类头是资源受限的近似；它不是论文端到端训练的独立视觉语言价值模型。
继承的 MEM/G1 实验代码不属于本次验证范围。不要把安装完成或原仓库测试通过当作模型效果复现。
旧参考说明保留在 [docs/REFERENCE_README.md](docs/REFERENCE_README.md)，以本 README 为当前使用入口。

## 环境安装（不运行模型）

目标平台：Linux x86_64、Python 3.11、NVIDIA CUDA 12 驱动。
锁文件固定 PyTorch 2.7.1+cu128、JAX 0.5.3、Flax 0.10.2、PyAV 14.2.0；GPU 实际兼容性尚未运行验证。

```bash
bash deploy/install.sh
source deploy/env.sh
```

安装脚本只同步依赖、检查包依赖关系和解析 Python 语法，不初始化 GPU、不加载权重。
现有服务器目录：`/home/andy/projects/pi05-recap`。所有环境都在该目录下的 `.venv`，未修改系统驱动或其他项目环境。

服务器无法直接访问 GitHub 时，本次通过经过固定提交校验的本地 Git bundle 安装 LeRobot；
可选镜像在 `.cache/git/lerobot.git`，安装脚本会仅对本次命令使用该镜像，不改全局 Git 配置。
其他机器通常直接访问锁文件中的原始 GitHub 地址即可。
本次还保留了 `.cache/wheels` 中的 PyTorch 官方 wheel；安装脚本校验其锁文件 SHA-256 后可离线安装。

## 以后运行时的数据要求

以下命令只是说明，本次没有执行。

先准备 LIBERO 格式的 **train-only LeRobot 数据集**，图像、状态、动作字段沿用 openpi LIBERO 转换例子。
每个任务都需要真实成功/失败标注，最好同时包含 demonstrations、自主失败/成功轨迹和人工纠正。
仅有成功演示不能代替论文在线经验采集。评估轨迹应独立，不能并入策略训练集。

对应 JSON 格式：

```json
[{
  "episode_id": "ep000", "task": "pick up the cup", "success": true,
  "timeout": false, "max_episode_length": 300,
  "frames": [{
    "t": 0,
    "observation": {"state": [0,0,0,0,0,0,0,0], "image_path": "ep000/000.png", "wrist_image_path": "ep000/000_wrist.png"},
    "action": [0,0,0,0,0,0,0], "is_human_intervention": false
  }]
}]
```

这只是字段示例，不是可用于训练的真实数据。帧号从 0 连续递增；`max_episode_length` 是该任务的固定最大步数。
`episode_map.json` 显式给出 JSON ID 到 LeRobot `episode_index` 的映射，例如 `{"ep000": 0}`。
标签工具会验证训练 JSON 与 LeRobot 的帧集合完全一致。不要裁剪 JSON 后继续使用未裁剪的 LeRobot 数据。

## 以后运行时的步骤

1. 准备官方 `gs://openpi-assets/checkpoints/pi05_base`（策略初始化）和
   `gs://openpi-assets/checkpoints/pi05_libero`（冻结视觉/语言特征提取，包含归一化 assets）。
   安装环境后，可用 `.venv/bin/python deploy/download_weights.py --root /home/andy/models/openpi` 下载并校验文件。
   下载器支持断点续传，仅获取权重和随附的归一化元数据；不会加载模型。
   服务器统一存放于 `/home/andy/models/openpi/{pi05_base,pi05_libero}`，Tezoi 副本也可直接使用这些路径。
   完整性记录为该目录中的 `download-verification.json`，全部文件通过校验后才会生成。
2. 为 train/eval JSON 分别提取冻结特征：

```bash
source deploy/env.sh
python scripts/extract_recap_features.py \
  --episodes data/train.json --image-root data/images \
  --checkpoint /home/andy/models/openpi/pi05_libero --output data/train_features.json
# 对独立 eval.json 同样执行，输出 eval_features.json。
```

3. 复制并修改 `deploy/offline_manifest.example.json`。设置数据路径、真实 repo ID、episode_map 和固定 base params。
   在这台服务器上，将 `base_params` 设为 `/home/andy/models/openpi/pi05_base/params` 即可使用本地权重。
   后续迭代的训练 JSON/LeRobot 数据必须累积已有经验；评估集合固定不变。

```bash
# 默认仅打印命令计划：
python scripts/recap_train.py --manifest deploy/my_manifest.json --output outputs/run01
# 以后确实要训练时才加 --execute：
python scripts/recap_train.py --manifest deploy/my_manifest.json --output outputs/run01 --execute
```

执行顺序为：训练价值分类头 → 用预测值计算优势 → 按 LeRobot 帧 ID 对齐 → 计算归一化统计 → 微调策略。
`--execute` 会启动实际计算和潜在的权重/数据下载；安装脚本不会调用它。
没有自动 rollout collector，也没有把离线 value MAE 当成策略成功率。
如果要做明确的 state-only 消融，可在 manifest 设置 `state_only: true`；默认必须提供视觉语言特征。

4. 训练后手动启动 positive-conditioned 策略服务：

```bash
python scripts/serve_recap_policy.py \
  --checkpoint /path/to/recap_checkpoint --repo-id your_account/libero_train_only \
  --host 127.0.0.1 --port 8016
```

传入的是 **训练后的 RECAP checkpoint**，不能把未训练 π0.5 加上 positive token 就称为 π0.6。

## 资源与验证边界

官方 openpi 给出的参考显存需求为：推理 >8 GB、LoRA >22.5 GB、全参数微调 >70 GB。
本项目新增路径的实际需求尚未测量。两张 32 GB 卡不等于可任意合并为单张 64 GB；
当前 LoRA 配置默认单 GPU、batch size 1，需要后续运行前检查空闲显存。

部署阶段只做静态语法、依赖一致性、源码与版本检查。
`tests/` 提供新增回归检查，未在本次“只部署”阶段执行；GPU/权重加载/训练/仿真验证均待后续进行。
原参考仓库在修改前曾通过 14 项 NumPy 离线工具测试，不代表本仓库的新实现或 GPU 路径已通过测试。

## 参考

- [π*0.6: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759)
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- [hzm8341/pi0.6](https://github.com/hzm8341/pi0.6)

继承 Apache-2.0 代码许可证与 `LICENSE_GEMMA.txt`。预训练权重、数据集和第三方组件各自的许可仍然适用。
