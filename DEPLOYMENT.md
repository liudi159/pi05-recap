# 部署记录 · 2026-09-26

## 结果

代码和独立 Python 环境已部署至服务器 `/home/andy/projects/pi05-recap`。
官方 π0.5 权重统一存放于 `/home/andy/models/openpi`，原项目与 Tezoi 副本共用。
权重下载及完整性校验由 `deploy/download_weights.py` 完成；全部文件校验通过后生成
`/home/andy/models/openpi/download-verification.json`。下载不需要初始化 GPU。

| 权重 | 本地目录 | 大小（十进制） | 用途 |
|---|---|---|---|
| 官方 π0.5 base | `/home/andy/models/openpi/pi05_base` | 12.442 GB | RECAP 策略训练初始化 |
| 官方 π0.5 LIBERO | `/home/andy/models/openpi/pi05_libero` | 12.439 GB | 冻结特征提取，包含归一化 assets |

来源为官方 `gs://openpi-assets/checkpoints/`；按对象 generation 固定下载版本，逐文件核对大小和 CRC32C。
这两份是官方 π0.5 权重，尚无本项目训练生成的 RECAP 权重。实验数据另行准备。

2026-09-26 11:08:36 UTC 完成验收：45 个文件，共 24,880,835,062 字节，
全部匹配官方文件大小和 CRC32C；无残留下载分段。下载后的 `/home` 可用空间约 319 GiB。
因服务器直连官方存储速度不稳定，本次采用分段续传并经本机中转补齐，最终在服务器合并、完整校验。
Tezoi 副本通过上述绝对路径共用权重，不重复占用一份模型空间。

## 资源检查

- Ubuntu 24.04.3 LTS，x86_64；系统内存约 251 GiB。
- 系统盘：468 GB 总量，206 GB 可用。
- `/home`：约 1.8 TB 总量；首次检查 331 GB 可用，安装后检查约 341 GB 可用。
  这是共享文件系统的不同时间读数，不能据此计算本次安装的空间增量。
- NVIDIA 驱动 570.144，报告 CUDA 12.8。
- 2 × RTX 5090，每张 32607 MiB 显存。
- 两次检查分别占用 18523 / 16039 MiB；本次没有停止或修改现有 GPU 服务。
- 新项目 `.venv` 的 `du` 读数约 9.0 GB（uv 可能使用硬链接，非独占磁盘增量）。

## 安装版本

| 包 | 安装版本 |
|---|---|
| openpi | 0.1.0（本地适配源码，editable） |
| jax / jaxlib | 0.5.3 / 0.5.3 |
| torch | 2.7.1+cu128 |
| torchvision | 0.22.1 |
| flax | 0.10.2 |
| av | 14.2.0 |
| lerobot | 0.1.0，源码提交 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 |

完整依赖锁定于 `uv.lock`。PyAV 改用具有 Linux wheel 的 14.2.0，避免原 14.4.0 在服务器上编译失败。
LeRobot 通过固定提交的 Git bundle 在服务器本地建立只用于安装的镜像，绕过服务器直连 GitHub 的网络故障。
PyTorch 官方 wheel 使用分段下载；安装前的完整 SHA-256 与锁文件一致：
`c301dc280458afd95450af794924c98fe07522dd148ff384739b810e3e3179f2`。
缓存 wheel 留在项目 `.cache/wheels` 中，安装脚本支持该离线缓存，缓存不上传 GitHub。

## 已完成的验证

- `uv pip check`：242 个安装包，全部依赖约束兼容。
- Python AST：130 个源码文件可解析；安装脚本也在服务器 Python 环境解析源码。
- 新增脚本与价值模型模块：Ruff E4/E7/E9/F 静态检查通过。
- 安装和环境 shell 脚本：`bash -n` 通过。
- TOML 锁文件、项目配置和 JSON manifest：语法检查通过。
- 源码提交前扫描：未发现 GitHub token、OpenAI key 或私钥格式。

## 尚未验证

尚未执行新实现的回归测试、完整模块导入、JAX GPU 计算、权重加载、
价值模型训练、LoRA 反向传播、策略推理或 LIBERO 评估。
包依赖约束兼容不等于 GPU 运行兼容，也不代表复现论文效果。
新增 `tests/test_recap_adaptation.py` 留待后续运行；原参考代码修改前通过的 14 项离线工具测试不计入本实现验收。

## 后续入口

- 重装/静态验收：`bash deploy/install.sh`
- 环境变量：`source deploy/env.sh`
- 离线配置模板：`deploy/offline_manifest.example.json`
- 默认只打印计划：`python scripts/recap_train.py --manifest <配置文件>`
- 真正训练需明确添加 `--execute`；训练前需要准备匹配的数据、标注、权重和足够空闲显存。

算法复现边界及示例命令见 [README.md](README.md)。
