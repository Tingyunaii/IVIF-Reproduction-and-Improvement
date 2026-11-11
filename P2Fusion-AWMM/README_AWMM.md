# P²Fusion → AWMM：配对干净参考引导的天气适应

此版本在上传的 P²Fusion 源码上新增 AWMM 训练入口，原 `train.py`、`test.py`、`models/P2Fusion.py` 与原配置保留不动。请使用 **`train_awmm.py` / `infer_awmm.py`** 运行新模型。没有随包提供数据集或权重。

## 1. 放到服务器

把代码包解压到单独目录 `P2Fusion-AWMM`。沿用现有 `p2fusion` conda 环境和原教师权重：

```bash
conda activate p2fusion
cd /你解压的位置/P2Fusion-AWMM
```

不要直接重新安装原来长达百项的 `requirements.txt`，以免改变已经可用的环境。新增代码只使用原依赖和标准库。以下是导入检查：

```bash
python -c "import torch, torchvision, timm, pyiqa, numpy, PIL, pytorch_msssim; print(torch.__version__, torch.version.cuda)"
```

若确实缺少某项，补装那一项。原服务器 `PyTorch 2.0.0+cu118` 与 A40 驱动支持的 CUDA 12.4 兼容，不需要据此升级 CUDA/PyTorch。

将原来的 `nets/best_epoch_weights.pth` 复制到新项目的同名位置，或者修改 `configs/awmm.json` 中 `teachers.ir_checkpoint`。真实训练严格加载两类 SegFormer-B2 权重，不会忽略不匹配参数而继续用随机教师。BRISQUE 沿用 `pyiqa`，其首次运行可能下载模型资源；离线服务器需使用已有的 pyiqa 资源缓存。

## 2. 生成固定训练／验证清单

当前没有标准划分，因此生成的是**本实验的固定划分，不宣称是 AWMM 官方协议**。推荐先用 Scene 候选池，覆盖雨、雾、雪及可用的 Light/Medium/Heavy，避免将 AWMM-Text 与候选池直接合并、重复采样。

```bash
python tools/prepare_awmm.py \
  --root /home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k \
  --out splits/awmm \
  --source scene --val-ratio 0.1 --seed 3407
```

脚本认识以下目录：

```text
AWMM-100k/
  Haze|Rain|Snow/Scene_*/Train/
    ir/  vi_Light/  vi_Medium/  vi_Heavy/  gt_ir/  gt_vi/
  AWMM-Text/Train/
    ir/  vi/  gt_ir/  gt_vi/
  AWMM-Text/Test/Haze|Rain|Snow/
    ir/  vi/  gt_ir/  gt_vi/
```

`--source text` 仅使用 AWMM-Text/Train；`--source all` 合并两者（可能重复变体，初版不推荐）。不要求文件扩展名相同，但四路图像的文件名主干必须严格相同；缺文件、同一路重名主干或四图尺寸不等会报错。

首次生成需要扫描图像尺寸并计算干净参考哈希，可能需要几分钟。生成 `train.json / val.json / test.json / excluded.json / report.json`。划分依据是干净图像**解码后的 RGB 像素内容哈希**，不是全局数字文件名：

- 任一相同干净 IR 或 VIS 会把样本连接为同组，连接关系具有传递性。
- 同组天气和程度变体放在同一侧；按组约90%训练、10%验证，因此样本比例未必正好90/10。
- 与官方测试集干净参考重复的整组候选样本写入 `excluded.json`，不进入训练／验证。
- 测试集参考仅用于重复筛查，不生成训练目标、不用于选最优模型。
- 已有清单默认拒绝覆盖；改变实验协议时使用新目录，或明确传 `--overwrite`。

**边界：** 精确哈希不能识别重压缩、轻微位移、相邻视频帧或同一场景的不同视角；因此它确保的是同一干净参考及其变体不泄漏，不能代替人工场景／序列级协议。初次生成后检查 `report.json` 的天气／程度数量和被排除数量，并保存清单用于复现实验。

## 3. 先运行无需权重的合成冒烟检查

```bash
CUDA_VISIBLE_DEVICES=0 python train_awmm.py \
  --config configs/awmm.json --device cuda --smoke-test
```

这个模式明确使用合成图像和模拟教师，只检查真实网络、八项损失与梯度是否可运行，不训练研究模型、不生成正式权重，也不验证真实教师质量或 AWMM 效果。CPU 也可以运行：`--device cpu`。

## 4. 用真实数据、真实教师做一次短跑

若已有原 P²Fusion 训练好的融合权重，推荐用它初始化。以下路径需要替换成**实际存在的原融合权重**，不是红外教师权重：

```bash
CUDA_VISIBLE_DEVICES=0 python train_awmm.py \
  --config configs/awmm.json \
  --init /实际路径/原P2Fusion融合模型.pth \
  --run-dir runs/awmm_check --max-steps 1 --val-batches 2
```

原模型权重支持裸 state_dict 和 `params / params_ema / state_dict / model` 包装，兼容 `module.` 前缀。只允许新增适应模块缺失；原网络参数缺失、未知参数或形状不符会报错。

如果没有原融合权重，可以明确传 `--from-scratch` 替代 `--init ...`，但这属于 AWMM 上从头训练，不能与预训练模型适应混为一谈。红外教师权重在两种情况下都必须提供。

默认 batch_size=2、crop_size=256、AMP 开启，作为单张 A40 48GB 的保守起点。这里没有实测你的服务器显存，确认短跑后再调整。四图同步裁剪、翻转；不会独立位移、遮挡或额外添加未同步退化。crop_size 必须是32的倍数。

## 5. 正式训练与恢复

```bash
CUDA_VISIBLE_DEVICES=0 python train_awmm.py \
  --config configs/awmm.json \
  --init /实际路径/原P2Fusion融合模型.pth \
  --run-dir runs/awmm
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_awmm.py \
  --config configs/awmm.json \
  --resume runs/awmm/last.pt
```

默认100轮。新模块学习率1e-4，原网络1e-5；编码主干前500步冻结，然后解冻。验证使用固定中心裁剪，报告八项损失，按验证总损失保存 `best.pt`；`last.pt` 用于继续训练。它是选权重的代理指标，不能代替全分辨率融合指标和分天气／程度的最终评估。不要把训练损失最低当作最优模型。

恢复会检查清单内容签名，恢复模型、优化器、调度器、AMP、训练步数和随机状态。多进程加载中断后的增强序列不承诺逐位重放。新的训练实验请使用新的 run-dir。当前新入口针对单卡，不使用旧脚本的 DDP 参数。

## 6. 推理只需要恶劣天气双模态

```bash
CUDA_VISIBLE_DEVICES=0 python infer_awmm.py \
  --checkpoint runs/awmm/best.pt \
  --ir-dir /home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k/AWMM-Text/Test/Haze/ir \
  --vis-dir /home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k/AWMM-Text/Test/Haze/vi \
  --output results/awmm/Haze --device cuda
```

Rain/Snow 只需替换目录。推理没有干净图像参数，不加载双教师，不运行辅助重建头。输出是原模型的一通道融合 PNG，不把恶劣可见光的色彩直接贴回结果。默认整图推理；超大图像应另外评估显存需求。

### 可选彩色输出（兼容已训练权重）

更新后的 `infer_awmm.py` 可追加 `--color-source input` 或 `--color-source reconstructed` 输出 RGB PNG。默认 `none` 保留灰度行为。

- `input`：使用实际恶劣 VIS 的色度，可能保留天气偏色。
- `reconstructed`：在同一次恶劣双模态前向中启用已有辅助重建头，以模型估计的干净 VIS RGB 提供色度。这一推理选项会运行辅助头，但不读取真实干净图像或双教师。重建色度是否更可靠需检查实际图像；此头原来是辅助训练目标，不能保证最终 RGB 质量。

两种方式都属于颜色后处理，不是新训练的三通道融合解码器；无需重训现有权重。实现给色度参考的 RGB 三通道添加相同偏移，保留其色差（裁剪前）。对于默认 `first_channel` 训练，以源 R 为偏移锚点，使结果 R 等于融合标量；对于 `luminance` 训练，以 Y 为锚点。不会把默认按 R 监督的融合标量直接声称为 Y。RGB 超出范围会被裁剪，可能改变局部色度。不得用测试 `gt_vi` 提供推理色度。

建议先用新的结果目录输出两种色源，保留灰度结果，分别评估天气残留、偏色与细节。灰度标准融合指标与彩色后处理结果应分开报告。新增5项颜色检查及端到端 RGB 保存检查已在 CPU 合成输入上通过；真实已训练模型的彩色观感仍需服务器验证。

## 7. 实现对应关系与实验设定

对 IR/VIS 各自执行：原共享编码器 → 空间特征 → FDN(U,V) → DWAM → 原 `proj` / `prompt_head` → 原交互/GDER/融合解码。两种天气共享对应参数；IR/VIS 解耦头独立。不引入 DINOv2 或新的基础模型。

DWAM 严格使用 `U + U * sigmoid(g_weather(V.detach())) * sigmoid(g_region(U))`。私有特征只在这条门控路径 detach，其正交与退化重建路径保持梯度。门控是相对选择性增强（倍数1～2），不单独保证绝对噪声衰减。

三项新增损失：

1. 同模态跨天气一致性：恶劣 U 与停止梯度的干净 U 的 MSE。
2. 正交约束：U/V 逐位置通道向量的平方余弦，不能通过负相关作弊。
3. 辅助重建：U 重建该模态干净图；V 重建恶劣减干净的有符号残差，干净 V 对应零残差。它为 private 提供独立学习信号，避免只有正交+detach而退化为零。这是成对参考监督，不声称得到严格可辨识的纯天气因素。

原五项与新增三项共八项，日志值未乘权重，total 对每项只加权一次：

| 项 | 初始权重 |
|---|---:|
| intensity | 8 |
| ssim | 25 |
| gradient | 20 |
| distill_ir | 0.5 |
| distill_vis | 5 |
| consistency | 1 |
| orthogonality | 0.01 |
| self_supervision | 1 |

前三项参考配对干净 IR/VIS；两项提示教师也读取干净图，红外 BCE、可见光图像级分数 MSE。可见光目标是 `(100-BRISQUE)/100` 并裁剪到[0,1]，含义为干净参考质量蒸馏。八项权重是初始实验配置，新增项需要消融／调参，不宣称最优。

保留上传训练代码 `image[:, :1]` 作为融合参考的通道约定，即 RGB 的第一通道；可在 `loss.fusion_reference` 显式改为 `luminance` 做单独实验。辅助重建和教师使用 RGB。不要在不同对照中无记录地混用通道约定。

**显式修正：** 上传源码的 SSIM 在[0,1]输入上隐式使用库默认 `data_range=255`；此版设为1。`loss.ssim_data_range=255` 可恢复原范围约定。全零参考的 SSIM 权重采用0.5/0.5，避免恒定损失。原文件不改，修正只用于新入口；基线对照应使用一致设定。

## 8. 验证范围

本包的33项自动检查全部通过，覆盖：四图严格配对、组级划分与测试重复排除、同步几何增强、模块插入位置、天气门控梯度阻断、私有分支可训练性、八项损失合成、原权重加载、新权重恢复以及恶劣双模态推理输出。CPU 验证环境为 Python3.12 / PyTorch2.5.1；代码按 Python3.8+ / PyTorch2.0 兼容接口编写。

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

没有真实 AWMM 图像或原模型／教师权重随源码上传，所以无法在此环境完成真实数据训练、真实教师数值检查或 A40 CUDA 检查；第4步的真实短跑用于验证这些服务器条件。合成检查通过不代表融合效果提高，效果需要实际训练和独立测试。
