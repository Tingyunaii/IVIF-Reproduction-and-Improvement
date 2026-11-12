# AWMM 融合 Y 与色彩还原诊断（无需训练）

使用现有作者权重与 `original_y` 实验的 `best.pt`，对 AWMM 的三种天气各50组重新推理。每个模型的 RGB 与 Y 来自同一次前向计算；Y 是原版量化步骤后的单通道结果，尚未拼接 Cr/Cb，也没有经过 RGB 裁剪。它不是从已有彩色 PNG 反算的 Y，也不对每张图做最大最小值归一化。

## 离线服务器更新

在联网电脑下载仓库 `awmm-adaptation` 分支 ZIP。解压后，从包内 `P2Fusion-AWMM/` 上传以下4个文件到服务器现有 `P2Fusion-AWMM-originalY/` 的相同相对位置：

- `diagnose_awmm_y.py`（新增）
- `infer_awmm.py`
- `infer_original_p2fusion.py`
- `awmm/preprocessing.py`

已有 `compare_awmm.py` 和 `evaluate_p2fusion.py` 继续使用。无需上传训练、网络或配置文件；不修改 `runs/`、`splits/`、数据集、已有推理和评估结果。若希望保留脚本旧版本，可在上传前复制这3个已有文件。新增脚本仅加载 checkpoint，不写入权重，不启动训练，不加载教师，不联网下载权重。

## 运行

```bash
cd /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-originalY

CUDA_VISIBLE_DEVICES=0 python -u diagnose_awmm_y.py \
  --data-root /home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k \
  --author-checkpoint /home/zj/miniconda3/envs/p2fusion/project/checkpoints/author_original/p2fusion_author_best_E.pth \
  --adapted-checkpoint runs/awmm_original_y_bs32_300/best.pt \
  --out-dir evaluation/awmm_y_diagnostic \
  --expected-count 50 --vif-mode mean --device cuda
```

输出目录必须是新的，避免覆盖和混入旧图；如失败后重试，换用 `evaluation/awmm_y_diagnostic_v2`。两模型都只接收恶劣 IR/VIS；干净图仅供评估。严格核对全部四路样本 ID 和尺寸，不取交集、不自动缩放。推理仍保持原版 Y 输入和 Y/CrCb 色彩还原。

## 如何解读

`summary.csv` 每行标记天气、参考集合、协议和模型，三套协议保持分开：

| protocol | 融合结果 | 参考图像 | 用途 |
| --- | --- | --- | --- |
| `rgb_l` | 原版彩色还原，再经 Pillow 转 L | Pillow L | 与上次评估口径一致 |
| `direct_y` | 网络融合 Y 量化为8位灰度 | **与 rgb_l 完全相同的 Pillow L** | 与 rgb_l 对比，单独测量输出后处理影响 |
| `direct_y_opencv_reference` | 同一网络融合 Y | 原版 OpenCV Y | 补充查看与训练输入一致的亮度评估；同时改变了参考图转换方式，不能单独用来判断后处理影响 |

`clean_reference` 使用配对干净 IR/VIS，`degraded_reference` 使用恶劣输入。干净双模态也不是真实融合 GT；恶劣参考指标可能奖励残留天气。三套协议使用同一四指标函数与 VIF 选择，不打印 AWMM 对应不了的论文参考数值。

- `postprocess_delta_summary.csv`：按天气、参考集合、模型汇总 **direct_y 减 rgb_l**。正值表示导出直接 Y 后指标更高，负值表示更低。若两者接近，先前指标下降主要由颜色还原导致的解释就缺乏支持；若变化明显，需要结合各天气、各指标和逐图差值判断。
- `postprocess_delta.csv`：同一方向的逐图差值。
- `paired_deltas.csv`：每套协议内 **adapted 减 author**，不可与上一项的差值方向混淆。
- `per_image.csv`：每个样本的完整指标；`summary.csv` 也保留 `MI_ir`、`MI_vi` 的均值，帮助定位雾天 MI 差距来自哪一路参考。
- `pixel_changes.csv`：每张图的 RGB→L 与直接 Y 的像素差异，包括0..255尺度的平均绝对差、最大差和绝对差大于1的像素比例。像素差较小也可能改变离散直方图 MI，需结合实际指标变化。
- `run.json`：checkpoint 与脚本 SHA256、参考协议和差值方向。
- `images/y/{author,adapted}/{Haze,Rain,Snow}/`：色彩还原前的灰度 Y；`images/rgb/...` 是同次推理的彩色结果。

`ALL` 按图像数汇总150组。先返回 `summary.csv`、`postprocess_delta_summary.csv`、`paired_deltas.csv`、`run.json` 即可分析，不需要先上传全部图像。

## 单独导出

两个推理入口现在都接受 `--save-y-dir 新目录`，在保留原彩色输出的同时保存 Y。适配入口只允许将 `original_y` 训练的输出标记为 Y，防止把旧 RGB/R 实验误当作亮度模型。未传此参数时保留原有推理行为。与上传原版 `test.py` 的逐像素还原对照测试继续保留。
