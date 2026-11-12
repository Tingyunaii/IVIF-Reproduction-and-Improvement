# 第一次 RGB/R 单通道实验：色彩还原前后对比

第一次300轮实验保存于旧项目 `runs/awmm_bs32_300/best.pt`。该模型读取 RGB，融合损失默认使用第一通道 R；“单通道输出”不等于“按 Y 训练”。后来的 `P2Fusion-AWMM-originalY` 是独立的原版 Y 实验，两者都输出一个通道。

本脚本使用现有推理入口的 `--color-source input`：给恶劣可见光 RGB 三通道添加相同偏移 `scalar - R_vis`，再裁剪到 [0,1]。裁剪前保留输入的通道色差，输出 R 等于预测标量。它符合旧实验的 R 约定，是颜色后处理；不等同于原版 Y/CrCb 还原，也不是训练了一个 RGB 解码器。

同时重新导出旧模型的原始灰度结果，评估色彩还原前后的变化。原始灰度只是标量的灰度显示，不能称为网络融合 Y。整个过程无需重新训练，不改权重，不修改训练配置，不使用测试干净图作为颜色来源。

## 离线更新与运行

已完成上一项 Y 诊断的服务器，只需从最新 `awmm-adaptation` 分支下载新增的 `P2Fusion-AWMM/compare_awmm_legacy.py`，放进 **当前 `P2Fusion-AWMM-originalY/`** 项目根目录。其余脚本沿用刚才 Y 诊断版本，不需要进入或覆盖第一次训练的旧项目。

```bash
cd /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-originalY

CUDA_VISIBLE_DEVICES=0 python -u compare_awmm_legacy.py \
  --legacy-checkpoint /home/zj/miniconda3/envs/p2fusion/project/IVIF-Reproduction-and-Improvement-awmm-adaptation/P2Fusion-AWMM/runs/awmm_bs32_300/best.pt \
  --baseline-dir evaluation/awmm_y_diagnostic \
  --out-dir evaluation/awmm_legacy_comparison \
  --expected-count 50 --vif-mode mean --device cuda
```

`--data-root` 默认仍为 `/home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k`。如果第一次权重或上一项 Y 诊断的目录实际位置不同，修改对应参数；例如上次输出用了 `evaluation/awmm_y_diagnostic_v2`，这里也要使用相同目录。`--out-dir` 必须是新目录。

## 报告

复用上一项完整 Y 诊断中的作者和当前模型彩色图，只对旧模型分别运行三种天气的灰度/彩色推理。四路结果共用同一套 Pillow L 参考和同一四指标实现，干净/恶劣参考仍分开汇总。严格匹配全部 ID 与尺寸，权重严格加载；旧 checkpoint 必须确实是 RGB/R 实验。重新计算复用结果，并与上次 `rgb_l` 汇总核对，发现变化就报错。

| model | 含义 |
| --- | --- |
| `author` | 作者权重，原版 Y/CrCb 还原 |
| `original_y` | 当前严格 Y 实验，原版 Y/CrCb 还原 |
| `legacy_gray` | 第一次 RGB/R 模型的原始单通道灰度显示 |
| `legacy_color` | 同一旧模型，使用恶劣可见光按 R 锚点还原颜色 |

`summary.csv` 含四模型、三种天气与 ALL150组的均值；`per_image.csv` 含逐图完整指标和 MI 分量。`paired_deltas.csv` 与 `delta_summary.csv` 的 comparison 列区分三个差值：

- `legacy_color_minus_legacy_gray`：旧模型色彩还原后减还原前。
- `legacy_color_minus_original_y`：旧模型色彩还原后减当前 Y 实验。
- `legacy_color_minus_author`：旧模型色彩还原后减作者模型。

`run.json` 记录旧权重、原对照权重、脚本和复用图像的哈希，以及四模型的输入/颜色约定。`images/legacy_gray/`、`images/legacy_color/` 保存旧模型新导出的图；作者和当前模型图保留在原 Y 诊断目录。

请返回 `summary.csv`、`delta_summary.csv`、`paired_deltas.csv`、`run.json`。这是完整流程对比；旧实验还包含最初 batch8 到 batch32 的训练历史，不应将全部差异归因于输入通道或单个模块。干净双模态只是评估参考，不是真实融合 GT。
