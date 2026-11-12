# 旧 RGB/R 模型：四基准彩色推理与评估

使用第一次300轮实验 `runs/awmm_bs32_300/best.pt`，沿用刚完成 AWMM 对比时的 `--color-source input` 还原：`RGB_vis + scalar - R_vis` 后裁剪并量化。旧模型输出按 R 监督，不能直接称为 Y；此后处理与原版 Y/CrCb 还原有区别，但保持旧实验的训练约定。

## 离线运行

已运行上一项旧模型对比的服务器，只需下载新增的 `P2Fusion-AWMM/evaluate_legacy_benchmarks.py`，上传到当前 `P2Fusion-AWMM-originalY/` 根目录。已有 `infer_awmm.py`、`evaluate_p2fusion.py`、`awmm/`、`compare_awmm.py` 沿用当前版本。无需修改训练、配置、模型或权重。

```bash
cd /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-originalY

CUDA_VISIBLE_DEVICES=0 python -u evaluate_legacy_benchmarks.py \
  --checkpoint /home/zj/miniconda3/envs/p2fusion/project/IVIF-Reproduction-and-Improvement-awmm-adaptation/P2Fusion-AWMM/runs/awmm_bs32_300/best.pt \
  --test-root /home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets \
  --out-dir evaluation/legacy_color_benchmarks \
  --vif-mode mean --device cuda
```

默认依次推理 MSRS361、M3FD300、FMB280、RoadScene221，共1162组。输入目录均为 `<test-root>/<dataset>/ir` 与 `vi`。输出目录必须不存在；失败后可换新目录重新运行，也可指定 `--datasets RoadScene` 仅补跑一个数据集。原目录保留，不覆盖已有图像或报告。

脚本先核对四套输入 ID 和数量，再推理、还原颜色、检查输出均为 RGB，最后调用现有 `evaluate_p2fusion.py` 计算 FQI_E、VIF、Qabf、MI。按无扩展名 ID 配对，兼容 RoadScene 的 JPG/BMP 输入与 PNG 输出，不删除缺失样本、不自动缩放。任何子任务报错即停止，已完成数据集报告保留。

使用与之前一致的 Pillow RGB→L 灰度和 VIF mean；每套数据集打印其自身论文参考值。旧模型在 AWMM 上训练，因此这些是跨数据集结果；参考评估器仍非已核实的官方评估器。灰度指标衡量的是保存后的彩色结果转 L。

## 输出

- `images/{MSRS,M3FD,FMB,RoadScene}/`：旧模型还原后的彩色 PNG。
- `metrics/<dataset>/per_image.csv`、`summary.json`、`summary.md`：每套数据集的完整评估报告，包括 MI 和 VIF 分量。
- 根目录 `summary.csv`：四数据集四指标均值，各自独立，不混为一个跨数据集均值。
- 根目录 `per_image.csv`：全部逐图指标。
- 根目录 `run.json`：权重与脚本 SHA256、输入和色彩约定、配对口径。
- 根目录 `summary.json`：汇总完整报告。

运行完返回根目录的 `summary.csv`、`run.json`；若要进一步做逐图比较，再提供 `per_image.csv`。不需要先上传1162张输出图。
