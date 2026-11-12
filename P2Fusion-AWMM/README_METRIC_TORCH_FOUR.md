# 用服务器 Metric_torch.py 计算 CC、PSNR、SCD、SSIM

只读取已有旧模型彩色图，不加载融合模型、不重新推理、不训练、不修改原指标代码或旧结果。默认五个数据集为 AWMM、MSRS、M3FD、FMB、RoadScene。

## 离线使用

下载新增 `P2Fusion-AWMM/evaluate_metric_torch_four.py` 到当前 `P2Fusion-AWMM-originalY/` 项目根目录。服务器原文件保持在 `/home/zj/IVIF_ZOO/Metric/Metric_torch.py`，SSIM 的实际实现保持在同目录 `ssim.py`；如果位置不同，可用 `--metric-file`、`--ssim-file` 指定。

```bash
cd /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-originalY

python -u evaluate_metric_torch_four.py \
  --metric-file /home/zj/IVIF_ZOO/Metric/Metric_torch.py \
  --awmm-images evaluation/awmm_legacy_comparison/images/legacy_color \
  --benchmark-images evaluation/legacy_color_benchmarks/images \
  --out-dir evaluation/legacy_color_metric_torch_four \
  --device cpu
```

默认数据根目录仍为之前的 AWMM-100k 和 valsets，可用 `--awmm-root`、`--test-root` 修改。若前面运行时改过输出目录，也需修改两个 `--*-images` 参数。输出目录必须不存在，以保留之前所有报告。可用 `--device cuda`；默认 CPU，避免依赖外部 SSIM 实现的 CUDA 支持。

## 保留的计算方式

从指定 `Metric_torch.py` 的 AST 原样编译 `CC_function`、`PSNR_function`、`SCD_function`、`SSIM_function` 以及 SCD 所需的 `corr2`。不复制或改写公式，也不执行未选指标的导入，因此不需要 Qabf、Nabf、scikit-learn 等未使用依赖。

SSIM 仍导入服务器真正的 `ssim.py::ssim`。先用固定格式样本依次检查4种输入：float32张量 `[1,1,H,W]`、float32张量 `[H,W]`、uint8 NumPy灰度数组、PIL L图像，记录选定格式及探测失败信息；可用 `--ssim-layout nchw|hw|numpy_hw|pil` 固定。全部不接受时明确报错，不替换另一种 SSIM 算法。服务器 SSIM 源文件未上传，因此本地只验证了适配器与测试后端，最终 SSIM 实现以服务器文件和记录的哈希为准。

服务器出现 `pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>` 时，说明其 SSIM 内部会做图像到张量的转换。修正版自动使用 NumPy uint8 或 PIL 输入；不要传 float32 NumPy 数组冒充原始8位图，因为 torchvision ToTensor 对 uint8 会除255、对浮点数组不会。脚本保留后端内部的转换，不再提前额外除255。原图像像素值逐位置保留。

三路图像均先经 Pillow `convert('L')`，不做逐图最大最小值归一化。CC、SCD 接收0..255的2维float32张量；PSNR 按原函数内部除255并计算两路均方误差的平均。SSIM 按后端接受的格式传入等值的float32张量、uint8数组或PIL图像；包装函数分别计算两路 SSIM 后取平均，保留服务器 `ssim()` 自身默认参数及内部归一化，不额外修改其范围参数。实际使用的格式、源码及脚本 SHA256 全部写入报告。

## 参考集合与输出

- AWMM：每种天气50组，分别使用干净 `gt_ir/gt_vi` 与恶劣 `ir/vi` 作为参考，另外汇总各150组。两套参考分开，不重复计算图像数量；干净双模态不是真实融合 GT。
- MSRS361、M3FD300、FMB280、RoadScene221：分别以各自测试集 IR/VIS 为参考。
- 按无扩展名 ID 配对，不取交集、不缩放；数量、缺失或尺寸错误会报错。

输出：

- `summary.csv`：五个数据集汇总，AWMM有干净/恶劣两行，因此共6行结果。
- `summary_by_subset.csv`：另外列出 AWMM 雨、雾、雪各自结果。
- `per_image.csv`：逐图四指标，共1462行（AWMM150组评估两套参考300行，加四个常规集1162行）。
- `run.json`：实际 Metric_torch.py、ssim.py 和脚本哈希、输入范围与形状、原结果来源记录。

原函数若产生 NaN/Inf，逐图和均值如实保留，不改成0、不偷偷丢样本；`*_finite` 列显示有限值数量。这样组会表格可判断均值是否覆盖完整测试集。未定义值可能来自常量图或 SCD 的零方差残差，不能据此修改原公式后仍称为同一实现。

运行后返回 `summary.csv`、`summary_by_subset.csv`、`run.json`。需要排查异常时再提供 `per_image.csv`。这些新指标结果与此前 FQI_E/VIF/Qabf/MI 报告分别保存。
