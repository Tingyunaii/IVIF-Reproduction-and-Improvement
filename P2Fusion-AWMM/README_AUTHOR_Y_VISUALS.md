# 作者模型与当前 Y 模型：AWMM和四基准可视化

只比较作者 P²Fusion 与 `runs/awmm_original_y_bs32_300/best.pt`。明确排除旧 RGB/R 模型，不改变训练、权重、输入图或旧结果。

## 离线运行

从最新分支下载新增 `P2Fusion-AWMM/visualize_author_vs_y.py`，上传到当前 `P2Fusion-AWMM-originalY/` 根目录。现有两份推理入口、`awmm/` 和模型代码继续使用。

```bash
cd /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-originalY

CUDA_VISIBLE_DEVICES=0 python -u visualize_author_vs_y.py \
  --author-checkpoint /home/zj/miniconda3/envs/p2fusion/project/checkpoints/author_original/p2fusion_author_best_E.pth \
  --y-checkpoint runs/awmm_original_y_bs32_300/best.pt \
  --awmm-cache evaluation/awmm_y_diagnostic \
  --out-dir visualizations/author_vs_current_y \
  --device cuda
```

默认生成全部1312组并排图：AWMM150组、MSRS361、M3FD300、FMB280、RoadScene221。输出目录必须不存在。先快速浏览时，可额外加 `--limit 6`，每组按自然排序后的ID固定间隔选择6张（七组共42张），四基准也仅推理选中的图；重新全量运行时换新的输出目录。选样不依赖分数，不自动挑选有利结果。

默认数据根目录沿用之前的 AWMM-100k 与 valsets，分别可用 `--awmm-root`、`--test-root` 修改。AWMM若使用不同的 Y 诊断目录，修改 `--awmm-cache`。

## 两模型结果来源

AWMM优先复用 `evaluation/awmm_y_diagnostic/images/rgb/{author,adapted}/{Rain,Haze,Snow}/`。先比较缓存报告中两份权重的 SHA256 与本次指定权重，再核对全部图像ID。缺少缓存报告或权重不匹配时，在新的可视化输出目录重新推理。若缓存报告存在但文件缺失或ID不齐，会明确报错，不混用不完整结果。

四个常规数据集使用两份指定权重新推理，写入新目录 `predictions/`，避免将旧 RGB/R 结果当成当前 Y。两个模型都使用原版 Y 输入与观察可见光的 Cr/Cb 还原，当前模型被检查必须是 `original_y` 实验。AWMM干净图只展示，不进入推理。

## 图像排列

AWMM按 Rain、Haze、Snow 分开，每张6列：

```text
Clean IR | Clean VIS | Degraded IR | Degraded VIS | Author P2Fusion | Current Y model
```

四个常规数据集各自独立，每张4列：

```text
IR input | VIS input | Author P2Fusion | Current Y model
```

常规数据集没有这里对应的配对恶劣/干净图，不生成或伪造额外参考。模型输出必须是实际还原后的RGB。原始六/四路图尺寸不一致则报错，不通过缩放掩盖错位；展示时整组采用相同等比例缩小，不调整亮度、对比度或饱和度。

预览并排图采用高质量JPEG，不用于重新计算指标；新推理模型原图仍为 `predictions/` 下PNG，复用的AWMM原图保留在原目录。可用 `--panel-width 640` 增加展示分辨率（不放大小于该宽度的原图）。

## 输出与查看

```text
visualizations/author_vs_current_y/
  index.html                       # 全部七组的离线图库入口
  AWMM_Rain/                       # 各样本并排图、overview.jpg、index.html
  AWMM_Haze/
  AWMM_Snow/
  MSRS/
  M3FD/
  FMB/
  RoadScene/
  predictions/                     # 本次补跑的两模型原始彩色预测
  preview/                         # 七张概览图与独立离线index.html
  preview.zip                      # 便于下载/上传的概览包
  manifest.csv                     # 每张并排图的两个模型结果路径
  run.json                         # 权重、推理脚本、源图哈希与选样记录
```

可在VS Code直接打开各组JPG查看；也可下载 `preview.zip` 到联网电脑，解压后双击 `index.html`，离线查看七组概览。每组概览默认4张固定间隔样本，可用 `--overview-rows` 修改；其余所有样本保存在各组图库，不会因为未进入概览而被删除。

若希望继续分析图像，返回 `preview.zip` 即可；需要排查结果来源时附 `run.json`。完整图库也可从服务器下载各组目录查看，无须一次上传全部原始预测。
