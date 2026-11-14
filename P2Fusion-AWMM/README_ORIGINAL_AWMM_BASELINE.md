# 同条件原 P²Fusion 在 AWMM 上从头训练

本实验为“原P²Fusion网络 + 与当前Y实验相同的AWMM数据/干净参考/双教师训练协议”。它用来衡量增加FDN、DWAM及辅助约束的整体收益，不是复现原论文在MSRS/FMB上的训练结果，也不是给原P²Fusion作者权重做AWMM微调。

网络直接继承 `models/P2Fusion.py::p2fusion`，没有FDN、DWAM、辅助重建头。只改forward的返回格式，保留原计算顺序并暴露原本已计算但未返回的VIS质量标量。全部参数名与原网络一致，可直接严格加载进原网络推理入口。仅保留原五项损失；五项实现与改进模型共用，保持干净参考、冻结IST/BRISQUE教师、Y输入和SSIM data_range=1一致。

## 离线更新

从最新 `awmm-adaptation` 分支ZIP解压，将包内 `P2Fusion-AWMM/` 作为一个新项目上传到服务器，命名为：

`/home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-baseline`

不覆盖已有 `P2Fusion-AWMM-originalY` 项目。新包包含训练所需的原网络、基线适配器、公共训练/损失入口及配置生成工具；不含用户训练权重和划分。进入新项目后，链接复用旧项目已有的划分和教师：

```bash
cd /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-baseline

ln -s /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-originalY/splits splits
ln -s /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-originalY/nets/best_epoch_weights.pth nets/best_epoch_weights.pth
```

这些命令在新项目中首次执行即可。原数据目录不复制、不改动；BRISQUE继续使用同一用户已有缓存，无需联网重新下载。

## 生成匹配配置

从实际完成训练的Y实验best.pt提取配置，只复用配置，不加载其融合权重。原数据根目录、manifest、裁剪、seed、batch、轮数、学习率、调度、优化器设置及五项权重均保留。删除三个辅助损失权重并切换模型类型，写入新配置和独立run目录：

```bash
cd /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-baseline

python tools/prepare_original_awmm_baseline.py \
  --reference-checkpoint /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-originalY/runs/awmm_original_y_bs32_300/best.pt \
  --out configs/awmm_original_baseline_matched.json \
  --run-dir runs/p2fusion_original_awmm_y_bs32_300
```

已有配置不覆盖，再次运行需换新配置路径或直接使用已生成文件。默认对应batch32、300轮，但以实际参考checkpoint的配置为准。保留现有较早衰减的学习率计划用于对照；若以后调整，应为两模型分别进行匹配实验。

## 先做合成检查，再开始真实训练

```bash
CUDA_VISIBLE_DEVICES=0 python train_awmm.py \
  --config configs/awmm_original_baseline_matched.json \
  --device cuda --smoke-test

CUDA_VISIBLE_DEVICES=0 python -u train_awmm.py \
  --config configs/awmm_original_baseline_matched.json \
  --from-scratch --device cuda
```

合成检查明确标记synthetic_smoke_only、不读真实数据或教师、不保存训练权重；它应只报告5项损失和total，model_variant应为original_p2fusion。真实训练使用已有6318训练/711验证划分（以实际manifest为准），完整验证，不传 `--val-batches 2`。`--from-scratch` 不加载作者或改进融合权重，冻结教师仍需已有预训练文件。

若batch32且训练样本6318、drop_last=False，每轮198批、300轮59400批；AMP溢出会跳过参数更新，实际成功step可能略少。不要给基线单独改batch、学习率或轮数后仍称同条件对照。

## 保存与恢复

```text
runs/p2fusion_original_awmm_y_bs32_300/
  config_resolved.json
  metrics.jsonl
  best.pt
  last.pt
```

断点恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_awmm.py \
  --config configs/awmm_original_baseline_matched.json \
  --resume runs/p2fusion_original_awmm_y_bs32_300/last.pt --device cuda
```

恢复时核对模型类型、配置和manifest，不允许把改进模型checkpoint直接resume到原网络。旧实验目录不改动，保存目录如已有权重则必须resume或选择新目录。

## 后续推理

`infer_awmm.py` 现在按checkpoint的model_variant自动构建正确网络，原版Y/CrCb颜色还原继续生效。`infer_original_p2fusion.py` 也能严格加载本基线的model参数，因为其网络参数名与原版完全一致。

基线best.pt按五项验证总损失选择，而已有改进best.pt按八项选择，checkpoint选择标准存在这一差别；不能直接比较两者训练total的大小。最终用相同保留测试集和相同指标评估。若要进一步隔离模块本身贡献，还需FDN、DWAM和辅助损失的分步消融。

## 训练完成：AWMM三模型对比

新增 `compare_awmm_original_baseline.py`，只需下载这个文件到基线项目根目录。它对新基线的best.pt进行150组测试推理，按原版流程还原彩色；复用原Y项目已完成的作者与当前Y模型彩色结果，并重新计算其分数核对一致性。

```bash
CUDA_VISIBLE_DEVICES=0 python -u compare_awmm_original_baseline.py \
  --checkpoint runs/p2fusion_original_awmm_y_bs32_300/best.pt \
  --comparison-dir /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-originalY/evaluation/awmm_y_diagnostic \
  --out-dir evaluation/original_awmm_vs_current_y \
  --vif-mode mean --device cuda
```

默认数据根目录保持AWMM-100k，三天气各50组。两套参考（干净/恶劣）分开，使用此前相同FQI_E、VIF mean、Qabf、MI参考评估器；保留MI分量。干净图仅供评估，不传入推理。输出目录必须不存在。

`summary.csv` 汇总三天气及ALL；`per_image.csv` 为逐图指标；`paired_deltas.csv` 与 `delta_summary.csv` 分别记录“当前Y改进模型减AWMM训练的原网络”和“AWMM训练的原网络减作者权重”。其中 `author` 是作者原权重，`original_awmm` 是本次新基线，`current_y` 是现有FDN/DWAM模型。

新彩色图保存在 `images/original_awmm/{Haze,Rain,Snow}/`，复用图仍在原Y项目。`run.json` 记录权重、参考实验、脚本哈希和差值方向。返回 `summary.csv`、`delta_summary.csv`、`paired_deltas.csv`、`run.json` 即可继续分析。
