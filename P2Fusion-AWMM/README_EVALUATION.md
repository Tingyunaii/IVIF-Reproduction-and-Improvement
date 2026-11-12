# 四基准评估与同实现原模型对照

`evaluate_p2fusion.py` 来自用户提供的参考评估器，保留其算法、注释和研究用途声明。它不是已核实的P²Fusion官方评估器。其算法自检通过不能证明与论文评估口径相同。VIF默认选择两路平均，所有三种候选VIF诊断值仍在summary中保留；应固定口径，不能根据接近论文的程度挑选。

更新仅修正数据集参考表和文件配对：

- `--dataset MSRS|M3FD|FMB|RoadScene` 选择对应论文数值和默认数量。
- `--match-stem` 使用无扩展名的样本ID，兼容JPG/BMP输入和统一PNG结果。重复ID、缺失样本和尺寸不等仍报错，不静默取交集，不自动缩放。
- 四项指标函数与上传脚本逐值比较一致。旧脚本给所有数据集打印MSRS参考值，旧的其他数据集差值应忽略；旧`means`不因参考表错误而失效。

服务器离线使用：在联网电脑下载本文件对应分支的最新 `evaluate_p2fusion.py` 和 `infer_original_p2fusion.py`，上传到项目根目录。保留已有权重、结果和评估目录。

RoadScene可先补算：

```bash
python evaluate_p2fusion.py --dataset RoadScene --match-stem \
  --ir-dir /home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets/RoadScene/ir \
  --vi-dir /home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets/RoadScene/vi \
  --fused-dir results/cross_dataset/RoadScene \
  --out-dir evaluation/awmm_original_y/RoadScene_stem \
  --expected-count 221 --vif-mode mean
```

没有报错之前，不宣称已完成RoadScene评估。如果仍报配对错误，新的异常会显示三路数量及缺失样例，按实际文件核对，不删除图像凑数。

## 先用同一实现测试原始P²Fusion权重

论文模型在MSRS/FMB混合训练；AWMM模型训练数据、任务难度和优化过程不同。为了区分模型差距与评估差距，使用原始融合权重（不是SegFormer教师，也不是AWMM改造模型）生成一个MSRS对照。

```bash
CUDA_VISIBLE_DEVICES=0 python infer_original_p2fusion.py \
  --checkpoint /实际路径/原P2Fusion融合权重.pth \
  --ir-dir /home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets/MSRS/ir \
  --vis-dir /home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets/MSRS/vi \
  --output results/original_p2fusion/MSRS --device cuda

python evaluate_p2fusion.py --dataset MSRS --match-stem \
  --ir-dir /home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets/MSRS/ir \
  --vi-dir /home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets/MSRS/vi \
  --fused-dir results/original_p2fusion/MSRS \
  --out-dir evaluation/original_p2fusion/MSRS \
  --expected-count 361 --vif-mode mean
```

原模型入口使用上传源码的原网络、原Y输入和Cr/Cb还原，不带FDN/DWAM；权重严格加载，架构不符会报错，不忽略参数。CPU合成权重的入口检查通过，用户实际原权重仍需服务器验证。

如果原权重在本评估器下也低于论文，应先查指标实现、权重版本与测试划分。如果原权重明显优于AWMM模型，则支持常规基准性能下降的判断，再排查训练分布、模型结构、学习率曲线和辅助损失。单凭当前分数不能指定某一个模块为原因。

最终还需在AWMM恶劣天气保留测试集比较原模型和改进模型，衡量天气鲁棒性的目标是否达成；四个常规数据集是跨数据集性能的另一部分。模块贡献需要与相同AWMM数据/监督下训练的原模型基线比较。
