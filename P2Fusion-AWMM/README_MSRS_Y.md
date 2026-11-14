# MSRS完整Y模型：轻度光照配对训练

使用MSRS的975训练、108验证、361测试固定清单，从头训练完整P²Fusion-AWMM Y架构（FDN、DWAM、双提示与辅助重建全部保留）。这是正常场景融合加光照一致性正则实验，不合成雨雾雪，也不声称采用了真实天气配对。

普通融合可以只用配准IR/VIS计算原五项损失；为了让完整Y模型的跨条件一致性和私有残差重建有实际目标，本实验在训练时在线构造轻度光照差异。原图作参考，扰动图作学生输入，无需额外图片文件。

## 独立项目与复用数据

下载最新分支ZIP，把包内P2Fusion-AWMM目录作为新项目上传到：

`/home/zj/miniconda3/envs/p2fusion/project/P2Fusion-MSRS-Y`

保留现有AWMM项目与已训练权重。新项目不包含用户原始数据、清单和教师权重，首次建立以下链接即可复用：

```bash
cd /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-MSRS-Y

ln -s /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-baseline/data_splits data_splits
ln -s /home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-originalY/nets/best_epoch_weights.pth nets/best_epoch_weights.pth
```

使用刚完成的 `data_splits/msrs_y`，不重新划分或把测试图加入训练。数据根目录仍为原Dataset；同一用户的BRISQUE缓存继续使用，不需要联网重新下载。

## 训练设置

- 输入：原版OpenCV uint8 Y提取、重复3通道、除255；IR/VIS及原图/扰动图的裁剪翻转完全同步。
- 75%的训练对施加扰动，25%保持原条件；两模态光照参数独立，任何像素坐标均不移动。
- 变换：围绕图像均值作对比度变化，再clamp、gamma、gain、offset、clamp。gain/contrast/gamma范围均为[0.95,1.05]，offset为[-0.025,0.025]。
- 教师目标来自未扰动的参考图；内容重建目标为参考图，私有重建目标为“扰动图减参考图”，两者都是复制Y。保留原8项损失及原权重。
- 验证和测试不加光照扰动；验证固定中心裁剪256。此时一致性项通常为0，因为两条输入相同，这不是训练配对无效，也不能作为学习到完美不变性的证据。
- batch32、crop256、300轮、AdamW、初始lr1e-4、AMP，完整验证108对。975/32向上取整为每轮31批，约9300次训练批次，AMP跳步会略减少成功更新数。
- 此新实验的MultiStepLR在约第60/120/180/240/270轮衰减0.5，对应成功更新step1860/3720/5580/7440/8370。不是照搬AWMM较早衰减的调度；最终比较应记录此差别。

## 先检查再训练

```bash
CUDA_VISIBLE_DEVICES=0 python train_awmm.py \
  --config configs/msrs_y_light_bs32_300.json --device cuda --smoke-test

CUDA_VISIBLE_DEVICES=0 python -u train_awmm.py \
  --config configs/msrs_y_light_bs32_300.json --from-scratch --device cuda
```

合成smoke仅验证架构/损失/反传，不加载真实教师、不保存权重；真实训练使用已有IST权重和BRISQUE缓存。--from-scratch仅指融合网络随机初始化，不加载AWMM融合权重，教师仍为冻结预训练模型。

启动日志应包含model_variant=awmm、input_mode=original_y、data_task=msrs_light、train_samples=975、val_samples=108、val_batch_limit=0。真实训练日志仍有原8项损失。原始参考图不被扰动操作原地修改。

保存：`runs/msrs_y_light_bs32_300/{best.pt,last.pt,config_resolved.json,metrics.jsonl}`。

恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_awmm.py \
  --config configs/msrs_y_light_bs32_300.json \
  --resume runs/msrs_y_light_bs32_300/last.pt --device cuda
```

恢复时验证训练任务、扰动参数和三个清单的哈希，不能静默从AWMM断点或更改光照配置继续。

## 测试

```bash
CUDA_VISIBLE_DEVICES=0 python infer_awmm.py \
  --checkpoint runs/msrs_y_light_bs32_300/best.pt \
  --ir-dir /home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets/MSRS/ir \
  --vis-dir /home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets/MSRS/vi \
  --output results/msrs_y_light/MSRS --device cuda
```

测试只输入原始IR/VIS，保持原版Y/CrCb彩色还原。361对全部保留；best根据108对验证选择，测试图不用于选模型。固定清单仍只是精确内容分组，不能排除相邻帧等近重复。
