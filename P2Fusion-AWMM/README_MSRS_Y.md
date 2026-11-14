# MSRS当前Y模型实验：数据准备

训练候选为 Dataset/trainsets/MSRS/{ir,vi} 的1083对；原测试集为 Dataset/valsets/MSRS/{ir,vi} 的361对。测试图不加入训练；默认在训练候选中按固定seed3407分出约10%的验证组。

当前只完成数据准备，尚不提供未经确定的训练目标或启动命令。正常MSRS图像不自动伪装成天气退化/干净四元组。完整FDN/DWAM训练的配对策略需要确定，可选择轻度光照变化或明确的合成天气退化，二者对应不同实验任务。

在已有项目根目录放入新增的 `tools/prepare_msrs_pairs.py` 后运行：

```bash
python tools/prepare_msrs_pairs.py \
  --root /home/zj/miniconda3/envs/p2fusion/project/Dataset \
  --out data_splits/msrs_y \
  --val-ratio 0.1 --seed 3407
```

脚本读取所有配对图检查ID、尺寸，以解码RGB内容相同的IR或VIS连接分组；同一组不会跨训练和验证，连接到测试图的训练候选从训练/验证排除。相邻帧、重压缩或不同视角的近重复无法靠精确哈希完全识别，因此不能据此声称绝对场景独立。

输出train.json、val.json、test.json、excluded.json、summary.json。每条记录只描述真实IR/VIS对，不包含虚构的clean_ir/clean_vis。约975训练/108验证只是没有重复组影响时的估计，以实际summary为准；测试仍保留361对。脚本不修改原始图像，也不覆盖已有清单。

不要将这套双图清单直接交给尚未适配的AWMM四图加载器；需要配对策略和相应训练入口接好后再启动MSRS实验。
