# 严格对齐上传原 P²Fusion 的亮度处理与色彩还原

## 修正范围与已完成训练的含义

原版 `utils/utils_image.py::imread_uint(..., n_channels=3)` 对 IR 和 VIS 都执行：OpenCV IMREAD_COLOR → BGR转RGB → RGB转YCrCb → 提取8位Y → 复制为3通道。原版 `test.py` 对模型的单通道结果先取整为8位Y，再合并**实际输入可见光**的Cr/Cb，转换为RGB保存。

初版 AWMM 加载器实际使用 RGB，第一通道融合参考因而是R；初版推理还漏掉了色彩还原。此前“完全保留原亮度约定”的说法不准确。旧300轮权重应标记为 RGB/R 实验，不能仅修改配置或检查点元数据就声称已按Y训练。旧文件和实验兼容性保留，`input_mode` 缺省按 `rgb` 解释。

新模式 `dataset.input_mode = original_y` 在**恶劣/干净、IR/VIS四路**全部执行原8位OpenCV Y处理；再做四图同步几何增强和除255。双教师接收同样的干净Y三通道，融合强度/梯度/SSIM的第一通道参考即原Y。辅助内容重建和退化残差目标也变为三份相同Y，不预测色度。

这次对齐的是亮度输入与输出色彩协议。仍保留AWMM配对干净监督、FDN/DWAM、三项解耦损失及此前明确的SSIM data_range=1修正，不声称复现原论文全部训练设置。不要将原版额外随机增强或超参数混同为这次对齐范围。

## 离线更新

联网电脑从仓库 `awmm-adaptation` 分支下载最新ZIP。在服务器原项目中备份旧代码后，上传覆盖以下内容：

- `awmm/` （包括新增 `preprocessing.py`）
- `train_awmm.py`、`infer_awmm.py`
- `configs/awmm_original_y_bs32_300.json`、`configs/awmm_original_y_finetune.json`
- `tests/`、本说明

不覆盖 `runs/`、`splits/`、数据集和 `nets/best_epoch_weights.pth`。无需重新划分数据，原清单继续使用；BRISQUE离线缓存仍有效。OpenCV沿用原p2fusion环境已有依赖。

先运行原版逐像素对照检查：

```bash
python -m unittest discover -s tests -p test_original_y.py -v
```

该测试从上传的原版文件中抽取实际 `imread_uint`、`tensor2uint` 及 `test.py` 色彩还原代码块作为对照，不是另写一个相近公式。检查彩色PNG/JPEG输入的模型数组逐像素相同，以及输出色彩还原（包括8位取整、Y/Cr/Cb顺序和必要尺寸匹配）逐像素相同。

## 路线A：严格Y协议从头训练

配置为batch32、256裁剪、300轮、学习率1e-4、编码器不冻结。

```bash
CUDA_VISIBLE_DEVICES=0 python train_awmm.py \
  --config configs/awmm_original_y_bs32_300.json \
  --from-scratch --run-dir runs/awmm_original_y_check \
  --max-steps 1 --val-batches 2
```

通过后，正式训练使用新目录，去掉两个短跑限制：

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_awmm.py \
  --config configs/awmm_original_y_bs32_300.json \
  --from-scratch --run-dir runs/awmm_original_y_bs32_300
```

此配置的里程碑为750、1500、2250、3000、3750，是原batch8的3000、6000、9000、12000、15000按样本量折算；仍是较早完成衰减，不代表已为300轮优化调度。若设计新的长周期调度，需单独记录并在各对照中保持一致。

## 路线B：保留旧权重，显式开展Y微调实验

这不是恢复原RGB实验，而是新的输入协议迁移阶段；模型参数初始化自旧权重，优化器、调度器和最优验证值重新建立。以50轮、1e-5作为可调整的初始微调方案，不保证50轮足够，应依据验证结果与图像评估。

```bash
CUDA_VISIBLE_DEVICES=0 python train_awmm.py \
  --config configs/awmm_original_y_finetune.json \
  --init runs/awmm_bs32_300/best.pt --allow-input-mode-transfer \
  --run-dir runs/awmm_original_y_finetune_check \
  --max-steps 1 --val-batches 2
```

通过后正式微调：

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_awmm.py \
  --config configs/awmm_original_y_finetune.json \
  --init runs/awmm_bs32_300/best.pt --allow-input-mode-transfer \
  --run-dir runs/awmm_original_y_finetune
```

完整新实验应记录为“旧RGB/R训练 + original_y微调”，不能报告为300轮纯Y从头训练。输入协议不同的检查点禁止直接 `--resume`；`--init` 也要求显式的迁移参数，避免误操作。仅给旧推理权重指定新Y配置会报错，而不会伪装成对齐模型。

## 严格原版彩色推理

以路线B的新Y微调权重为例：

```bash
AWMM_ROOT=/home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k
for WEATHER in Haze Rain Snow
do
  CUDA_VISIBLE_DEVICES=0 python infer_awmm.py \
    --checkpoint runs/awmm_original_y_finetune/best.pt \
    --ir-dir "$AWMM_ROOT/AWMM-Text/Test/$WEATHER/ir" \
    --vis-dir "$AWMM_ROOT/AWMM-Text/Test/$WEATHER/vi" \
    --output "results/awmm_original_y/$WEATHER" \
    --color-source original --device cuda
done
```

路线A将检查点改为 `runs/awmm_original_y_bs32_300/best.pt`。original_y检查点默认即采用 `original` 色彩还原；可用 `--color-source none` 另存融合Y。此模式不采用重建色度，不读取测试gt，不加载双教师，不运行辅助重建头。输出颜色取自恶劣可见光，所以原版色彩还原本身可能携带天气偏色，这是忠实复现的边界，不能把“严格对齐”当成“颜色退化已消除”。

已进行CPU端原版逐像素审计及完整Y数据流训练/恢复/推理合成检查。没有用户真实权重与AWMM图像在本工作区，真实Y模型质量仍需在服务器训练和评估。
