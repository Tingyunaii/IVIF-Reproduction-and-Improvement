# 当前 Y 模型结构与数据流

图依据本仓库实际 `models/awmm_network.py`、`models/P2Fusion.py`、`awmm/losses.py`、`awmm/runtime.py` 与 `train_awmm.py` 绘制。它描述当前 `original_y` 实验；旧 RGB/R 实验的新增模块结构相同，但图像输入、监督和颜色后处理约定不同。

- `current_y_architecture.png`：4000×2400，适合组会直接插入PPT。
- `current_y_architecture.svg`：矢量版，可放大及在支持SVG的软件中编辑。
- `tools/draw_current_y_architecture.py`：绘图源程序，需要Matplotlib、Pillow与Noto CJK字体；不修改或运行训练。

## 推理主干

1. 读取退化IR和VIS。按原版uint8 OpenCV Y提取方式转换，分别将Y复制3份后除255。VIS的原始Cr/Cb另外保留用于最后颜色还原。
2. 两模态通过同一套 `conv_first → PatchEmbed(stride4) → RSTB → LayerNorm` 编码参数。必要时先补边至32的倍数，得到 `F ∈ R^(B×60×h×w)`，h、w为补边后尺寸的四分之一。
3. 每个模态有独立FDN：`U=F+R(F)`，`V=P(F)`。R和P各为 `Conv1×1 → GroupNorm → GELU → Conv3×3`，同模态的干净/退化输入共享这些参数。U为期望学习的域不变内容特征，V为期望学习的域私有条件特征；损失用于促进分离，不保证完全解耦。
4. 对应DWAM接收U/V。天气门控 `Gw = sigmoid(Conv1×1(ReLU(Conv1×1(sg(V)))))`，维度为B×60×h×w，隐藏通道为15，**没有全局池化**。区域门控 `Gr = sigmoid(Conv7×7(cat(mean_channel(U),max_channel(U))))`，维度为B×1×h×w。广播相乘后残差调制：`Uhat=U+U*Gw*Gr`。仅V进入天气门控的那条路径detach，私有编码器仍能通过正交和残差重建学习。
5. **FDN与DWAM均在原 `proj` / `prompt_head` 分流之前**。两条头都读取同一Uhat：proj输出融合特征，prompt_head输出模态提示。红外提示先得到单通道图，再复制为8通道供融合；红外蒸馏只使用其中一份。可见光提示图经score_head、sigmoid、GAP输出每张图一个质量分数，用于蒸馏。
6. 两路融合特征及两路提示进入原P²Fusion的4个渐进融合块。两路最终特征拼接后由重建头上采样4倍，输出单通道融合Y；裁掉补边。最后按原版量化并与**输入退化VIS**的Cr/Cb合并，转换得到RGB融合图。

## 训练额外数据流

每组训练样本还包含对齐的干净IR/VIS。它们采用相同Y预处理，并通过相同E和对应FDN取Uc/Vc及辅助重建结果，不进入提示头或融合解码器。实现会计算clean DWAM的临时输出，但该输出不被clean辅助损失使用，图仅绘出有效依赖路径。

- **跨域一致性**：两模态分别 `MSE(Ud, sg(Uc))` 后取平均。只有一致性目标路径上的Uc detach；干净分支仍通过其他辅助损失训练。
- **正交约束**：对每个位置沿通道归一化U/V，计算余弦相似度平方；覆盖两模态、干净与退化两种条件。
- **自监督**：内容头DU从Ud和Uc都重建干净输入；私有头DV从Vd重建有符号差值 `Id−Ic`，从Vc重建0。均使用L1，`Lself=Lcontent+Lresidual`。当前Y实验这些3通道目标都是复制Y，不是学习真实彩色重建；DU使用sigmoid，DV使用tanh。
- **双教师**：干净IR送入冻结SegFormer IST，得到显著性概率图；干净VIS送入冻结BRISQUE，得到 `clip((100−score)/100,0,1)` 的图像级质量目标。红外提示对显著性图做BCE；VIS提示汇聚出的质量标量对质量目标做MSE。BRISQUE标量不是逐像素质量真值。
- **融合约束**：单通道融合Y与配对干净IR/VIS的Y计算强度、梯度及SSIM损失。不存在真实融合图GT。

总损失按当前配置为：

`L=8Lint+25Lssim+20Lgrad+0.5Ldistill_IR+5Ldistill_VIS+Lcon+0.01Lorth+Lself`。

图中的训练SSIM显式使用归一化图像的 `data_range=1`；它与最近单独调用的 `Metric_torch.py / ssim.py` 离线评估实现不是同一个配置，不能混淆。

推理仍需要私有特征V来生成天气门控，停用的是干净分支、双教师与辅助重建头。两Sigmoid门控令逐元素残差倍率为 `1+GwGr∈[1,2]`，属于相对增强而非直接绝对衰减；实际去天气效果由整个网络及训练约束共同学习。
