"""Draw the code-aligned AWMM original_y architecture as an editable scientific figure."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'architecture'
OUT.mkdir(parents=True, exist_ok=True)
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(),
                     'svg.fonttype': 'none', 'axes.unicode_minus': False})
fig, ax = plt.subplots(figsize=(25, 15))
fig.patch.set_facecolor('#ffffff')
ax.set(xlim=(0, 250), ylim=(0, 150))
ax.axis('off')
fig.subplots_adjust(left=.012, right=.988, bottom=.012, top=.988)
INK = '#20334b'
BLUE = '#3678b4'
PURPLE = '#9164b9'
GREEN = '#238b75'
ORANGE = '#d58a30'


def label(x, y, value, size=13, color=INK, ha='center', va='center', weight='normal'):
    ax.text(x, y, value, fontsize=size, color=color, ha=ha, va=va, weight=weight,
            linespacing=1.35, zorder=8)


def box(x, y, w, h, value='', fill='#eff5fb', edge=BLUE, size=13, dashed=False, radius=.8):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.18,rounding_size='+str(radius),
                          linewidth=1.45, edgecolor=edge, facecolor=fill,
                          linestyle=(0, (5, 3)) if dashed else '-', zorder=3 if value else 0)
    ax.add_patch(patch)
    if value:
        label(x+w/2, y+h/2, value, size)
    return patch


def arrow(points, color=BLUE, dashed=False, width=1.5, head=True):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=width,
                linestyle=(0, (4, 3)) if dashed else '-', zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>' if head else '-',
                                mutation_scale=12, linewidth=width, color=color,
                                linestyle=(0, (4, 3)) if dashed else '-', zorder=2))


def circle(x, y, value, color=GREEN, radius=2):
    ax.add_patch(Circle((x, y), radius, facecolor='white', edgecolor=color, linewidth=1.5, zorder=5))
    label(x, y, value, 17, color)


label(3, 145, 'P²Fusion-AWMM：当前 Y 模型与新增模块', 27, ha='left', weight='bold')
label(3, 139, '实线：推理数据流    虚线：训练监督 / 参数关系    sg：仅在指定路径停止梯度', 14, ha='left', color='#607086')

# A: complete model. All adaptation lies before the original two-head split.
box(2, 29, 159, 106, fill='#fbfcfe', edge='#d5dfe9', radius=1.3)
label(5, 131, 'A  整体框架：退化双模态 → 提示引导渐进融合 → 彩色结果', 17, ha='left', weight='bold')
label(5, 125, '编码器 E 在 IR/VIS 间共享；FDN 与 DWAM 按模态独立；干净/退化条件共享对应模块参数。', 11.3, ha='left')

for modality, y in [('IR', 108), ('VIS', 80)]:
    box(5, y, 17, 11, '退化 '+modality+'\n取 Y 并复制3份', '#eef4fa', BLUE, 11.5)
    box(28, y, 19, 11, '共享编码器 E\nConv / Patch / RSTB', '#eef4fa', BLUE, 11.5)
    box(53, y, 17, 11, 'FDN_'+modality+'\nU / V', '#fff3df', ORANGE, 12)
    box(76, y, 18, 11, 'DWAM_'+modality+'\n调整后特征 Û', '#e9f7f1', GREEN, 11.5)
    arrow([(22, y+5.5), (28, y+5.5)])
    arrow([(47, y+5.5), (53, y+5.5)])
    arrow([(70, y+5.5), (76, y+5.5)], GREEN)
    arrow([(94, y+5.5), (98, y+5.5)], GREEN, head=False)
    for head_y, title in [(y+7, 'proj\nF_'+modality), (y-6, 'prompt_head\nP_'+modality)]:
        box(101, head_y, 17, 9, title, '#eff4fb', BLUE, 11.5)
        arrow([(98, y+5.5), (98, head_y+4.5), (101, head_y+4.5)], GREEN)
arrow([(37.5, 108), (37.5, 91)], '#7f91a5', dashed=True, head=False)
label(39.5, 99.5, '同一 E', 9.5, '#607086', ha='left')

box(125, 81, 14, 39, '原 P²Fusion\n提示引导\n渐进融合\n×4', '#e9f0fd', BLUE, 13)
arrow([(118, 119.5), (121, 119.5), (125, 113)])
arrow([(118, 106.5), (121, 106.5), (125, 104)])
arrow([(118, 91.5), (121, 91.5), (125, 95)])
arrow([(118, 78.5), (121, 78.5), (125, 86)])

box(144, 102, 14, 14, '重建解码器\n拼接 / 上采样', '#e9f0fd', BLUE, 11.5)
arrow([(139, 109), (144, 109)])
box(144, 87, 14, 9, '融合亮度 Y_f\n1通道', '#e8f6ed', GREEN, 11.5)
arrow([(151, 102), (151, 96)], GREEN)
box(142, 64, 16, 12, '原版色彩还原\nY_f + Cr/Cb', '#e8f6ed', GREEN, 11.5)
arrow([(151, 87), (151, 76)], GREEN)
label(151, 59.5, '彩色融合图', 12, GREEN)
arrow([(13.5, 80), (13.5, 70), (142, 70)], GREEN)
label(52, 72.7, '色度旁路：原始退化 VIS 的 Cr/Cb', 10.5, GREEN)
box(101, 60, 17, 9, 'score_head＋σ\nGAP → q_VIS', '#f4edfa', PURPLE, 10.5)
arrow([(109.5, 74), (109.5, 69)], PURPLE)

# Supervision-only paths. The clean forward does not use proj/prompt/fusion.
box(5, 32, 153, 24, fill='#fbf7ff', edge=PURPLE, dashed=True)
label(8, 53, '训练专用：干净分支、双教师与辅助重建', 12.5, PURPLE, ha='left', weight='bold')
box(8, 36, 22, 12, '配对干净 IR/VIS\n同样取 Y×3', '#f3eafb', PURPLE, 11.5)
box(36, 36, 35, 12, '同 E ＋对应 FDN\n得到 U_c、V_c', '#f3eafb', PURPLE, 11.5)
arrow([(30, 42), (36, 42)], PURPLE, dashed=True)
label(53.5, 33.7, '一致性 / 正交 / 自监督：见 B', 9, PURPLE)
box(78, 36, 28, 12, '冻结 IST：红外显著性\n冻结 BRISQUE：可见光质量', '#f3eafb', PURPLE, 10.1)
arrow([(30, 44.5), (32, 44.5), (32, 50), (76, 50), (78, 46)], PURPLE, dashed=True)
box(113, 36, 42, 12, 'P_IR 与干净教师目标：BCE\nq_VIS 与干净质量标量：MSE', '#f3eafb', PURPLE, 10.8)
arrow([(106, 42), (113, 42)], PURPLE, dashed=True)
arrow([(109.5, 60), (109.5, 56.7), (131, 56.7), (131, 48)], PURPLE, dashed=True)
arrow([(118, 106.5), (120, 106.5), (120, 57.8), (144, 57.8), (144, 48)], PURPLE, dashed=True)

# B: disentanglement, with exact residual/content and private branches.
box(165, 78, 83, 57, fill='#fffdf8', edge='#e7cfaa', radius=1.3)
label(168, 131, 'B  特征解耦模块 FDN', 17, ha='left', weight='bold')
label(168, 125.5, 'C=60；每模态独立，退化/干净条件共享本模态参数', 10.8, ha='left')
box(168, 107, 11, 10, 'F\nC×h×w', '#fff3df', ORANGE, 11)
box(184, 115, 40, 8, '1×1 Conv → GN → GELU → 3×3 Conv', '#fff3df', ORANGE, 10.3)
box(184, 101, 40, 8, '1×1 Conv → GN → GELU → 3×3 Conv', '#fff3df', ORANGE, 10.3)
arrow([(179, 112), (181, 112), (181, 119), (184, 119)], ORANGE)
arrow([(179, 112), (181, 112), (181, 105), (184, 105)], ORANGE)
circle(229, 119, '+', ORANGE, 2)
arrow([(224, 119), (227, 119)], ORANGE)
arrow([(174, 117), (174, 123.7), (229, 123.7), (229, 121)], ORANGE)
box(235, 114.5, 10, 9, 'U\n不变特征', '#e9f7f1', GREEN, 10.5)
box(235, 100.5, 10, 9, 'V\n私有特征', '#fff3df', ORANGE, 10.5)
arrow([(231, 119), (235, 119)], ORANGE)
arrow([(224, 105), (235, 105)], ORANGE)
label(168, 96, r'$U=F+R(F),\quad V=P(F)$', 15, ha='left')
label(168, 90.5, '跨域一致性：MSE(U_d, sg(U_c))；正交：mean cos²(U,V)', 10.8, ha='left', color=PURPLE)
label(168, 85.5, '自监督：D_U(U_d)、D_U(U_c) → 干净 Y×3（L1）', 10.6, ha='left', color=PURPLE)
label(168, 80.8, 'D_V(V_d) → I_d−I_c；D_V(V_c) → 0（L1；有符号残差）', 10.6, ha='left', color=PURPLE)

# C: spatial weather gate, region gate, and residual modulation. No GAP on V.
box(165, 29, 83, 46, fill='#f9fdfb', edge='#b7d7ce', radius=1.3)
label(168, 71, 'C  动态天气感知门控 DWAM', 17, ha='left', weight='bold')
box(168, 56, 10, 8, 'V', '#fff3df', ORANGE, 13)
box(182, 56, 7, 8, 'sg', '#f3eafb', PURPLE, 12)
box(193, 54, 33, 12, '1×1 Conv → ReLU\n1×1 Conv → Sigmoid\n天气门控 G_w：C×h×w', '#e9f7f1', GREEN, 10.1)
arrow([(178, 60), (182, 60)], ORANGE)
arrow([(189, 60), (193, 60)], GREEN)
box(168, 40, 10, 8, 'U', '#e9f7f1', GREEN, 13)
box(183, 38, 21, 12, '通道 Mean / Max\n拼接：2×h×w', '#e9f7f1', GREEN, 10.3)
box(209, 38, 17, 12, '7×7 Conv＋σ\nG_r：1×h×w', '#e9f7f1', GREEN, 10.3)
arrow([(178, 44), (183, 44)], GREEN)
arrow([(204, 44), (209, 44)], GREEN)
circle(233, 55, '×', GREEN, 2)
arrow([(226, 60), (233, 60), (233, 57)], GREEN)
arrow([(226, 44), (229, 44), (229, 52), (233, 53)], GREEN)
box(231, 36, 14, 10, 'U + U⊙G\n输出 Û', '#e9f7f1', GREEN, 10.6)
arrow([(235, 55), (238, 55), (238, 46)], GREEN)
arrow([(173, 40), (173, 34.5), (238, 34.5), (238, 36)], GREEN)
label(168, 31.4, r'$\widehat U=U+U\odot G_w\odot G_r$', 14.5, ha='left')

# Training objective and key reading notes.
box(2, 2, 246, 23, fill='#f4f6fa', edge='#d5dfe9', radius=1.3)
label(5, 21, '训练目标：原5项融合 / 提示损失 ＋ 3项解耦约束', 15, ha='left', weight='bold')
label(5, 14, r'$\mathcal{L}=8\mathcal{L}_{int}+25\mathcal{L}_{ssim}+20\mathcal{L}_{grad}'
      r'+0.5\mathcal{L}_{distill}^{IR}+5\mathcal{L}_{distill}^{VIS}'
      r'+\mathcal{L}_{con}+0.01\mathcal{L}_{orth}+\mathcal{L}_{self}$', 20, ha='left')
label(5, 6.6, '强度 / 梯度 / SSIM 使用干净 Y 参考；教师、干净分支及重建头仅训练时使用。', 11.3, ha='left')
label(166, 6.6, '门控倍率 1+G_wG_r ∈ [1,2]：相对增强，非直接衰减。', 10.3, ha='left', color=GREEN)

fig.savefig(OUT / 'current_y_architecture.svg', facecolor='white')
fig.savefig(OUT / 'current_y_architecture.png', dpi=160, facecolor='white')
plt.close(fig)
from PIL import Image
with Image.open(OUT / 'current_y_architecture.png') as image:
    image.quantize(colors=192).save(OUT / 'current_y_architecture.png', optimize=True)
print(OUT / 'current_y_architecture.svg')
print(OUT / 'current_y_architecture.png')
