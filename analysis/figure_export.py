"""项目统一绘图导出规范（2026-10-10 起，所有新图必须使用）。
每张图只导出两种格式：
  1) <stem>.pdf  —— Adobe Illustrator 可编辑矢量：pdf.fonttype=42（TrueType 嵌入，文字保持为可编辑文本，非 Type 3/轮廓），
                    不栅格化任何图元；
  2) <stem>.jpg  —— 位图预览/投稿用，600 dpi（≥300 dpi），质量 95，白底。
用法：
    from figure_export import apply_style, save_figure
    apply_style()            # 在建图前调用
    ...
    save_figure(fig, outdir / 'Figure2_xxx')
"""
from pathlib import Path
import matplotlib

JPEG_DPI = 600
JPEG_QUALITY = 95
FONT = 'Liberation Sans'   # 与 Arial 字宽一致；AI 中若提示缺字体，替换为 Arial 版式不变

RC = {
    'pdf.fonttype': 42,        # TrueType 嵌入 -> AI 中文字可编辑
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'font.family': FONT,
    'savefig.facecolor': 'white',
    'savefig.transparent': False,
    'image.composite_image': False,  # 不合并图像层，便于 AI 中分别选取
}


def apply_style(extra=None):
    matplotlib.rcParams.update(RC)
    if extra:
        matplotlib.rcParams.update(extra)
    # 用户样式可能覆盖导出关键项，这里强制回写
    matplotlib.rcParams.update({k: RC[k] for k in ('pdf.fonttype', 'ps.fonttype')})


def save_figure(fig, stem):
    stem = Path(stem)
    assert matplotlib.rcParams['pdf.fonttype'] == 42, 'pdf.fonttype must be 42'
    for ax in fig.axes:   # 拒绝栅格化图元，保证 PDF 全矢量
        for art in ax.get_children():
            if getattr(art, 'get_rasterized', lambda: False)():
                art.set_rasterized(False)
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.jpg'), dpi=JPEG_DPI,
                pil_kwargs={'quality': JPEG_QUALITY, 'optimize': True})
    return [stem.with_suffix('.pdf'), stem.with_suffix('.jpg')]
