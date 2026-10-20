"""Shared settings for the main and supplementary figures (2026-10-12 restructuring): paths, colours, font sizes."""
import sys
from pathlib import Path
A = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(A), str(A / 'independent_reproduction_20261009')]
from figure_export import apply_style, save_figure as _save_figure
import matplotlib.pyplot as plt
MM = 1 / 25.4
apply_style({'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
             'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
             'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
HIGH, LOW, MID, GREY = '#B2182B', '#2166AC', '#9E9E9E', '#BDBDBD'
DZ = {302: 'Diarrhoeal disease', 322: 'LRI', 332: 'Meningitis'}
DZC = {302: '#E08214', 322: '#542788', 332: '#1B7837'}
PER = ['1990–2000', '2001–2010', '2011–2023']
_LABELS = []
def label(ax, s, x=None, y=None):
    """Register a panel letter. Letters are placed by place_labels() at save time: left edge of the panel's
    content (tick labels, axis label, side annotations) and top edge of its title, so no letter sits on text."""
    _LABELS.append((ax, s))
def place_labels(fig, pad_pt=1.0):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    mine = [(ax, s) for ax, s in _LABELS if ax.figure is fig]
    for ax, s in mine:
        boxes = [ax.get_window_extent(r), ax.yaxis.get_tightbbox(r), ax.title.get_window_extent(r),
                 ax._left_title.get_window_extent(r)]
        boxes += [t.get_window_extent(r) for t in ax.texts if t.get_visible() and t.get_text()]
        boxes = [b for b in boxes if b is not None and b.width > 0]
        x0 = min(b.x0 for b in boxes); y1 = max(b.y1 for b in boxes)
        fx, fy = inv.transform((x0, y1))
        # 2026-10-20: panels have no titles; put the letter just above the panel's content so it never sits on an axis label
        fy2 = inv.transform((x0, y1 + 3))[1]
        if fy2 < 0.975: t = fig.text(max(fx, 0.004), fy2, s, fontsize=9, fontweight='bold', ha='left', va='bottom')
        else: t = fig.text(max(fx, 0.004), min(fy, 0.997), s, fontsize=9, fontweight='bold', ha='left', va='top')
        tb = t.get_window_extent(r); gap = 3
        for ti in (ax._left_title, ax.title):          # never let the letter run into a title on its own row
            if ti.get_text():
                bb = ti.get_window_extent(r)
                if bb.x0 < tb.x1 + gap and bb.y1 > tb.y0 and bb.y0 < tb.y1:
                    t.set_x(inv.transform((bb.x0 - gap - tb.width, 0))[0]); tb = t.get_window_extent(r)
    _LABELS[:] = [(ax, s) for ax, s in _LABELS if ax.figure is not fig]
def save_figure(fig, stem):
    place_labels(fig)
    return _save_figure(fig, stem)
MCOL = {'Mortality': '#1A1A1A', 'Incidence': '#8C8C8C', 'Case fatality': '#C2410C'}
OUTS = A / 'supp_figures_20261012'
