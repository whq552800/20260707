"""正文主图共用：数据路径、配色、字号（投稿用英文标签）。"""
import sys
from pathlib import Path
A = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(A), str(A / 'independent_reproduction_20261009')]
from figure_export import apply_style, save_figure
import matplotlib.pyplot as plt
MM = 1 / 25.4
apply_style({'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
             'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
             'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
HIGH, LOW, MID, GREY = '#B2182B', '#2166AC', '#9E9E9E', '#BDBDBD'
DZ = {302: 'Diarrhoeal disease', 322: 'LRI', 332: 'Meningitis'}
DZC = {302: '#E08214', 322: '#542788', 332: '#1B7837'}
PER = ['1990–2000', '2001–2010', '2011–2023']
def label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')
