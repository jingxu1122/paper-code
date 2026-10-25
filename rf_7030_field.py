# -*- coding: utf-8 -*-
"""Stratified 70/30 random forest helpers for the field spectra."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

RF_TREES = 500


def split_7030(y, seed):
    idx = np.arange(len(y))
    tr, te = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=seed)
    return tr, te


def fit_predict(x_tr, y_tr, x_te):
    rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=42, n_jobs=1)
    rf.fit(x_tr.reshape(-1, 1), y_tr)
    return rf.predict(x_te.reshape(-1, 1))


def class_rates(y_true, yhat):
    cm = confusion_matrix(y_true, yhat, labels=[0, 1])
    low = cm[0, 0] / cm[0].sum() if cm[0].sum() else np.nan
    high = cm[1, 1] / cm[1].sum() if cm[1].sum() else np.nan
    return float(accuracy_score(y_true, yhat)), float(low), float(high), cm


def plot_bars(summary, n_tr, n_te, path, raw=False):
    names = [r['Index'] for r in summary]
    means = np.array([r['OA_mean'] for r in summary], float)
    stds = np.array([r['OA_std'] for r in summary], float)
    order = np.argsort(means)[::-1]
    names = [names[i] for i in order]
    means, stds = means[order], stds[order]
    clrs = ['#C0392B' if n == 'LSDI' else '#BDC3C7' for n in names]

    fig, ax = plt.subplots(figsize=(4.8, 2.95))
    x = np.arange(len(names))
    bars = ax.bar(x, means, 0.62, yerr=stds, color=clrs, edgecolor='white',
                  linewidth=0.4, zorder=3, capsize=2,
                  error_kw=dict(lw=0.7, ecolor='0.35'))
    for bar, v, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, min(1.08, v + s + 0.018),
                f'{v:.0%}', ha='center', va='bottom', fontsize=6.5, color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5, rotation=35, ha='right')
    ax.set_ylabel('Test accuracy', fontsize=8)
    ax.set_ylim(0.0, 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.yaxis.grid(True, lw=0.4, color='0.88', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor='#C0392B', edgecolor='none', label='LSDI'),
        Patch(facecolor='#BDC3C7', edgecolor='none', label='Comparison indices'),
    ], fontsize=6.5, loc='upper right', handlelength=1.0, borderpad=0.4)
    ax.text(0.0, 1.02,
            f'Train {n_tr} / test {n_te}, stratified 70% / 30%   |   '
            f'Binary classification (≥60% vs. <60%)',
            transform=ax.transAxes, fontsize=6, va='bottom', ha='left', color='0.35')
    fig.tight_layout(pad=0.55)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'  saved {path}')


def plot_confusion(y_true, pred, acc, names, n_tr, n_te, path):
    cmap_cm = LinearSegmentedColormap.from_list('cm_blue', ['#FDFEFE', '#1A5276'], N=256)
    fig, axes = plt.subplots(2, 5, figsize=(8.8, 4.15),
                             gridspec_kw={'wspace': 0.42, 'hspace': 0.62})
    axes = axes.ravel()
    labels = ['Low', 'High']
    for k, name in enumerate(names):
        ax = axes[k]
        cm = confusion_matrix(y_true, pred[name], labels=[0, 1])
        vmax = cm.max() if cm.max() else 1
        ax.imshow(cm, cmap=cmap_cm, aspect='equal', vmin=0, vmax=vmax)
        thresh = vmax / 2.0
        for i in range(2):
            row_sum = cm[i].sum()
            for j in range(2):
                val = cm[i, j]
                pct = val / row_sum * 100 if row_sum else 0.0
                ax.text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center',
                        fontsize=7, color='white' if val > thresh else '#2C3E50',
                        linespacing=1.25)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=8)
        if k % 5 == 0:
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_ylabel('True label', fontsize=8)
        else:
            ax.set_yticklabels([''] * 2)
        ax.set_title(f'{name}  OA={acc[name]:.0%}', fontsize=9, pad=4)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        f'Train {n_tr} / test {n_te}   |   '
        'Binary classification (≥60% vs. <60%)   |   seed 0',
        fontsize=9, color='0.25', y=1.02,
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'  saved {path}')


