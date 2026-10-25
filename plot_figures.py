# -*- coding: utf-8 -*-
"""All indices, including LSDI, computed on the same SG-smoothed spectra."""

import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score

from litter_index_2 import (
    ALL_COVERAGES, XLSX_MEAN_PATH, XLSX_SAMPLES_PATH, SIM_EXPORT_PKL,
    OUT_DIR, step1_read_measured_mean, step1_read_measured_samples,
    step3_accuracy_filter, calc_base_vi, BASE_VI, _band_at,
)

warnings.filterwarnings('ignore')

OUT = OUT_DIR / 'compare_npv'
OUT.mkdir(parents=True, exist_ok=True)

SG_WIN, SG_POLY = 9, 3
LSDI_BANDS = (760, 780, 473)
ORDER_VI = ['NDVI', 'WDRVI', 'GDVI', 'GRNDVI', 'PSRI', 'MINI', 'MSAVI', 'ARVI', 'EVI']


def sg_mat(X):
    return np.array([savgol_filter(s.astype(float), SG_WIN, SG_POLY) for s in X])


def lsdi_vals(mat, wl):
    a = _band_at(mat, wl, LSDI_BANDS[0])
    b = _band_at(mat, wl, LSDI_BANDS[1])
    c = _band_at(mat, wl, LSDI_BANDS[2])
    return (a - b) / (a + b + c + 1e-10)


def fit_predict(x_tr, y_tr, x_te):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(x_tr.reshape(-1, 1), y_tr)
    return rf.predict(x_te.reshape(-1, 1))


def scores(y_true, yhat):
    cm = confusion_matrix(y_true, yhat, labels=[0, 1])
    low = float(cm[0, 0] / cm[0].sum())
    high = float(cm[1, 1] / cm[1].sum())
    return dict(oa=float(accuracy_score(y_true, yhat)),
                bal=float(balanced_accuracy_score(y_true, yhat)),
                low=low, high=high, pred=yhat)


def plot_oa_bars(names, accs, n, path):
    clrs = ['#C0392B' if n_ == 'LSDI' else '#BDC3C7' for n_ in names]
    order = np.argsort(accs)[::-1]
    names_s = [names[i] for i in order]
    accs_s = [accs[i] for i in order]
    clrs_s = [clrs[i] for i in order]
    best_other = max(a for n_, a in zip(names, accs) if n_ != 'LSDI')

    fig, ax = plt.subplots(figsize=(4.6, 2.85))
    x = np.arange(len(names_s))
    bars = ax.bar(x, accs_s, 0.62, color=clrs_s, edgecolor='white', linewidth=0.4, zorder=3)
    for bar, v in zip(bars, accs_s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f'{v:.0%}',
                ha='center', va='bottom', fontsize=6.5, color='#2C3E50')
    ax.axhline(best_other, color='#2C3E50', lw=0.8, ls='--', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names_s, fontsize=7.5, rotation=35, ha='right')
    ax.set_ylabel('Classification accuracy', fontsize=8)
    ax.set_ylim(0.0, min(1.0, max(accs_s) + 0.16))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.yaxis.grid(True, lw=0.4, color='0.88', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor='#C0392B', edgecolor='none', label='LSDI'),
            Patch(facecolor='#BDC3C7', edgecolor='none', label='Comparison indices'),
        ],
        fontsize=6.5, loc='upper right', handlelength=1.0, borderpad=0.5,
    )
    ax.text(0.02, 1.03,
            f'$n$ = {n} field spectra   |   Binary classification (≥60% vs. <60%)',
            transform=ax.transAxes, fontsize=6, va='bottom', ha='left', color='0.4')
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'  saved {path}')


def plot_oa_and_high(names, accs, highs, n, path):
    order = np.argsort(accs)[::-1]
    names_s = [names[i] for i in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    panels = [
        (axes[0], [accs[i] for i in order], 'Overall accuracy'),
        (axes[1], [highs[i] for i in order], 'High-litter recall'),
    ]
    for ax, vals, ylabel in panels:
        clrs = ['#C0392B' if n_ == 'LSDI' else '#BDC3C7' for n_ in names_s]
        x = np.arange(len(names_s))
        bars = ax.bar(x, vals, 0.62, color=clrs, edgecolor='white', linewidth=0.4, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f'{v:.0%}',
                    ha='center', va='bottom', fontsize=6, color='#2C3E50')
        ax.set_xticks(x)
        ax.set_xticklabels(names_s, fontsize=7, rotation=40, ha='right')
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(0.0, 1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.yaxis.grid(True, lw=0.4, color='0.88', zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].legend(
        handles=[
            Patch(facecolor='#C0392B', edgecolor='none', label='LSDI'),
            Patch(facecolor='#BDC3C7', edgecolor='none', label='Comparison indices'),
        ],
        fontsize=6.5, loc='upper right', handlelength=1.0, borderpad=0.4,
    )
    fig.text(0.5, -0.02,
             f'$n$ = {n} field spectra. All indices computed after the same Savitzky-Golay filter.',
             ha='center', fontsize=7, color='0.35')
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'  saved {path}')


def plot_cm(y_true, pred, acc, names, path):
    cmap_cm = LinearSegmentedColormap.from_list('cm_blue', ['#FDFEFE', '#1A5276'], N=256)
    n_col = 5
    n_row = int(np.ceil(len(names) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(8.8, 3.6),
                             gridspec_kw={'wspace': 0.42, 'hspace': 0.55})
    axes = np.atleast_1d(axes).ravel()
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
                ax.text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center', fontsize=7,
                        color='white' if val > thresh else '#2C3E50', linespacing=1.25)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=8)
        if k % n_col == 0:
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_ylabel('True label', fontsize=8)
        else:
            ax.set_yticklabels([''] * 2)
        ax.set_title(f'{name}  OA={acc[name]:.1%}', fontsize=9, pad=4)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for k in range(len(names), len(axes)):
        fig.delaxes(axes[k])
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'  saved {path}')


def plot_boxes(values, y_cov, path):
    covers = [0, 20, 40, 60, 80, 100]
    keys = ['LSDI', 'NDVI', 'PSRI', 'MINI']
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.55))
    for ax, name in zip(axes, keys):
        data = [values[name][y_cov == c] for c in covers]
        ax.boxplot(data, positions=range(len(covers)), widths=0.55,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color='#1A5276', lw=1.1),
                   whiskerprops=dict(color='0.35', lw=0.7),
                   capprops=dict(color='0.35', lw=0.7),
                   boxprops=dict(facecolor='#D6EAF8', edgecolor='#1A5276', lw=0.7))
        rho = spearmanr(y_cov, values[name]).correlation
        ax.set_xticks(range(len(covers)))
        ax.set_xticklabels([str(c) for c in covers], fontsize=7)
        ax.set_title(f'{name}   rho = {rho:.2f}', fontsize=8.5, pad=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if name == 'LSDI':
            ax.set_ylabel('Index value', fontsize=8)
        ax.set_xlabel('Standing litter (%)', fontsize=7.5)
    fig.tight_layout(pad=0.45)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'  saved {path}')


def main():
    wl, measured_mean = step1_read_measured_mean(XLSX_MEAN_PATH)
    wl, X_meas, y_cov, y_bin = step1_read_measured_samples(XLSX_SAMPLES_PATH, wl)
    with open(SIM_EXPORT_PKL, 'rb') as f:
        sim_data = pickle.load(f)['sim_data']
    sim_data, _, _, _ = step3_accuracy_filter(wl, measured_mean, sim_data, out_dir=OUT)

    X_tr = sg_mat(np.concatenate([sim_data[c]['all'] for c in ALL_COVERAGES]))
    y_tr = np.concatenate([np.full(len(sim_data[c]['all']), int(c >= 60)) for c in ALL_COVERAGES])
    X_te = sg_mat(X_meas)

    vi_tr = calc_base_vi(X_tr, wl)
    vi_te = calc_base_vi(X_te, wl)
    vi_tr['LSDI'] = lsdi_vals(X_tr, wl)
    vi_te['LSDI'] = lsdi_vals(X_te, wl)

    names = ['LSDI'] + [v for v in ORDER_VI if v in BASE_VI]
    pred, acc, high, low, bal = {}, {}, {}, {}, {}
    print('\nAll indices on SG-smoothed spectra. PROSAIL train, field test.')
    for name in names:
        yhat = fit_predict(vi_tr[name], y_tr, vi_te[name])
        sc = scores(y_bin, yhat)
        pred[name] = yhat
        acc[name] = sc['oa']
        high[name] = sc['high']
        low[name] = sc['low']
        bal[name] = sc['bal']
        rho = spearmanr(y_cov, vi_te[name]).correlation
        print(f'  {name:<8} OA={sc["oa"]:.3f}  Bal={sc["bal"]:.3f}  '
              f'Low={sc["low"]:.3f}  High={sc["high"]:.3f}  rho={rho:+.3f}')

    n = len(y_bin)
    plot_oa_bars(names, [acc[k] for k in names], n, OUT / 'Fig_index_compare_all_SG.png')
    plot_oa_and_high(names, [acc[k] for k in names], [high[k] for k in names], n,
                     OUT / 'Fig_index_OA_high_all_SG.png')
    plot_cm(y_bin, pred, acc, names, OUT / 'Fig_confusion_all_SG.png')
    plot_boxes(vi_te, y_cov, OUT / 'Fig_index_vs_litter_box_all_SG.png')

    rows = []
    for name in names:
        rho = float(spearmanr(y_cov, vi_te[name]).correlation)
        kind = 'Standing-litter index' if name == 'LSDI' else (
            'NPV / senescence index' if name in ('PSRI', 'MINI') else 'Green-vegetation index')
        rows.append(dict(
            Index=name, Type=kind,
            Evaluation='SG-smoothed; PROSAIL train → field test',
            OA=round(acc[name], 4), Balanced=round(bal[name], 4),
            Low=round(low[name], 4), High=round(high[name], 4),
            Spearman=round(rho, 4),
            Correct=f'{acc[name] * n:.0f}/{n}',
        ))
    pd.DataFrame(rows).sort_values('OA', ascending=False).to_csv(
        OUT / 'all_SG_results.csv', index=False, encoding='utf-8-sig')
    print('\n  done')


if __name__ == '__main__':
    main()
