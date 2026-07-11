# -*- coding: utf-8 -*-
"""
=====================================================================
 芦苇立枯物指数（LSDI）构建与验证 —— 整合流程
=====================================================================
本脚本由以下两份原始代码重新整理合并而成：
  · main_15_efast_lhs_v6_SG_6.py   → PROSAIL模拟 / 敏感性分析 / 光谱预处理
  · litter_index_rf_3band.py       → LSDI指数构建 / 混淆矩阵 / 随机森林

整理原则：
  - 只保留研究流程主线所需的计算与关键图表，去除了与传感器波段
    重采样（Sentinel-2/Landsat）等旁支相关、与主线不直接相关的内容；
  - 每一步均可独立运行/调试，中间结果通过 dict/DataFrame 传递；
  - 所有绘图函数集中在每个 Step 末尾，便于统一风格管理。

流程总览（对应本文件中的 STEP 1~10）：
  STEP 1   读取实测数据
             (a) 6个覆盖度梯度的“代表光谱”（用于 PROSAIL 精度验证 / 敏感性分析）
             (b) 90条逐条实测光谱（用于 LSDI 构建 / 随机森林分类）
  STEP 2   PROSAIL 冠层光谱模拟（拉丁超立方采样 LHS，每个覆盖度 N_SAMPLES 次）
  STEP 3   精度筛选：逐样本计算模拟 vs 实测的 R²，仅保留 R² ≥ 0.95 的模拟样本
  STEP 4   光谱响应预处理：原始反射率 / 一阶导数 / 二阶导数
  STEP 5   Savitzky-Golay (SG) 滤波参数筛选（多组合窗口/阶数对比，确定最优参数）
  STEP 6   OAT（One-At-a-Time）局部敏感性分析
  STEP 7   EFAST（扩展傅里叶幅度检验）全局敏感性分析
  STEP 8   构建立枯物指数 LSDI（三波段 LOO-CV 嵌套搜索）
  STEP 9   LSDI 与 7 个常规植被指数（NDVI/GRNDVI/GDVI/MSAVI/ARVI/WDRVI/EVI）
             的二分类混淆矩阵对比
  STEP 10  随机森林二分类结果汇总（准确率对比图 + 结果表 CSV）
=====================================================================
"""

import sys
import time
import pickle
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              r2_score, mean_squared_error)

warnings.filterwarnings('ignore')

try:
    import prosail
    print(f"prosail {prosail.__version__} loaded")
except ImportError:
    print("提示：未安装 prosail，STEP 2/6/7（PROSAIL模拟与敏感性分析）无法运行。"
          "请先 `pip install prosail`。")
    prosail = None

try:
    from SALib.sample import latin as lhs_sampler
    from SALib.sample import fast_sampler
    from SALib.analyze import fast as fast_analyze
except ImportError:
    print("提示：未安装 SALib，STEP 2/7（LHS采样 / EFAST敏感性分析）无法运行。"
          "请先 `pip install SALib`。")

# ─────────────────────────────────────────────────────────────
# 期刊绘图风格（Times New Roman + 内向刻度）
# ─────────────────────────────────────────────────────────────
matplotlib.rcParams['font.family']        = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus']  = False
matplotlib.rcParams['axes.linewidth']      = 0.8
matplotlib.rcParams['xtick.direction']     = 'in'
matplotlib.rcParams['ytick.direction']     = 'in'
matplotlib.rcParams['xtick.major.size']    = 3.5
matplotlib.rcParams['ytick.major.size']    = 3.5
matplotlib.rcParams['xtick.labelsize']     = 8
matplotlib.rcParams['ytick.labelsize']     = 8
matplotlib.rcParams['legend.frameon']      = True
matplotlib.rcParams['legend.framealpha']   = 0.92
matplotlib.rcParams['legend.edgecolor']    = '0.75'
matplotlib.rcParams['legend.fontsize']     = 7.5
matplotlib.rcParams['figure.dpi']          = 150


# =====================================================================
# CONFIG —— 请根据实际路径修改
# =====================================================================
BASE_DIR = Path(__file__).parent

# (a) 6个覆盖度“代表光谱”数据（宽表：波段列 + 6个覆盖度列）
XLSX_MEAN_PATH = BASE_DIR / '20190521lu.xlsx'

# (b) 90条逐条实测光谱数据（宽表：波段列 + 90个样本列，列名编码覆盖度）
XLSX_SAMPLES_PATH = BASE_DIR / '201905.xlsx'

# PROSAIL模拟结果导出（如果已有，可直接读取跳过 STEP 2，与 litter_index_rf_3band.py 对接）
SIM_EXPORT_PKL = BASE_DIR / 'sim_data_export.pkl'

OUT_DIR = BASE_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_COVERAGES = [0, 20, 40, 60, 80, 100]
COL_MAP = {0: 'Litter_0%', 20: 'Litter_20%', 40: 'Litter_40%',
           60: 'Litter_60%', 80: 'Litter_80%', 100: 'Litter_100%'}

N_SAMPLES   = 2000            # 每个覆盖度 LHS 采样数
PARAM_NAMES = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm', 'prot', 'cbc', 'lai']
R2_THRESHOLD = 0.95            # STEP 3 精度筛选阈值

COLORS = [
    (0.13, 0.55, 0.13), (0.60, 0.76, 0.20), (0.93, 0.78, 0.05),
    (0.90, 0.55, 0.10), (0.78, 0.28, 0.08), (0.50, 0.18, 0.05),
]

GEO = dict(
    hspot=0.01, tts=30.0, tto=0.0, psi=0.0, ant=0.0,
    prospect_version='PRO', rsoil=1.0, psoil=0.85,
)

# lidfa（平均叶倾角）采样范围：随覆盖度线性插值
LIDFA_RANGE = {
    0:   (40, 70),  20:  (39, 72),  40:  (38, 74),
    60:  (37, 76),  80:  (36, 78), 100:  (35, 80),
}

# PROSPECT-PRO + 4SAIL 各覆盖度参数先验范围
TABLE1 = {
    0:   dict(n=(1.5, 2.5),    cab=(15.0, 70.0), car=(1.5, 20.0), cbrown=(0.0, 0.0),
              cw=(0.01, 0.05), cm=(0.005, 0.010), prot=(0.0015, 0.0020), cbc=(0.0035, 0.007),  lai=(0.9, 2.5)),
    20:  dict(n=(1.2, 3.42),   cab=(12.0, 63),   car=(1.5, 16.4), cbrown=(0.0, 0.4),
              cw=(0.008, 0.041), cm=(0.006, 0.012), prot=(0.0010, 0.0016), cbc=(0.0048, 0.0096), lai=(1.0, 2.8)),
    40:  dict(n=(1.5, 2.85),   cab=(9.0, 55),    car=(1.5, 12.8), cbrown=(0.0, 0.8),
              cw=(0.007, 0.014), cm=(0.005, 0.032), prot=(0.0005, 0.0013), cbc=(0.0075, 0.0147), lai=(1.2, 3.0)),
    60:  dict(n=(1.4, 2.27),   cab=(6.0, 50),    car=(1.5, 9.2),  cbrown=(0.0, 1.2),
              cw=(0.004, 0.023), cm=(0.008, 0.016), prot=(0.0003, 0.0008), cbc=(0.0075, 0.0147), lai=(1.2, 3.0)),
    80:  dict(n=(1.3, 1.70),   cab=(3, 40),      car=(1.5, 5.6),  cbrown=(0, 1.6),
              cw=(0.002, 0.014), cm=(0.009, 0.018), prot=(0.0002, 0.0005), cbc=(0.0088, 0.0172), lai=(1.0, 2.8)),
    100: dict(n=(1.2, 2),      cab=(0.0, 35),    car=(22, 22),    cbrown=(0, 2),
              cw=(0.001, 0.005), cm=(0.01, 0.02),  prot=(0.0000, 0.0004), cbc=(0.01, 0.0196),   lai=(0.8, 2.5)),
}

_SLICE = slice(0, 501)   # 400-900nm，501个波段


# =====================================================================
# STEP 1  读取实测数据
# =====================================================================

def step1_read_measured_mean(xlsx_path):
    """
    读取 6 个覆盖度梯度的代表光谱（宽表：第一列为波段nm，其余列为 Litter_X% ）
    用于：PROSAIL 精度验证 / OAT / EFAST 敏感性分析
    """
    print("\n" + "=" * 62)
    print("  STEP 1(a)  读取实测数据 —— 6组覆盖度代表光谱")
    print("=" * 62)

    df = pd.read_excel(xlsx_path, sheet_name='Sheet1', header=0)
    wl = df.iloc[:, 0].values.astype(float)
    measured_mean = {cov: df[COL_MAP[cov]].values.astype(float) for cov in ALL_COVERAGES}

    for cov in ALL_COVERAGES:
        v = measured_mean[cov]
        print(f"  {COL_MAP[cov]:12s}: VIS={v[wl <= 700].mean():.4f}  NIR={v[wl > 700].mean():.4f}")

    return wl, measured_mean


def step1_read_measured_samples(xlsx_path, wl_ref):
    """
    读取 90 条逐条实测光谱（波段400~900nm，覆盖度梯度100/80/60/40/20/0%）
    列名规则：
      1, 1.1, 1.2, ...       → 100%
      0.8, 0.8.1, ...        → 80%
      0.6, 0.6.1, ...        → 60%
      0.4, 0.4.1, ...        → 40%
      0.2, 0.2.1, ...        → 20%
      0,   0.1,   0.2, ...   → 0%
    用于：LSDI 构建（LOO-CV） / 随机森林二分类测试
    """
    print("\n" + "=" * 62)
    print("  STEP 1(b)  读取实测数据 —— 90条逐条光谱（用于LSDI构建/分类）")
    print("=" * 62)

    df_raw = pd.read_excel(xlsx_path, sheet_name='Sheet1')
    mask_wl = (df_raw['波段nm'] >= 400) & (df_raw['波段nm'] <= 900)
    df = df_raw[mask_wl].copy().reset_index(drop=True)
    wl = df['波段nm'].values.astype(float)

    assert np.allclose(wl, wl_ref, atol=0.5), \
        f"波段不一致！samples={wl[0]}~{wl[-1]}，mean_spec={wl_ref[0]}~{wl_ref[-1]}"

    cov_map = {'1': 100, '0.8': 80, '0.6': 60, '0.4': 40, '0.2': 20, '0': 0}

    def parse_cov(col_name):
        parts = str(col_name).split('.')
        if len(parts) == 1:
            return cov_map.get(parts[0])
        prefix2 = parts[0] + '.' + parts[1]
        return cov_map[prefix2] if prefix2 in cov_map else cov_map.get(parts[0])

    spec_cols = [c for c in df.columns if c != '波段nm']
    cov_labels = [parse_cov(c) for c in spec_cols]
    valid_pairs = [(c, lbl) for c, lbl in zip(spec_cols, cov_labels) if lbl is not None]
    spec_cols = [p[0] for p in valid_pairs]
    cov_labels = [p[1] for p in valid_pairs]

    X_meas = np.array([df[c].values.astype(float) for c in spec_cols])   # (90, 501)
    y_meas_cov = np.array(cov_labels)
    y_meas_bin = (y_meas_cov >= 60).astype(int)                          # 二分类：高(>=60%) / 低(<60%)

    print(f"总样本数: {len(X_meas)}")
    for cov in sorted(set(y_meas_cov), reverse=True):
        n = (y_meas_cov == cov).sum()
        print(f"  覆盖度 {cov:3d}%: {n} 条")
    print(f"二分类: 高覆盖(≥60%) = {y_meas_bin.sum()}  低覆盖(<60%) = {(1 - y_meas_bin).sum()}")

    return wl, X_meas, y_meas_cov, y_meas_bin


def sg_smooth(spec, win=9, p=3):
    w = win if len(spec) >= win else (len(spec) // 2 * 2 + 1)
    return savgol_filter(spec.astype(float), window_length=w, polyorder=p)


# =====================================================================
# STEP 2  PROSAIL 冠层光谱模拟（LHS采样）
# =====================================================================

def run_prosail_once(params: dict, lidfa: float = None) -> np.ndarray:
    lidfa_val = lidfa if lidfa is not None else 57.0
    rho_full = prosail.run_prosail(
        n=float(params['n']), cab=float(params['cab']), car=float(params['car']),
        cbrown=float(params['cbrown']), cw=float(params['cw']), cm=0.0,
        prot=float(params['prot']), cbc=float(params['cbc']), lai=float(params['lai']),
        lidfa=lidfa_val, hspot=GEO['hspot'], tts=GEO['tts'], tto=GEO['tto'], psi=GEO['psi'],
        ant=GEO['ant'], prospect_version=GEO['prospect_version'],
        rsoil=GEO['rsoil'], psoil=GEO['psoil'],
    )
    return rho_full[_SLICE]   # 400-900nm, 501波段


def step2_prosail_simulation(wl):
    """对每个覆盖度做 LHS 采样 + PROSAIL 模拟，返回 sim_data[cov] = dict(all, pmat, ...)"""
    print("\n" + "=" * 62)
    print(f"  STEP 2  PROSAIL冠层光谱模拟（LHS采样，每个覆盖度 {N_SAMPLES} 次）")
    print("=" * 62)

    sim_data = {}
    t0 = time.time()
    for cov in ALL_COVERAGES:
        ranges = TABLE1[cov]
        lidfa_lo, lidfa_hi = LIDFA_RANGE[cov]

        lhs_names = PARAM_NAMES + ['lidfa']
        lhs_bounds = []
        for k in PARAM_NAMES:
            lo, hi = ranges[k][0], ranges[k][1]
            if lo >= hi:
                lo, hi = lo - 1e-8, lo + 1e-8
            lhs_bounds.append([lo, hi])
        lhs_bounds.append([lidfa_lo, lidfa_hi])

        lhs_problem = {'num_vars': len(lhs_names), 'names': lhs_names, 'bounds': lhs_bounds}
        lhs_mat = lhs_sampler.sample(lhs_problem, N_SAMPLES, seed=42)

        specs = np.zeros((N_SAMPLES, len(wl)))
        pmat = np.zeros((N_SAMPLES, len(PARAM_NAMES)))

        for j in range(N_SAMPLES):
            row = lhs_mat[j]
            p = {}
            for pi, k in enumerate(PARAM_NAMES):
                lo0, hi0 = ranges[k][0], ranges[k][1]
                p[k] = float(lo0) if lo0 >= hi0 else float(row[pi])
                pmat[j, pi] = p[k]
            lidfa_val = float(row[len(PARAM_NAMES)])
            specs[j] = run_prosail_once(p, lidfa=lidfa_val)
            if (j + 1) % 500 == 0:
                print(f"  Coverage {cov:3d}%  {j + 1}/{N_SAMPLES}  ({time.time() - t0:.0f}s)")

        sim_data[cov] = dict(
            all=specs, mean=specs.mean(0), std=specs.std(0),
            p5=np.percentile(specs, 5, 0), p95=np.percentile(specs, 95, 0),
            pmat=pmat,
        )
        print(f"  Coverage {cov:3d}%  done ({N_SAMPLES} runs)")

    print(f"\n  完成：共 {N_SAMPLES * len(ALL_COVERAGES)} 次PROSAIL运行，耗时 {time.time() - t0:.0f}s")
    return sim_data


def sam_angle(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def spectral_metrics(meas, sim, vis_mask, nir_mask):
    return dict(
        R2=r2_score(meas, sim), RMSE=np.sqrt(mean_squared_error(meas, sim)), SAM=sam_angle(meas, sim),
        R2_vis=r2_score(meas[vis_mask], sim[vis_mask]), R2_nir=r2_score(meas[nir_mask], sim[nir_mask]),
        RMSE_vis=np.sqrt(mean_squared_error(meas[vis_mask], sim[vis_mask])),
        RMSE_nir=np.sqrt(mean_squared_error(meas[nir_mask], sim[nir_mask])),
    )


# =====================================================================
# STEP 3  精度筛选（逐样本 R² ≥ 0.95）
# =====================================================================

def step3_accuracy_filter(wl, measured_mean, sim_data, out_dir=OUT_DIR):
    """
    对每个覆盖度：
      1) 计算筛选前均值光谱精度（基准）
      2) 逐条模拟光谱与实测计算 R²，保留 R² >= R2_THRESHOLD 的样本
      3) 更新 sim_data 为筛选后的子集，并计算筛选后精度
    返回：更新后的 sim_data（就地更新）、acc_before、acc_after、filter_stats
    """
    print("\n" + "=" * 62)
    print(f"  STEP 3  精度筛选（NSE-R² ≥ {R2_THRESHOLD}）")
    print("=" * 62)

    vis_mask = (wl >= 400) & (wl <= 700)
    nir_mask = (wl > 700) & (wl <= 900)

    acc_before = {cov: spectral_metrics(measured_mean[cov], sim_data[cov]['all'].mean(0), vis_mask, nir_mask)
                  for cov in ALL_COVERAGES}

    print(f"\n  筛选前均值光谱精度（全部 {N_SAMPLES} 条模拟样本）:")
    for cov in ALL_COVERAGES:
        m = acc_before[cov]
        print(f"  {cov:4d}%  R²={m['R2']:.4f}  RMSE={m['RMSE']:.5f}  SAM={m['SAM']:.3f}°")

    filter_stats = {}
    for cov in ALL_COVERAGES:
        specs_all = sim_data[cov]['all']
        pmat_all = sim_data[cov]['pmat']
        meas = measured_mean[cov]
        n_tot = len(specs_all)

        meas_mean = meas.mean()
        ss_tot = ((meas - meas_mean) ** 2).sum()
        if ss_tot < 1e-12:
            r2_arr = np.zeros(n_tot)
        else:
            ss_res = ((specs_all - meas[np.newaxis, :]) ** 2).sum(axis=1)
            r2_arr = np.clip(1.0 - ss_res / ss_tot, -np.inf, 1.0)

        mask = r2_arr >= R2_THRESHOLD
        n_kept = int(mask.sum())
        if n_kept == 0:
            print(f"  ⚠ Coverage {cov}%: 无样本达标，回退为全部样本")
            mask = np.ones(n_tot, dtype=bool)
            n_kept = n_tot

        specs_f, pmat_f, r2_f = specs_all[mask], pmat_all[mask], r2_arr[mask]

        sim_data[cov].update(dict(
            all=specs_f, mean=specs_f.mean(0), std=specs_f.std(0),
            p5=np.percentile(specs_f, 5, 0), p95=np.percentile(specs_f, 95, 0),
            pmat=pmat_f, r2_per_sample=r2_f,
        ))
        filter_stats[cov] = dict(n_total=n_tot, n_kept=n_kept, ratio=n_kept / n_tot,
                                  r2_all=r2_arr, r2_kept=r2_f,
                                  r2_mean=float(r2_f.mean()), r2_min=float(r2_f.min()))
        print(f"  Coverage {cov:3d}%  保留 {n_kept:4d}/{n_tot} ({n_kept / n_tot * 100:.1f}%)  "
              f"R²: mean={r2_f.mean():.4f} min={r2_f.min():.4f}")

    acc_after = {cov: spectral_metrics(measured_mean[cov], sim_data[cov]['mean'], vis_mask, nir_mask)
                 for cov in ALL_COVERAGES}

    print(f"\n  筛选后均值光谱精度（R²≥{R2_THRESHOLD} 保留样本）:")
    for cov in ALL_COVERAGES:
        m = acc_after[cov]
        print(f"  {cov:4d}%  R²={m['R2']:.4f}  RMSE={m['RMSE']:.5f}  SAM={m['SAM']:.3f}°")

    # ── 筛选前后精度对比图 ──
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw={'wspace': 0.40})
    labels = [f'{c}%' for c in ALL_COVERAGES]
    x = np.arange(len(ALL_COVERAGES)); w = 0.35
    for ax, key, ylabel, hl in [
        (axes[0], 'R2', 'R²', 0.85), (axes[1], 'RMSE', 'RMSE', None), (axes[2], 'SAM', 'SAM (°)', 10),
    ]:
        b_vals = [acc_before[c][key] for c in ALL_COVERAGES]
        a_vals = [acc_after[c][key] for c in ALL_COVERAGES]
        ax.bar(x - w / 2, b_vals, w, label=f'Before (n={N_SAMPLES})', color='#5B9BD5',
               edgecolor='white', lw=0.3, zorder=3)
        ax.bar(x + w / 2, a_vals, w, label=f'After (R²≥{R2_THRESHOLD})', color='#C0392B',
               edgecolor='white', lw=0.3, zorder=3)
        if hl:
            ax.axhline(hl, color='#2C3E50', lw=0.8, ls='--', zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlabel('Litter coverage (%)', fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
        ax.yaxis.grid(True, lw=0.4, color='0.88', zorder=0); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=6.5, loc='upper right', borderpad=0.4)
    plt.tight_layout(pad=0.5)
    fig.savefig(out_dir / 'step3_before_after_accuracy.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  已保存 step3_before_after_accuracy.png")

    return sim_data, acc_before, acc_after, filter_stats


# =====================================================================
# STEP 4  光谱响应预处理：原始反射率 / 一阶导数 / 二阶导数
# =====================================================================

def spectral_transform(spec, wl, order=0, sg_win=11, sg_poly=3):
    """
    order=0: 原始反射率
    order=1: 一阶导数（中心差分）
    order=2: 二阶导数（先SG平滑，再两次差分；SG参数由 STEP 5 确认）
    """
    if order == 0:
        return spec.astype(float)
    if order == 1:
        return np.gradient(spec, wl)
    if order == 2:
        win = sg_win if len(spec) >= sg_win else (len(spec) // 2 * 2 + 1)
        sm = savgol_filter(spec, window_length=win, polyorder=sg_poly)
        d1 = np.gradient(sm, wl)
        return np.gradient(d1, wl)
    raise ValueError(f"未知 order={order}")


def step4_spectral_response(wl, measured_mean, sim_data, out_dir=OUT_DIR,
                             sg_win=11, sg_poly=3, n_draw=50):
    """绘制 原始反射率 / 一阶导数 / 二阶导数 三张对比图（低立枯物 vs 高立枯物）"""
    print("\n" + "=" * 62)
    print("  STEP 4  光谱响应预处理（原始 / 一阶导数 / 二阶导数）")
    print("=" * 62)

    rng_draw = np.random.default_rng(123)

    def _plot_panel(ax, wl_mask, order):
        wl_plot = wl if wl_mask is None else wl[wl_mask]
        for cov, color in [(0, 'black'), (20, 'black')]:
            specs_all = sim_data[cov]['all']
            idx = rng_draw.choice(len(specs_all), size=min(n_draw, len(specs_all)), replace=False)
            for j in idx:
                y = spectral_transform(specs_all[j], wl, order, sg_win, sg_poly)
                d = y if wl_mask is None else y[wl_mask]
                ax.plot(wl_plot, d, color=color, alpha=0.15, lw=0.5)
        for cov, color in [(80, 'darkorange'), (100, 'darkorange')]:
            specs_all = sim_data[cov]['all']
            idx = rng_draw.choice(len(specs_all), size=min(n_draw, len(specs_all)), replace=False)
            for j in idx:
                y = spectral_transform(specs_all[j], wl, order, sg_win, sg_poly)
                d = y if wl_mask is None else y[wl_mask]
                ax.plot(wl_plot, d, color=color, alpha=0.15, lw=0.5)
        for cov in [0, 20]:
            y = spectral_transform(measured_mean[cov], wl, order, sg_win, sg_poly)
            d = y if wl_mask is None else y[wl_mask]
            ax.plot(wl_plot, d, color='black', alpha=0.75, lw=1.4, zorder=5)
        for cov in [80, 100]:
            y = spectral_transform(measured_mean[cov], wl, order, sg_win, sg_poly)
            d = y if wl_mask is None else y[wl_mask]
            ax.plot(wl_plot, d, color='darkorange', alpha=0.75, lw=1.4, zorder=5)

    def _make_figure(order, ylabel, fname, panel_b_range, panel_b_title,
                      panel_c_range, panel_c_title, mark_rededge=True):
        fig = plt.figure(figsize=(7.0, 4.5))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.35, wspace=0.30)

        ax_a = fig.add_subplot(gs[0, :])
        _plot_panel(ax_a, None, order)
        legend_lines = [
            Line2D([0], [0], color='black', lw=1.5, label='Low litter (0–20%)'),
            Line2D([0], [0], color='darkorange', lw=1.5, label='High litter (80–100%)'),
        ]
        ax_a.legend(handles=legend_lines, fontsize=9, loc='center right',
                    bbox_to_anchor=(1.0, 0.65), framealpha=0.8)
        ax_a.set_xlabel('Wavelength (nm)', fontsize=9); ax_a.set_ylabel(ylabel, fontsize=9)
        ax_a.set_title(f'(a) {ylabel} (400–900 nm)', fontsize=9)
        ax_a.set_xlim(400, 900); ax_a.grid(True, alpha=0.3)
        if mark_rededge:
            ax_a.axvspan(700, 780, color='red', alpha=0.05)
            ax_a.axvspan(750, 900, color='salmon', alpha=0.06)

        ax_b = fig.add_subplot(gs[1, 0])
        mask_b = (wl >= panel_b_range[0]) & (wl <= panel_b_range[1])
        _plot_panel(ax_b, mask_b, order)
        ax_b.set_xlabel('Wavelength (nm)', fontsize=9); ax_b.set_ylabel(ylabel, fontsize=9)
        ax_b.set_title(f'(b) {panel_b_title}', fontsize=9); ax_b.grid(True, alpha=0.3)

        ax_c = fig.add_subplot(gs[1, 1])
        mask_c = (wl >= panel_c_range[0]) & (wl <= panel_c_range[1])
        _plot_panel(ax_c, mask_c, order)
        ax_c.set_xlabel('Wavelength (nm)', fontsize=9); ax_c.set_ylabel(ylabel, fontsize=9)
        ax_c.set_title(f'(c) {panel_c_title}', fontsize=9); ax_c.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"  已保存 {fname}")

    _make_figure(0, 'Reflectance', 'step4_reflectance_spectra.png',
                 (400, 750), 'VIS region', (750, 900), 'NIR plateau', mark_rededge=False)
    _make_figure(1, 'First derivative reflectance', 'step4_first_derivative_spectra.png',
                 (680, 780), 'Red-edge region (zoom)', (750, 900), 'NIR plateau')
    _make_figure(2, 'Second derivative reflectance', 'step4_second_derivative_spectra.png',
                 (680, 780), 'Red-edge region (zoom)', (750, 900), 'NIR plateau')


# =====================================================================
# STEP 5  SG（Savitzky-Golay）滤波参数筛选
# =====================================================================

SG_COMBOS = [
    (5,  2, 'Narrow/low'),
    (7,  2, 'Narrow/low'),
    (9,  2, 'Chen2004'),
    (9,  3, 'Mid/mid'),
    (11, 2, 'Wide/low'),
    (11, 3, 'Shah1990★'),   # 默认参数
    (13, 3, 'Wide/mid'),
    (15, 3, 'Extra-wide/mid'),
    (15, 4, 'Extra-wide/high'),
]


def step5_sg_selection(wl, test_spectrum, out_dir=OUT_DIR):
    """
    评价指标：
      (1) 红边斜率保留度 —— 680-750nm 区间二阶导数与"无平滑基准"的相关系数
      (2) 高频噪声抑制   —— 750-900nm（NIR平台，理论应≈0）二阶导数标准差
      (3) 信号保真度     —— 平滑反射率与原始反射率的 RMSE
    参考：Tsai & Philpot(1998) w=9-15,p=2-4；Demetriades-Shah et al.(1990) w=11,p=3
    返回：推荐的 (win, poly)
    """
    print("\n" + "=" * 62)
    print("  STEP 5  SG滤波参数筛选（多组合窗口/阶数对比）")
    print("=" * 62)

    _wl = wl.astype(float)
    _spec = test_spectrum.astype(float)

    d1_raw = np.gradient(_spec, _wl)
    d2_raw = np.gradient(d1_raw, _wl)

    re_mask = (_wl >= 680) & (_wl <= 750)
    nir_mask = (_wl >= 750) & (_wl <= 900)

    results = []
    for win, poly, label in SG_COMBOS:
        sm = savgol_filter(_spec, window_length=win, polyorder=poly)
        d1 = np.gradient(sm, _wl)
        d2 = np.gradient(d1, _wl)
        re_corr = float(np.corrcoef(d2[re_mask], d2_raw[re_mask])[0, 1])
        nir_noise = float(np.std(d2[nir_mask]))
        smth_rmse = float(np.sqrt(np.mean((sm - _spec) ** 2)))
        results.append((win, poly, label, re_corr, nir_noise, smth_rmse))

    print(f"\n  {'w':>3} {'p':>2}  {'Label':<14}  {'Red-edge r':>10}  {'NIR noise σ':>11}  {'Smooth RMSE':>11}")
    for win, poly, label, rc, nn, sr in results:
        mark = ' ◀default' if (win == 11 and poly == 3) else ''
        print(f"  {win:>3} {poly:>2}  {label:<14}  {rc:>10.4f}  {nn:>11.6f}  {sr:>11.6f}{mark}")

    noises = np.array([r[4] for r in results])
    candidates = [r for r in results if r[4] <= noises.mean()]
    if candidates:
        best = max(candidates, key=lambda r: r[3])
        print(f"\n  → 推荐参数: w={best[0]}, p={best[1]} ({best[2]})  "
              f"red-edge r={best[3]:.4f}, NIR noise={best[4]:.6f}")
        best_w, best_p = best[0], best[1]
    else:
        best_w, best_p = 11, 3

    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.5), gridspec_kw={'hspace': 0.55, 'wspace': 0.38})
    for ax, (win, poly, label, rc, nn, sr) in zip(axes.flat, results):
        sm = savgol_filter(_spec, window_length=win, polyorder=poly)
        d2 = np.gradient(np.gradient(sm, _wl), _wl)
        ax.plot(_wl, d2_raw, color='0.80', lw=0.7, label='Unsmoothed', zorder=1)
        ax.plot(_wl, d2, color='#2980B9', lw=1.0, label=f'w={win}, p={poly}', zorder=2)
        ax.axvspan(680, 750, color='#E74C3C', alpha=0.06)
        ax.set_xlim(400, 900)
        ax.set_xlabel('Wavelength (nm)', fontsize=7.5); ax.set_ylabel('2nd derivative', fontsize=7.5)
        ax.set_title(f'w={win}, p={poly}  r={rc:.3f}  σ={nn:.2e}', fontsize=7.0, pad=2)
        ax.yaxis.grid(True, lw=0.35, color='0.88', zorder=0); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(fontsize=6, borderpad=0.3, handlelength=1.0, loc='upper right')
    plt.tight_layout(pad=0.5)
    fig.savefig(out_dir / 'step5_SG_param_comparison.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  已保存 step5_SG_param_comparison.png")

    return best_w, best_p


# =====================================================================
# STEP 6  OAT（One-At-a-Time）局部敏感性分析
# =====================================================================

def step6_oat_sensitivity(wl, out_dir=OUT_DIR):
    """对每个覆盖度：取参数区间中值为基准，逐参数 +20% 扰动，计算归一化敏感性曲线"""
    print("\n" + "=" * 62)
    print("  STEP 6  OAT局部敏感性分析（±20%，全部覆盖度）")
    print("=" * 62)

    oat_all = {}
    for cov in ALL_COVERAGES:
        ranges = TABLE1[cov]
        lo_l, hi_l = LIDFA_RANGE[cov]
        lidfa_mid = (lo_l + hi_l) / 2.0
        base = {p: (ranges[p][0] + ranges[p][1]) / 2.0 for p in PARAM_NAMES}
        base_s = run_prosail_once(base, lidfa=lidfa_mid)

        oat = {}
        for p in PARAM_NAMES:
            pu = dict(**base)
            pu[p] = base[p] * 1.20 if base[p] != 0 else 0.01
            oat[p] = np.abs(run_prosail_once(pu, lidfa=lidfa_mid) - base_s) / (np.abs(base_s) + 1e-10)
        oat_all[cov] = oat
        ranked = sorted(PARAM_NAMES, key=lambda p: oat[p].mean(), reverse=True)
        print(f"  Coverage {cov:3d}%  敏感性排序: {' > '.join(ranked[:4])} ...")

    # ── 绘图：3行2列，每格一个覆盖度 ──
    oat_c = plt.cm.tab10(np.linspace(0, 1, len(PARAM_NAMES)))
    fig, axes = plt.subplots(3, 2, figsize=(5.5, 6.5), gridspec_kw={'hspace': 0.55, 'wspace': 0.38})
    pids = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for ax, cov, pid in zip(axes.flat, ALL_COVERAGES, pids):
        oat = oat_all[cov]
        for i, p in enumerate(PARAM_NAMES):
            ax.plot(wl, oat[p], lw=0.85, color=oat_c[i], label=p)
        ax.set_title(f'{pid} Litter {cov}% (OAT ±20%)', fontsize=7.5, pad=2)
        ax.set_xlabel('Wavelength (nm)', fontsize=7.5); ax.set_ylabel('Norm. sensitivity', fontsize=7.5)
        ax.set_xlim(400, 900)
        ax.yaxis.grid(True, lw=0.35, color='0.88', zorder=0); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if ax is axes.flat[0]:
            ax.legend(fontsize=5.8, loc='upper right', ncol=2, handlelength=1.0,
                      borderpad=0.3, columnspacing=0.4)
    plt.tight_layout(pad=0.5)
    fig.savefig(out_dir / 'step6_OAT_sensitivity.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  已保存 step6_OAT_sensitivity.png")

    return oat_all


# =====================================================================
# STEP 7  EFAST 全局敏感性分析
# =====================================================================

def step7_efast_sensitivity(wl, efast_n=2000, out_dir=OUT_DIR):
    """对每个覆盖度独立EFAST采样，逐波长计算一阶敏感性 S1 与总敏感性 ST"""
    print("\n" + "=" * 62)
    print(f"  STEP 7  EFAST全局敏感性分析（{efast_n} runs/覆盖度）")
    print("=" * 62)

    efast_S1, efast_ST = {}, {}
    t0 = time.time()

    for cov in ALL_COVERAGES:
        ranges = TABLE1[cov]
        lidfa_lo, lidfa_hi = LIDFA_RANGE[cov]

        all_names = PARAM_NAMES + ['lidfa']
        all_bounds = []
        for k in PARAM_NAMES:
            lo, hi = ranges[k][0], ranges[k][1]
            if lo >= hi:
                lo, hi = lo - 1e-6, lo + 1e-6
            all_bounds.append([lo, hi])
        all_bounds.append([lidfa_lo, lidfa_hi])

        problem = {'num_vars': len(all_names), 'names': all_names, 'bounds': all_bounds}
        param_values = fast_sampler.sample(problem, efast_n, seed=42)
        n_runs = param_values.shape[0]

        specs_efast = np.zeros((n_runs, len(wl)))
        for j in range(n_runs):
            row = param_values[j]
            p = {k: float(row[i]) for i, k in enumerate(PARAM_NAMES)}
            lidfa_val = float(row[len(PARAM_NAMES)])
            specs_efast[j] = run_prosail_once(p, lidfa=lidfa_val)
            if (j + 1) % 5000 == 0:
                print(f"  Coverage {cov:3d}%  EFAST {j + 1}/{n_runs}  ({time.time() - t0:.0f}s)")

        n_params_all = len(all_names)
        S1_mat = np.zeros((n_params_all, len(wl)))
        ST_mat = np.zeros((n_params_all, len(wl)))
        for wi in range(len(wl)):
            y = specs_efast[:, wi]
            if y.std() < 1e-10:
                continue
            si = fast_analyze.analyze(problem, y, print_to_console=False)
            S1_mat[:, wi] = np.clip(si['S1'], 0, 1)
            ST_mat[:, wi] = np.clip(si['ST'], 0, 1)

        n_p = len(PARAM_NAMES)
        efast_S1[cov] = S1_mat[:n_p, :]
        efast_ST[cov] = ST_mat[:n_p, :]

        order = np.argsort(efast_S1[cov].mean(1))[::-1]
        print(f"  Coverage {cov:3d}%  Top S1: {' > '.join([PARAM_NAMES[i] for i in order[:3]])}"
              f"  ({time.time() - t0:.0f}s)")

    print(f"\n  EFAST完成，总耗时 {time.time() - t0:.0f}s")

    # ── 绘图：S1 和 ST 各一张 3行2列图 ──
    cc = plt.cm.tab10(np.linspace(0, 1, len(PARAM_NAMES)))
    pids = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for mat_dict, label, fname in [(efast_S1, 'S1', 'step7_efast_S1.png'),
                                    (efast_ST, 'ST', 'step7_efast_ST.png')]:
        fig, axes = plt.subplots(3, 2, figsize=(5.5, 6.5), gridspec_kw={'hspace': 0.55, 'wspace': 0.38})
        for idx, (cov, pid) in enumerate(zip(ALL_COVERAGES, pids)):
            ax = axes[idx // 2, idx % 2]
            for pi, pname in enumerate(PARAM_NAMES):
                ax.plot(wl, mat_dict[cov][pi], lw=0.85, color=cc[pi], label=pname)
            ax.set_xlim(400, 900); ax.set_ylim(0, 1)
            ax.set_title(f'{pid} Litter {cov}% — {label}', fontsize=7.5, pad=2)
            ax.set_xlabel('Wavelength (nm)', fontsize=7.5)
            ax.set_ylabel(f'{"First-order" if label=="S1" else "Total"} sensitivity {label}', fontsize=7.5)
            ax.yaxis.grid(True, lw=0.35, color='0.88', zorder=0); ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            if ax is axes.flat[0]:
                ax.legend(fontsize=5.8, loc='upper right', ncol=2, handlelength=1.0,
                          borderpad=0.3, columnspacing=0.4)
        plt.tight_layout(pad=0.5)
        fig.savefig(out_dir / fname, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"  已保存 {fname}")

    return efast_S1, efast_ST


# =====================================================================
# STEP 8  构建立枯物指数 LSDI（三波段 LOO-CV 嵌套搜索）
# =====================================================================
#
#   LSDI = (ρ_B1 - ρ_B2) / (ρ_B1 + ρ_B2 + ρ_B3)
#
#   外层 LOO（共90折）：留出第 i 条
#     内层：仅用剩余89条
#       Step A：搜索最优波段对 (B1,B2) —— 使 (B1-B2)/(|B1|+|B2|) 与覆盖度 Spearman|ρ| 最大
#       Step B：固定 B1/B2，搜索最优 B3 —— 使 LSDI 与覆盖度 Spearman|ρ| 最大
#       训练随机森林（二分类）→ 预测第 i 条
#   汇总 → LOO 准确率 + 波段稳定性统计
# =====================================================================

SEARCH_RANGE = (440, 780)   # 候选波段搜索范围 (nm)
MIN_SEP = 20                # 最小波段间隔 (nm)


def _find_best_band_pair(spec_mat, cov_vals, wl, candidate_idx, min_sep=MIN_SEP):
    best_rho, best_pair = -1, (candidate_idx[0], candidate_idx[1])
    for ii in range(len(candidate_idx)):
        for jj in range(ii + 1, len(candidate_idx)):
            ia, ib = candidate_idx[ii], candidate_idx[jj]
            if abs(wl[ia] - wl[ib]) < min_sep:
                continue
            ba, bb = spec_mat[:, ia].astype(float), spec_mat[:, ib].astype(float)
            denom = np.abs(ba) + np.abs(bb)
            if np.any(denom < 1e-10):
                continue
            vi = (ba - bb) / (denom + 1e-10)
            if not np.all(np.isfinite(vi)):
                continue
            rho, _ = spearmanr(cov_vals, vi)
            if abs(rho) > best_rho:
                best_rho, best_pair = abs(rho), (ia, ib)
    return best_pair[0], best_pair[1], best_rho


def _find_best_b3(spec_mat, cov_vals, wl, ia, ib, candidate_idx, min_sep=MIN_SEP):
    best_rho, best_ic = -1, candidate_idx[0]
    ba, bb = spec_mat[:, ia].astype(float), spec_mat[:, ib].astype(float)
    for ic in candidate_idx:
        if ic == ia or ic == ib:
            continue
        if abs(wl[ic] - wl[ia]) < min_sep or abs(wl[ic] - wl[ib]) < min_sep:
            continue
        bc = spec_mat[:, ic].astype(float)
        denom = ba + bb + bc
        if np.any(np.abs(denom) < 1e-10):
            continue
        lsdi = (ba - bb) / (denom + 1e-10)
        if not np.all(np.isfinite(lsdi)):
            continue
        rho, _ = spearmanr(cov_vals, lsdi)
        if abs(rho) > best_rho:
            best_rho, best_ic = abs(rho), ic
    return best_ic, best_rho


def step8_build_lsdi(wl, X_meas, y_meas_cov, y_meas_bin, sg_win=9, sg_poly=3, out_dir=OUT_DIR):
    """三波段 LOO-CV 嵌套搜索，构建 LSDI 指数并返回 LOO 预测结果、最终波段组合"""
    print("\n" + "=" * 62)
    print("  STEP 8  构建立枯物指数 LSDI（三波段 LOO-CV 嵌套搜索，共90折）")
    print("=" * 62)
    print("  指数：LSDI = (ρ_B1 - ρ_B2) / (ρ_B1 + ρ_B2 + ρ_B3)")
    print(f"  搜索范围：{SEARCH_RANGE[0]}~{SEARCH_RANGE[1]} nm，最小波段间隔 {MIN_SEP} nm\n")

    n_samples = len(X_meas)
    X_meas_sm = np.array([sg_smooth(X_meas[i], win=sg_win, p=sg_poly) for i in range(n_samples)])

    search_mask = (wl >= SEARCH_RANGE[0]) & (wl <= SEARCH_RANGE[1])
    search_idx = np.where(search_mask)[0]

    loo_pred = np.zeros(n_samples, dtype=int)
    band_log = []

    for i in range(n_samples):
        tr_idx = [j for j in range(n_samples) if j != i]
        X_tr, X_te = X_meas_sm[tr_idx], X_meas_sm[[i]]
        y_tr, y_tr_cov = y_meas_bin[tr_idx], y_meas_cov[tr_idx]

        ia, ib, rho_ab = _find_best_band_pair(X_tr, y_tr_cov, wl, search_idx)
        ic, rho_3 = _find_best_b3(X_tr, y_tr_cov, wl, ia, ib, search_idx)
        band_log.append((round(float(wl[ia])), round(float(wl[ib])), round(float(wl[ic]))))

        def lsdi_val(mat, ia=ia, ib=ib, ic=ic):
            ba, bb, bc = mat[:, ia].astype(float), mat[:, ib].astype(float), mat[:, ic].astype(float)
            return ((ba - bb) / (ba + bb + bc + 1e-10)).reshape(-1, 1)

        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(lsdi_val(X_tr), y_tr)
        loo_pred[i] = int(rf.predict(lsdi_val(X_te))[0])

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  折 {i + 1:2d}/{n_samples}  B1={wl[ia]:.0f}nm  B2={wl[ib]:.0f}nm  B3={wl[ic]:.0f}nm  "
                  f"|ρ|_pair={rho_ab:.3f}  |ρ|_LSDI={rho_3:.3f}")

    acc_lsdi = accuracy_score(y_meas_bin, loo_pred)
    print(f"\nLOO 准确率（共{n_samples}条）: LSDI = {acc_lsdi:.3f}  ({acc_lsdi * n_samples:.0f}/{n_samples})")

    # ── 波段稳定性统计 ──
    cnt_all = Counter(band_log)
    top_combo = cnt_all.most_common(1)[0][0]
    print(f"\n最终指数波段定义（最高频三波段组合）:")
    print(f"  B1={top_combo[0]}nm  B2={top_combo[1]}nm  B3={top_combo[2]}nm")
    print(f"  LSDI = (ρ_{top_combo[0]} - ρ_{top_combo[1]}) / (ρ_{top_combo[0]} + ρ_{top_combo[1]} + ρ_{top_combo[2]})")

    cnt_b1 = Counter([b[0] for b in band_log])
    cnt_b2 = Counter([b[1] for b in band_log])
    cnt_b3 = Counter([b[2] for b in band_log])

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw={'wspace': 0.38})
    panel_titles = ['B1 (numerator, +)', 'B2 (numerator, −)', 'B3 (denominator)']
    panel_colors = ['#2980B9', '#27AE60', '#E67E22']
    for ax, cnt, color, title in zip(axes, [cnt_b1, cnt_b2, cnt_b3], panel_colors, panel_titles):
        freq = np.zeros(len(wl))
        for wl_val, c in cnt.items():
            freq[np.argmin(np.abs(wl - wl_val))] += c
        mask = (wl >= SEARCH_RANGE[0]) & (wl <= SEARCH_RANGE[1])
        ax.fill_between(wl[mask], freq[mask], alpha=0.18, color=color, zorder=2)
        ax.plot(wl[mask], freq[mask], color=color, lw=1.1, zorder=3)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlabel('Wavelength (nm)', fontsize=8)
        ax.set_ylabel(f'Selection count ({n_samples} folds)', fontsize=7.5)
        ax.set_xlim(*SEARCH_RANGE)
        ax.yaxis.grid(True, lw=0.4, color='0.88', zorder=0); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.5)
    fig.savefig(out_dir / 'step8_band_stability.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  已保存 step8_band_stability.png")

    # ── LSDI 散点图（最终波段）──
    ia_f = np.argmin(np.abs(wl - top_combo[0]))
    ib_f = np.argmin(np.abs(wl - top_combo[1]))
    ic_f = np.argmin(np.abs(wl - top_combo[2]))
    ba_f, bb_f, bc_f = X_meas_sm[:, ia_f], X_meas_sm[:, ib_f], X_meas_sm[:, ic_f]
    lsdi_full = (ba_f - bb_f) / (ba_f + bb_f + bc_f + 1e-10)
    rho_full, pval = spearmanr(y_meas_cov, lsdi_full)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    cov_levels = sorted(set(y_meas_cov))
    cov_palette = ['#1a6b2a', '#4caf50', '#a5d63f', '#ffc107', '#e65100', '#7b1a1a']
    for ci, cov in enumerate(cov_levels):
        m = y_meas_cov == cov
        rng_j = np.random.default_rng(cov)
        jitter = rng_j.uniform(-3.5, 3.5, m.sum())
        ax.scatter(cov + jitter, lsdi_full[m], color=cov_palette[ci], s=16,
                   edgecolors='white', linewidths=0.3, zorder=3, alpha=0.85, label=f'{cov}%')
    ax.axvline(50, color='0.7', lw=0.6, ls=':')
    ax.set_xlabel('Litter coverage (%)', fontsize=8); ax.set_ylabel('LSDI', fontsize=8)
    ax.set_xticks(cov_levels); ax.set_xticklabels([f'{c}' for c in cov_levels], fontsize=7.5)
    ax.yaxis.grid(True, lw=0.4, color='0.88', zorder=0); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    sign_str = '< 0.001' if pval < 0.001 else f'= {pval:.3f}'
    ax.text(0.97, 0.04, fr'$\rho$ = {rho_full:.3f}' + '\n' + fr'$p$ {sign_str}',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.8', lw=0.5))
    ax.legend(title='Coverage', fontsize=6.5, title_fontsize=6.5, loc='upper right', ncol=2)
    plt.tight_layout(pad=0.5)
    fig.savefig(out_dir / 'step8_lsdi_scatter.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  已保存 step8_lsdi_scatter.png")

    return dict(loo_pred=loo_pred, acc_lsdi=acc_lsdi, top_combo=top_combo,
                band_log=band_log, X_meas_sm=X_meas_sm, lsdi_full=lsdi_full)


# =====================================================================
# STEP 9  LSDI 与 7个常规植被指数的二分类混淆矩阵对比
# =====================================================================

BASE_VI = ['NDVI', 'GRNDVI', 'GDVI', 'MSAVI', 'ARVI', 'WDRVI', 'EVI']


def _band_at(spec_mat, wl, target_nm):
    return spec_mat[:, np.argmin(np.abs(wl - target_nm))].astype(float)


def calc_base_vi(spec_mat, wl):
    """7个常规植被指数（NDVI/GRNDVI/GDVI/MSAVI/ARVI/WDRVI/EVI），标准可见光-近红外波段"""
    R, G, N, B = (_band_at(spec_mat, wl, 670), _band_at(spec_mat, wl, 560),
                  _band_at(spec_mat, wl, 860), _band_at(spec_mat, wl, 490))
    eps = 1e-10
    return {
        'NDVI':   (N - R) / (N + R + eps),
        'GRNDVI': (N - G - R) / (N + G + R + eps),
        'GDVI':   (N ** 2 - R ** 2) / (N ** 2 + R ** 2 + eps),
        'MSAVI':  (2 * N + 1 - np.sqrt(np.maximum(0, (2 * N + 1) ** 2 - 8 * (N - R)))) / 2,
        'ARVI':   (N - (2 * R - B)) / (N + (2 * R - B) + eps),
        'WDRVI':  (0.2 * N - R) / (0.2 * N + R + eps),
        'EVI':    2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1 + eps),
    }


def step9_confusion_matrices(wl, X_meas, y_meas_bin, sim_data, lsdi_result, out_dir=OUT_DIR):
    """
    常规VI：PROSAIL（筛选后）训练随机森林 → 90条实测测试
    LSDI  ：使用 STEP 8 的 LOO-CV 预测结果
    绘制 LSDI + 7个常规指数 共8张混淆矩阵（同一网格图）
    """
    print("\n" + "=" * 62)
    print("  STEP 9  LSDI 与 7个常规指数的二分类混淆矩阵对比")
    print("=" * 62)

    n_samples = len(X_meas)
    X_train_all = np.concatenate([sim_data[cov]['all'] for cov in ALL_COVERAGES])
    y_train_cov = np.concatenate([np.full(len(sim_data[cov]['all']), cov) for cov in ALL_COVERAGES])
    y_train_bin = (y_train_cov >= 60).astype(int)

    vi_tr = calc_base_vi(X_train_all, wl)
    vi_te = calc_base_vi(X_meas, wl)

    base_pred, base_results = {}, {}
    for vi in BASE_VI:
        rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42, n_jobs=-1)
        rf.fit(vi_tr[vi].reshape(-1, 1), y_train_bin)
        pred = rf.predict(vi_te[vi].reshape(-1, 1))
        base_pred[vi] = pred
        base_results[vi] = accuracy_score(y_meas_bin, pred)
        print(f"  {vi:<8} ACC={base_results[vi]:.3f}  ({base_results[vi] * n_samples:.0f}/{n_samples})")

    panel_names = BASE_VI + ['LSDI']
    panel_pred = {**base_pred, 'LSDI': lsdi_result['loo_pred']}
    panel_acc = {**base_results, 'LSDI': lsdi_result['acc_lsdi']}

    cm_labels = ['Low', 'High']
    cmap_cm = LinearSegmentedColormap.from_list('cm_blue', ['#FDFEFE', '#1A5276'], N=256)
    n_col = 4
    n_row = int(np.ceil(len(panel_names) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(7.2, 3.7), gridspec_kw={'wspace': 0.45, 'hspace': 0.55})
    axes = np.atleast_1d(axes).ravel()

    for k, name in enumerate(panel_names):
        ax = axes[k]
        cm = confusion_matrix(y_meas_bin, panel_pred[name], labels=[0, 1])
        vmax = cm.max() if cm.max() > 0 else 1
        ax.imshow(cm, cmap=cmap_cm, aspect='equal', vmin=0, vmax=vmax)
        thresh = vmax / 2.0
        for i in range(2):
            row_sum = cm[i].sum()
            for j in range(2):
                val = cm[i, j]
                pct = val / row_sum * 100 if row_sum > 0 else 0.0
                ax.text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center', fontsize=7.5,
                        color='white' if val > thresh else '#2C3E50', linespacing=1.3)
        ax.set_xticks(range(2)); ax.set_yticks(range(2))
        ax.set_xticklabels(cm_labels, fontsize=8); ax.set_xlabel('Predicted', fontsize=8)
        if k % n_col == 0:
            ax.set_yticklabels(cm_labels, fontsize=8, rotation=0); ax.set_ylabel('True label', fontsize=8)
        else:
            ax.set_yticklabels([''] * 2)
        ax.set_title(f'{name}  OA={panel_acc[name]:.1%}', fontsize=9, pad=5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for k in range(len(panel_names), len(axes)):
        fig.delaxes(axes[k])
    plt.tight_layout(pad=0.6)
    fig.savefig(out_dir / 'step9_confusion_all.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  已保存 step9_confusion_all.png")

    return base_results, base_pred


# =====================================================================
# STEP 10  随机森林二分类结果汇总
# =====================================================================

def step10_summary(lsdi_result, base_results, n_samples, out_dir=OUT_DIR):
    """汇总 LSDI(LOO-CV) 与 7个常规VI(PROSAIL训练→实测测试) 的分类准确率，输出对比图与CSV"""
    print("\n" + "=" * 62)
    print("  STEP 10  随机森林二分类结果汇总")
    print("=" * 62)

    acc_lsdi = lsdi_result['acc_lsdi']
    top_combo = lsdi_result['top_combo']

    all_names = ['LSDI'] + BASE_VI
    all_accs = [acc_lsdi] + [base_results[v] for v in BASE_VI]
    all_clrs = ['#C0392B'] + ['#BDC3C7'] * len(BASE_VI)

    order = np.argsort(all_accs)[::-1]
    names_s = [all_names[i] for i in order]
    accs_s = [all_accs[i] for i in order]
    clrs_s = [all_clrs[i] for i in order]
    best_base = max(base_results.values())

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(names_s))
    bars = ax.bar(x, accs_s, 0.62, color=clrs_s, edgecolor='white', linewidth=0.4, zorder=3)
    for bar, v in zip(bars, accs_s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f'{v:.0%}',
                ha='center', va='bottom', fontsize=6.5, color='#2C3E50')
    ax.axhline(best_base, color='#2C3E50', lw=0.8, ls='--', zorder=4,
               label=f'Best conventional VI ({best_base:.0%})')
    ax.set_xticks(x); ax.set_xticklabels(names_s, fontsize=7.5, rotation=35, ha='right')
    ax.set_ylabel('Classification accuracy', fontsize=8)
    ax.set_ylim(0.0, min(1.0, max(accs_s) + 0.18))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.yaxis.grid(True, lw=0.4, color='0.88', zorder=0); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    legend_handles = [
        Patch(facecolor='#C0392B', edgecolor='none', label='LSDI (LOO-CV)'),
        Patch(facecolor='#BDC3C7', edgecolor='none', label='Conventional VIs'),
    ]
    ax.legend(handles=legend_handles, fontsize=6.5, loc='upper right', handlelength=1.0, borderpad=0.5)
    ax.text(0.02, 0.97, f'$n$ = {n_samples} field spectral\nBinary classification (≥60% vs. <60%)',
            transform=ax.transAxes, fontsize=6, va='top', color='0.4', linespacing=1.4)
    plt.tight_layout(pad=0.5)
    fig.savefig(out_dir / 'step10_rf_compare.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  已保存 step10_rf_compare.png")

    rows = [{
        'Index': 'LSDI', 'Type': 'LOO-CV litter index (3-band)',
        'Evaluation': f'LOO-CV (n={n_samples} measured)',
        'Accuracy': round(acc_lsdi, 4), 'Correct': f'{acc_lsdi * n_samples:.0f}/{n_samples}',
        'Final_B1(nm)': top_combo[0], 'Final_B2(nm)': top_combo[1], 'Final_B3(nm)': top_combo[2],
    }]
    for vi in BASE_VI:
        rows.append({
            'Index': vi, 'Type': 'Conventional VI', 'Evaluation': 'PROSAIL-trained → measured test',
            'Accuracy': round(base_results[vi], 4), 'Correct': f'{base_results[vi] * n_samples:.0f}/{n_samples}',
            'Final_B1(nm)': '', 'Final_B2(nm)': '', 'Final_B3(nm)': '',
        })
    pd.DataFrame(rows).sort_values('Accuracy', ascending=False).to_csv(
        out_dir / 'step10_results.csv', index=False, encoding='utf-8-sig')
    print("  已保存 step10_results.csv")

    print("\n" + "=" * 62)
    print(f"  LSDI 最终结果: OA = {acc_lsdi:.1%}  ({acc_lsdi * n_samples:.0f}/{n_samples})")
    print(f"  最终波段：B1={top_combo[0]}nm  B2={top_combo[1]}nm  B3={top_combo[2]}nm")
    print(f"  最佳常规指数: {max(base_results, key=base_results.get)} {best_base:.1%}")
    print("=" * 62)


# =====================================================================
# 主流程入口
# =====================================================================

def main():
    print("=" * 62)
    print("  芦苇立枯物指数（LSDI）构建与验证 —— 整合流程")
    print("=" * 62)

    # ---- STEP 1  读取实测数据 ----
    wl, measured_mean = step1_read_measured_mean(XLSX_MEAN_PATH)
    wl_s, X_meas, y_meas_cov, y_meas_bin = step1_read_measured_samples(XLSX_SAMPLES_PATH, wl)

    # ---- STEP 2  PROSAIL 模拟 ----
    if SIM_EXPORT_PKL.exists():
        # 若已有导出的模拟结果（例如来自单独的批处理任务），可直接加载，跳过重新模拟
        print(f"\n检测到已有模拟结果 {SIM_EXPORT_PKL.name}，直接加载...")
        with open(SIM_EXPORT_PKL, 'rb') as f:
            export = pickle.load(f)
        sim_data = export['sim_data']
        assert np.allclose(wl, export['WL'], atol=0.5), "波段不一致！"
    else:
        sim_data = step2_prosail_simulation(wl)
        with open(SIM_EXPORT_PKL, 'wb') as f:
            pickle.dump({'sim_data': sim_data, 'WL': wl, 'ALL_COVERAGES': ALL_COVERAGES}, f)

    # ---- STEP 3  精度筛选 R²≥0.95 ----
    sim_data, acc_before, acc_after, filter_stats = step3_accuracy_filter(wl, measured_mean, sim_data)

    # ---- STEP 4  光谱响应（原始/一阶/二阶导数）----
    step4_spectral_response(wl, measured_mean, sim_data, sg_win=11, sg_poly=3)

    # ---- STEP 5  SG滤波参数筛选 ----
    best_w, best_p = step5_sg_selection(wl, measured_mean[0])

    # ---- STEP 6  OAT局部敏感性分析 ----
    oat_all = step6_oat_sensitivity(wl)

    # ---- STEP 7  EFAST全局敏感性分析 ----
    efast_S1, efast_ST = step7_efast_sensitivity(wl, efast_n=2000)

    # ---- STEP 8  构建立枯物指数 LSDI ----
    lsdi_result = step8_build_lsdi(wl, X_meas, y_meas_cov, y_meas_bin, sg_win=best_w, sg_poly=best_p)

    # ---- STEP 9  LSDI 与 7个常规指数混淆矩阵对比 ----
    base_results, base_pred = step9_confusion_matrices(wl, X_meas, y_meas_bin, sim_data, lsdi_result)

    # ---- STEP 10  随机森林二分类结果汇总 ----
    step10_summary(lsdi_result, base_results, n_samples=len(X_meas))

    print("\n全部流程完成！结果已保存至:", OUT_DIR)


if __name__ == '__main__':
    main()
