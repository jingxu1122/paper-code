#!/usr/bin/env python3

import sys, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error

# ──  prosail  ──────────────────────────────────────────────
try:
    import prosail
    print(f"prosail {prosail.__version__} loaded")
except ImportError:
    print("Error: please run  pip install prosail")
    sys.exit(1)

# ── SALib (for LHS) ────────────────────────────────────────
from SALib.sample import latin as lhs_sampler

# ── 绘图风格──────────────────────────────────
matplotlib.rcParams['font.family']       = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.linewidth']    = 0.8
matplotlib.rcParams['xtick.direction']   = 'in'
matplotlib.rcParams['ytick.direction']   = 'in'
matplotlib.rcParams['xtick.major.size']  = 3.5
matplotlib.rcParams['ytick.major.size']  = 3.5
matplotlib.rcParams['xtick.minor.size']  = 2.0
matplotlib.rcParams['ytick.minor.size']  = 2.0
matplotlib.rcParams['xtick.labelsize']   = 8
matplotlib.rcParams['ytick.labelsize']   = 8
matplotlib.rcParams['legend.frameon']    = True
matplotlib.rcParams['legend.framealpha'] = 0.92
matplotlib.rcParams['legend.edgecolor']  = '0.75'
matplotlib.rcParams['legend.fontsize']   = 7.5
matplotlib.rcParams['figure.dpi']        = 150

# ── 路径与文件 ──────────────────────────────────────────────
_BASE     = Path(__file__).parent
DATA_FILE = _BASE / '20190521lu.xlsx'          # 实测数据文件
OUT_DIR   = _BASE / 'results_simulation'       # 输出目录
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_COVERAGES = [0, 20, 40, 60, 80, 100]
COL_MAP = {
    0:   'Litter_0%',   20:  'Litter_20%',
    40:  'Litter_40%',  60:  'Litter_60%',
    80:  'Litter_80%',  100: 'Litter_100%',
}
N_SAMPLES = 2000
PARAM_NAMES = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm', 'prot', 'cbc', 'lai']
COLORS = [
    (0.13,0.55,0.13), (0.60,0.76,0.20), (0.93,0.78,0.05),
    (0.90,0.55,0.10), (0.78,0.28,0.08), (0.50,0.18,0.05),
]

# ── 几何与土壤参数（固定） ──────────────────────────────────
GEO = dict(
    hspot=0.01, tts=30.0, tto=0.0, psi=0.0,
    ant=0.0, prospect_version='PRO', rsoil=1.0, psoil=0.85,
)

# ── LIDFA 范围（按覆盖度线性插值） ──────────────────────────
LIDFA_RANGE = {
    0:   (40, 70),   20: (39, 72),   40: (38, 74),
    60:  (37, 76),   80: (36, 78),  100: (35, 80),
}

# ── 各覆盖度参数范围（PROSPECT-PRO + LAI） ───────────────────
TABLE1 = {
    0:   dict(n=(1.5,  2.5),  cab=(15.0, 70.0), car=(1.5,  20.0),
              cbrown=(0.0, 0.0), cw=(0.01, 0.05), cm=(0.005, 0.010),
              prot=(0.0015, 0.0020), cbc=(0.0035, 0.007), lai=(0.9, 2.5)),
    20:  dict(n=(1.2, 3.42), cab=(12.0, 63.0), car=(1.5, 16.4),
              cbrown=(0.0, 0.4), cw=(0.008, 0.041), cm=(0.006, 0.012),
              prot=(0.0010, 0.0016), cbc=(0.0048, 0.0096), lai=(1.0, 2.8)),
    40:  dict(n=(1.5, 2.85), cab=(9.0, 55.0), car=(1.5, 12.8),
              cbrown=(0.0, 0.8), cw=(0.007, 0.014), cm=(0.005, 0.032),
              prot=(0.0005, 0.0013), cbc=(0.0075, 0.0147), lai=(1.2, 3.0)),
    60:  dict(n=(1.4, 2.27), cab=(6.0, 50.0), car=(1.5, 9.2),
              cbrown=(0.0, 1.2), cw=(0.004, 0.023), cm=(0.008, 0.016),
              prot=(0.0003, 0.0008), cbc=(0.0075, 0.0147), lai=(1.2, 3.0)),
    80:  dict(n=(1.3, 1.70), cab=(3.0, 40.0), car=(1.5, 5.6),
              cbrown=(0.0, 1.6), cw=(0.002, 0.014), cm=(0.009, 0.018),
              prot=(0.0002, 0.0005), cbc=(0.0088, 0.0172), lai=(1.0, 2.8)),
    100: dict(n=(1.2, 2.0),  cab=(0.0, 35.0), car=(22.0, 22.0),
              cbrown=(0.0, 2.0), cw=(0.001, 0.005), cm=(0.01, 0.02),
              prot=(0.0000, 0.0004), cbc=(0.01, 0.0196), lai=(0.8, 2.5)),
}

# ── 波长切片 (400–900 nm) ──────────────────────────────────
_SLICE = slice(0, 501)   # prosail 输出 400–2500nm，取前501个

# ── 辅助函数 ──────────────────────────────────────────────
def run_prosail_once(params: dict, lidfa: float = None) -> np.ndarray:
    """调用 prosail.run_prosail，返回 400–900nm 反射率"""
    lidfa_val = lidfa if lidfa is not None else 57.0
    rho_full = prosail.run_prosail(
        n       = float(params['n']),
        cab     = float(params['cab']),
        car     = float(params['car']),
        cbrown  = float(params['cbrown']),
        cw      = float(params['cw']),
        cm      = 0.0,                     # 使用 prot+cbc 替代 cm
        prot    = float(params['prot']),
        cbc     = float(params['cbc']),
        lai     = float(params['lai']),
        lidfa   = lidfa_val,
        hspot   = GEO['hspot'],
        tts     = GEO['tts'],
        tto     = GEO['tto'],
        psi     = GEO['psi'],
        ant     = GEO['ant'],
        prospect_version = GEO['prospect_version'],
        rsoil   = GEO['rsoil'],
        psoil   = GEO['psoil'],
    )
    return rho_full[_SLICE]

def sam(a, b):
    """光谱角距离 (度)"""
    c = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-10)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

def metrics(meas, sim, vis_mask, nir_mask):
    """计算 R², RMSE, SAM 及分波段指标"""
    return dict(
        R2=r2_score(meas, sim),
        RMSE=np.sqrt(mean_squared_error(meas, sim)),
        SAM=sam(meas, sim),
        R2_vis=r2_score(meas[vis_mask], sim[vis_mask]),
        R2_nir=r2_score(meas[nir_mask], sim[nir_mask]),
        RMSE_vis=np.sqrt(mean_squared_error(meas[vis_mask], sim[vis_mask])),
        RMSE_nir=np.sqrt(mean_squared_error(meas[nir_mask], sim[nir_mask])),
    )

# ─── 主程序 ────────────────────────────────────────────────
def main():
    print("="*62)
    print("  PROSAIL simulation & validation (core)")
    print("  Chongming Dongtan wetland – litter coverage gradient")
    print("="*62)

    # ── 1. 读取实测数据 ─────────────────────────────────────
    print("\n[1] Loading measured spectra...")
    df  = pd.read_excel(DATA_FILE, sheet_name='Sheet1', header=0)
    WL  = df.iloc[:,0].values.astype(float)
    measured = {cov: df[COL_MAP[cov]].values.astype(float) for cov in ALL_COVERAGES}
    vis_mask = (WL>=400)&(WL<=700)
    nir_mask = (WL>700)&(WL<=900)
    for cov in ALL_COVERAGES:
        v = measured[cov]
        print(f"  {COL_MAP[cov]:12s}: VIS={v[vis_mask].mean():.4f}  NIR={v[nir_mask].mean():.4f}")

    # ── 2. 实测光谱图 ──────────────────────────────────────
    print("\n[2] Plotting measured spectra...")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, cov in enumerate(ALL_COVERAGES):
        ax.plot(WL, measured[cov], lw=1.2, color=COLORS[i], label=f'{cov}%')
    ax.set_xlabel('Wavelength (nm)', fontsize=8)
    ax.set_ylabel('Reflectance', fontsize=8)
    ax.set_xlim(400, 900); ax.set_ylim(0, 0.55)
    ax.grid(True, lw=0.4, color='0.88', zorder=0); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(title='Coverage', fontsize=7, title_fontsize=7,
              loc='upper left', ncol=2, handlelength=1.2)
    plt.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR/'measured_spectra.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  measured_spectra.png")

    # ── 3. LHS 采样 + PROSAIL 模拟 ──────────────────────
    print(f"\n[3] Monte Carlo simulation ({N_SAMPLES} runs/coverage, LHS)...")
    sim_data = {}
    t0 = time.time()
    for cov in ALL_COVERAGES:
        ranges   = TABLE1[cov]
        lidfa_lo, lidfa_hi = LIDFA_RANGE[cov]

        # 构建 SALib 问题（包含 lidfa）
        lhs_names = PARAM_NAMES + ['lidfa']
        lhs_bounds = []
        for k in PARAM_NAMES:
            lo, hi = ranges[k][0], ranges[k][1]
            if lo >= hi:          # 固定参数，加极小扰动
                lo, hi = lo - 1e-8, lo + 1e-8
            lhs_bounds.append([lo, hi])
        lhs_bounds.append([lidfa_lo, lidfa_hi])
        lhs_problem = {
            'num_vars': len(lhs_names),
            'names':    lhs_names,
            'bounds':   lhs_bounds,
        }
        lhs_mat = lhs_sampler.sample(lhs_problem, N_SAMPLES, seed=42)

        specs = np.zeros((N_SAMPLES, len(WL)))
        pmat  = np.zeros((N_SAMPLES, len(PARAM_NAMES)))
        for j in range(N_SAMPLES):
            row = lhs_mat[j]
            p = {}
            for pi, k in enumerate(PARAM_NAMES):
                lo0, hi0 = ranges[k][0], ranges[k][1]
                p[k] = float(lo0) if lo0 >= hi0 else float(row[pi])
                pmat[j, pi] = p[k]
            lidfa_val = float(row[len(PARAM_NAMES)])
            specs[j] = run_prosail_once(p, lidfa=lidfa_val)
            if (j+1) % 500 == 0:
                print(f"  Coverage {cov:3d}%  {j+1}/{N_SAMPLES}  ({time.time()-t0:.0f}s)")

        sim_data[cov] = dict(
            all=specs, mean=specs.mean(0), std=specs.std(0),
            p5=np.percentile(specs,5,0), p95=np.percentile(specs,95,0),
            pmat=pmat
        )
        print(f"  Coverage {cov:3d}%  VIS={specs[:,vis_mask].mean():.4f}  "
              f"NIR={specs[:,nir_mask].mean():.4f}  done")
    print(f"\n  Simulation finished ({N_SAMPLES*len(ALL_COVERAGES)} runs, {time.time()-t0:.0f}s)")

    # ── 4. 光谱筛选 (R² ≥ 0.95) ──────────────────────────
    R2_THRESHOLD = 0.95
    print(f"\n[4] Filtering by NSE-R² ≥ {R2_THRESHOLD} (simulated vs. measured)...")
    acc_before = {}
    acc_after  = {}
    filter_stats = {}
    param_ranges = {}

    # 筛选前精度
    for cov in ALL_COVERAGES:
        meas = measured[cov]
        mean_before = sim_data[cov]['all'].mean(axis=0)
        acc_before[cov] = metrics(meas, mean_before, vis_mask, nir_mask)

    print(f"\n  Pre-filtering mean spectral accuracy (all {N_SAMPLES} samples):")
    print(f"  {'Coverage':>8}  {'R²':>8}  {'RMSE':>9}  {'SAM(°)':>7}")
    print("  " + "─"*38)
    for cov in ALL_COVERAGES:
        m = acc_before[cov]
        print(f"  {cov:4d}%  {m['R2']:8.4f}  {m['RMSE']:9.5f}  {m['SAM']:7.3f}")

    # 逐样本筛选
    for cov in ALL_COVERAGES:
        specs_all = sim_data[cov]['all']
        pmat_all  = sim_data[cov]['pmat']
        meas      = measured[cov]
        n_tot     = len(specs_all)

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
            print(f"  ⚠ Coverage {cov}%: no samples passed, using all")
            mask = np.ones(n_tot, dtype=bool)
            n_kept = n_tot

        specs_f = specs_all[mask]
        pmat_f  = pmat_all[mask]
        r2_f    = r2_arr[mask]

        # 参数范围统计
        prange = {}
        for pi, pname in enumerate(PARAM_NAMES):
            col = pmat_f[:, pi]
            lo_in, hi_in = TABLE1[cov][pname]
            span_in = hi_in - lo_in if hi_in > lo_in else 1e-10
            p5_a  = float(np.percentile(col,  5))
            p95_a = float(np.percentile(col, 95))
            prange[pname] = dict(
                min=float(col.min()), max=float(col.max()),
                mean=float(col.mean()), std=float(col.std()),
                p5=p5_a, p95=p95_a,
                shrink=round(1.0 - (p95_a - p5_a) / span_in, 3),
            )
        param_ranges[cov] = prange

        # 包络覆盖率
        p5_spec  = np.percentile(specs_f,  5, axis=0)
        p95_spec = np.percentile(specs_f, 95, axis=0)
        envelope_cov = float(((meas >= p5_spec) & (meas <= p95_spec)).mean() * 100)

        filter_stats[cov] = dict(
            n_total=n_tot, n_kept=n_kept, ratio=n_kept/n_tot,
            r2_all=r2_arr, r2_kept=r2_f,
            r2_mean=float(r2_f.mean()), r2_median=float(np.median(r2_f)),
            r2_min=float(r2_f.min()), r2_std=float(r2_f.std()),
        )

        # 更新 sim_data 为筛选后子集
        sim_data[cov].update(dict(
            all=specs_f, mean=specs_f.mean(0), std=specs_f.std(0),
            p5=p5_spec, p95=p95_spec, pmat=pmat_f, r2_per_sample=r2_f,
        ))
        acc_after[cov] = metrics(meas, sim_data[cov]['mean'], vis_mask, nir_mask)

        print(f"  Coverage {cov:3d}%  kept {n_kept:4d}/{n_tot} ({n_kept/n_tot*100:.1f}%)  "
              f"R²: mean={r2_f.mean():.4f} min={r2_f.min():.4f}  envelope={envelope_cov:.1f}%")

    # 筛选后精度
    print(f"\n  Post-filtering mean spectral accuracy (R²≥{R2_THRESHOLD} retained):")
    print(f"  {'Coverage':>8}  {'R²':>8}  {'RMSE':>9}  {'SAM(°)':>7}")
    print("  " + "─"*38)
    for cov in ALL_COVERAGES:
        m = acc_after[cov]
        print(f"  {cov:4d}%  {m['R2']:8.4f}  {m['RMSE']:9.5f}  {m['SAM']:7.3f}")

    # ── 5. 绘图：筛选前后精度对比 ──────────────────────────
    print("\n[5] Plotting pre/post filtering comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4),
                             gridspec_kw={'wspace': 0.40})
    labels = [f'{c}%' for c in ALL_COVERAGES]
    x = np.arange(len(ALL_COVERAGES)); w = 0.35
    for ax, key, ylabel, title, hl in [
        (axes[0], 'R2',   'R²',      'R²',   (0.85, 'R²=0.85')),
        (axes[1], 'RMSE', 'RMSE',    'RMSE', None),
        (axes[2], 'SAM',  'SAM (°)', 'SAM',  (10, 'SAM=10°')),
    ]:
        b_vals = [acc_before[c][key] for c in ALL_COVERAGES]
        a_vals = [acc_after[c][key]  for c in ALL_COVERAGES]
        ax.bar(x - w/2, b_vals, w, label=f'Before (n={N_SAMPLES})',
               color='#5B9BD5', alpha=0.88, edgecolor='white', lw=0.3, zorder=3)
        ax.bar(x + w/2, a_vals, w, label=f'After (R²≥{R2_THRESHOLD})',
               color='#C0392B', alpha=0.90, edgecolor='white', lw=0.3, zorder=3)
        if hl:
            ax.axhline(hl[0], color='#2C3E50', lw=0.8, ls='--', zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlabel('Litter coverage (%)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f'{title}', fontsize=8, pad=3)
        ax.grid(True, lw=0.4, color='0.88', zorder=0); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=6.5, loc='upper right', borderpad=0.4)
    plt.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR/'before_after_accuracy.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  before_after_accuracy.png")

    # R²分布直方图
    print("  Plotting R² distribution histograms...")
    fig, axes = plt.subplots(3, 2, figsize=(5.5, 6.5),
                             gridspec_kw={'hspace': 0.55, 'wspace': 0.38})
    for idx, cov in enumerate(ALL_COVERAGES):
        ax  = axes[idx // 2, idx % 2]
        fs  = filter_stats[cov]
        bins = np.linspace(max(fs['r2_all'].min(), -0.2), 1.0, 35)
        ax.hist(fs['r2_all'],  bins=bins, color='0.82', edgecolor='white',
                lw=0.2, label=f'All (n={fs["n_total"]})', zorder=1)
        ax.hist(fs['r2_kept'], bins=bins, color=COLORS[idx], alpha=0.88,
                edgecolor='white', lw=0.2, label=f'Retained (n={fs["n_kept"]})', zorder=2)
        ax.axvline(R2_THRESHOLD, color='#C0392B', lw=0.9, ls='--', zorder=4)
        ax.axvline(fs['r2_mean'], color='#2C3E50', lw=0.9, ls='-',  zorder=4)
        ax.set_title(f'Litter {cov}%  (retained {fs["ratio"]*100:.0f}%)', fontsize=7.5, pad=2)
        ax.set_xlabel('NSE-R² (simulated vs. measured)', fontsize=7.5)
        ax.set_ylabel('Count', fontsize=7.5)
        ax.grid(True, lw=0.35, color='0.88', zorder=0); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(fontsize=5.5, loc='upper left', borderpad=0.4, handlelength=1.0)
    fig.savefig(OUT_DIR/'r2_distribution.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  r2_distribution.png")

    # 参数收缩热力图
    print("  Plotting parameter shrinkage heatmap...")
    shrink_mat = np.array([[param_ranges[c][p]['shrink'] for c in ALL_COVERAGES]
                           for p in PARAM_NAMES])
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    im = ax.imshow(shrink_mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(ALL_COVERAGES)))
    ax.set_xticklabels([f'{c}%' for c in ALL_COVERAGES], fontsize=7.5)
    ax.set_yticks(range(len(PARAM_NAMES)))
    ax.set_yticklabels(PARAM_NAMES, fontsize=7.5)
    ax.tick_params(length=0)
    for i in range(len(PARAM_NAMES)):
        for j in range(len(ALL_COVERAGES)):
            val = shrink_mat[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6.5, color='white' if val > 0.55 else '#2C3E50')
    cb = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Shrinkage ratio', fontsize=7.5)
    ax.set_xlabel('Litter coverage (%)', fontsize=8)
    ax.set_ylabel('Parameter', fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR/'param_shrinkage.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  param_shrinkage.png")

    # ── 6. 模拟光谱图（筛选后） ──────────────────────────────
    print("\n[6] Plotting simulated spectra (post-filtering)...")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, cov in enumerate(ALL_COVERAGES):
        ax.plot(WL, sim_data[cov]['mean'], lw=1.2, color=COLORS[i], label=f'{cov}%')
        ax.fill_between(WL, sim_data[cov]['p5'], sim_data[cov]['p95'],
                        color=COLORS[i], alpha=0.10)
    ax.set_xlabel('Wavelength (nm)', fontsize=8)
    ax.set_ylabel('Reflectance', fontsize=8)
    ax.set_xlim(400, 900); ax.set_ylim(0, 0.60)
    ax.grid(True, lw=0.4, color='0.88', zorder=0); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(title='Coverage', fontsize=7, title_fontsize=7,
              loc='upper left', ncol=2, handlelength=1.2)
    plt.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR/'simulated_spectra.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  simulated_spectra.png")

    # ── 7. 精度评价条状图 ──────────────────────────────────
    print("\n[7] Accuracy evaluation bar charts...")
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.4),
                             gridspec_kw={'hspace': 0.48, 'wspace': 0.38})
    r2s   = [acc_after[c]['R2']       for c in ALL_COVERAGES]
    rmses = [acc_after[c]['RMSE']     for c in ALL_COVERAGES]
    sams  = [acc_after[c]['SAM']      for c in ALL_COVERAGES]
    rv    = [acc_after[c]['RMSE_vis'] for c in ALL_COVERAGES]
    rn    = [acc_after[c]['RMSE_nir'] for c in ALL_COVERAGES]
    labels = [f'{c}%' for c in ALL_COVERAGES]

    for ax, vals, title, ylabel, hl in [
        (axes[0,0], r2s,  'R²',    'R²',     (0.85, 'R²=0.85')),
        (axes[0,1], rmses,'RMSE',  'RMSE',   None),
        (axes[1,0], sams, 'SAM',   'SAM (°)',(10, 'SAM=10°')),
        (axes[1,1], None, 'RMSE by band','RMSE', None),
    ]:
        if vals is not None:
            bars = ax.bar(labels, vals, color=COLORS, edgecolor='white', lw=0.3, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v*1.018,
                        f'{v:.3f}', ha='center', fontsize=6, color='#2C3E50')
            if hl:
                ax.axhline(hl[0], color='#C0392B', lw=0.8, ls='--', zorder=4)
        else:
            x = np.arange(len(ALL_COVERAGES)); w = 0.35
            ax.bar(x-w/2, rv, w, label='VIS (400–700 nm)',
                   color='#2980B9', alpha=0.88, edgecolor='white', lw=0.2, zorder=3)
            ax.bar(x+w/2, rn, w, label='NIR (700–900 nm)',
                   color='#C0392B', alpha=0.88, edgecolor='white', lw=0.2, zorder=3)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
            ax.legend(fontsize=6.5, borderpad=0.4)
        ax.set_title(f'{title}', fontsize=8, pad=3)
        ax.set_xlabel('Litter coverage (%)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if vals is not None:
            ax.set_xticklabels(labels, fontsize=7)
        ax.grid(True, lw=0.35, color='0.88', zorder=0); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR/'accuracy_bars.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  accuracy_bars.png")

    # ── 8. 实测 vs 模拟对比子图（6个覆盖度） ──────────────
    print("\n[8] Measured vs simulated comparison (6 panels)...")
    fig, axes = plt.subplots(3, 2, figsize=(5.5, 7.0),
                             gridspec_kw={'hspace': 0.55, 'wspace': 0.38})
    for i, cov in enumerate(ALL_COVERAGES):
        ax = axes[i // 2, i % 2]
        ax.fill_between(WL, sim_data[cov]['p5'], sim_data[cov]['p95'],
                        color='0.80', alpha=0.50, label='Simulated 5–95%', zorder=1)
        ax.plot(WL, sim_data[cov]['mean'], color='0.45', lw=0.9,
                ls='--', label='Simulated mean', zorder=2)
        ax.plot(WL, measured[cov], color=COLORS[i], lw=1.3,
                label='Measured', zorder=3)
        m = acc_after[cov]
        ax.set_title(f'Litter {cov}%  R²={m["R2"]:.3f}  RMSE={m["RMSE"]:.4f}  SAM={m["SAM"]:.1f}°',
                     fontsize=7.0, pad=2)
        ax.set_xlim(400, 900); ax.set_ylim(0, 0.52)
        ax.set_xlabel('Wavelength (nm)', fontsize=7.5)
        ax.set_ylabel('Reflectance', fontsize=7.5)
        ax.grid(True, lw=0.35, color='0.88', zorder=0); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if i == 0:
            ax.legend(fontsize=6, loc='upper left', borderpad=0.4, handlelength=1.2)
    plt.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR/'comparison_panels.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("  comparison_panels.png")

    # ── 9. 保存筛选后参数范围 CSV ──────────────────────────
    rows = []
    for cov in ALL_COVERAGES:
        for pname in PARAM_NAMES:
            r = param_ranges[cov][pname]
            rows.append({
                'Coverage(%)': cov, 'Param': pname,
                'min': r['min'], 'max': r['max'],
                'mean': r['mean'], 'std': r['std'],
                'p5': r['p5'], 'p95': r['p95'],
                'Shrinkage': r['shrink'],
                'n_kept': filter_stats[cov]['n_kept'],
                'n_total': filter_stats[cov]['n_total'],
                'Retention': round(filter_stats[cov]['ratio'], 4),
            })
    pd.DataFrame(rows).to_csv(OUT_DIR/'filtered_param_ranges.csv', index=False, encoding='utf-8-sig')
    print("  filtered_param_ranges.csv")

    # ── 10. 保存精度表 ─────────────────────────────────────
    pd.DataFrame([{
        'Coverage(%)':c, 'R2':acc_after[c]['R2'], 'RMSE':acc_after[c]['RMSE'],
        'SAM(deg)':acc_after[c]['SAM'], 'VIS_RMSE':acc_after[c]['RMSE_vis'],
        'NIR_RMSE':acc_after[c]['RMSE_nir'], 'R2_VIS':acc_after[c]['R2_vis'],
        'R2_NIR':acc_after[c]['R2_nir'],
    } for c in ALL_COVERAGES]).to_csv(
        OUT_DIR/'accuracy_evaluation.csv', index=False, encoding='utf-8-sig')
    print("  accuracy_evaluation.csv")

    # ── 11. 保存光谱对比数据 ──────────────────────────────
    sim_df = pd.DataFrame({'wavelength_nm': WL})
    for cov in ALL_COVERAGES:
        sim_df[f'meas_{cov}%']    = measured[cov]
        sim_df[f'sim_mean_{cov}%']= sim_data[cov]['mean']
        sim_df[f'sim_p5_{cov}%']  = sim_data[cov]['p5']
        sim_df[f'sim_p95_{cov}%'] = sim_data[cov]['p95']
    sim_df.to_csv(OUT_DIR/'spectra_comparison.csv', index=False, encoding='utf-8-sig')
    print("  spectra_comparison.csv")

    # ── 12. 生成简短的报告 ──────────────────────────────────
    with open(OUT_DIR/'simulation_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*58 + "\nPROSAIL simulation summary (core)\n")
        f.write(f"Data: 20190521lu.xlsx, {len(WL)} bands (400–900 nm)\n")
        f.write(f"Monte Carlo: {N_SAMPLES} runs/coverage, LHS sampling\n")
        f.write(f"Filtering: NSE-R² ≥ {R2_THRESHOLD}\n\n")
        f.write("Post-filtering accuracy:\n")
        f.write(f"  {'Coverage':>8}  {'R²':>8}  {'RMSE':>9}  {'SAM(°)':>7}\n")
        f.write("  " + "─"*38 + "\n")
        for c in ALL_COVERAGES:
            m = acc_after[c]
            f.write(f"  {c:4d}%  {m['R2']:8.4f}  {m['RMSE']:9.5f}  {m['SAM']:7.3f}\n")
        f.write("\nOutput files:\n")
        f.write("  - measured_spectra.png\n  - simulated_spectra.png\n")
        f.write("  - comparison_panels.png\n  - accuracy_bars.png\n")
        f.write("  - before_after_accuracy.png\n  - r2_distribution.png\n")
        f.write("  - param_shrinkage.png\n  - filtered_param_ranges.csv\n")
        f.write("  - accuracy_evaluation.csv\n  - spectra_comparison.csv\n")
    print("  simulation_report.txt")

    print("\n" + "="*62)
    print("  Simulation & plotting complete.")
    print(f"  All outputs saved to: {OUT_DIR}")
    print("="*62)

if __name__ == '__main__':
    main()
