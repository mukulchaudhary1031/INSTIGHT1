import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

BG='#07090F'; BG2='#0D1120'; BG3='#111827'; BORDER='#1E2D45'
TEXT='#E8EDF7'; MUTED='#6B7A99'
PALETTE=['#00E5C8','#00BFFF','#FF8C42','#9B5DE5','#FF4D6D','#3DDB8A','#FFD166','#A8DADC','#FF6B9D','#C4E538']


# ════════════════════════════════════════
# FIX #2: Thread-safe figure creation
# Apply theme per-figure, never globally
# ════════════════════════════════════════
def _make_fig(*args, **kwargs):
    """Create figure with theme applied locally — thread safe"""
    fig = plt.figure(*args, **kwargs)
    fig.patch.set_facecolor(BG)
    return fig

def _style_ax(ax):
    """Apply dark theme to a single axes object"""
    ax.set_facecolor(BG2)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, alpha=0.25, color=BORDER)
    return ax

def _styled_subplots(nrows=1, ncols=1, figsize=(10,5), **kwargs):
    """Create subplots with dark theme applied per-axes — thread safe"""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor=BG, **kwargs)
    if isinstance(axes, np.ndarray):
        for ax in axes.flatten():
            _style_ax(ax)
    elif hasattr(axes, '__iter__'):
        for ax in axes:
            _style_ax(ax)
    else:
        _style_ax(axes)
    return fig, axes

def _to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return result

def _hide_extra(axes, used):
    for j in range(used, len(axes)):
        axes[j].set_visible(False)

def _fmt(v):
    """FIX #17: Single consistent formatter used everywhere"""
    if v is None: return '—'
    n = float(v)
    if abs(n) >= 1e9:  return f'{n/1e9:.1f}B'
    if abs(n) >= 1e6:  return f'{n/1e6:.1f}M'
    if abs(n) >= 1e3:  return f'{n/1e3:.1f}K'
    return f'{n:.2f}'


# ════════════════════════════════════════
# SMART FEATURE SELECTOR — FIX #3
# ════════════════════════════════════════
def _is_id_col(series):
    """FIX: Float columns are NEVER IDs"""
    if series.dtype in ['float64', 'float32']: return False
    n = len(series); u = series.nunique()
    if series.dtype == object and u > 0.95 * n and n > 20: return True
    if series.dtype in ['int64','int32','int16'] and u == n:
        try:
            s_sorted = series.dropna().sort_values().values
            diffs = np.diff(s_sorted)
            if len(diffs) > 0 and np.all(diffs == diffs[0]): return True
        except: pass
    return False

def _smart_num(df, exclude=[]):
    return [c for c in df.select_dtypes(include=np.number).columns
            if c not in exclude and not _is_id_col(df[c])]

def _smart_cat(df, exclude=[]):
    return [c for c in df.select_dtypes(include=['object','category']).columns
            if c not in exclude and not _is_id_col(df[c])]


# ════════════════════════════════════════
# DATE DETECTION — FIX #20
# ════════════════════════════════════════
def _detect_date_col(df):
    keywords = ['date','time','month','year','week','day','period','quarter']
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True,
                                        errors='coerce')
                if parsed.notna().sum() > 0.7 * len(df):
                    return col, parsed
            except: pass
        if 'datetime' in str(df[col].dtype):
            return col, df[col]
    return None, None


# ════════════════════════════════════════
# INSIGHTS — English only
# ════════════════════════════════════════
def _ins_hist(df, col):
    data = df[col].dropna()
    if data.nunique() < 2:
        return f"'{col}' has only one value — not useful for analysis."
    skew = data.skew()
    q1,q3 = data.quantile(0.25), data.quantile(0.75)
    iqr   = q3 - q1
    n_out = int(((data < q1-1.5*iqr) | (data > q3+1.5*iqr)).sum())
    if abs(skew) < 0.5:   s = f"✅ <b>{col}</b> is evenly distributed. Average: <b>{_fmt(data.mean())}</b>"
    elif skew > 0.5:      s = f"📊 <b>{col}</b> is right-skewed — most values are small, a few are very large."
    else:                  s = f"📊 <b>{col}</b> is left-skewed — most values are large, a few are very small."
    if n_out > 0: s += f" ⚠️ <b>{n_out} outliers</b> detected."
    return s

def _ins_corr(df, cols):
    if len(cols) < 2: return "Not enough numeric columns for correlation."
    corr = df[cols].corr(); pairs = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = corr.iloc[i,j]
            if abs(r) > 0.6:
                d  = "move together" if r > 0 else "move in opposite directions"
                st = "very strong" if abs(r) > 0.8 else "strong"
                pairs.append(f"<b>{cols[i]}</b> & <b>{cols[j]}</b>: {st} link (r={r:.2f}) — they {d}")
    return ("🔗 " + '<br>'.join(pairs[:3])) if pairs else "ℹ️ No strong correlations — columns are mostly independent."

def _ins_missing(df):
    m = df.isnull().sum(); m = m[m > 0]
    if m.empty: return "✅ <b>No missing data!</b> Dataset is clean."
    w   = m.idxmax(); wp = round(m.max()/len(df)*100, 1)
    return f"⚠️ <b>{len(m)} columns</b> have missing values. Worst: <b>'{w}'</b> ({wp}% missing). Auto-filled."

# FIX #7: Proper imbalance ratio — considers all minority classes
def _ins_balance(df, target):
    vc  = df[target].value_counts()
    total = len(df)
    majority_pct = vc.iloc[0] / total * 100
    minority_pct = vc.iloc[-1] / total * 100
    # Compare majority vs average of others (not just min)
    avg_others = vc.iloc[1:].mean() if len(vc) > 1 else vc.iloc[0]
    ir = vc.iloc[0] / max(avg_others, 1)

    if ir > 5:
        return (f"🚨 <b>Severely imbalanced!</b> '{vc.index[0]}' = {majority_pct:.1f}% of data. "
                f"Minority class '{vc.index[-1]}' = only {minority_pct:.1f}%. "
                f"Model will strongly favor the majority class. Fix: use class_weight='balanced'.")
    elif ir > 2:
        return (f"⚠️ <b>Moderately imbalanced</b> — '{vc.index[0]}' ({majority_pct:.1f}%) "
                f"dominates. Monitor minority class metrics carefully.")
    return f"✅ <b>Well balanced!</b> All classes are roughly equal — predictions will be fair across all classes."


# ════════════════════════════════════════
# MAIN EDA GENERATOR
# ════════════════════════════════════════
def generate_eda_visualizations(df: pd.DataFrame, target_col: str, task_type: str) -> list:
    charts = []

    # Clean numeric-looking class labels
    if task_type == 'classification' and df[target_col].dtype == object:
        bad = df[target_col].astype(str).str.strip().str.match(r'^-?\d+(\.\d+)?$')
        if bad.sum() > 0: df = df[~bad].copy()

    feat_num = _smart_num(df, exclude=[target_col])
    feat_cat = _smart_cat(df, exclude=[target_col])
    tgt_is_num = target_col in df.select_dtypes(include=np.number).columns
    all_num = feat_num + ([target_col] if tgt_is_num else [])

    # ── 1. CLASS BALANCE — FIX #7 ──────────────────────────
    if task_type == 'classification':
        vc     = df[target_col].value_counts()
        n_cls  = len(vc)
        avg_others = vc.iloc[1:].mean() if n_cls > 1 else vc.iloc[0]
        ir     = vc.iloc[0] / max(avg_others, 1)
        imb_color = '#FF4D6D' if ir>5 else '#FF8C42' if ir>2 else '#3DDB8A'
        imb_label = 'SEVERELY IMBALANCED' if ir>5 else 'MODERATELY IMBALANCED' if ir>2 else 'BALANCED ✅'
        colors = [PALETTE[i%len(PALETTE)] for i in range(n_cls)]

        if n_cls > 8:
            fig, ax = plt.subplots(figsize=(11, max(5, n_cls*0.52)), facecolor=BG)
            _style_ax(ax)
            bars = ax.barh(vc.index.astype(str)[::-1], vc.values[::-1],
                           color=colors[::-1], edgecolor='none', alpha=0.85, height=0.65)
            for bar,val in zip(bars, vc.values[::-1]):
                ax.text(bar.get_width()+max(vc.values)*0.01,
                        bar.get_y()+bar.get_height()/2,
                        f'{val:,}  ({val/len(df)*100:.1f}%)',
                        va='center', fontsize=8.5, color=TEXT, fontweight='600')
            ax.set_xlabel('Count', fontsize=10)
            ax.set_xlim(0, max(vc.values)*1.3)
            ax.set_title(f'Target: {target_col}   |   {imb_label}',
                         fontsize=11, color=imb_color, pad=10, fontweight='bold')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
            ax = axes[0]; _style_ax(ax); _style_ax(axes[1])
            bars = ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor='none', width=0.55)
            for bar,val in zip(bars, vc.values):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+max(vc.values)*0.015,
                        f'{val:,}\n({val/len(df)*100:.1f}%)',
                        ha='center', va='bottom', fontsize=9, color=TEXT, fontweight='600')
            ax.set_xlabel('Class', fontsize=10); ax.set_ylabel('Count', fontsize=10)
            ax.set_ylim(0, max(vc.values)*1.22)
            ax.set_title(f'Target: {target_col}   |   {imb_label}',
                         fontsize=11, color=imb_color, pad=10, fontweight='bold')
            ax2 = axes[1]
            exp = [0.05 if v == vc.min() else 0 for v in vc.values]
            _, _, autotexts = ax2.pie(vc.values, labels=vc.index.astype(str),
                autopct='%1.1f%%', colors=colors,
                textprops={'color': TEXT, 'fontsize': 9},
                wedgeprops={'linewidth': 2, 'edgecolor': BG},
                startangle=90, explode=exp)
            for at in autotexts: at.set_color(TEXT); at.set_fontweight('bold')
            ax2.set_title('Class Proportion', fontsize=11, color=TEXT)

        plt.tight_layout()
        charts.append({"title":"Class Balance Analysis","image":_to_b64(fig),
                       "insight": _ins_balance(df, target_col)})

    # ── 2. HISTOGRAMS — KDE normalized to count scale ──────
    if feat_num:
        n = len(feat_num); ncols = min(3,n); nrows = (n+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5*ncols, 5*nrows), facecolor=BG)
        axs = np.array(axes).flatten() if n > 1 else [axes]
        [_style_ax(a) for a in axs]
        insights = []
        for i, col in enumerate(feat_num):
            ax = axs[i]; color = PALETTE[i%len(PALETTE)]
            data = df[col].dropna()
            if data.nunique() < 2:
                ax.text(0.5, 0.5, f'{col}\n(constant — no variation)',
                        ha='center', va='center', color=MUTED,
                        fontsize=10, transform=ax.transAxes)
                insights.append(f"'{col}' has no variation — consider dropping it.")
                continue

            if data.skew() > 2 and data.min() > 0:
                ax.hist(data, bins=40, color=color, alpha=0.75, edgecolor='none')
                ax.set_xscale('log')
                note = ' (log scale)'
            else:
                counts, edges, _ = ax.hist(data, bins=35, color=color, alpha=0.75, edgecolor='none')
                note = ''
                # FIX: KDE normalized to histogram count scale
                try:
                    kde = scipy_stats.gaussian_kde(data)
                    xr  = np.linspace(data.min(), data.max(), 300)
                    bw  = edges[1] - edges[0]
                    ax.plot(xr, kde(xr)*len(data)*bw, color='white', linewidth=2, alpha=0.85)
                except: pass

            ax.axvline(data.mean(),   color='#FFD166', linewidth=1.8, linestyle='--', alpha=0.9)
            ax.axvline(data.median(), color='#FF4D6D', linewidth=1.8, linestyle=':',  alpha=0.9)
            ax.text(0.97, 0.97, f'Avg: {_fmt(data.mean())}\nMed: {_fmt(data.median())}',
                    transform=ax.transAxes, va='top', ha='right', fontsize=8, color=MUTED,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=BG3, alpha=0.8, edgecolor=BORDER))
            ax.set_title(f'{col}{note}   (skew={data.skew():.2f})', fontsize=10, color=TEXT, pad=5)
            ax.set_xlabel(col, fontsize=9); ax.set_ylabel('Count', fontsize=9)
            insights.append(_ins_hist(df, col))

        _hide_extra(axs, n)
        fig.legend(handles=[
            mpatches.Patch(color='#FFD166', label='Mean'),
            mpatches.Patch(color='#FF4D6D', label='Median'),
        ], loc='lower center', ncol=2, fontsize=8, framealpha=0.7,
           bbox_to_anchor=(0.5, -0.02), facecolor=BG3, edgecolor=BORDER,
           labelcolor=TEXT)
        fig.suptitle('Distribution — How Values Are Spread', fontsize=14, color=TEXT,
                     y=1.01, fontweight='bold')
        plt.tight_layout()
        charts.append({"title":"Value Distribution","image":_to_b64(fig),
                       "insight":'<br>'.join(insights[:4])})

    # ── 3. CORRELATION HEATMAP — FIX #16 ───────────────────
    if len(all_num) >= 2:
        corr  = df[all_num].corr()
        n_c   = len(all_num)
        ann_fs = max(6, min(11, int(110/max(n_c,1))))
        fig_w  = max(8, min(20, n_c*1.2+2))
        fig_h  = max(7, min(18, n_c*1.0+1))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
        ax.set_facecolor(BG2)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(10, 180, s=90, l=45, as_cmap=True)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, mask=mask, ax=ax,
                    linewidths=0.4, linecolor=BG,
                    annot_kws={'size': ann_fs, 'weight': 'bold'},
                    vmin=-1, vmax=1, center=0,
                    cbar_kws={'shrink': 0.7, 'label': 'Pearson r'})
        # FIX #16: Auto-rotate for many columns
        rot = 45 if n_c > 6 else 0
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rot, ha='right',
                           fontsize=max(6, ann_fs-1))
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=max(6, ann_fs-1))
        ax.set_title('Correlation Matrix — How Columns Relate',
                     fontsize=13, color=TEXT, pad=12, fontweight='bold')
        # FIX #9: Clear note that correlation ≠ causation
        fig.text(0.5, -0.02,
                 '⚠️ Correlation shows relationship strength, NOT causation. '
                 'High correlation does not mean one column causes the other.',
                 ha='center', fontsize=8, color=MUTED,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=BG3,
                           alpha=0.8, edgecolor=BORDER))
        plt.tight_layout()
        charts.append({"title":"Correlation Heatmap","image":_to_b64(fig),
                       "insight":_ins_corr(df, all_num)})

    # ── 4. BOX PLOTS — clipped for visibility ──────────────
    if feat_num:
        n = len(feat_num); ncols = min(3,n); nrows = (n+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5*ncols, 5*nrows), facecolor=BG)
        axs = np.array(axes).flatten() if n > 1 else [axes]
        [_style_ax(a) for a in axs]
        vi = []
        for i, col in enumerate(feat_num):
            ax = axs[i]; color = PALETTE[i%len(PALETTE)]
            data = df[col].dropna()
            if data.nunique() < 5:
                vc2 = data.value_counts().sort_index()
                ax.barh([str(v) for v in vc2.index], vc2.values,
                        color=color, alpha=0.8, edgecolor='none')
                ax.set_title(f'{col}  ({data.nunique()} unique values)',
                             fontsize=10, color=TEXT, pad=5)
                ax.set_xlabel('Count', fontsize=9)
                vi.append(f"'{col}' has only {data.nunique()} unique values.")
                continue

            vals = data.values
            lo, hi = np.percentile(vals, [0.5, 99.5])
            bp = ax.boxplot(vals, patch_artist=True, vert=True,
                            boxprops=dict(facecolor=color, alpha=0.6, linewidth=1.5),
                            medianprops=dict(color='white', linewidth=2.5),
                            whiskerprops=dict(color=MUTED, linewidth=1.2),
                            capprops=dict(color=MUTED, linewidth=1.5),
                            flierprops=dict(marker='.', color=color, alpha=0.3, markersize=4))
            ax.set_ylim(lo-(hi-lo)*0.12, hi+(hi-lo)*0.18)
            q1,q3 = np.percentile(vals,[25,75]); iqr = q3-q1
            n_out = int(((vals<q1-1.5*iqr)|(vals>q3+1.5*iqr)).sum())
            ax.text(1.35, np.median(vals),
                    f'Med: {_fmt(np.median(vals))}\nIQR: {_fmt(iqr)}\nOutliers: {n_out}',
                    transform=ax.get_yaxis_transform(), va='center', fontsize=8, color=MUTED,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=BG3, alpha=0.8, edgecolor=BORDER))
            tc = '#FF8C42' if n_out > 0 else TEXT
            ax.set_title(col, fontsize=10, color=tc, pad=5)
            ax.set_xticks([])
            # FIX #11: Show actual outlier values in insight
            if n_out > 0:
                out_vals = vals[(vals < q1-1.5*iqr) | (vals > q3+1.5*iqr)]
                out_sample = [_fmt(v) for v in sorted(out_vals)[:3]]
                vi.append(f"<b>{col}</b>: {n_out} outliers — e.g. {', '.join(out_sample)}. "
                           f"These were capped to IQR bounds during cleaning.")
            else:
                vi.append(f"<b>{col}</b>: No outliers. Clean distribution ✅")

        _hide_extra(axs, n)
        fig.suptitle('Box Plots — Distribution & Outlier Detection',
                     fontsize=14, color=TEXT, y=1.01, fontweight='bold')
        plt.tight_layout()
        charts.append({"title":"Box Plots & Outliers","image":_to_b64(fig),
                       "insight":'<br>'.join(vi[:4])})

    # ── 5. SCATTER — FIX #9 Correlation ≠ Causation ────────
    if task_type == 'regression' and tgt_is_num and feat_num:
        corrs = {}
        for col in feat_num:
            try:
                mask = df[col].notna() & df[target_col].notna()
                r, _ = scipy_stats.pearsonr(df.loc[mask,col], df.loc[mask,target_col])
                corrs[col] = abs(r)
            except: pass
        top4 = sorted(corrs, key=corrs.get, reverse=True)[:4]
        if top4:
            n = len(top4)
            fig, axes = plt.subplots(1, n, figsize=(6.5*n, 5.5), facecolor=BG)
            axs = [axes] if n==1 else list(axes)
            [_style_ax(a) for a in axs]
            sc_ins = []
            for i, col in enumerate(top4):
                ax = axs[i]; color = PALETTE[i%len(PALETTE)]
                mask = df[col].notna() & df[target_col].notna()
                x, y = df.loc[mask,col], df.loc[mask,target_col]
                xlo,xhi = np.percentile(x,[0.5,99.5]); ylo,yhi = np.percentile(y,[0.5,99.5])
                dm = (x>=xlo)&(x<=xhi)&(y>=ylo)&(y<=yhi)
                ax.scatter(x[dm], y[dm], alpha=0.3, color=color, s=14, edgecolors='none')
                try:
                    z = np.polyfit(x[dm], y[dm], 1)
                    xr = np.linspace(x[dm].min(), x[dm].max(), 200)
                    ax.plot(xr, np.poly1d(z)(xr), color='white', linewidth=2,
                            alpha=0.85, linestyle='--')
                except: pass
                r = corrs.get(col, 0)
                sig = 'Strong ✅' if r > 0.6 else 'Moderate' if r > 0.3 else 'Weak'
                # FIX #9: Clear correlation ≠ causation
                ax.set_title(f'{col}  →  {target_col}\nr = {r:.2f}  ({sig} correlation)',
                             fontsize=10, color=TEXT, pad=6)
                ax.set_xlabel(col, fontsize=9); ax.set_ylabel(target_col, fontsize=9)
                note = "correlated with" if r > 0.5 else "weakly related to"
                sc_ins.append(
                    f"<b>{col}</b> is {note} <b>{target_col}</b> (r={r:.2f}). "
                    f"Note: correlation ≠ causation — {col} may not directly cause changes in {target_col}."
                )
            fig.suptitle('Feature vs Target — Correlation Analysis\n'
                         '(Higher r = stronger relationship, but not necessarily causal)',
                         fontsize=13, color=TEXT, y=1.01, fontweight='bold')
            plt.tight_layout()
            charts.append({"title":"Feature vs Target (Scatter)","image":_to_b64(fig),
                           "insight":'<br>'.join(sc_ins)})

    # ── 6. FEATURE PER CLASS — clipped + FIX #10 KDE min 5 ─
    if task_type == 'classification' and feat_num:
        top4 = feat_num[:4]; n = len(top4)
        classes = df[target_col].dropna().unique()
        colors  = [PALETTE[j%len(PALETTE)] for j in range(len(classes))]
        fig, axes = plt.subplots(1, n, figsize=(6.5*n, 5.5), facecolor=BG)
        axs = [axes] if n==1 else list(axes)
        [_style_ax(a) for a in axs]
        cs_ins = []
        for i, col in enumerate(top4):
            ax = axs[i]
            data_per = [df[df[target_col]==cls][col].dropna().values for cls in classes]
            labels   = [str(c) for c in classes]
            all_vals = np.concatenate([d for d in data_per if len(d) > 0])
            if len(all_vals) < 2: continue
            skew_val = pd.Series(all_vals).skew()
            clip_lo  = 2 if abs(skew_val) > 2 else 1
            clip_hi  = 95 if abs(skew_val) > 2 else 99
            lo, hi   = np.percentile(all_vals, [clip_lo, clip_hi])
            clipped  = [np.clip(d, lo, hi) for d in data_per]
            bp = ax.boxplot(clipped, patch_artist=True, labels=labels,
                            medianprops=dict(color='white', linewidth=2.5),
                            whiskerprops=dict(color=MUTED, linewidth=1.2),
                            capprops=dict(color=MUTED, linewidth=1.5),
                            flierprops=dict(marker='.', markersize=3, alpha=0.3))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color); patch.set_alpha(0.65)
            for j, (d, color) in enumerate(zip(clipped, colors)):
                if len(d) == 0: continue
                jitter = np.random.normal(0, 0.07, len(d))
                ax.scatter(np.full(len(d),j+1)+jitter, d,
                           alpha=0.2, color=color, s=8, edgecolors='none')
            if len(labels) > 5:
                ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
            ax.set_title(col, fontsize=10, color=TEXT, pad=6)
            ax.set_ylabel(col, fontsize=9); ax.set_xlabel(target_col, fontsize=9)
            means = [np.mean(d) for d in data_per if len(d) > 0]
            if len(means) > 1:
                spread = max(means) - min(means)
                if spread > df[col].std() * 0.5:
                    cs_ins.append(f"✅ <b>{col}</b> differs clearly across classes — strong predictor!")
                else:
                    cs_ins.append(f"ℹ️ <b>{col}</b> looks similar across classes — may be weak predictor.")
        fig.suptitle('Feature Distribution by Class',
                     fontsize=14, color=TEXT, y=1.01, fontweight='bold')
        plt.tight_layout()
        charts.append({"title":"Feature by Class","image":_to_b64(fig),
                       "insight":'<br>'.join(cs_ins)})

    # ── 7. CLASS KDE — FIX #10: min 5 samples ──────────────
    if task_type == 'classification' and feat_num:
        top4 = feat_num[:4]; n = len(top4)
        classes = df[target_col].dropna().unique()
        fig, axes = plt.subplots(1, n, figsize=(6.5*n, 5), facecolor=BG)
        axs = [axes] if n==1 else list(axes)
        [_style_ax(a) for a in axs]
        for i, col in enumerate(top4):
            ax = axs[i]
            for j, cls in enumerate(classes):
                sub = df[df[target_col]==cls][col].dropna()
                # FIX #10: Minimum 5 samples for KDE — skip if less
                if len(sub) < 5:
                    ax.text(0.5, 0.5 - j*0.1,
                            f"Class '{cls}': only {len(sub)} samples — KDE skipped",
                            ha='center', va='center', color=MUTED,
                            fontsize=7, transform=ax.transAxes)
                    continue
                color = PALETTE[j%len(PALETTE)]
                try:
                    lo2, hi2 = sub.quantile(0.01), sub.quantile(0.99)
                    sub_clip = sub.clip(lo2, hi2)
                    kde = scipy_stats.gaussian_kde(sub_clip)
                    xr  = np.linspace(lo2, hi2, 200)
                    ax.fill_between(xr, kde(xr), alpha=0.2, color=color)
                    ax.plot(xr, kde(xr), color=color, linewidth=2, alpha=0.9,
                            label=f'{cls} (n={len(sub)})')
                except: pass
            ax.set_title(col, fontsize=10, color=TEXT, pad=5)
            ax.set_xlabel(col, fontsize=9); ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=7.5, title=target_col, title_fontsize=8,
                      facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)
        fig.suptitle('Density by Class — Less Overlap = Easier to Predict',
                     fontsize=14, color=TEXT, y=1.01, fontweight='bold')
        plt.tight_layout()
        charts.append({"title":"Class-wise Density","image":_to_b64(fig),
                       "insight":"Less overlap between class curves = model can predict more accurately. More overlap = harder to distinguish."})

    # ── 8. CATEGORICAL COUNTS ──────────────────────────────
    if feat_cat:
        n = min(4, len(feat_cat)); ncols = min(2,n); nrows = (n+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(10*ncols, 5.5*nrows), facecolor=BG)
        axs = np.array(axes).flatten() if n > 1 else [axes]
        [_style_ax(a) for a in axs]
        for i, col in enumerate(feat_cat[:n]):
            ax = axs[i]; vc = df[col].value_counts().head(15)
            colors = [PALETTE[j%len(PALETTE)] for j in range(len(vc))]
            ax.barh(vc.index.astype(str)[::-1], vc.values[::-1],
                    color=colors[::-1], alpha=0.85, edgecolor='none', height=0.65)
            for j, (idx,val) in enumerate(zip(vc.index[::-1], vc.values[::-1])):
                ax.text(val+max(vc.values)*0.01, j,
                        f'{val:,} ({val/len(df)*100:.1f}%)',
                        va='center', fontsize=8.5, color=TEXT)
            ax.set_title(col, fontsize=11, color=TEXT, pad=6)
            ax.set_xlabel('Count', fontsize=9)
            ax.set_xlim(0, max(vc.values)*1.3)
        _hide_extra(axs, n)
        fig.suptitle('Category Counts — Most Frequent Values',
                     fontsize=14, color=TEXT, y=1.01, fontweight='bold')
        plt.tight_layout()
        charts.append({"title":"Category Breakdown","image":_to_b64(fig),
                       "insight":f"Most common: '<b>{df[feat_cat[0]].value_counts().index[0]}</b>'. "
                                  "If one category dominates, it may bias the model."})

    # ── 9. MISSING VALUES ──────────────────────────────────
    missing = df.isnull().sum(); missing = missing[missing > 0]
    if not missing.empty:
        pct = (missing/len(df)*100).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(4, len(pct)*0.6+2)), facecolor=BG)
        _style_ax(ax)
        colors = ['#FF4D6D' if v>30 else '#FF8C42' if v>10 else '#FFD166' for v in pct.values]
        ax.barh(pct.index, pct.values, color=colors, alpha=0.85, edgecolor='none', height=0.6)
        for i,(name,val) in enumerate(pct.items()):
            ax.text(val+0.5, i, f'{val:.1f}%', va='center', fontsize=9, color=TEXT, fontweight='600')
        patches = [mpatches.Patch(color='#FF4D6D', label='>30% Critical'),
                   mpatches.Patch(color='#FF8C42', label='>10% Warning'),
                   mpatches.Patch(color='#FFD166', label='<10% OK')]
        ax.legend(handles=patches, fontsize=8.5, loc='lower right',
                  facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)
        ax.set_xlabel('Missing %', fontsize=10)
        ax.set_title('Missing Values — Which Columns Have Gaps?',
                     fontsize=13, color=TEXT, pad=10, fontweight='bold')
        ax.set_xlim(0, max(pct.values)*1.25)
        plt.tight_layout()
        charts.append({"title":"Missing Data","image":_to_b64(fig),
                       "insight":_ins_missing(df)})

    # ── 10. TREND CHART — FIX #20 date crash, #2 thread safe ─
    if feat_num and len(df) <= 8000:
        date_col, date_series = _detect_date_col(df)
        cols_to_plot = feat_num[:3]
        n = len(cols_to_plot)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4*n), facecolor=BG, sharex=False)
        axs = [axes] if n==1 else list(axes)
        [_style_ax(a) for a in axs]

        # FIX #20: Safe date parsing with fallback
        use_dates = False
        x_dates   = None
        if date_series is not None:
            try:
                x_dates   = pd.to_datetime(date_series, errors='coerce').reset_index(drop=True)
                valid_pct = x_dates.notna().sum() / len(x_dates)
                use_dates = valid_pct > 0.7
            except: use_dates = False

        x_label = f'Date ({date_col})' if use_dates else 'Row Index'

        scale_notes = []
        for i, col in enumerate(cols_to_plot):
            ax = axs[i]; color = PALETTE[i%len(PALETTE)]
            y_raw = df[col].values
            s     = pd.Series(y_raw)

            # Auto log scale per subplot (no global state)
            log_applied = False
            clean = s.dropna()
            if len(clean) >= 5 and clean.min() > 0 and clean.max()/(clean.min()+1e-9) > 100:
                ax.set_yscale('log')
                log_applied = True
                scale_notes.append(f'{col}: log scale')

            roll = s.rolling(max(1, len(s)//60), min_periods=1).mean()

            if use_dates and x_dates is not None:
                x_vals = x_dates.values
            else:
                x_vals = np.arange(len(df))

            ax.plot(x_vals, s.values, color=color, linewidth=0.6, alpha=0.2)
            ax.plot(x_vals, roll.values, color=color, linewidth=2.2, alpha=0.95, label=col)
            ax.fill_between(x_vals, roll.values, alpha=0.07, color=color)

            # Outlier markers
            q1,q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3-q1
            out_mask = (s < q1-3*iqr) | (s > q3+3*iqr)
            for idx in out_mask[out_mask].index.tolist()[:5]:
                if idx < len(x_vals):
                    ax.annotate('⚠', xy=(x_vals[idx], s.iloc[idx]),
                                fontsize=10, ha='center', color='#FF4D6D', alpha=0.9,
                                xytext=(0,12), textcoords='offset points')

            ax.set_ylabel(col, fontsize=9, color=color)
            ax.legend(fontsize=9, loc='upper right',
                      facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)

        if use_dates:
            try: fig.autofmt_xdate()
            except: pass

        axs[-1].set_xlabel(x_label, fontsize=10, color=MUTED)
        title = 'Trend Chart'
        if use_dates and date_col: title += f' — Date: {date_col}'
        if scale_notes: title += '  |  ' + ', '.join(scale_notes)
        fig.suptitle(title, fontsize=14, color=TEXT, y=1.01, fontweight='bold')
        plt.tight_layout()
        note = f"Date column '{date_col}' detected." if use_dates else "No date column found — using row index."
        charts.append({"title":"Trend Chart","image":_to_b64(fig),
                       "insight":f"📈 {note} ⚠ markers = unusual values worth investigating."})

    return charts


# ════════════════════════════════════════
# FIX #15: KPI — proper fallback, no wrong cols
# FIX #17: Use _fmt consistently
# ════════════════════════════════════════
def compute_kpis(df: pd.DataFrame) -> dict:
    kpis = {}
    revenue_kws = ['revenue','sales','income','amount','profit','turnover','gmv','earning','gross']
    target_kws  = ['target','goal','quota','plan','budget','forecast']
    actual_kws  = ['actual','achieved','real','result','attained']
    num_cols    = _smart_num(df)

    # FIX #15: Only use revenue cols if keyword matched — don't blindly take first 3
    rev_cols = [c for c in num_cols if any(k in c.lower() for k in revenue_kws)]
    has_revenue = len(rev_cols) > 0

    for col in rev_cols[:3]:
        data = df[col].dropna()
        kpis[col] = {
            "total":   round(float(data.sum()), 2),
            "average": round(float(data.mean()), 2),
            "max":     round(float(data.max()), 2),
            "min":     round(float(data.min()), 2),
            "type":    "revenue"
        }

    # Target vs Actual
    t_cols = [c for c in num_cols if any(k in c.lower() for k in target_kws)]
    a_cols = [c for c in num_cols if any(k in c.lower() for k in actual_kws)]
    if t_cols and a_cols:
        tc, ac = t_cols[0], a_cols[0]
        ach = round(float(df[ac].sum()/(df[tc].sum()+1e-9)*100), 2)
        kpis["target_vs_actual"] = {
            "target_col": tc, "actual_col": ac,
            "target_total": round(float(df[tc].sum()), 2),
            "actual_total": round(float(df[ac].sum()), 2),
            "achievement_pct": ach
        }

    # Growth
    date_col, date_series = _detect_date_col(df)
    if date_col and rev_cols:
        try:
            tmp = df.copy()
            tmp['_dt'] = pd.to_datetime(date_series, errors='coerce')
            tmp = tmp.dropna(subset=['_dt']).sort_values('_dt')
            if len(tmp) >= 4:
                mid = len(tmp)//2
                f  = tmp.iloc[:mid][rev_cols[0]].mean()
                s2 = tmp.iloc[mid:][rev_cols[0]].mean()
                kpis["growth"] = {
                    "column": rev_cols[0],
                    "first_half_avg":  round(float(f), 2),
                    "second_half_avg": round(float(s2), 2),
                    "growth_pct": round((s2-f)/(f+1e-9)*100, 2)
                }
        except: pass

    kpis["summary"] = {
        "total_rows":    len(df),
        "total_cols":    len(df.columns),
        "numeric_cols":  len(num_cols),
        "missing_cells": int(df.isnull().sum().sum()),
        "has_revenue":   has_revenue
    }
    return kpis


# ════════════════════════════════════════
# COMPARISON VISUALIZATIONS
# ════════════════════════════════════════
def generate_comparison_viz(df_old, df_new, label_old, label_new):
    charts = []; stats = {}
    common     = [c for c in df_old.columns if c in df_new.columns]
    all_df     = pd.concat([df_old[common], df_new[common]])
    num_common = _smart_num(all_df)
    num_common = [c for c in num_common if c in common]

    stats["shape"] = {
        label_old: {"rows": int(df_old.shape[0]), "cols": int(df_old.shape[1])},
        label_new: {"rows": int(df_new.shape[0]), "cols": int(df_new.shape[1])}
    }
    stats["drift"] = {}

    for col in num_common:
        try:
            stat, pval = scipy_stats.ks_2samp(df_old[col].dropna(), df_new[col].dropna())
            drift = "High Drift 🚨" if pval<0.01 else "Moderate Drift ⚠️" if pval<0.05 else "Stable ✅"
            stats["drift"][col] = {
                "ks_stat": round(float(stat), 4),
                "p_value": round(float(pval), 4),
                "status":  drift
            }
        except: pass

    # Mean comparison
    if num_common:
        cols_show = num_common[:5]; n = len(cols_show)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5.5), facecolor=BG)
        axs = [axes] if n==1 else list(axes)
        [_style_ax(a) for a in axs]
        for i, col in enumerate(cols_show):
            ax = axs[i]
            means = [df_old[col].mean(), df_new[col].mean()]
            bars  = ax.bar([label_old, label_new], means,
                           color=[PALETTE[0], PALETTE[1]], width=0.55, edgecolor='none', alpha=0.85)
            for bar,v in zip(bars,means):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+abs(max(means))*0.02,
                        _fmt(v), ha='center', va='bottom', fontsize=9,
                        color=TEXT, fontweight='700')
            pct = ((means[1]-means[0])/(means[0]+1e-10))*100
            arrow = '↑' if pct>0 else '↓'
            chg_col = '#3DDB8A' if pct>0 else '#FF4D6D'
            drift_s = stats["drift"].get(col, {}).get("status", "")
            ax.set_title(f'{col}\n{arrow}{abs(pct):.1f}%  {drift_s}',
                         fontsize=10, color=chg_col, pad=6)
            ax.set_ylabel('Average', fontsize=9)
        fig.suptitle(f'Average Comparison: {label_old}  vs  {label_new}',
                     fontsize=14, color=TEXT, y=1.01, fontweight='bold')
        plt.tight_layout()
        charts.append({"title": "Average Comparison", "image": _to_b64(fig)})

    # KDE overlay
    if num_common:
        n = min(4, len(num_common))
        fig, axes = plt.subplots(1, n, figsize=(6.5*n, 5), facecolor=BG)
        axs = [axes] if n==1 else list(axes)
        [_style_ax(a) for a in axs]
        for i, col in enumerate(num_common[:n]):
            ax = axs[i]
            for lbl, df_x, color in [(label_old,df_old,PALETTE[0]),(label_new,df_new,PALETTE[1])]:
                data = df_x[col].dropna()
                lo2, hi2 = data.quantile(0.01), data.quantile(0.99)
                data_clip = data.clip(lo2, hi2)
                ax.hist(data_clip, bins=25, alpha=0.3, color=color, density=True, edgecolor='none')
                # FIX #10: min 5 samples for KDE
                if len(data_clip) >= 5:
                    try:
                        kde = scipy_stats.gaussian_kde(data_clip)
                        xr  = np.linspace(lo2, hi2, 200)
                        ax.fill_between(xr, kde(xr), alpha=0.12, color=color)
                        ax.plot(xr, kde(xr), color=color, linewidth=2.2, alpha=0.9, label=lbl)
                    except: pass
            drift_s = stats["drift"].get(col, {}).get("status", "")
            ax.set_title(f'{col}   {drift_s}', fontsize=10, color=TEXT, pad=5)
            ax.legend(fontsize=8, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)
        fig.suptitle('Distribution Comparison', fontsize=14, color=TEXT, y=1.01, fontweight='bold')
        plt.tight_layout()
        charts.append({"title": "Distribution Comparison", "image": _to_b64(fig)})

    # Missing comparison
    show = [c for c in common
            if df_old[c].isnull().sum()>0 or df_new[c].isnull().sum()>0]
    if show:
        miss_old = (df_old[common].isnull().sum()/len(df_old)*100)
        miss_new = (df_new[common].isnull().sum()/len(df_new)*100)
        fig, ax  = plt.subplots(figsize=(max(8, len(show)*0.9), 5), facecolor=BG)
        _style_ax(ax)
        x = np.arange(len(show)); w = 0.38
        ax.bar(x-w/2, miss_old[show].values, w, label=label_old,
               color=PALETTE[0], alpha=0.85, edgecolor='none')
        ax.bar(x+w/2, miss_new[show].values, w, label=label_new,
               color=PALETTE[1], alpha=0.85, edgecolor='none')
        ax.set_xticks(x); ax.set_xticklabels(show, rotation=40, ha='right', fontsize=8.5)
        ax.set_ylabel('Missing %', fontsize=10)
        ax.set_title('Missing Values Comparison', fontsize=13, color=TEXT,
                     fontweight='bold', pad=10)
        ax.legend(fontsize=9, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)
        plt.tight_layout()
        charts.append({"title": "Missing Values Comparison", "image": _to_b64(fig)})

    for col in num_common[:6]:
        stats[col] = {
            label_old: {
                "mean":   round(float(df_old[col].mean()), 3),
                "std":    round(float(df_old[col].std()), 3),
                "median": round(float(df_old[col].median()), 3),
                "min":    round(float(df_old[col].min()), 3),
                "max":    round(float(df_old[col].max()), 3),
            },
            label_new: {
                "mean":   round(float(df_new[col].mean()), 3),
                "std":    round(float(df_new[col].std()), 3),
                "median": round(float(df_new[col].median()), 3),
                "min":    round(float(df_new[col].min()), 3),
                "max":    round(float(df_new[col].max()), 3),
            },
        }
    return charts, stats