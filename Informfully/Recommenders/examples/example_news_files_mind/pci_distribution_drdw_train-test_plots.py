import json, os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

_dir     = os.path.dirname(os.path.abspath(__file__))
run_dir  = os.path.join(_dir, "plots_drdw", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(run_dir, exist_ok=True)
plot_dir = run_dir

# ── Tee stdout to a log file inside the run folder ────────────────────────────
class _Tee:
    def __init__(self, stream, path):
        self._stream = stream
        self._log    = open(path, "w", encoding="utf-8", buffering=1)
    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
    def flush(self):
        self._stream.flush()
        self._log.flush()
    def close(self):
        self._log.close()

_log_path  = os.path.join(run_dir, "analysis.log")
_tee       = _Tee(sys.stdout, _log_path)
sys.stdout = _tee

print(f"Run started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output dir  : {run_dir}")
print(f"Log file    : {_log_path}")
print()

# ── Load category map ──────────────────────────────────────────────────────────
with open(os.path.join(_dir, "example_category.json")) as f:
    cat_map = json.load(f)  # article_id -> category

# ── Helper: compute all stats for one split ────────────────────────────────────
def compute_stats(csv_path):
    df     = pd.read_csv(csv_path)
    df_pos = df[df["Rating"] == 1]

    user_pcis   = []
    user_clicks = []

    for _, grp in df_pos.groupby("UserID"):
        click_count = len(grp)
        user_clicks.append(click_count)
        cats = [cat_map[iid] for iid in grp["ItemID"] if iid in cat_map]
        if len(cats) < 3:
            continue
        counts = Counter(cats)
        total  = sum(counts.values())
        pci    = sum((c / total) ** 2 for c in counts.values())
        user_pcis.append(pci)

    pci_s    = pd.Series(user_pcis)
    clicks_s = pd.Series(user_clicks)

    article_cats = pd.Series(
        [cat_map[iid] for iid in df_pos["ItemID"] if iid in cat_map]
    )
    cat_counts = article_cats.value_counts()

    # PCI vs clicks rows (for scatter)
    pci_click_rows = []
    for _, grp in df_pos.groupby("UserID"):
        cats = [cat_map[iid] for iid in grp["ItemID"] if iid in cat_map]
        if len(cats) < 3:
            continue
        counts = Counter(cats)
        total  = sum(counts.values())
        pci    = sum((c / total) ** 2 for c in counts.values())
        pci_click_rows.append({"clicks": len(grp), "pci": pci})
    pc_df = pd.DataFrame(pci_click_rows)

    # Average PCI by dominant category
    dom_cat_pci = {}
    for _, grp in df_pos.groupby("UserID"):
        cats = [cat_map[iid] for iid in grp["ItemID"] if iid in cat_map]
        if len(cats) < 3:
            continue
        counts  = Counter(cats)
        total   = sum(counts.values())
        pci     = sum((c / total) ** 2 for c in counts.values())
        dom_cat = counts.most_common(1)[0][0]
        dom_cat_pci.setdefault(dom_cat, []).append(pci)
    dom_df = pd.Series(
        {cat: np.mean(vals) for cat, vals in dom_cat_pci.items()}
    ).sort_values(ascending=False)

    return dict(
        df=df, df_pos=df_pos,
        pci_s=pci_s, clicks_s=clicks_s,
        article_cats=article_cats, cat_counts=cat_counts,
        pc_df=pc_df, dom_df=dom_df,
    )

# ── Load both splits ───────────────────────────────────────────────────────────
train = compute_stats(os.path.join(_dir, "example_training_graph_uir_top3.csv"))
test  = compute_stats(os.path.join(_dir, "example_impression_test_uir.csv"))

# ==============================================================================
# Console output
# ==============================================================================
SEP = "=" * 60

def print_overview(label, s):
    df, df_pos     = s["df"], s["df_pos"]
    clicks_s       = s["clicks_s"]
    pci_s          = s["pci_s"]
    article_cats   = s["article_cats"]
    cat_counts     = s["cat_counts"]

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    print("\n[1] DATASET OVERVIEW")
    print(f"  Total interactions        : {len(df):,}")
    print(f"  Positive (clicked)        : {len(df_pos):,}  ({100*len(df_pos)/len(df):.1f}%)")
    print(f"  Unique users              : {df['UserID'].nunique():,}")
    print(f"  Unique articles           : {df['ItemID'].nunique():,}")
    print(f"  Unique categories         : {article_cats.nunique()}")

    print("\n[2] CLICKS PER USER")
    print(clicks_s.describe().rename({
        "count": "users", "mean": "mean_clicks", "std": "std",
        "min": "min", "25%": "p25", "50%": "median", "75%": "p75", "max": "max"
    }).to_string())
    print(f"  Average clicks per user   : {clicks_s.mean():.2f}")
    print(f"  Median  clicks per user   : {clicks_s.median():.0f}")
    print(f"  Users with ≥ 10 clicks    : {(clicks_s >= 10).sum():,}  ({100*(clicks_s >= 10).mean():.1f}%)")
    print(f"  Users with ≥ 20 clicks    : {(clicks_s >= 20).sum():,}  ({100*(clicks_s >= 20).mean():.1f}%)")

    print("\n[3] CATEGORY DISTRIBUTION (by click count)")
    total_clicks = cat_counts.sum()
    for cat, cnt in cat_counts.items():
        bar = "█" * int(40 * cnt / cat_counts.max())
        print(f"  {cat:<20s} {cnt:>7,}  ({100*cnt/total_clicks:5.1f}%)  {bar}")

    print(f"\n[4] PCI DISTRIBUTION  (users with ≥ 3 clicks, n={len(pci_s):,})")
    print(pci_s.describe().to_string())
    print(f"\n  % users PCI > 0.25 : {(pci_s > 0.25).mean()*100:.1f}%")
    print(f"  % users PCI > 0.35 : {(pci_s > 0.35).mean()*100:.1f}%")
    print(f"  % users PCI > 0.50 : {(pci_s > 0.50).mean()*100:.1f}%")
    print(f"  % users PCI = 1.00 : {(pci_s == 1.00).mean()*100:.1f}%  (single-category readers)")

print(SEP)
print("  D-RDW MIND DATASET — EXPLORATORY ANALYSIS (TRAIN vs TEST)")
print(SEP)
print_overview("TRAINING SET", train)
print_overview("TEST SET", test)
print(f"\nPlots saved to: {plot_dir}\n")

# ==============================================================================
# Plots — train (left) vs test (right)
# ==============================================================================
TITLE_PAD  = 14
LABEL_SIZE = 11
PALETTE    = plt.cm.tab20.colors

TRAIN_COLOR = "#4C72B0"
TEST_COLOR  = "#DD8452"

# ── Align categories across both splits ───────────────────────────────────────
all_cats = train["cat_counts"].index.union(test["cat_counts"].index)
tr_cat   = train["cat_counts"].reindex(all_cats, fill_value=0)
te_cat   = test["cat_counts"].reindex(all_cats, fill_value=0)
cat_colors = [PALETTE[i % len(PALETTE)] for i in range(len(all_cats))]

# ── Plot 1: Category distribution bar chart ───────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
for ax, cat_s, label in [(ax1, tr_cat, "Training Set"), (ax2, te_cat, "Test Set")]:
    bars = ax.bar(cat_s.index, cat_s.values, color=cat_colors, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Category", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Clicks", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + cat_s.max() * 0.01,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=6)
    ax.annotate(label, xy=(0.5, 1.01), xycoords="axes fraction",
                ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "1_category_distribution.png"), dpi=150)
plt.close()

# ── Plot 2: Category share pie chart ──────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
for ax, cat_s, label in [(ax1, tr_cat, "Training Set"), (ax2, te_cat, "Test Set")]:
    _, texts, autotexts = ax.pie(
        cat_s.values,
        labels=cat_s.index,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        colors=cat_colors,
        startangle=140,
        pctdistance=0.82,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.annotate(label, xy=(0.5, 1.01), xycoords="axes fraction",
                ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "2_category_share_pie.png"), dpi=150)
plt.close()

# ── Plot 3: PCI histogram ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
for ax, s, color, label in [
    (ax1, train, TRAIN_COLOR, "Training Set"),
    (ax2, test,  TEST_COLOR,  "Test Set"),
]:
    pci_s = s["pci_s"]
    ax.hist(pci_s, bins=40, color=color, edgecolor="white", linewidth=0.5)
    ax.axvline(pci_s.mean(),   color="#DD4444", linestyle="--", linewidth=1.5,
               label=f"Mean = {pci_s.mean():.3f}")
    ax.axvline(pci_s.median(), color="#44AA44", linestyle="--", linewidth=1.5,
               label=f"Median = {pci_s.median():.3f}")
    ax.axvline(0.10, color="orange", linestyle=":", linewidth=1.5, label="PCI threshold = 0.10")
    ax.set_xlabel("PCI (Perspective Concentration Index)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Users", fontsize=LABEL_SIZE)
    ax.legend(fontsize=9)
    ax.annotate(label, xy=(0.5, 1.01), xycoords="axes fraction",
                ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "3_pci_histogram.png"), dpi=150)
plt.close()

# ── Plot 4: PCI CDF ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for ax, s, color, label in [
    (ax1, train, TRAIN_COLOR, "Training Set"),
    (ax2, test,  TEST_COLOR,  "Test Set"),
]:
    pci_s      = s["pci_s"]
    sorted_pci = np.sort(pci_s)
    cdf        = np.arange(1, len(sorted_pci) + 1) / len(sorted_pci)
    ax.plot(sorted_pci, cdf, color=color, linewidth=2)
    for thresh, col, lbl in [(0.25, "#DD4444", "0.25"), (0.35, "#FF9900", "0.35"), (0.50, "#44AA44", "0.50")]:
        pct = (pci_s > thresh).mean() * 100
        ax.axvline(thresh, color=col, linestyle="--", linewidth=1.2,
                   label=f"PCI > {lbl}: {pct:.1f}% of users")
    ax.axvline(0.10, color="grey", linestyle=":", linewidth=1.2, label="Threshold = 0.10")
    ax.set_xlabel("PCI", fontsize=LABEL_SIZE)
    ax.set_ylabel("Cumulative Fraction of Users", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    ax.annotate(label, xy=(0.5, 1.01), xycoords="axes fraction",
                ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "4_pci_cdf.png"), dpi=150)
plt.close()

# ── Plot 5: Clicks per user distribution ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
for ax, s, color, label in [
    (ax1, train, "#55A868", "Training Set"),
    (ax2, test,  "#C44E52", "Test Set"),
]:
    clicks_s = s["clicks_s"]
    cap      = int(clicks_s.quantile(0.99))
    ax.hist(clicks_s.clip(upper=cap), bins=50, color=color, edgecolor="white", linewidth=0.5)
    ax.axvline(clicks_s.mean(),   color="#DD4444", linestyle="--", linewidth=1.5,
               label=f"Mean = {clicks_s.mean():.1f}")
    ax.axvline(clicks_s.median(), color="#4C72B0", linestyle="--", linewidth=1.5,
               label=f"Median = {clicks_s.median():.0f}")
    ax.set_xlabel("Number of Clicks (capped at 99th pct)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Users", fontsize=LABEL_SIZE)
    ax.legend(fontsize=9)
    ax.annotate(label, xy=(0.5, 1.01), xycoords="axes fraction",
                ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "5_clicks_per_user.png"), dpi=150)
plt.close()

# ── Plot 6: PCI vs clicks scatter ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for ax, s, color, label in [
    (ax1, train, TRAIN_COLOR, "Training Set"),
    (ax2, test,  TEST_COLOR,  "Test Set"),
]:
    pc_df = s["pc_df"]
    cap   = int(pc_df["clicks"].quantile(0.99))
    ax.scatter(pc_df["clicks"].clip(upper=cap), pc_df["pci"],
               alpha=0.15, s=8, color=color)
    ax.axhline(0.10, color="orange", linestyle="--", linewidth=1.2, label="PCI threshold = 0.10")
    ax.set_xlabel("Number of Clicks (capped at 99th pct)", fontsize=LABEL_SIZE)
    ax.set_ylabel("PCI", fontsize=LABEL_SIZE)
    ax.legend(fontsize=9)
    ax.annotate(label, xy=(0.5, 1.01), xycoords="axes fraction",
                ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "6_pci_vs_clicks.png"), dpi=150)
plt.close()

# ── Plot 7: Average PCI per dominant category ─────────────────────────────────
all_dom_cats = train["dom_df"].index.union(test["dom_df"].index)
tr_dom = train["dom_df"].reindex(all_dom_cats, fill_value=0).sort_values(ascending=False)
te_dom = test["dom_df"].reindex(all_dom_cats, fill_value=0)
te_dom = te_dom.reindex(tr_dom.index)   # same order as train

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
dom_colors = [PALETTE[i % len(PALETTE)] for i in range(len(tr_dom))]
for ax, dom_s, pci_s, color, label in [
    (ax1, tr_dom, train["pci_s"], TRAIN_COLOR, "Training Set"),
    (ax2, te_dom, test["pci_s"],  TEST_COLOR,  "Test Set"),
]:
    ax.bar(dom_s.index, dom_s.values, color=dom_colors, edgecolor="white", linewidth=0.6)
    ax.axhline(pci_s.mean(), color="#DD4444", linestyle="--", linewidth=1.2,
               label=f"Overall mean PCI = {pci_s.mean():.3f}")
    ax.set_xlabel("Dominant Category", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mean PCI", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.legend(fontsize=9)
    ax.annotate(label, xy=(0.5, 1.01), xycoords="axes fraction",
                ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "7_avg_pci_by_dominant_category.png"), dpi=150)
plt.close()

# ==============================================================================
# Parameter recommendations — derived from TRAINING SET only
# ==============================================================================
pci_s    = train["pci_s"]
clicks_s = train["clicks_s"]

n_categories  = int(train["article_cats"].nunique())
n_users       = int(train["df"]["UserID"].nunique())
median_clicks = float(clicks_s.median())
mean_clicks   = float(clicks_s.mean())

pci_threshold      = float(np.round(max(pci_s.quantile(0.25) * 0.40, 0.08), 2))
min_history_single = max(1, int(np.floor(clicks_s.quantile(0.10))))
min_history_multi  = max(2, min_history_single)
penalty_single     = float(np.round(min(max(4.0, pci_s.median() * 10.0), 8.0), 1))
penalty_multi      = float(np.round(penalty_single / 2.0, 1))
top_k_single       = 10
top_k_recommend    = 10
num_sim_users      = min(1000, max(200, int(n_users * 0.10)))
top_k_click        = max(2, int(round(mean_clicks * 0.20)))
num_sessions       = int(np.clip(np.ceil(median_clicks / top_k_click), 15, 25))

print()
print(SEP)
print("  PARAMETER RECOMMENDATIONS (derived from TRAINING SET only)")
print(SEP)
print(f"""
  Dataset facts used:
    Unique categories        : {n_categories}
    Total users              : {n_users:,}
    Median clicks / user     : {median_clicks:.0f}
    Mean   clicks / user     : {mean_clicks:.1f}
    PCI p25 / median / p75   : {pci_s.quantile(0.25):.3f} / {pci_s.median():.3f} / {pci_s.quantile(0.75):.3f}
    % users PCI > 0.25       : {(pci_s > 0.25).mean()*100:.1f}%
    % single-category readers: {(pci_s == 1.0).mean()*100:.1f}%
""")

print("  ── Single-session ablation study (NRMS+PCI / D_RDW+PCI) ──────────────")
print(f"    TOP_K              = {top_k_single}")
print(f"    PCI_THRESHOLD      = {pci_threshold}")
print(f"      → targets {(pci_s > pci_threshold).mean()*100:.1f}% of non-empty-history users")
print(f"    PCI_MIN_HISTORY    = {min_history_single}")
print(f"      → includes users with as few as {min_history_single} click(s) in history")
print(f"    PCI_PENALTY_WEIGHT = {penalty_single}")
print(f"      → derived from median PCI {pci_s.median():.3f} × 10, capped at 8.0")

print()
print("  ── Multi-session simulation (D_RDW+PCI) ──────────────────────────────")
print(f"    TOP_K_RECOMMEND    = {top_k_recommend}")
print(f"    PCI_THRESHOLD      = {pci_threshold}  (same as single-session)")
print(f"    PCI_MIN_HISTORY    = {min_history_multi}")
print(f"    PCI_PENALTY_WEIGHT = {penalty_multi}")
print(f"      → half of single-session value (LP formulation is more sensitive)")
print(f"    NUM_SIM_USERS      = {num_sim_users}")
print(f"      → 10% of {n_users:,} total users, capped at 1,000")
print(f"    NUM_SESSIONS       = {num_sessions}")
print(f"      → enough sessions to cover median history ({median_clicks:.0f} clicks)")
print(f"         at {top_k_click} clicks/session")
print(f"    TOP_K_CLICK        = {top_k_click}")
print(f"      → ~20% of mean clicks/session ({mean_clicks:.1f}), minimum 2")

print()
print(SEP)

# ==============================================================================
print("\nDone.")
print(f"\nRun finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
sys.stdout = _tee._stream
_tee.close()
