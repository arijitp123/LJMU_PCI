import json, os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

_dir     = os.path.dirname(os.path.abspath(__file__))
run_dir  = os.path.join(_dir, "plots_drdw", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(run_dir, exist_ok=True)
plot_dir = run_dir   # all plots go into the timestamped folder

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

# ── Load data ──────────────────────────────────────────────────────────────────
with open(os.path.join(_dir, "example_category.json")) as f:
    cat_map = json.load(f)  # article_id -> category

df     = pd.read_csv(os.path.join(_dir, "example_impression_test_uir.csv"))
df_pos = df[df["Rating"] == 1]   # only clicked articles

# ── Compute per-user PCI ───────────────────────────────────────────────────────
user_pcis   = []
user_clicks = []

for user, grp in df_pos.groupby("UserID"):
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

# ── Article-level category series ─────────────────────────────────────────────
article_cats = pd.Series(
    [cat_map[iid] for iid in df_pos["ItemID"] if iid in cat_map]
)
cat_counts = article_cats.value_counts()

# ==============================================================================
# Console output
# ==============================================================================
SEP = "=" * 60

print(SEP)
print("  D-RDW MIND DATASET — EXPLORATORY ANALYSIS")
print(SEP)

# ── Dataset overview ──────────────────────────────────────────────────────────
print("\n[1] DATASET OVERVIEW")
print(f"  Total interactions        : {len(df):,}")
print(f"  Positive (clicked)        : {len(df_pos):,}  ({100*len(df_pos)/len(df):.1f}%)")
print(f"  Unique users              : {df['UserID'].nunique():,}")
print(f"  Unique articles           : {df['ItemID'].nunique():,}")
print(f"  Unique categories         : {article_cats.nunique()}")

# ── Clicks per user ───────────────────────────────────────────────────────────
print("\n[2] CLICKS PER USER")
print(clicks_s.describe().rename({
    "count": "users", "mean": "mean_clicks", "std": "std",
    "min": "min", "25%": "p25", "50%": "median", "75%": "p75", "max": "max"
}).to_string())
print(f"  Average clicks per user   : {clicks_s.mean():.2f}")
print(f"  Median  clicks per user   : {clicks_s.median():.0f}")
print(f"  Users with ≥ 10 clicks    : {(clicks_s >= 10).sum():,}  ({100*(clicks_s >= 10).mean():.1f}%)")
print(f"  Users with ≥ 20 clicks    : {(clicks_s >= 20).sum():,}  ({100*(clicks_s >= 20).mean():.1f}%)")

# ── Category distribution ─────────────────────────────────────────────────────
print("\n[3] CATEGORY DISTRIBUTION (by click count)")
total_clicks = cat_counts.sum()
for cat, cnt in cat_counts.items():
    bar = "█" * int(40 * cnt / cat_counts.max())
    print(f"  {cat:<20s} {cnt:>7,}  ({100*cnt/total_clicks:5.1f}%)  {bar}")

# ── PCI distribution ──────────────────────────────────────────────────────────
print(f"\n[4] PCI DISTRIBUTION  (users with ≥ 3 clicks, n={len(pci_s):,})")
print(pci_s.describe().to_string())
print(f"\n  % users PCI > 0.25 : {(pci_s > 0.25).mean()*100:.1f}%")
print(f"  % users PCI > 0.35 : {(pci_s > 0.35).mean()*100:.1f}%")
print(f"  % users PCI > 0.50 : {(pci_s > 0.50).mean()*100:.1f}%")
print(f"  % users PCI = 1.00 : {(pci_s == 1.00).mean()*100:.1f}%  (single-category readers)")

print(f"\nPlots saved to: {plot_dir}\n")

# ==============================================================================
# Plots
# ==============================================================================
TITLE_PAD  = 14
LABEL_SIZE = 11
PALETTE    = plt.cm.tab20.colors


# ── Plot 1: Category distribution bar chart ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors  = [PALETTE[i % len(PALETTE)] for i in range(len(cat_counts))]
bars    = ax.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor="white", linewidth=0.6)
# ax.set_title("Click Distribution by News Category", fontsize=13, pad=TITLE_PAD, fontweight="bold")
ax.set_xlabel("Category", fontsize=LABEL_SIZE)
ax.set_ylabel("Number of Clicks", fontsize=LABEL_SIZE)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.xticks(rotation=40, ha="right", fontsize=9)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + cat_counts.max() * 0.01,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "1_category_distribution.png"), dpi=150)
plt.close()

# ── Plot 2: Category share pie chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    cat_counts.values,
    labels=cat_counts.index,
    autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
    colors=colors,
    startangle=140,
    pctdistance=0.82,
)
for t in autotexts:
    t.set_fontsize(8)
# ax.set_title("Category Share of Total Clicks", fontsize=13, pad=TITLE_PAD, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "2_category_share_pie.png"), dpi=150)
plt.close()

# ── Plot 3: PCI histogram ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(pci_s, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
ax.axvline(pci_s.mean(),   color="#DD4444", linestyle="--", linewidth=1.5, label=f"Mean = {pci_s.mean():.3f}")
ax.axvline(pci_s.median(), color="#44AA44", linestyle="--", linewidth=1.5, label=f"Median = {pci_s.median():.3f}")
ax.axvline(0.10, color="orange", linestyle=":", linewidth=1.5, label="PCI threshold = 0.10")
# ax.set_title("Distribution of User PCI Scores (MIND Training Set)", fontsize=13, pad=TITLE_PAD, fontweight="bold")
ax.set_xlabel("PCI (Perspective Concentration Index)", fontsize=LABEL_SIZE)
ax.set_ylabel("Number of Users", fontsize=LABEL_SIZE)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "3_pci_histogram.png"), dpi=150)
plt.close()

# ── Plot 4: PCI CDF ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
sorted_pci = np.sort(pci_s)
cdf        = np.arange(1, len(sorted_pci) + 1) / len(sorted_pci)
ax.plot(sorted_pci, cdf, color="#4C72B0", linewidth=2)
for thresh, col, lbl in [(0.25, "#DD4444", "0.25"), (0.35, "#FF9900", "0.35"), (0.50, "#44AA44", "0.50")]:
    pct = (pci_s > thresh).mean() * 100
    ax.axvline(thresh, color=col, linestyle="--", linewidth=1.2,
               label=f"PCI > {lbl}: {pct:.1f}% of users")
ax.axvline(0.10, color="grey", linestyle=":", linewidth=1.2, label="Threshold = 0.10")
# ax.set_title("Cumulative Distribution of User PCI Scores", fontsize=13, pad=TITLE_PAD, fontweight="bold")
ax.set_xlabel("PCI", fontsize=LABEL_SIZE)
ax.set_ylabel("Cumulative Fraction of Users", fontsize=LABEL_SIZE)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "4_pci_cdf.png"), dpi=150)
plt.close()

# ── Plot 5: Clicks per user distribution ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
cap = int(clicks_s.quantile(0.99))   # cap at 99th percentile for readability
ax.hist(clicks_s.clip(upper=cap), bins=50, color="#55A868", edgecolor="white", linewidth=0.5)
ax.axvline(clicks_s.mean(),   color="#DD4444", linestyle="--", linewidth=1.5, label=f"Mean = {clicks_s.mean():.1f}")
ax.axvline(clicks_s.median(), color="#4C72B0", linestyle="--", linewidth=1.5, label=f"Median = {clicks_s.median():.0f}")
# ax.set_title("Distribution of Clicks per User (capped at 99th pct)", fontsize=13, pad=TITLE_PAD, fontweight="bold")
ax.set_xlabel("Number of Clicks", fontsize=LABEL_SIZE)
ax.set_ylabel("Number of Users", fontsize=LABEL_SIZE)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "5_clicks_per_user.png"), dpi=150)
plt.close()

# ── Plot 6: PCI vs clicks scatter ─────────────────────────────────────────────
pci_click_rows = []
for user, grp in df_pos.groupby("UserID"):
    cats = [cat_map[iid] for iid in grp["ItemID"] if iid in cat_map]
    if len(cats) < 3:
        continue
    counts = Counter(cats)
    total  = sum(counts.values())
    pci    = sum((c / total) ** 2 for c in counts.values())
    pci_click_rows.append({"clicks": len(grp), "pci": pci})

pc_df = pd.DataFrame(pci_click_rows)
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(pc_df["clicks"].clip(upper=int(pc_df["clicks"].quantile(0.99))),
           pc_df["pci"], alpha=0.15, s=8, color="#4C72B0")
ax.axhline(0.10, color="orange", linestyle="--", linewidth=1.2, label="PCI threshold = 0.10")
# ax.set_title("PCI vs Number of Clicks per User", fontsize=13, pad=TITLE_PAD, fontweight="bold")
ax.set_xlabel("Number of Clicks (capped at 99th pct)", fontsize=LABEL_SIZE)
ax.set_ylabel("PCI", fontsize=LABEL_SIZE)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "6_pci_vs_clicks.png"), dpi=150)
plt.close()

# ── Plot 7: Average PCI per dominant category ─────────────────────────────────
dom_cat_pci = {}
for user, grp in df_pos.groupby("UserID"):
    cats = [cat_map[iid] for iid in grp["ItemID"] if iid in cat_map]
    if len(cats) < 3:
        continue
    counts  = Counter(cats)
    total   = sum(counts.values())
    pci     = sum((c / total) ** 2 for c in counts.values())
    dom_cat = counts.most_common(1)[0][0]
    dom_cat_pci.setdefault(dom_cat, []).append(pci)

dom_means = {cat: np.mean(vals) for cat, vals in dom_cat_pci.items()}
dom_df    = pd.Series(dom_means).sort_values(ascending=False)
fig, ax   = plt.subplots(figsize=(10, 5))
dom_colors = [PALETTE[i % len(PALETTE)] for i in range(len(dom_df))]
ax.bar(dom_df.index, dom_df.values, color=dom_colors, edgecolor="white", linewidth=0.6)
ax.axhline(pci_s.mean(), color="#DD4444", linestyle="--", linewidth=1.2,
           label=f"Overall mean PCI = {pci_s.mean():.3f}")
# ax.set_title("Average User PCI by Dominant Reading Category", fontsize=13, pad=TITLE_PAD, fontweight="bold")
ax.set_xlabel("Dominant Category", fontsize=LABEL_SIZE)
ax.set_ylabel("Mean PCI", fontsize=LABEL_SIZE)
ax.set_ylim(0, 1.05)
plt.xticks(rotation=40, ha="right", fontsize=9)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "7_avg_pci_by_dominant_category.png"), dpi=150)
plt.close()

# ==============================================================================
# Parameter recommendations — derived from dataset statistics
# ==============================================================================

n_categories  = int(article_cats.nunique())
n_users       = int(df["UserID"].nunique())
median_clicks = float(clicks_s.median())
mean_clicks   = float(clicks_s.mean())

# ── Derive values ──────────────────────────────────────────────────────────────

# PCI_THRESHOLD: set just below p25 so ~75%+ of non-empty-history users are
# eligible for reranking. Floor at 0.08 to avoid over-triggering on noise.
pci_threshold = float(np.round(max(pci_s.quantile(0.25) * 0.40, 0.08), 2))

# MIN_HISTORY (single-session): 10th-percentile of user click counts, floor 1.
# Ensures we don't exclude the shortest-history users who are most concentrated.
min_history_single = max(1, int(np.floor(clicks_s.quantile(0.10))))

# MIN_HISTORY (multi-session): same logic, but multi-session updates history
# each round, so a slightly higher floor (2) is acceptable.
min_history_multi = max(2, min_history_single)

# PENALTY_WEIGHT (single-session post-hoc reranker):
# Stronger penalty needed to overcome model confidence scores.
# Scale with median PCI: highly concentrated datasets need a larger exponent.
penalty_single = float(np.round(max(4.0, pci_s.median() * 10.0), 1))
penalty_single = min(penalty_single, 8.0)   # cap to avoid total suppression

# PENALTY_WEIGHT (multi-session LP-based):
# LP formulation is more sensitive than post-hoc division; use half the value.
penalty_multi = float(np.round(penalty_single / 2.0, 1))

# TOP_K (single-session): 10 — consistent with standard MIND evaluation.
top_k_single = 10

# TOP_K (multi-session recommend): same 10 for consistency.
top_k_recommend = 10

# NUM_SIM_USERS: 10% of total users, capped at 1000 for runtime.
num_sim_users = min(1000, max(200, int(n_users * 0.10)))

# NUM_SESSIONS: enough sessions to double the median user history via clicks.
# sessions = median_history / TOP_K_CLICK, minimum 15, maximum 25.
top_k_click_derived = max(2, int(round(mean_clicks * 0.20)))   # 20% of avg clicks
num_sessions = int(np.clip(np.ceil(median_clicks / top_k_click_derived), 15, 25))

# TOP_K_CLICK: ~20% of mean clicks per session (realistic CTR proxy), min 2.
top_k_click = top_k_click_derived

# ── Print recommendations ──────────────────────────────────────────────────────
print()
print(SEP)
print("  PARAMETER RECOMMENDATIONS (derived from this dataset)")
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
