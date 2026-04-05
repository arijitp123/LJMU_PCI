"""
NRMS Multi-Session PCI Ablation Study

Demonstrates that NRMS+PCI progressively reduces viewpoint concentration
(PCI decreases over sessions) while NRMS-Baseline stays flat or increases.

Two models are compared across multiple simulated recommendation sessions:
  - NRMS-Baseline : vanilla NRMS scoring, no diversity correction
  - NRMS+PCI      : same NRMS scores + PCIReRanker post-processing

Each session:
  1. Score all items for each simulation user (NRMS neural encoder)
  2. Baseline: rank by raw NRMS score → top-K recommendations
     PCI:      rank by raw score → apply PCI penalty reranking → top-K
  3. Simulate user "clicks": user reads the top-K_CLICK recommended items
  4. Batch-update both models' user histories with clicked items
  5. Record session-level metrics for both variants

Metrics recorded per session:
  - cumulative_pci      : PCI(HHI) over recs + full reading history
  - session_pci         : PCI over just the current session's recommendations
  - ild                 : Intra-List Diversity (higher = more diverse)
  - pci_triggered_users : number of users whose PCI exceeded the threshold

Usage:
    1. First run: python mind_to_nrms_preprocessor.py
       (generates input files from MIND data into examples/example_news_files/)
    2. Then run: python example_nrms_multi_session_pci.py
"""

# ── TensorFlow / logging suppression ─────────────────────────────────────────
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

import logging
tf.get_logger().setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# ── Standard imports ──────────────────────────────────────────────────────────
import json
import random
import time
import copy
import numpy as np
import pandas as pd
import sys
from collections import Counter

# Use local cornac instead of installed site-packages
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
try:
    import cornac_local_src as _cornac_local
    sys.modules.setdefault('cornac', _cornac_local)
except ImportError:
    pass

from cornac.data import Reader
from cornac.eval_methods import BaseMethod
from cornac.datasets import mind as mind_utils
from cornac.metrics import ILD, PCI
from cornac.models import NRMS
from cornac.rerankers import ReRanker


# =============================================================================
# Configuration
# =============================================================================
current_dir   = os.path.dirname(os.path.abspath(__file__))
NEWS_FILES_DIR = os.path.join(current_dir, "example_news_files")
OUTPUT_DIR     = os.path.join(current_dir, "output_nrms_multi_session_pci")

NUM_SIM_USERS    = 200   # Users in multi-session simulation
NUM_SESSIONS     = 7     # Recommendation sessions to simulate
TOP_K_RECOMMEND  = 10    # Items recommended per session
TOP_K_CLICK      = 3     # Items each user "clicks" per session
PCI_THRESHOLD    = 0.15  # tau: trigger reranking when PCI exceeds this
PENALTY_WEIGHT   = 2.0   # lambda: overrepresentation penalty exponent
MIN_HISTORY      = 5     # Minimum history length for meaningful PCI
MAX_TRAIN_USERS  = 10000  # Random sample of training users (reproducible via seed=42)
MAX_TEST_USERS   = 5000  # Random sample of test users    (reproducible via seed=42)
NRMS_EPOCHS      = 5
NRMS_HEAD_NUM    = 16
NRMS_HEAD_DIM    = 16
NRMS_NPRATIO     = 6
NRMS_HISTORY_SIZE= 50
NRMS_TITLE_SIZE  = 30
NRMS_BATCH_SIZE  = 64
NRMS_SEED        = 42


# =============================================================================
# PCIReRanker — PCI-based post-processing reranker  (same as in ablation file)
# =============================================================================

class PCIReRanker(ReRanker):
    """PCI-aware post-processing reranker used for NRMS+PCI variant.

    After NRMS scores items, this reranker:
      1. Computes PCI (HHI over categories) from the user's reading history.
      2. If PCI > threshold AND history length >= min_history, applies a
         multiplicative penalty to items from over-represented categories:
             score_j /= (overrep_ratio ** penalty_weight)
      3. Re-sorts by adjusted scores and returns top_k items.
    """

    def __init__(
        self,
        name="NRMS+PCI",
        item_dataframe=None,
        category_column="category",
        top_k=10,
        pool_size=-1,
        threshold=0.15,
        penalty_weight=2.0,
        min_history=5,
        user_item_history=None,
        rerankers_item_pool=None,
    ):
        super().__init__(
            name=name,
            item_dataframe=item_dataframe,
            diversity_dimension=[category_column],
            top_k=top_k,
            pool_size=pool_size,
            user_item_history=user_item_history,
            rerankers_item_pool=rerankers_item_pool,
        )
        self.category_column = category_column
        self.threshold = threshold
        self.penalty_weight = penalty_weight
        self.min_history = min_history

    def _compute_pci(self, history_items):
        """HHI over news categories in history_items. Returns 0.0 if empty."""
        categories = []
        for iid in history_items:
            if self.item_dataframe is not None and iid in self.item_dataframe.index:
                cat = self.item_dataframe.at[iid, self.category_column]
                if pd.notna(cat):
                    categories.append(cat)
        if not categories:
            return 0.0
        counts = Counter(categories)
        total = len(categories)
        return sum((c / total) ** 2 for c in counts.values())

    def rerank(
        self,
        user_idx,
        interaction_history=None,
        candidate_items=None,
        prediction_scores=None,
        filtering_rules=None,
        **kwargs,
    ):
        """PCI-aware reranking for a single user."""
        # Save session-accumulated history before execute_filters overwrites it
        # (execute_filters resets self.user_history[user_idx] to training positives only)
        saved_session_history = list(self.user_history.get(user_idx, []))

        super().rerank(
            user_idx, interaction_history, candidate_items,
            prediction_scores, filtering_rules, **kwargs
        )
        self.execute_filters(user_idx, filtering_rules)
        self.filter_items_in_additional_history(user_idx)

        # Restore accumulated session history for PCI computation
        if saved_session_history:
            self.user_history[user_idx] = saved_session_history

        items = list(self.candidate_items[user_idx])
        if not items:
            self.ranked_items[user_idx] = np.array([], dtype=int)
            return self.ranked_items[user_idx]

        if prediction_scores is not None and candidate_items is not None:
            score_dict = {
                int(item): float(sc)
                for item, sc in zip(candidate_items, prediction_scores)
            }
            scores = np.array(
                [score_dict.get(int(i), 0.0) for i in items], dtype=float
            )
        else:
            scores = np.arange(len(items), 0, -1, dtype=float)

        user_history = self.user_history.get(user_idx, [])
        pci_val = self._compute_pci(user_history)

        if pci_val > self.threshold and len(user_history) >= self.min_history:
            cat_counts = Counter()
            for iid in user_history:
                if self.item_dataframe is not None and iid in self.item_dataframe.index:
                    cat = self.item_dataframe.at[iid, self.category_column]
                    if pd.notna(cat):
                        cat_counts[cat] += 1

            if cat_counts:
                total_hist = sum(cat_counts.values())
                expected = 1.0 / len(cat_counts)
                for j, iid in enumerate(items):
                    if self.item_dataframe is not None and iid in self.item_dataframe.index:
                        cat = self.item_dataframe.at[iid, self.category_column]
                        if pd.notna(cat) and cat in cat_counts:
                            actual_prop = cat_counts[cat] / total_hist
                            if actual_prop > expected:
                                overrep = actual_prop / expected
                                scores[j] /= (overrep ** self.penalty_weight)

        ranked_indices = np.argsort(-scores)
        ranked_items = np.array(items)[ranked_indices][: self.top_k]
        self.ranked_items[user_idx] = ranked_items
        return ranked_items


# =============================================================================
# Metric helpers
# =============================================================================

def compute_session_pci(rec_items, item_categories):
    """HHI over the categories of a single recommendation list."""
    categories = [item_categories[i] for i in rec_items if i in item_categories]
    if not categories:
        return None
    counts = Counter(categories)
    total = len(categories)
    return sum((c / total) ** 2 for c in counts.values())


def compute_session_metrics(
    recs_dict,
    histories_cornac,
    item_categories,
    item_feature,
    threshold,
):
    """
    Compute per-session aggregate metrics over a set of users.

    Parameters
    ----------
    recs_dict : dict  {user_cornac_idx: [cornac_item_idx, ...]}
    histories_cornac : dict  {user_cornac_idx: [cornac_item_idx, ...]}
    item_categories : dict  {cornac_item_idx: category_str}
    item_feature : dict  {cornac_item_idx: np.ndarray}  (for ILD)
    threshold : float  PCI trigger threshold

    Returns
    -------
    dict with keys: avg_pci, avg_session_pci, avg_ild, pci_triggered
    """
    pci_metric = PCI(item_categories=item_categories, k=TOP_K_RECOMMEND)
    ild_metric = ILD(item_feature=item_feature, k=TOP_K_RECOMMEND)

    cumulative_pci_values = []
    session_pci_values    = []
    ild_values            = []
    pci_triggered         = 0

    for user_idx, top_recs in recs_dict.items():
        if not top_recs:
            continue

        # Cumulative PCI: recs + full history
        user_history = histories_cornac.get(user_idx, [])
        cum_pci = pci_metric.compute(
            np.array(top_recs),
            user_history=np.array(user_history)
        )
        if cum_pci is not None:
            cumulative_pci_values.append(cum_pci)

        # Session PCI: recs only
        sess_pci = compute_session_pci(top_recs, item_categories)
        if sess_pci is not None:
            session_pci_values.append(sess_pci)

        # ILD
        ild_val = ild_metric.compute(np.array(top_recs))
        if ild_val is not None:
            ild_values.append(ild_val)

        # PCI triggered?
        user_pci_val = compute_session_pci(user_history, item_categories) if user_history else 0.0
        if user_pci_val is not None and user_pci_val > threshold:
            pci_triggered += 1

    return {
        "avg_pci":         np.mean(cumulative_pci_values) if cumulative_pci_values else None,
        "avg_session_pci": np.mean(session_pci_values)    if session_pci_values    else None,
        "avg_ild":         np.mean(ild_values)             if ild_values             else None,
        "pci_triggered":   pci_triggered,
    }


# =============================================================================
# History update helpers
# =============================================================================

def update_history_raw(history_raw, recs_cornac, clicks_per_user, item_idx2id):
    """
    Add clicked raw item IDs to the raw history dict (in-place).

    Parameters
    ----------
    history_raw : dict  {raw_user_id: [raw_item_id, ...]}
    recs_cornac : dict  {cornac_user_idx: [cornac_item_idx, ...]}
    clicks_per_user : int   number of top items the user "clicks"
    item_idx2id : dict  {cornac_item_idx: raw_item_id}
    """
    for user_cornac_idx, top_recs in recs_cornac.items():
        clicked_cornac = top_recs[:clicks_per_user]
        clicked_raw    = [item_idx2id[i] for i in clicked_cornac if i in item_idx2id]
        if clicked_raw:
            raw_uid = history_raw.get("_idx2uid", {}).get(user_cornac_idx)
            if raw_uid and raw_uid in history_raw:
                history_raw[raw_uid] = history_raw[raw_uid] + clicked_raw


def update_history_cornac(history_cornac, recs_cornac, clicks_per_user):
    """
    Add clicked cornac item indices to the cornac history dict (in-place).

    Parameters
    ----------
    history_cornac : dict  {cornac_user_idx: [cornac_item_idx, ...]}
    recs_cornac    : dict  {cornac_user_idx: [cornac_item_idx, ...]}
    clicks_per_user : int
    """
    for user_cornac_idx, top_recs in recs_cornac.items():
        clicked = list(top_recs[:clicks_per_user])
        if clicked:
            history_cornac.setdefault(user_cornac_idx, [])
            history_cornac[user_cornac_idx] = history_cornac[user_cornac_idx] + clicked


# =============================================================================
# Single-session scoring
# =============================================================================

def run_nrms_session_baseline(nrms_model, sim_users, history_raw_current,
                               item_idx2id, user_idx2id):
    """
    Run one NRMS-Baseline session: update model click history, get top-K recs.

    Returns dict: {user_cornac_idx: [cornac_item_idx, ...]}
    """
    recs = {}
    for user_cornac_idx in sim_users:
        raw_uid = user_idx2id.get(user_cornac_idx)
        if raw_uid is None:
            continue

        # Refresh cached click-title encoding to reflect latest history
        raw_history = history_raw_current.get(raw_uid, [])
        if raw_history:
            nrms_model.news_organizer.click_title_all_users[user_cornac_idx] = (
                nrms_model.news_organizer.process_history_news_title(
                    raw_history, nrms_model.history_size
                )
            )

        try:
            all_items = np.arange(nrms_model.total_items)
            scores    = nrms_model.score(user_cornac_idx)
            ranked    = np.argsort(-scores)[:TOP_K_RECOMMEND]
            recs[user_cornac_idx] = list(ranked)
        except Exception:
            continue

    return recs


def run_nrms_session_pci(nrms_model, pci_reranker, sim_users, history_raw_current,
                          history_cornac_current, item_idx2id, user_idx2id, train_set):
    """
    Run one NRMS+PCI session: update histories, score, apply PCI reranking.

    Returns dict: {user_cornac_idx: [cornac_item_idx, ...]}
    """
    recs = {}
    for user_cornac_idx in sim_users:
        raw_uid = user_idx2id.get(user_cornac_idx)
        if raw_uid is None:
            continue

        # Refresh NRMS cached click-title encoding
        raw_history = history_raw_current.get(raw_uid, [])
        if raw_history:
            nrms_model.news_organizer.click_title_all_users[user_cornac_idx] = (
                nrms_model.news_organizer.process_history_news_title(
                    raw_history, nrms_model.history_size
                )
            )

        # Sync cornac history into PCIReRanker
        pci_reranker.user_history[user_cornac_idx] = (
            history_cornac_current.get(user_cornac_idx, [])
        )

        try:
            scores    = nrms_model.score(user_cornac_idx)
            all_items = np.arange(len(scores))
            ranked_raw = list(np.argsort(-scores))  # full ranked list

            # Pass all items + scores to the reranker
            pci_reranker.rerank(
                user_idx            = user_cornac_idx,
                interaction_history = train_set,
                candidate_items     = all_items,
                prediction_scores   = scores,
            )
            top_recs = list(pci_reranker.ranked_items.get(user_cornac_idx, []))
            recs[user_cornac_idx] = top_recs
        except Exception:
            continue

    return recs


# =============================================================================
# Report generation
# =============================================================================

def generate_report(results, output_dir):
    """Print console tables, save CSV, and save trend plot."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(results["NRMS-Baseline"])

    W = 112
    print("\n" + "=" * W)
    print("MULTI-SESSION PCI RESULTS  —  NRMS-Baseline  vs  NRMS+PCI")
    print("=" * W)

    # ── Combined metrics table ────────────────────────────────────────────────
    # Column layout (112 chars total):
    #   Session(9) | CumPCI(32) | SessionPCI(32) | ILD(26) | Triggered(10)
    def _v(val, w=9):
        return f"{val:{w}.4f}" if val is not None else f"{'N/A':>{w}}"
    def _d(val, w=9):
        return f"{val:>+{w}.4f}" if val is not None else f"{'N/A':>{w}}"

    hdr1 = (f"{'':9}|{'Cumulative PCI (recs+history)':^32}"
            f"|{'Session PCI (recs only)':^32}"
            f"|{'ILD (higher=diverse)':^26}"
            f"|{'Triggered':^10}")
    hdr2 = (f"{'Session':>9}| {'Baseline':>9} {'NRMS+PCI':>9} {'Delta':>9} "
            f"| {'Baseline':>9} {'NRMS+PCI':>9} {'Delta':>9} "
            f"| {'Baseline':>7} {'NRMS+PCI':>8} {'Delta':>7} "
            f"|{'(PCI)':^10}")
    sep  = "-" * W

    print(f"\n{hdr1}")
    print(hdr2)
    print(sep)

    rows = []
    for i in range(n):
        base = results["NRMS-Baseline"][i]
        pci  = results["NRMS+PCI"][i]

        b_cpci = base["avg_pci"];         p_cpci = pci["avg_pci"]
        b_spci = base["avg_session_pci"]; p_spci = pci["avg_session_pci"]
        b_ild  = base["avg_ild"];         p_ild  = pci["avg_ild"]

        cpci_d = (p_cpci - b_cpci) if (b_cpci is not None and p_cpci is not None) else None
        spci_d = (p_spci - b_spci) if (b_spci is not None and p_spci is not None) else None
        ild_d  = (p_ild  - b_ild)  if (b_ild  is not None and p_ild  is not None) else None

        print(f"{i+1:>9}| {_v(b_cpci)} {_v(p_cpci)} {_d(cpci_d)} "
              f"| {_v(b_spci)} {_v(p_spci)} {_d(spci_d)} "
              f"| {_v(b_ild,7)} {_v(p_ild,8)} {_d(ild_d,7)} "
              f"|{pci['pci_triggered']:^10}")

        rows.append({
            "session":                 i + 1,
            "nrms_cumulative_pci":     b_cpci,
            "nrms_pci_cumulative_pci": p_cpci,
            "nrms_session_pci":        b_spci,
            "nrms_pci_session_pci":    p_spci,
            "nrms_ild":                b_ild,
            "nrms_pci_ild":            p_ild,
            "pci_triggered_users":     pci["pci_triggered"],
        })

    print(sep)

    # ── Summary row ───────────────────────────────────────────────────────────
    b_first = results["NRMS-Baseline"][0]["avg_pci"]
    b_last  = results["NRMS-Baseline"][-1]["avg_pci"]
    p_first = results["NRMS+PCI"][0]["avg_pci"]
    p_last  = results["NRMS+PCI"][-1]["avg_pci"]
    if all(v is not None for v in [b_first, b_last, p_first, p_last]):
        base_delta = b_last - b_first
        pci_delta  = p_last - p_first
        print(f"\nCumulative PCI change over {n} sessions:")
        print(f"  NRMS-Baseline : {base_delta:+.4f}")
        print(f"  NRMS+PCI      : {pci_delta:+.4f}")
        if pci_delta < base_delta:
            print(f"  -> NRMS+PCI reduces concentration by "
                  f"{abs(base_delta - pci_delta):.4f} more than baseline")
    print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "nrms_pci_trends.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV saved to: {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        sessions = list(range(1, n + 1))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "NRMS Multi-Session Ablation Study\nNRMS-Baseline vs NRMS+PCI",
            fontsize=14, fontweight='bold'
        )

        # -- Cumulative PCI ---------------------------------------------------
        axes[0, 0].plot(
            sessions,
            [r["avg_pci"] for r in results["NRMS-Baseline"]],
            'o-', label='NRMS-Baseline', color='#e74c3c', linewidth=2
        )
        axes[0, 0].plot(
            sessions,
            [r["avg_pci"] for r in results["NRMS+PCI"]],
            's-', label='NRMS+PCI', color='#2ecc71', linewidth=2
        )
        axes[0, 0].set_xlabel('Session')
        axes[0, 0].set_ylabel('Average Cumulative PCI')
        axes[0, 0].set_title('Cumulative PCI Over Sessions\n(Lower = Less Concentrated)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # -- Session PCI -------------------------------------------------------
        axes[0, 1].plot(
            sessions,
            [r["avg_session_pci"] for r in results["NRMS-Baseline"]],
            'o-', label='NRMS-Baseline', color='#e74c3c', linewidth=2
        )
        axes[0, 1].plot(
            sessions,
            [r["avg_session_pci"] for r in results["NRMS+PCI"]],
            's-', label='NRMS+PCI', color='#2ecc71', linewidth=2
        )
        axes[0, 1].set_xlabel('Session')
        axes[0, 1].set_ylabel('Average Session PCI')
        axes[0, 1].set_title('Per-Session PCI Over Sessions\n(Lower = More Diverse Recs)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # -- ILD ---------------------------------------------------------------
        axes[1, 0].plot(
            sessions,
            [r["avg_ild"] for r in results["NRMS-Baseline"]],
            'o-', label='NRMS-Baseline', color='#e74c3c', linewidth=2
        )
        axes[1, 0].plot(
            sessions,
            [r["avg_ild"] for r in results["NRMS+PCI"]],
            's-', label='NRMS+PCI', color='#2ecc71', linewidth=2
        )
        axes[1, 0].set_xlabel('Session')
        axes[1, 0].set_ylabel('Average ILD')
        axes[1, 0].set_title('ILD Over Sessions\n(Higher = More Diverse)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # -- PCI Triggered Users -----------------------------------------------
        axes[1, 1].bar(
            sessions,
            [r["pci_triggered"] for r in results["NRMS+PCI"]],
            color='#3498db', alpha=0.75, label='NRMS+PCI triggered'
        )
        axes[1, 1].set_xlabel('Session')
        axes[1, 1].set_ylabel('Users with PCI > threshold')
        axes[1, 1].set_title(f'PCI Triggered Users per Session\n(threshold={PCI_THRESHOLD})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "nrms_pci_trends.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Chart saved to: {chart_path}")

    except ImportError:
        print("[INFO] matplotlib not available — skipping chart generation")


# =============================================================================
# Main
# =============================================================================

def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("NRMS MULTI-SESSION PCI ABLATION STUDY")
    print("Comparing: NRMS-Baseline  vs  NRMS+PCI over multiple sessions")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Simulation  : {NUM_SIM_USERS} users x {NUM_SESSIONS} sessions")
    print(f"  Rec / Click : top_k_recommend={TOP_K_RECOMMEND}, top_k_click={TOP_K_CLICK}")
    print(f"  Data caps   : max_train_users={MAX_TRAIN_USERS}, max_test_users={MAX_TEST_USERS}, rng_seed=42")
    print(f"  PCI reranker: threshold={PCI_THRESHOLD}, penalty_weight={PENALTY_WEIGHT}, min_history={MIN_HISTORY}")
    print(f"  NRMS model  : epochs={NRMS_EPOCHS}, head_num={NRMS_HEAD_NUM}, head_dim={NRMS_HEAD_DIM},")
    print(f"                npratio={NRMS_NPRATIO}, history_size={NRMS_HISTORY_SIZE},")
    print(f"                title_size={NRMS_TITLE_SIZE}, batch_size={NRMS_BATCH_SIZE}, seed={NRMS_SEED}")
    print()

    # ── Data paths ────────────────────────────────────────────────────────────
    train_uir_path    = os.path.join(NEWS_FILES_DIR, "example_training_graph_uir_top3.csv")
    test_uir_path     = os.path.join(NEWS_FILES_DIR, "example_impression_test_uir.csv")
    article_pool_path = os.path.join(NEWS_FILES_DIR, "example_article.csv")
    user_history_path = os.path.join(NEWS_FILES_DIR, "example_user_history.json")
    title_dict_path   = os.path.join(NEWS_FILES_DIR, "example_title.json")
    word_dict_path    = os.path.join(NEWS_FILES_DIR, "word_index_dict.json")
    word_emb_path     = os.path.join(NEWS_FILES_DIR, "embedding_matrix.npy")
    sentiment_path    = os.path.join(NEWS_FILES_DIR, "example_sentiment.json")
    category_path     = os.path.join(NEWS_FILES_DIR, "example_category.json")
    party_path        = os.path.join(NEWS_FILES_DIR, "example_party.json")

    for p in [train_uir_path, test_uir_path, article_pool_path,
              user_history_path, title_dict_path, word_dict_path,
              word_emb_path, sentiment_path, category_path]:
        if not os.path.exists(p):
            print(f"[FAIL] Required file not found: {p}")
            print("Please run mind_to_nrms_preprocessor.py first.")
            return False

    # ── Load feedback ─────────────────────────────────────────────────────────
    print("Loading feedback data...")
    feedback_train = mind_utils.load_feedback(fpath=train_uir_path)
    feedback_test  = mind_utils.load_feedback(fpath=test_uir_path)
    print(f"  Train interactions : {len(feedback_train)}")
    print(f"  Test  interactions : {len(feedback_test)}")

    rng = np.random.default_rng(42)

    all_train_users = list({uid for uid, _, _ in feedback_train})
    sampled_train   = set(rng.choice(all_train_users, size=min(MAX_TRAIN_USERS, len(all_train_users)), replace=False))
    feedback_train  = [(u, i, r) for u, i, r in feedback_train if u in sampled_train]
    print(f"  Train interactions : {len(feedback_train)} (sampled {len(sampled_train):,} users)")

    all_test_users = list({uid for uid, _, _ in feedback_test})
    sampled_users  = set(rng.choice(all_test_users, size=min(MAX_TEST_USERS, len(all_test_users)), replace=False))
    feedback_test  = [(u, i, r) for u, i, r in feedback_test if u in sampled_users]
    print(f"  Test  interactions : {len(feedback_test)} (sampled {len(sampled_users):,} users)")

    impression_items_df = pd.read_csv(article_pool_path, dtype={"id": str})
    impression_iid_list = impression_items_df["id"].tolist()

    with open(user_history_path, "r") as f:
        user_item_history_raw = json.load(f)   # {raw_uid: [raw_iid, ...]}

    # ── Train / test split ────────────────────────────────────────────────────
    print("\nCreating train/test split...")
    rs = BaseMethod.from_splits(
        train_data=feedback_train,
        test_data=feedback_test,
        exclude_unknowns=False,
        verbose=True,
        rating_threshold=0.5,
    )
    print(f"  Users: {rs.train_set.num_users}   Items: {rs.train_set.num_items}\n")

    # ID mapping helpers
    iid_map      = rs.train_set.iid_map        # raw_iid → cornac_idx
    uid_map      = rs.train_set.uid_map        # raw_uid → cornac_idx
    item_idx2id  = {v: k for k, v in iid_map.items()}   # cornac_idx → raw_iid
    user_idx2id  = {v: k for k, v in uid_map.items()}   # cornac_idx → raw_uid

    # ── Load article features ─────────────────────────────────────────────────
    print("Loading article features...")
    sentiment      = mind_utils.load_sentiment(fpath=sentiment_path)
    category       = mind_utils.load_category(fpath=category_path)
    category_multi = mind_utils.load_category_multi(fpath=category_path)

    item_categories = mind_utils.build(data=category, id_map=rs.global_iid_map)
    item_feature    = mind_utils.build(data=category_multi, id_map=rs.global_iid_map)

    # DataFrame indexed by cornac int IDs for PCIReRanker
    pci_item_df = pd.Series(item_categories).to_frame("category")

    print(f"  Articles with category : {len(item_categories)}")
    print(f"  Feature vectors        : {len(item_feature)}\n")

    # ── Build NRMS model ──────────────────────────────────────────────────────
    print("Initializing and training NRMS model (trained once)...")
    nrms_model = NRMS(
        wordEmb_file  = word_emb_path,
        wordDict_file = word_dict_path,
        newsTitle_file= title_dict_path,
        userHistory   = user_item_history_raw,
        epochs        = NRMS_EPOCHS,
        head_num      = NRMS_HEAD_NUM,
        head_dim      = NRMS_HEAD_DIM,
        npratio       = NRMS_NPRATIO,
        history_size  = NRMS_HISTORY_SIZE,
        title_size    = NRMS_TITLE_SIZE,
        batch_size    = NRMS_BATCH_SIZE,
        seed          = NRMS_SEED,
    )
    t0 = time.time()
    nrms_model.fit(rs.train_set)
    print(f"[OK] NRMS trained in {time.time() - t0:.1f}s\n")

    # ── Build PCIReRanker ─────────────────────────────────────────────────────
    print("Initializing PCIReRanker (NRMS+PCI)...")
    pci_reranker = PCIReRanker(
        name               = "NRMS+PCI",
        item_dataframe     = pci_item_df,
        category_column    = "category",
        top_k              = TOP_K_RECOMMEND,
        pool_size          = -1,
        threshold          = PCI_THRESHOLD,
        penalty_weight     = PENALTY_WEIGHT,
        min_history        = MIN_HISTORY,
        user_item_history  = user_item_history_raw,
        rerankers_item_pool= impression_iid_list,
    )
    # Provide cornac ID maps so reranker can look up users/items
    pci_reranker.num_users = rs.train_set.num_users
    pci_reranker.num_items = rs.train_set.num_items
    pci_reranker.uid_map   = uid_map
    pci_reranker.iid_map   = iid_map
    print(f"[OK] PCIReRanker initialized  "
          f"(threshold={PCI_THRESHOLD}, penalty_weight={PENALTY_WEIGHT})\n")

    # ── Select simulation users ───────────────────────────────────────────────
    # Require enough history for meaningful PCI and NRMS encoding
    eligible_users = [
        cornac_uid
        for raw_uid, cornac_uid in uid_map.items()
        if len(user_item_history_raw.get(raw_uid, [])) >= MIN_HISTORY
    ]
    print(f"Eligible users (history >= {MIN_HISTORY} items): {len(eligible_users)}")

    random.seed(42)
    if len(eligible_users) > NUM_SIM_USERS:
        sim_users = sorted(random.sample(eligible_users, NUM_SIM_USERS))
    else:
        sim_users = sorted(eligible_users)
    print(f"Simulation users: {len(sim_users)}\n")

    # ── Initialise per-model history stores ───────────────────────────────────
    # Raw histories for NRMS scoring (keyed by raw user ID)
    baseline_history_raw = {
        raw_uid: list(hist)
        for raw_uid, hist in user_item_history_raw.items()
    }
    pci_history_raw = {
        raw_uid: list(hist)
        for raw_uid, hist in user_item_history_raw.items()
    }

    # Cornac-indexed histories for metric computation & PCIReRanker
    baseline_history_cornac: dict = {}
    pci_history_cornac: dict = {}

    # Seed cornac histories from raw histories
    for cornac_uid in sim_users:
        raw_uid = user_idx2id.get(cornac_uid)
        if raw_uid and raw_uid in user_item_history_raw:
            raw_items = user_item_history_raw[raw_uid]
            cornac_items = [iid_map[rid] for rid in raw_items if rid in iid_map]
            baseline_history_cornac[cornac_uid] = list(cornac_items)
            pci_history_cornac[cornac_uid]      = list(cornac_items)

    # Diagnostic: initial PCI distribution
    above_thresh = sum(
        1 for uid in sim_users
        if (compute_session_pci(
            baseline_history_cornac.get(uid, []), item_categories
        ) or 0.0) > PCI_THRESHOLD
    )
    print(f"Users with initial PCI > {PCI_THRESHOLD}: "
          f"{above_thresh}/{len(sim_users)}\n")

    # ── Multi-session simulation ───────────────────────────────────────────────
    results = {"NRMS-Baseline": [], "NRMS+PCI": []}

    for session in range(NUM_SESSIONS):
        print(f"{'='*70}")
        print(f"SESSION {session + 1}/{NUM_SESSIONS}")
        print(f"{'='*70}")

        # ── NRMS-Baseline session ─────────────────────────────────────────
        t0 = time.time()
        base_recs = run_nrms_session_baseline(
            nrms_model, sim_users,
            baseline_history_raw,
            item_idx2id, user_idx2id,
        )
        rec_time = time.time() - t0

        base_metrics = compute_session_metrics(
            base_recs, baseline_history_cornac,
            item_categories, item_feature, PCI_THRESHOLD
        )
        results["NRMS-Baseline"].append(base_metrics)

        pci_str  = f"{base_metrics['avg_pci']:.4f}"          if base_metrics['avg_pci']         else "N/A"
        spci_str = f"{base_metrics['avg_session_pci']:.4f}"  if base_metrics['avg_session_pci'] else "N/A"
        ild_str  = f"{base_metrics['avg_ild']:.4f}"          if base_metrics['avg_ild']         else "N/A"
        print(f"  NRMS-Baseline: CumPCI={pci_str}  SessPCI={spci_str}  "
              f"ILD={ild_str}  triggered={base_metrics['pci_triggered']}  "
              f"({rec_time:.1f}s)")

        # Update baseline histories
        update_history_cornac(baseline_history_cornac, base_recs, TOP_K_CLICK)
        for cornac_uid, top_recs in base_recs.items():
            raw_uid = user_idx2id.get(cornac_uid)
            if raw_uid:
                clicked_raw = [item_idx2id[i] for i in top_recs[:TOP_K_CLICK] if i in item_idx2id]
                if clicked_raw:
                    baseline_history_raw.setdefault(raw_uid, [])
                    baseline_history_raw[raw_uid] = baseline_history_raw[raw_uid] + clicked_raw

        # ── NRMS+PCI session ──────────────────────────────────────────────
        t0 = time.time()
        pci_recs = run_nrms_session_pci(
            nrms_model, pci_reranker, sim_users,
            pci_history_raw, pci_history_cornac,
            item_idx2id, user_idx2id, rs.train_set,
        )
        rec_time = time.time() - t0

        pci_metrics = compute_session_metrics(
            pci_recs, pci_history_cornac,
            item_categories, item_feature, PCI_THRESHOLD
        )
        results["NRMS+PCI"].append(pci_metrics)

        pci_str  = f"{pci_metrics['avg_pci']:.4f}"          if pci_metrics['avg_pci']         else "N/A"
        spci_str = f"{pci_metrics['avg_session_pci']:.4f}"  if pci_metrics['avg_session_pci'] else "N/A"
        ild_str  = f"{pci_metrics['avg_ild']:.4f}"          if pci_metrics['avg_ild']         else "N/A"
        print(f"  NRMS+PCI     : CumPCI={pci_str}  SessPCI={spci_str}  "
              f"ILD={ild_str}  triggered={pci_metrics['pci_triggered']}  "
              f"({rec_time:.1f}s)")

        # Update PCI histories
        update_history_cornac(pci_history_cornac, pci_recs, TOP_K_CLICK)
        for cornac_uid, top_recs in pci_recs.items():
            raw_uid = user_idx2id.get(cornac_uid)
            if raw_uid:
                clicked_raw = [item_idx2id[i] for i in top_recs[:TOP_K_CLICK] if i in item_idx2id]
                if clicked_raw:
                    pci_history_raw.setdefault(raw_uid, [])
                    pci_history_raw[raw_uid] = pci_history_raw[raw_uid] + clicked_raw

        print()

    # ── Report ────────────────────────────────────────────────────────────────
    generate_report(results, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print()
    print("Interpretation:")
    print("  - Cumulative PCI : viewpoint concentration in recs + full history")
    print("  - Session PCI    : diversity of just the current session's recs")
    print("  - ILD            : intra-list diversity (higher = more diverse)")
    print("  - PCI Triggered  : users where history PCI exceeds threshold tau")
    print()
    print("Expected outcome:")
    print("  - NRMS+PCI should show LOWER cumulative PCI over time")
    print("    (echo chamber effect is mitigated by PCI reranking)")
    print("  - NRMS+PCI should show LOWER session PCI")
    print("    (individual recommendation lists are more category-diverse)")
    print("  - NRMS+PCI should show HIGHER ILD")
    print("    (items in the list are more dissimilar from each other)")
    print()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
