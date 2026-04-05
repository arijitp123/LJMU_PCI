"""
NRMS Model — PCI Ablation Study (MIND Dataset)

This script compares:
  1. NRMS-Baseline  (vanilla NRMS, no diversity correction)
  2. NRMS+PCI       (NRMS + PCI-based category diversity reranker)

The Perspective Centration Index (PCI) measures viewpoint concentration in a
user's reading history using the Herfindahl-Hirschman Index (HHI) over news
categories.  When PCI exceeds a threshold, the NRMS+PCI reranker applies a
multiplicative penalty to items belonging to overrepresented categories,
nudging the recommendation list toward a more balanced viewpoint distribution.

Metrics evaluated:
  AUC | HitRatio@10 | NDCG@10 | Precision@10 | Recall@10 |
  GiniCoeff@10 | ILD@10 | PCI@10 | Train (s) | Test (s)

Usage:
    1. First run: python mind_to_nrms_preprocessor.py
       (generates input files from MIND data into examples/example_news_files/)
    2. Then run: python example_nrms_news_reranking.py
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
import numpy as np
import pandas as pd
import sys
import time
from collections import Counter


class _Tee:
    """Mirrors writes to both the original stream and a log file."""
    def __init__(self, stream, log_path):
        self._stream = stream
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def close(self):
        self._log.close()

# Use local cornac instead of installed site-packages (has custom metrics/models)
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
from cornac.experiment.experiment import Experiment

# Accuracy metrics
from cornac.metrics import AUC, HitRatio, NDCG, Precision, Recall

# Diversity metrics
from cornac.metrics import GiniCoeff, ILD, PCI

# Dataset utilities
from cornac.datasets import mind as mind

# Recommender model
from cornac.models import NRMS

# Reranker base class
from cornac.rerankers import ReRanker

# =============================================================================
# Configuration
# =============================================================================
TOP_K            = 10    # Cutoff for all ranking metrics and recommendation list size
MAX_TRAIN_USERS  = 10000  # Random sample of training users (seed=42)
MAX_TEST_USERS   = 5000  # Random sample of test users    (seed=42)
NRMS_EPOCHS      = 5
NRMS_HEAD_NUM    = 16
NRMS_HEAD_DIM    = 16
NRMS_NPRATIO     = 6
NRMS_HISTORY_SIZE= 50
NRMS_TITLE_SIZE  = 30
NRMS_BATCH_SIZE  = 64
NRMS_SEED        = 42
PCI_THRESHOLD    = 0.10
PCI_PENALTY_WEIGHT = 6.0
PCI_MIN_HISTORY  = 1


# =============================================================================
# PCIReRanker — PCI-based post-processing reranker
# =============================================================================

class PCIReRanker(ReRanker):
    """PCI-aware post-processing reranker for the NRMS+PCI ablation study.

    After NRMS scores items, this reranker:
      1. Computes PCI (HHI over categories) from the user's interaction history.
      2. If PCI > threshold AND history length >= min_history, applies a
         multiplicative penalty to items in over-represented categories.
         The penalty is proportional to the degree of over-representation:
             score_j /= (overrep_ratio ** penalty_weight)
         where overrep_ratio = actual_proportion / expected_uniform_proportion.
      3. Re-sorts by adjusted scores and returns the top-k items.

    Parameters
    ----------
    name : str
        Name shown in the experiment results table (default: "NRMS+PCI").
    item_dataframe : pd.DataFrame
        DataFrame indexed by cornac integer item IDs with a column named
        `category_column`.
    category_column : str
        Column in `item_dataframe` holding the news category string.
    top_k : int
        Number of items to return after reranking.
    pool_size : int
        Candidate pool size passed to the base ReRanker (-1 = all items).
    threshold : float
        PCI value above which the diversity penalty is triggered (τ).
    penalty_weight : float
        Exponent controlling the strength of the penalty (λ).
    min_history : int
        Minimum number of history items required to compute a meaningful PCI.
    user_item_history : dict
        {raw_user_id: [raw_item_ids]} — pre-training reading history.
    rerankers_item_pool : list
        Raw item IDs that define the impression pool to rerank within.
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

        # ── Diagnostic counters (reset each run) ──────────────────────────────
        self._diag_total          = 0   # rerank() calls
        self._diag_empty_history  = 0   # user_history == []  (ID mismatch risk)
        self._diag_zero_pci       = 0   # pci_val == 0.0 (no category resolved)
        self._diag_below_thresh   = 0   # pci_val <= threshold
        self._diag_below_minhist  = 0   # pci_val OK but len(history) < min_history
        self._diag_no_cat_counts  = 0   # penalty gate open but cat_counts empty
        self._diag_penalised      = 0   # at least one item was penalised
        self._diag_id_samples     = []  # (user_idx type, first history key type)
        self._diag_pci_vals       = []  # pci_val for users with non-empty history

    # ------------------------------------------------------------------
    def _compute_pci(self, history_items):
        """HHI over news categories present in `history_items`.

        PCI = Σ (count_c / total)²  for each category c.
        Returns 0.0 if no categories can be resolved.
        """
        categories = []
        for iid in history_items:
            if (
                self.item_dataframe is not None
                and iid in self.item_dataframe.index
            ):
                cat = self.item_dataframe.at[iid, self.category_column]
                if pd.notna(cat):
                    categories.append(cat)
        if not categories:
            return 0.0
        counts = Counter(categories)
        total = len(categories)
        return sum((c / total) ** 2 for c in counts.values())

    # ------------------------------------------------------------------
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
        # ── 1. Parent setup: candidate_items / scores / history ──────────────
        super().rerank(
            user_idx, interaction_history, candidate_items,
            prediction_scores, filtering_rules, **kwargs
        )

        # ── 2. Filter already-seen and out-of-pool items ─────────────────────
        self.execute_filters(user_idx, filtering_rules)
        self.filter_items_in_additional_history(user_idx)

        items = list(self.candidate_items[user_idx])
        if not items:
            self.ranked_items[user_idx] = np.array([], dtype=int)
            return self.ranked_items[user_idx]

        # ── 3. Build score array from NRMS prediction scores ─────────────────
        if prediction_scores is not None and candidate_items is not None:
            score_dict = {
                int(item): float(sc)
                for item, sc in zip(candidate_items, prediction_scores)
            }
            scores = np.array(
                [score_dict.get(int(i), 0.0) for i in items], dtype=float
            )
        else:
            # Fallback: items are already rank-ordered; use reverse-rank proxy
            scores = np.arange(len(items), 0, -1, dtype=float)

        # ── 4. Compute PCI from user history ─────────────────────────────────
        self._diag_total += 1
        user_history = self.user_history.get(user_idx, [])

        # Collect ID type samples (first 5 users only, to avoid overhead)
        if len(self._diag_id_samples) < 5:
            hist_key_type = type(next(iter(self.user_history), None)).__name__
            self._diag_id_samples.append(
                (type(user_idx).__name__, hist_key_type, len(user_history))
            )

        if not user_history:
            self._diag_empty_history += 1

        pci_val = self._compute_pci(user_history)
        if user_history:
            self._diag_pci_vals.append(pci_val)
        if pci_val == 0.0 and user_history:
            self._diag_zero_pci += 1

        # ── 5. Apply category penalty when PCI exceeds threshold ─────────────
        if pci_val > self.threshold and len(user_history) >= self.min_history:
            cat_counts = Counter()
            for iid in user_history:
                if (
                    self.item_dataframe is not None
                    and iid in self.item_dataframe.index
                ):
                    cat = self.item_dataframe.at[iid, self.category_column]
                    if pd.notna(cat):
                        cat_counts[cat] += 1

            if not cat_counts:
                self._diag_no_cat_counts += 1
            else:
                item_penalised = False
                total_hist = sum(cat_counts.values())
                n_cats = max(len(cat_counts), 2)   # avoid expected=1.0 trap for single-category histories
                expected = 1.0 / n_cats
                for j, iid in enumerate(items):
                    if (
                        self.item_dataframe is not None
                        and iid in self.item_dataframe.index
                    ):
                        cat = self.item_dataframe.at[iid, self.category_column]
                        if pd.notna(cat) and cat in cat_counts:
                            actual_prop = cat_counts[cat] / total_hist
                            if actual_prop > expected:
                                overrep = actual_prop / expected
                                scores[j] /= (overrep ** self.penalty_weight)
                                item_penalised = True
                if item_penalised:
                    self._diag_penalised += 1
        elif pci_val <= self.threshold:
            self._diag_below_thresh += 1
        else:
            self._diag_below_minhist += 1

        # ── 6. Sort by adjusted scores, keep top_k ───────────────────────────
        ranked_indices = np.argsort(-scores)
        ranked_items = np.array(items)[ranked_indices][: self.top_k]
        self.ranked_items[user_idx] = ranked_items
        return ranked_items

    # ------------------------------------------------------------------
    def print_diagnostics(self):
        """Print a gating funnel showing how many users reached each stage."""
        n = self._diag_total
        if n == 0:
            print("[PCIReRanker] No rerank() calls recorded.")
            return

        def pct(x): return f"{x:>6,}  ({100*x/n:5.1f}%)"

        pci_arr = np.array(self._diag_pci_vals) if self._diag_pci_vals else np.array([0.0])

        print()
        print("=" * 60)
        print(f"  PCIReRanker diagnostic — '{self.name}'")
        print("=" * 60)
        print(f"  threshold={self.threshold}, penalty_weight={self.penalty_weight}, "
              f"min_history={self.min_history}")
        print()
        print(f"  Total rerank() calls          : {n:>6,}")
        print(f"  ├─ Empty user_history (→ skip): {pct(self._diag_empty_history)}"
              "  ← ID mismatch if high")
        print(f"  ├─ History present, PCI=0.0   : {pct(self._diag_zero_pci)}"
              "  ← category lookup failing")
        print(f"  ├─ PCI ≤ threshold (no-op)    : {pct(self._diag_below_thresh)}")
        print(f"  ├─ PCI > thresh, hist too short: {pct(self._diag_below_minhist)}")
        print(f"  ├─ Gate passed, cat_counts=∅  : {pct(self._diag_no_cat_counts)}")
        print(f"  └─ Penalty actually applied   : {pct(self._diag_penalised)}"
              "  ← should be high for effect")
        print()
        print(f"  PCI distribution (non-empty histories):")
        print(f"    min={pci_arr.min():.4f}  p25={np.percentile(pci_arr,25):.4f}"
              f"  median={np.median(pci_arr):.4f}"
              f"  p75={np.percentile(pci_arr,75):.4f}  max={pci_arr.max():.4f}")
        print()
        print("  ID type samples (user_idx_type, history_key_type, history_len):")
        for s in self._diag_id_samples:
            mismatch = " ← TYPE MISMATCH!" if s[0] != s[1] else ""
            print(f"    user_idx={s[0]}, history key={s[1]}, history_len={s[2]}{mismatch}")
        print("=" * 60)


# =============================================================================
# Main — NRMS-Baseline vs NRMS+PCI ablation study
# =============================================================================

def main():
    print("=" * 70)
    print("NRMS MODEL — PCI ABLATION STUDY (MIND DATASET)")
    print("Comparing: NRMS-Baseline  vs  NRMS+PCI")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Evaluation  : top_k={TOP_K}")
    print(f"  Data caps   : max_train_users={MAX_TRAIN_USERS}, max_test_users={MAX_TEST_USERS}, rng_seed=42")
    print(f"  NRMS model  : epochs={NRMS_EPOCHS}, head_num={NRMS_HEAD_NUM}, head_dim={NRMS_HEAD_DIM},")
    print(f"                npratio={NRMS_NPRATIO}, history_size={NRMS_HISTORY_SIZE},")
    print(f"                title_size={NRMS_TITLE_SIZE}, batch_size={NRMS_BATCH_SIZE}, seed={NRMS_SEED}")
    print(f"  PCI reranker: threshold={PCI_THRESHOLD}, penalty_weight={PCI_PENALTY_WEIGHT}, min_history={PCI_MIN_HISTORY}")
    print()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    news_files_dir = os.path.join(current_dir, "example_news_files")
    output_dir = os.path.join(current_dir, "output_nrms_reranking_result")

    # ── Data paths ────────────────────────────────────────────────────────────
    train_uir_path   = os.path.join(news_files_dir, "example_training_graph_uir_top3.csv")
    test_uir_path    = os.path.join(news_files_dir, "example_impression_test_uir.csv")
    article_pool_path= os.path.join(news_files_dir, "example_article.csv")
    user_history_path= os.path.join(news_files_dir, "example_user_history.json")
    title_dict_path  = os.path.join(news_files_dir, "example_title.json")
    word_dict_path   = os.path.join(news_files_dir, "word_index_dict.json")
    word_emb_path    = os.path.join(news_files_dir, "embedding_matrix.npy")
    sentiment_path   = os.path.join(news_files_dir, "example_sentiment.json")
    category_path    = os.path.join(news_files_dir, "example_category.json")
    party_path       = os.path.join(news_files_dir, "example_party.json")

    # ── Load feedback ─────────────────────────────────────────────────────────
    print("Loading feedback data...")
    feedback_train = mind.load_feedback(fpath=train_uir_path)
    feedback_test  = mind.load_feedback(fpath=test_uir_path)
    print(f"  Train interactions : {len(feedback_train)}")
    print(f"  Test  interactions : {len(feedback_test)}")

    rng = np.random.default_rng(42)

    # ── Limit training to a random sample of train users ──────────────────────
    all_train_users   = list({uid for uid, _, _ in feedback_train})
    sampled_train     = set(rng.choice(all_train_users, size=min(MAX_TRAIN_USERS, len(all_train_users)), replace=False))
    feedback_train    = [(u, i, r) for u, i, r in feedback_train if u in sampled_train]
    print(f"  Train interactions : {len(feedback_train)} (sampled {len(sampled_train):,} users)")

    # ── Limit evaluation to a random sample of test users ─────────────────────
    all_test_users = list({uid for uid, _, _ in feedback_test})
    sampled_users  = set(rng.choice(all_test_users, size=min(MAX_TEST_USERS, len(all_test_users)), replace=False))
    feedback_test  = [(u, i, r) for u, i, r in feedback_test if u in sampled_users]
    print(f"  Test  interactions : {len(feedback_test)} (sampled {len(sampled_users):,} users)")

    # Impression pool (used to restrict reranking to known candidate articles)
    impression_items_df  = pd.read_csv(article_pool_path, dtype={"id": str})
    impression_iid_list  = impression_items_df["id"].tolist()

    # User pre-training reading history (raw IDs)
    with open(user_history_path, "r") as f:
        user_item_history = json.load(f)

    # ── Train / test split ────────────────────────────────────────────────────
    print("\nCreating train/test split...")
    rs = BaseMethod.from_splits(
        train_data=feedback_train,
        test_data=feedback_test,
        exclude_unknowns=False,
        verbose=True,
        rating_threshold=0.5,
    )

    # ── NRMS model ────────────────────────────────────────────────────────────
    print("\nInitializing NRMS model...")
    nrms_model = NRMS(
        wordEmb_file   =word_emb_path,
        wordDict_file  =word_dict_path,
        newsTitle_file =title_dict_path,
        userHistory    =user_item_history,
        epochs         =NRMS_EPOCHS,
        head_num       =NRMS_HEAD_NUM,
        head_dim       =NRMS_HEAD_DIM,
        npratio        =NRMS_NPRATIO,
        history_size   =NRMS_HISTORY_SIZE,
        title_size     =NRMS_TITLE_SIZE,
        batch_size     =NRMS_BATCH_SIZE,
        seed           =NRMS_SEED,
    )
    print("[OK] NRMS model initialized")

    # ── Article feature loading ───────────────────────────────────────────────
    print("\nLoading article features...")
    sentiment      = mind.load_sentiment(fpath=sentiment_path)
    category       = mind.load_category(fpath=category_path)
    category_multi = mind.load_category_multi(fpath=category_path)
    entities_nop   = mind.load_entities(fpath=party_path, keep_empty=True)

    Item_sentiment     = mind.build(data=sentiment,      id_map=rs.global_iid_map)
    Item_category      = mind.build(data=category,       id_map=rs.global_iid_map)
    Item_category_vec  = mind.build(data=category_multi, id_map=rs.global_iid_map)
    print(f"  Articles with category : {len(Item_category)}")
    print(f"  Articles with sentiment: {len(Item_sentiment)}")

    # ── Item feature dict for ILD@10 ──────────────────────────────────────────
    # One-hot encoded category vector (matches NRMS example convention)
    item_feature_ild = Item_category_vec   # {cornac_int_idx: one_hot_array}

    # ── Item categories dict for PCI@10 ───────────────────────────────────────
    item_categories_pci = Item_category    # {cornac_int_idx: category_string}

    # ── item_dataframe for PCIReRanker ────────────────────────────────────────
    # Build a DataFrame indexed by cornac integer IDs so the reranker
    # can look up category by integer index.
    pci_item_df = pd.Series(Item_category).to_frame("category")
    # pd.Series from a dict preserves integer keys → integer index ✓

    # ── PCI reranker (NRMS+PCI) ───────────────────────────────────────────────
    print("\nInitializing PCIReRanker (NRMS+PCI)...")
    pci_reranker = PCIReRanker(
        name              ="NRMS+PCI",
        item_dataframe    =pci_item_df,
        category_column   ="category",
        top_k             =TOP_K,
        pool_size         =-1,
        threshold         =PCI_THRESHOLD,
        penalty_weight    =PCI_PENALTY_WEIGHT,
        min_history       =PCI_MIN_HISTORY,
        user_item_history =user_item_history,
        rerankers_item_pool=impression_iid_list,
    )
    print("[OK] PCIReRanker initialized")

    # ── Evaluation metrics ────────────────────────────────────────────────────
    # Required: AUC | HitRatio@10 | NDCG@10 | Precision@10 | Recall@10 |
    #           ILD@10 | PCI@10 | Train(s) | Test(s)
    # Train(s) and Test(s) are reported automatically by cornac.Experiment.
    print("\nSetting up evaluation metrics...")
    metrics = [
        AUC(),
        HitRatio(k=TOP_K),
        NDCG(k=TOP_K),
        Precision(k=TOP_K),
        Recall(k=TOP_K),
        GiniCoeff(item_genre=Item_category_vec, k=TOP_K),
        ILD(name=f"ILD@{TOP_K}", item_feature=item_feature_ild, k=TOP_K),
        PCI(item_categories=item_categories_pci, k=TOP_K),
    ]
    print(f"[OK] Metrics: AUC | HitRatio@{TOP_K} | NDCG@{TOP_K} | Precision@{TOP_K} | Recall@{TOP_K} | GiniCoeff@{TOP_K} | ILD@{TOP_K} | PCI@{TOP_K}")

    # ── Run ablation experiment ───────────────────────────────────────────────
    # Passing pci_reranker as a static reranker produces two rows in the
    # results table:
    #   - "NRMS"       → baseline (vanilla NRMS, no PCI reranking)
    #   - "NRMS+PCI"   → NRMS recommendations re-ranked by PCIReRanker
    print()
    print("=" * 70)
    print("RUNNING ABLATION: NRMS-Baseline  vs  NRMS+PCI")
    print("=" * 70)

    Experiment(
        eval_method =rs,
        models      =[nrms_model],
        metrics     =metrics,
        rerankers   ={"static": [pci_reranker]},
        user_based  =True,
        save_dir    =output_dir,
    ).run()

    pci_reranker.print_diagnostics()

    print()
    print("=" * 70)
    print("ABLATION EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print()
    print("Interpretation:")
    print("  NRMS-Baseline  — standard neural news ranking, no viewpoint correction")
    print("  NRMS+PCI       — same NRMS scores, then PCI penalty applied at rerank time")
    print()
    print("Expected outcome:")
    print("  - NRMS+PCI should have LOWER PCI@10  (less viewpoint concentration)")
    print("  - NRMS+PCI should have HIGHER ILD@10  (more intra-list diversity)")
    print("  - Accuracy metrics (AUC, NDCG, HR, P, R) may decrease slightly")
    print("    due to the diversity-accuracy trade-off introduced by PCI.")


if __name__ == "__main__":
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _output_dir  = os.path.join(_current_dir, "output_nrms_reranking_result")
    os.makedirs(_output_dir, exist_ok=True)
    _log_path = os.path.join(_output_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")

    _tee = _Tee(sys.stdout, _log_path)
    sys.stdout = _tee

    _start = time.time()
    print(f"Run started  : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_start))}")
    print(f"Log file     : {_log_path}")
    print()

    try:
        main()
    finally:
        _end = time.time()
        print()
        print(f"Run finished : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_end))}")
        print(f"Total time   : {_end - _start:.1f} s  ({(_end - _start) / 60:.2f} min)")
        sys.stdout = _tee._stream
        _tee.close()
