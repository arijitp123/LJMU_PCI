"""
Multi-Session PCI Simulation

Demonstrates that D_RDW+PCI progressively reduces viewpoint concentration
(PCI decreases over sessions) while baseline D_RDW stays flat or increases.

Each session:
  1. Generate recommendations for all simulation users
  2. Simulate user clicking top-K of N recommended items
  3. Batch update user histories with clicked items
  4. Recompute PCI values
  5. Record session-level metrics (both cumulative and session-only PCI)

Usage:
    1. First run: python mind_to_drdw_preprocessor.py
       (generates input files from MIND data)

    2. Then run: python example_multi_session_pci.py
"""

import sys
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
try:
    import cornac_local_src as _cornac_local
    sys.modules.setdefault('cornac', _cornac_local)
except ImportError:
    pass

import cornac
from cornac.eval_methods import BaseMethod
from cornac.metrics import ILD, PCI
from cornac.models import D_RDW
from cornac.datasets import mind
import pandas as pd
import numpy as np
from collections import Counter
import os
import sys
import time
import logging
from datetime import datetime

# =============================================================================
# Configuration  (all tunable parameters live here)
# =============================================================================

# --- Paths ---
MIND_GENERATED_DATA_PATH = os.path.join(os.path.dirname(__file__), 'example_news_files_mind')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_multi_session_pci')

# --- Data loading ---
MAX_USERS        = 10000  # Max users to load from data
RANDOM_SEED      = 42     # Random seed for reproducibility

# --- Simulation ---
NUM_SIM_USERS    = 500    # Users in multi-session simulation
NUM_SESSIONS     = 15      # Number of recommendation sessions
TOP_K_RECOMMEND  = 10      # Items recommended per session
TOP_K_CLICK      = 2      # Items user "clicks" per session
MIN_USER_HISTORY = 5      # Minimum history length to be eligible for simulation

# --- Evaluation ---
RATING_THRESHOLD = 0.5    # Minimum rating to treat as positive feedback

# --- D_RDW model ---
DIVERSITY_DIMENSIONS = ["sentiment"]   # Feature columns to diversify by
MAX_HOPS         = 5
RANKING_TYPE     = "rdw_score"
SAMPLE_OBJECTIVE = "rdw_score"

# --- Sentiment target distribution (continuous) ---
SENTIMENT_DISTRIBUTION = [
    {"min": -1.0, "max": -0.5, "prob": 0.1},
    {"min": -0.5, "max":  0.0, "prob": 0.2},
    {"min":  0.0, "max":  0.5, "prob": 0.3},
    {"min":  0.5, "max":  1.01, "prob": 0.4},
]

# --- PCI configuration (used by D_RDW+PCI model) ---
PCI_CONFIG = {
    "enabled":         True,
    "threshold":       0.33,   # Session PCI score above which penalty is applied
    "category_column": "category",
    "penalty_weight":  1.5,
    "min_history":     5,
}

# =============================================================================


def setup_logging(base_output_dir):
    """Create a timestamped run folder and configure logging into it.

    Returns (run_dir, log_path) so all outputs for this run go to run_dir.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"run_{timestamp}.log")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Redirect print() output through logging
    class _PrintToLog:
        def __init__(self, original):
            self._original = original

        def write(self, msg):
            if msg.rstrip():
                logging.info(msg.rstrip())

        def flush(self):
            self._original.flush()

    sys.stdout = _PrintToLog(sys.stdout)
    return run_dir, log_path


def load_data():
    """Load preprocessed MIND data from generated files."""
    print("Loading preprocessed MIND data...\n")

    if not os.path.exists(MIND_GENERATED_DATA_PATH):
        raise FileNotFoundError(
            f"Data directory not found: {MIND_GENERATED_DATA_PATH}\n"
            f"Please run mind_to_drdw_preprocessor.py first."
        )

    train_uir_path = os.path.join(MIND_GENERATED_DATA_PATH, 'example_training_graph_uir_top3.csv')
    feedback_train = mind.load_feedback(fpath=train_uir_path)
    print(f"[OK] Loaded {len(feedback_train)} training interactions")

    test_uir_path = os.path.join(MIND_GENERATED_DATA_PATH, 'example_impression_test_uir.csv')
    feedback_test = mind.load_feedback(fpath=test_uir_path)
    print(f"[OK] Loaded {len(feedback_test)} test interactions")

    # Subsample users
    all_train_users = list(set(uid for uid, _, _ in feedback_train))
    if len(all_train_users) > MAX_USERS:
        import random as rng
        rng.seed(RANDOM_SEED)
        selected_users = set(rng.sample(all_train_users, MAX_USERS))
        feedback_train = [(u, i, r) for u, i, r in feedback_train if u in selected_users]
        feedback_test = [(u, i, r) for u, i, r in feedback_test if u in selected_users]
        print(f"[OK] Subsampled to {MAX_USERS} users")
    print()

    # Load article features
    sentiment = mind.load_sentiment(
        fpath=os.path.join(MIND_GENERATED_DATA_PATH, 'example_sentiment.json'))
    category = mind.load_category(
        fpath=os.path.join(MIND_GENERATED_DATA_PATH, 'example_category.json'))
    entities = mind.load_entities(
        fpath=os.path.join(MIND_GENERATED_DATA_PATH, 'example_party.json'), keep_empty=True)

    item_sentiment = mind.build(data=sentiment, id_map=None)
    item_category = mind.build(data=category, id_map=None)

    article_feature_dataframe = (
        pd.Series(item_sentiment).to_frame('sentiment')
        .join(pd.Series(item_category).to_frame('category'), how='outer')
        .join(pd.Series(entities).to_frame('party'), how='outer')
    )

    print(f"[OK] Feature dataframe shape: {article_feature_dataframe.shape}\n")
    return feedback_train, feedback_test, article_feature_dataframe


def compute_session_pci(rec_items, item_categories):
    """Compute PCI over just the recommended items (no history)."""
    categories = [item_categories[i] for i in rec_items if i in item_categories]
    if len(categories) == 0:
        return None
    counts = Counter(categories)
    total = len(categories)
    return sum((c / total) ** 2 for c in counts.values())


def run_session(model, user_indices, item_categories, item_feature):
    """
    Run one recommendation session for all users.

    Returns dict with:
      recs, avg_pci (cumulative), avg_session_pci (recs only),
      avg_ild, pci_triggered
    """
    cumulative_pci_values = []
    session_pci_values = []
    ild_values = []
    recs = {}
    pci_triggered = 0

    pci_metric = PCI(item_categories=item_categories, k=TOP_K_RECOMMEND)
    ild_metric = ILD(item_feature=item_feature, k=TOP_K_RECOMMEND)

    for user_idx in user_indices:
        try:
            ranked_items, scores = model.rank(user_idx)
            top_recs = list(ranked_items[:TOP_K_RECOMMEND])
            recs[user_idx] = top_recs

            # Cumulative PCI: recommendations + full user history
            user_history = model.train_set_dict.get(user_idx, [])
            cum_pci = pci_metric.compute(
                np.array(top_recs),
                user_history=np.array(user_history))
            if cum_pci is not None:
                cumulative_pci_values.append(cum_pci)

            # Session PCI: just the recommended items (shows per-session diversity)
            sess_pci = compute_session_pci(top_recs, item_categories)
            if sess_pci is not None:
                session_pci_values.append(sess_pci)

            # ILD
            ild_val = ild_metric.compute(np.array(top_recs))
            if ild_val is not None:
                ild_values.append(ild_val)

            # Check if PCI mechanism was triggered (based on session PCI,
            # not cumulative — avoids history-dilution silencing the trigger)
            if model.pci_config and sess_pci is not None:
                threshold = model.pci_config.get("threshold", 0.5)
                if sess_pci > threshold:
                    pci_triggered += 1

        except Exception:
            continue

    return {
        "recs": recs,
        "avg_pci": np.mean(cumulative_pci_values) if cumulative_pci_values else None,
        "avg_session_pci": np.mean(session_pci_values) if session_pci_values else None,
        "avg_ild": np.mean(ild_values) if ild_values else None,
        "pci_triggered": pci_triggered,
    }


def generate_report(results, output_dir):
    """Generate console report, CSV, and optional chart."""
    os.makedirs(output_dir, exist_ok=True)

    # Console table - Cumulative PCI
    print("\n" + "=" * 90)
    print("MULTI-SESSION PCI TREND REPORT")
    print("=" * 90)

    print(f"\n--- Cumulative PCI (recommendations + full history) ---")
    print(f"{'Session':>7} | {'D_RDW':>10} | {'D_RDW+PCI':>10} | {'Delta':>10} | {'Triggered':>10}")
    print("-" * 55)

    rows = []
    for i in range(len(results["D_RDW"])):
        base = results["D_RDW"][i]
        pci_m = results["D_RDW+PCI"][i]

        b_pci = base['avg_pci']
        p_pci = pci_m['avg_pci']
        delta = (p_pci - b_pci) if (b_pci is not None and p_pci is not None) else None

        b_str = f"{b_pci:>10.4f}" if b_pci is not None else f"{'N/A':>10}"
        p_str = f"{p_pci:>10.4f}" if p_pci is not None else f"{'N/A':>10}"
        d_str = f"{delta:>+10.4f}" if delta is not None else f"{'N/A':>10}"
        print(f"{i+1:>7} | {b_str} | {p_str} | {d_str} | {pci_m['pci_triggered']:>10}")

        rows.append({
            "session": i + 1,
            "drdw_cumulative_pci": b_pci,
            "drdw_pci_cumulative_pci": p_pci,
            "drdw_session_pci": base['avg_session_pci'],
            "drdw_pci_session_pci": pci_m['avg_session_pci'],
            "drdw_ild": base['avg_ild'],
            "drdw_pci_ild": pci_m['avg_ild'],
            "pci_triggered_users": pci_m['pci_triggered'],
        })

    print()

    # Session-only PCI table
    print(f"--- Session PCI (recommended items only, per session) ---")
    print(f"{'Session':>7} | {'D_RDW':>10} | {'D_RDW+PCI':>10} | {'Delta':>10}")
    print("-" * 45)
    for i in range(len(results["D_RDW"])):
        base = results["D_RDW"][i]
        pci_m = results["D_RDW+PCI"][i]
        b_spci = base['avg_session_pci']
        p_spci = pci_m['avg_session_pci']
        delta = (p_spci - b_spci) if (b_spci is not None and p_spci is not None) else None
        b_str = f"{b_spci:>10.4f}" if b_spci is not None else f"{'N/A':>10}"
        p_str = f"{p_spci:>10.4f}" if p_spci is not None else f"{'N/A':>10}"
        d_str = f"{delta:>+10.4f}" if delta is not None else f"{'N/A':>10}"
        print(f"{i+1:>7} | {b_str} | {p_str} | {d_str}")

    print()

    # ILD table
    print(f"--- ILD (Intra-List Diversity, higher = more diverse) ---")
    print(f"{'Session':>7} | {'D_RDW':>10} | {'D_RDW+PCI':>10} | {'Delta':>10}")
    print("-" * 45)
    for i in range(len(results["D_RDW"])):
        base = results["D_RDW"][i]
        pci_m = results["D_RDW+PCI"][i]
        b_ild = base['avg_ild']
        p_ild = pci_m['avg_ild']
        delta = (p_ild - b_ild) if (b_ild is not None and p_ild is not None) else None
        b_str = f"{b_ild:>10.4f}" if b_ild is not None else f"{'N/A':>10}"
        p_str = f"{p_ild:>10.4f}" if p_ild is not None else f"{'N/A':>10}"
        d_str = f"{delta:>+10.4f}" if delta is not None else f"{'N/A':>10}"
        print(f"{i+1:>7} | {b_str} | {p_str} | {d_str}")

    print()

    # Summary
    n = len(results["D_RDW"])
    if results["D_RDW"][0]['avg_pci'] and results["D_RDW"][-1]['avg_pci']:
        base_delta = results["D_RDW"][-1]['avg_pci'] - results["D_RDW"][0]['avg_pci']
        pci_delta = results["D_RDW+PCI"][-1]['avg_pci'] - results["D_RDW+PCI"][0]['avg_pci']
        print(f"Cumulative PCI change over {n} sessions:")
        print(f"  D_RDW baseline:  {base_delta:+.4f}")
        print(f"  D_RDW+PCI:       {pci_delta:+.4f}")
        if pci_delta < base_delta:
            print(f"  -> D_RDW+PCI reduces concentration by {abs(base_delta - pci_delta):.4f} more")
        print()

    if results["D_RDW"][0]['avg_session_pci'] and results["D_RDW"][-1]['avg_session_pci']:
        base_sdelta = results["D_RDW"][-1]['avg_session_pci'] - results["D_RDW"][0]['avg_session_pci']
        pci_sdelta = results["D_RDW+PCI"][-1]['avg_session_pci'] - results["D_RDW+PCI"][0]['avg_session_pci']
        print(f"Session PCI change over {n} sessions:")
        print(f"  D_RDW baseline:  {base_sdelta:+.4f}")
        print(f"  D_RDW+PCI:       {pci_sdelta:+.4f}")
        if pci_sdelta < base_sdelta:
            print(f"  -> D_RDW+PCI recommendations become more diverse per session")
        print()

    # Save CSV
    csv_path = os.path.join(output_dir, "pci_trends.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV saved to: {csv_path}")

    # Generate chart if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        import matplotlib.pyplot as plt

        sessions = list(range(1, n + 1))

        fig, axes_grid = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes_grid.flatten()

        # Cumulative PCI
        axes[0].plot(sessions, [r['avg_pci'] for r in results["D_RDW"]],
                     'o-', label='D_RDW (baseline)', color='#e74c3c', linewidth=2)
        axes[0].plot(sessions, [r['avg_pci'] for r in results["D_RDW+PCI"]],
                     's-', label='D_RDW+PCI', color='#2ecc71', linewidth=2)
        axes[0].set_xlabel('Session')
        axes[0].set_ylabel('Average Cumulative PCI')
        axes[0].set_title('Cumulative PCI Over Sessions\n(Lower = Less Concentrated)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Session PCI
        axes[1].plot(sessions, [r['avg_session_pci'] for r in results["D_RDW"]],
                     'o-', label='D_RDW (baseline)', color='#e74c3c', linewidth=2)
        axes[1].plot(sessions, [r['avg_session_pci'] for r in results["D_RDW+PCI"]],
                     's-', label='D_RDW+PCI', color='#2ecc71', linewidth=2)
        axes[1].set_xlabel('Session')
        axes[1].set_ylabel('Average Session PCI')
        axes[1].set_title('Per-Session PCI Over Sessions\n(Lower = More Diverse Recs)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # ILD
        axes[2].plot(sessions, [r['avg_ild'] for r in results["D_RDW"]],
                     'o-', label='D_RDW (baseline)', color='#e74c3c', linewidth=2)
        axes[2].plot(sessions, [r['avg_ild'] for r in results["D_RDW+PCI"]],
                     's-', label='D_RDW+PCI', color='#2ecc71', linewidth=2)
        axes[2].set_xlabel('Session')
        axes[2].set_ylabel('Average ILD')
        axes[2].set_title('ILD Over Sessions\n(Higher = More Diverse)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Triggered PCI users bar chart
        triggered_counts = [r['pci_triggered'] for r in results["D_RDW+PCI"]]
        bar_width = 0.6
        axes[3].bar(sessions, triggered_counts, width=bar_width, color='#3498db', alpha=0.8)
        for x, val in zip(sessions, triggered_counts):
            axes[3].text(x, val + 0.3, str(val), ha='center', va='bottom', fontsize=9)
        axes[3].set_xlabel('Session')
        axes[3].set_ylabel('Users with PCI Triggered')
        axes[3].set_title('PCI-Triggered Users per Session\n(D_RDW+PCI only)')
        axes[3].set_xticks(sessions)
        axes[3].grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "pci_trends.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Chart saved to: {chart_path}")

    except ImportError:
        print("[INFO] matplotlib not available, skipping chart generation")


def main():
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    run_dir, log_path = setup_logging(OUTPUT_DIR)

    start_time = datetime.now()
    print("=" * 70)
    print("MULTI-SESSION D-RDW+PCI SIMULATION")
    print("Demonstrating PCI echo chamber detection and correction over time")
    print(f"Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_path}")
    print("=" * 70)
    print()
    print("--- Parameter Settings ---")
    print(f"  Data:        MAX_USERS={MAX_USERS}  RANDOM_SEED={RANDOM_SEED}")
    print(f"  Simulation:  NUM_SIM_USERS={NUM_SIM_USERS}  NUM_SESSIONS={NUM_SESSIONS}")
    print(f"               TOP_K_RECOMMEND={TOP_K_RECOMMEND}  TOP_K_CLICK={TOP_K_CLICK}  MIN_USER_HISTORY={MIN_USER_HISTORY}")
    print(f"  Model:       DIVERSITY_DIMENSIONS={DIVERSITY_DIMENSIONS}  MAX_HOPS={MAX_HOPS}")
    print(f"               RANKING_TYPE={RANKING_TYPE}  SAMPLE_OBJECTIVE={SAMPLE_OBJECTIVE}")
    print(f"  PCI config:  threshold={PCI_CONFIG['threshold']}  penalty_weight={PCI_CONFIG['penalty_weight']}")
    print(f"               category_column={PCI_CONFIG['category_column']}  min_history={PCI_CONFIG['min_history']}")
    print("-" * 70)
    print()

    try:
        # ================================================================
        # Load data and create train/test split
        # ================================================================
        feedback_train, feedback_test, article_feature_dataframe = load_data()

        print("Creating train/test split...")
        rs = BaseMethod.from_splits(
            train_data=feedback_train,
            test_data=feedback_test,
            exclude_unknowns=False,
            verbose=False,
            rating_threshold=RATING_THRESHOLD,
        )
        print(f"[OK] Users: {rs.train_set.num_users}, Items: {rs.train_set.num_items}\n")

        # Re-index article dataframe to cornac internal indices
        iid_map = rs.train_set.iid_map
        idx_to_features = {}
        for orig_id, cornac_idx in iid_map.items():
            if orig_id in article_feature_dataframe.index:
                idx_to_features[cornac_idx] = article_feature_dataframe.loc[orig_id]
        article_feature_dataframe = pd.DataFrame.from_dict(idx_to_features, orient='index')
        article_feature_dataframe.sort_index(inplace=True)
        print(f"[OK] Re-indexed dataframe: {article_feature_dataframe.shape}\n")

        # ================================================================
        # Build feature dicts for metrics
        # ================================================================
        from sklearn.preprocessing import LabelEncoder

        item_categories = {}
        for idx, row in article_feature_dataframe.iterrows():
            if pd.notna(row['category']):
                item_categories[idx] = row['category']

        item_feature = {}
        cat_encoder = LabelEncoder()
        all_categories = article_feature_dataframe['category'].dropna().unique()
        cat_encoder.fit(all_categories)
        for idx, row in article_feature_dataframe.iterrows():
            sentiment_val = row['sentiment'] if pd.notna(row['sentiment']) else 0.0
            cat_val = cat_encoder.transform([row['category']])[0] if pd.notna(row['category']) else -1
            item_feature[idx] = np.array([sentiment_val, float(cat_val)])

        # ================================================================
        # Configure target distributions
        # ================================================================
        # Category distribution is uniform over categories found in the data;
        # the sentiment buckets come from SENTIMENT_DISTRIBUTION in config.
        num_categories = len(set(article_feature_dataframe['category'].dropna()))
        Target_distribution = {
            "sentiment": {
                "type": "continuous",
                "distr": SENTIMENT_DISTRIBUTION,
            },
            "category": {
                "type": "discrete",
                "distr": {
                    cat: 1.0 / num_categories
                    for cat in set(article_feature_dataframe['category'].dropna())
                },
            },
        }

        # ================================================================
        # Initialize models
        # ================================================================
        # Only diversify by sentiment (not category) so PCI's LP penalties
        # become the only force balancing category diversity. This lets us
        # clearly see PCI's effect: baseline has no category correction,
        # while D_RDW+PCI penalizes overrepresented categories in the LP.
        common_params = dict(
            item_dataframe=article_feature_dataframe,
            diversity_dimension=DIVERSITY_DIMENSIONS,
            target_distributions=Target_distribution,
            targetSize=TOP_K_RECOMMEND,
            maxHops=MAX_HOPS,
            filteringCriteria=None,
            rankingType=RANKING_TYPE,
            rankingObjectives=None,
            sampleObjective=SAMPLE_OBJECTIVE,
            verbose=False,
        )

        print("Initializing models...")
        drdw_baseline = D_RDW(name="D_RDW", pci_config=None, **common_params)
        drdw_pci = D_RDW(name="D_RDW+PCI", pci_config=PCI_CONFIG, **common_params)

        # ================================================================
        # Fit models on initial training data
        # ================================================================
        print("Fitting models on initial training data...")
        t0 = time.time()
        drdw_baseline.fit(rs.train_set)
        drdw_pci.fit(rs.train_set)
        print(f"[OK] Both models fitted in {time.time() - t0:.1f}s\n")

        # ================================================================
        # Select simulation users
        # ================================================================
        eligible_users = [
            u for u in drdw_baseline.train_set_dict
            if len(drdw_baseline.train_set_dict[u]) >= MIN_USER_HISTORY
        ]
        print(f"Eligible users (history >= {MIN_USER_HISTORY} items): {len(eligible_users)}")

        import random
        random.seed(RANDOM_SEED)
        if len(eligible_users) > NUM_SIM_USERS:
            sim_users = sorted(random.sample(eligible_users, NUM_SIM_USERS))
        else:
            sim_users = sorted(eligible_users)
        print(f"Simulation users: {len(sim_users)}")

        # Diagnostic: initial PCI distribution
        above_threshold = 0
        for u in sim_users:
            pci_val = drdw_pci.sampleRank.user_pci_values.get(u)
            if pci_val is not None and pci_val > PCI_CONFIG["threshold"]:
                above_threshold += 1
        print(f"Users with initial PCI > {PCI_CONFIG['threshold']}: "
              f"{above_threshold}/{len(sim_users)}")
        print()

        # ================================================================
        # Multi-session simulation loop
        # ================================================================
        results = {"D_RDW": [], "D_RDW+PCI": []}

        for session in range(NUM_SESSIONS):
            print(f"{'='*70}")
            print(f"SESSION {session + 1}/{NUM_SESSIONS}")
            print(f"{'='*70}")

            for model_name, model in [("D_RDW", drdw_baseline), ("D_RDW+PCI", drdw_pci)]:
                t0 = time.time()
                session_result = run_session(
                    model, sim_users, item_categories, item_feature)
                rec_time = time.time() - t0

                results[model_name].append(session_result)

                pci_str = f"{session_result['avg_pci']:.4f}" if session_result['avg_pci'] else "N/A"
                spci_str = f"{session_result['avg_session_pci']:.4f}" if session_result['avg_session_pci'] else "N/A"
                ild_str = f"{session_result['avg_ild']:.4f}" if session_result['avg_ild'] else "N/A"
                print(f"  {model_name:12s}: CumPCI={pci_str}  SessPCI={spci_str}  "
                      f"ILD={ild_str}  triggered={session_result['pci_triggered']}  "
                      f"({rec_time:.1f}s)")

                # Batch update: collect all clicks, then rebuild graph once
                t0 = time.time()
                updates = {}
                for user_idx, rec_list in session_result["recs"].items():
                    clicked = rec_list[:TOP_K_CLICK]
                    if clicked:
                        updates[user_idx] = clicked
                model.batch_update_history(updates)
                update_time = time.time() - t0

                print(f"  {'':12s}  Updated {len(updates)} users "
                      f"(+{TOP_K_CLICK} items each, rebuild {update_time:.1f}s)")

            print()

        # ================================================================
        # Generate report
        # ================================================================
        generate_report(results, run_dir)

        end_time = datetime.now()
        elapsed = end_time - start_time
        total_seconds = int(elapsed.total_seconds())
        h, rem = divmod(total_seconds, 3600)
        m, s = divmod(rem, 60)

        print("\n" + "=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {run_dir}")
        print(f"Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed : {h:02d}h {m:02d}m {s:02d}s")
        print()
        print("Interpretation:")
        print("  - Cumulative PCI: measures viewpoint concentration in recs + full history")
        print("  - Session PCI: measures diversity of just the current session's recommendations")
        print("  - If D_RDW+PCI session PCI is LOWER: PCI mechanism produces more diverse recs")
        print("  - If D_RDW+PCI cumulative PCI grows SLOWER: echo chamber is being mitigated")
        print()

        return True

    except FileNotFoundError as e:
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"\n[FAIL] {e}")
        print(f"Elapsed: {elapsed}")
        print("\nPlease run mind_to_drdw_preprocessor.py first.")
        return False
    except Exception as e:
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"\n[FAIL] Unexpected error: {e}")
        print(f"Elapsed: {elapsed}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
