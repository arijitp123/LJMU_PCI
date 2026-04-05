"""
D_RDW Model Example using MIND Dataset — PCI Ablation Study

This script runs an ablation study comparing:
  1. D_RDW baseline (static diversity via target distributions)
  2. D_RDW + PCI adaptive mechanism (dynamic diversity correction)

Both models are evaluated with accuracy and diversity metrics including the
Perspective Concentration Index (PCI).

Usage:
    1. First run: python mind_to_drdw_preprocessor.py
       (This generates the input files from actual MIND data)

    2. Then run: python drdw_example_mind.py
       (This trains and evaluates both models on the generated data)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cornac
from cornac.eval_methods import BaseMethod
from cornac.metrics import Precision, Recall, NDCG, AUC, HitRatio, ILD, PCI, GiniCoeff
from cornac.models import D_RDW
from cornac.datasets import mind
import pandas as pd
import numpy as np
import json
import logging
import time
from datetime import datetime

# Configuration
MIND_GENERATED_DATA_PATH = os.path.join(os.path.dirname(__file__), 'example_news_files_mind')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_drdw_mind_result')
MAX_USERS = 10000  # Limit users to avoid memory issues with graph operations
TOP_K = 10        # Cutoff for all ranking metrics (must be <= targetSize)
RATING_THRESHOLD = 0.5  # Minimum rating to count as a positive interaction


def setup_logging():
    """Set up logging to both console and a timestamped log file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_filename = os.path.join(
        OUTPUT_DIR,
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter('%(message)s')

    # File handler — captures everything
    fh = logging.FileHandler(log_filename, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — mirrors to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Redirect bare print() calls through the logger
    class _PrintToLogger:
        def __init__(self, orig): self._orig = orig
        def write(self, msg):
            if msg.rstrip():
                logging.info(msg.rstrip())
        def flush(self): self._orig.flush()
        def fileno(self): return self._orig.fileno()

    sys.stdout = _PrintToLogger(sys.stdout)

    return log_filename


def load_data():
    """
    Load preprocessed MIND data from generated files.

    Returns:
    --------
    tuple: (train_feedback, test_feedback, article_dataframe)
    """
    print("Loading preprocessed MIND data...\n")

    # Validate input directory
    if not os.path.exists(MIND_GENERATED_DATA_PATH):
        raise FileNotFoundError(
            f"Data directory not found: {MIND_GENERATED_DATA_PATH}\n"
            f"Please run mind_to_drdw_preprocessor.py first."
        )

    # Load training feedback
    train_uir_path = os.path.join(MIND_GENERATED_DATA_PATH, 'example_training_graph_uir_top3.csv')
    if not os.path.exists(train_uir_path):
        raise FileNotFoundError(f"Training file not found: {train_uir_path}")

    feedback_train = mind.load_feedback(fpath=train_uir_path)
    print(f"[OK] Loaded {len(feedback_train)} training interactions")

    # Load test feedback
    test_uir_path = os.path.join(MIND_GENERATED_DATA_PATH, 'example_impression_test_uir.csv')
    if not os.path.exists(test_uir_path):
        raise FileNotFoundError(f"Test file not found: {test_uir_path}")

    feedback_test = mind.load_feedback(fpath=test_uir_path)
    print(f"[OK] Loaded {len(feedback_test)} test interactions")

    # Subsample to limit memory usage for graph operations
    all_train_users = sorted(set(uid for uid, _, _ in feedback_train))
    if len(all_train_users) > MAX_USERS:
        import random as rng
        rng.seed(42)
        selected_users = set(rng.sample(all_train_users, MAX_USERS))
        feedback_train = [(u, i, r) for u, i, r in feedback_train if u in selected_users]
        feedback_test = [(u, i, r) for u, i, r in feedback_test if u in selected_users]
        print(f"[OK] Subsampled to {MAX_USERS} users: {len(feedback_train)} train, {len(feedback_test)} test interactions")
    print()

    # Load article features
    print("Loading article features...")

    sentiment_file_path = os.path.join(MIND_GENERATED_DATA_PATH, 'example_sentiment.json')
    sentiment = mind.load_sentiment(fpath=sentiment_file_path)
    print(f"[OK] Loaded sentiment for {len(sentiment)} articles")

    category_file_path = os.path.join(MIND_GENERATED_DATA_PATH, 'example_category.json')
    category = mind.load_category(fpath=category_file_path)
    print(f"[OK] Loaded categories for {len(category)} articles")

    party_file_path = os.path.join(MIND_GENERATED_DATA_PATH, 'example_party.json')
    entities = mind.load_entities(fpath=party_file_path, keep_empty=True)
    print(f"[OK] Loaded entities for {len(entities)} articles\n")

    # Build feature dataframe
    item_sentiment = mind.build(data=sentiment, id_map=None)
    item_category = mind.build(data=category, id_map=None)

    article_feature_dataframe = (
        pd.Series(item_sentiment).to_frame('sentiment')
        .join(pd.Series(item_category).to_frame('category'), how='outer')
        .join(pd.Series(entities).to_frame('party'), how='outer')
    )

    print(f"Feature dataframe shape: {article_feature_dataframe.shape}")
    print(f"Columns: {list(article_feature_dataframe.columns)}\n")

    return feedback_train, feedback_test, article_feature_dataframe


def main():
    """
    Main function: PCI ablation study comparing D_RDW vs D_RDW+PCI.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    log_file = setup_logging()

    start_time = time.time()
    start_dt   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("="*70)
    print("D_RDW MODEL - PCI ABLATION STUDY (MIND DATASET)")
    print("="*70)
    print(f"Start time : {start_dt}")
    print(f"Log file   : {log_file}")
    print()

    try:
        # Load data
        feedback_train, feedback_test, article_feature_dataframe = load_data()

        # Create train/test split
        print("Creating train/test split...")
        rs = BaseMethod.from_splits(
            train_data=feedback_train,
            test_data=feedback_test,
            exclude_unknowns=False,
            verbose=False,
            rating_threshold=RATING_THRESHOLD
        )
        print(f"[OK] Train set size: {rs.train_set.num_ratings}")
        print(f"[OK] Test set size: {rs.test_set.num_ratings}")
        print(f"[OK] Users: {rs.train_set.num_users}")
        print(f"[OK] Items: {rs.train_set.num_items}\n")

        # Re-index article dataframe to use cornac internal item indices
        iid_map = rs.train_set.iid_map  # maps original ID -> cornac int index
        idx_to_features = {}
        for orig_id, cornac_idx in iid_map.items():
            if orig_id in article_feature_dataframe.index:
                idx_to_features[cornac_idx] = article_feature_dataframe.loc[orig_id]
        article_feature_dataframe = pd.DataFrame.from_dict(idx_to_features, orient='index')
        article_feature_dataframe.sort_index(inplace=True)
        print(f"[OK] Re-indexed dataframe to cornac indices: {article_feature_dataframe.shape}\n")

        # Display sentiment statistics
        print("Sentiment statistics:")
        sentiment_values = [v for v in article_feature_dataframe['sentiment'].dropna()]
        if sentiment_values:
            print(f"  Range: [{min(sentiment_values):.2f}, {max(sentiment_values):.2f}]")
            print(f"  Mean: {sum(sentiment_values)/len(sentiment_values):.2f}")
        print()

        # Define target distributions for diversity
        print("Configuring diversity targets...")
        Target_distribution = {
            "sentiment": {
                "type": "continuous",
                "distr": [
                    {"min": -1.0, "max": -0.5, "prob": 0.1},    # Very negative
                    {"min": -0.5, "max":  0.0, "prob": 0.2},    # Negative
                    {"min":  0.0, "max":  0.5, "prob": 0.3},    # Positive
                    {"min":  0.5, "max":  1.01, "prob": 0.4}     # Very positive
                ]
            } #,
            # "category": {
            #     "type": "discrete",
            #     "distr": {
            #         cat: 1.0 / len(set(article_feature_dataframe['category'].dropna()))
            #         for cat in set(article_feature_dataframe['category'].dropna())
            #     }
            # }
        }
        print(f"[OK] Configured sentiment distribution")
        # print(f"[OK] Configured category distribution")
        print()

        # ================================================================
        # MODEL 1: D_RDW Baseline (no PCI mechanism)
        # ================================================================
        print("Initializing Model 1: D_RDW baseline...")
        drdw_baseline = D_RDW(
            name="D_RDW",
            item_dataframe=article_feature_dataframe,
            diversity_dimension=["sentiment"],  # category removed — not in target_distributions
            target_distributions=Target_distribution,
            targetSize=TOP_K,
            maxHops=5,
            filteringCriteria=None,
            rankingType="graph_coloring",
            rankingObjectives=["category"],
            sampleObjective="rdw_score",
            verbose=True,
        )
        print("[OK] D_RDW baseline initialized")

        # ================================================================
        # MODEL 2: D_RDW + PCI adaptive mechanism
        # ================================================================
        print("Initializing Model 2: D_RDW + PCI...")
        pci_config = {
            "enabled": True,
            "threshold": 0.35,            # tau: trigger when PCI > 0.5
            "category_column": "category",
            "penalty_weight": 0.01,        # LP penalty multiplier for overrepresented categories
            "min_history": 5,                # minimum history length to compute PCI
        }
        drdw_pci = D_RDW(
            name="D_RDW+PCI",
            item_dataframe=article_feature_dataframe,
            diversity_dimension=["sentiment"],  # category removed — PCI is the sole category diversity driver
            target_distributions=Target_distribution,
            targetSize=TOP_K,
            maxHops=5,
            filteringCriteria=None,
            rankingType="graph_coloring",
            rankingObjectives=["category"],
            sampleObjective="rdw_score",
            pci_config=pci_config,
            verbose=True,
        )
        print("[OK] D_RDW+PCI initialized\n")

        # ================================================================
        # Build feature dicts for evaluation metrics
        # ================================================================
        from sklearn.preprocessing import LabelEncoder

        # Item feature vectors for ILD (sentiment + encoded category)
        item_feature = {}
        cat_encoder = LabelEncoder()
        all_categories = article_feature_dataframe['category'].dropna().unique()
        cat_encoder.fit(all_categories)
        for idx, row in article_feature_dataframe.iterrows():
            sentiment_val = row['sentiment'] if pd.notna(row['sentiment']) else 0.0
            cat_val = cat_encoder.transform([row['category']])[0] if pd.notna(row['category']) else -1
            item_feature[idx] = np.array([sentiment_val, float(cat_val)])

        # Item categories dict for PCI metric
        item_categories = {}
        for idx, row in article_feature_dataframe.iterrows():
            if pd.notna(row['category']):
                item_categories[idx] = row['category']

        # Item genre one-hot vectors for Gini coefficient
        item_genre = {}
        sorted_cats = sorted(all_categories)
        cat_to_idx = {cat: i for i, cat in enumerate(sorted_cats)}
        for idx, row in article_feature_dataframe.iterrows():
            genre_vec = np.zeros(len(sorted_cats))
            if pd.notna(row['category']) and row['category'] in cat_to_idx:
                genre_vec[cat_to_idx[row['category']]] = 1
            item_genre[idx] = genre_vec

        # ================================================================
        # Evaluation metrics
        # ================================================================
        print("Setting up evaluation metrics...")
        metrics = [
            Precision(k=TOP_K), Recall(k=TOP_K), NDCG(k=TOP_K),
            AUC(),
            HitRatio(k=TOP_K),
            ILD(item_feature=item_feature, k=TOP_K),
            PCI(item_categories=item_categories, k=TOP_K),
            GiniCoeff(item_genre=item_genre, k=TOP_K),
        ]
        print(f"[OK] Metrics: Precision@{TOP_K}, Recall@{TOP_K}, NDCG@{TOP_K}, AUC, HitRatio@{TOP_K}, ILD@{TOP_K}, PCI@{TOP_K}, GiniCoeff@{TOP_K}\n")

        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ================================================================
        # Display all vital test parameters before running
        # ================================================================
        print("="*70)
        print("TEST PARAMETERS SUMMARY")
        print("="*70)
        print(f"\n[Data]")
        print(f"  Data path          : {MIND_GENERATED_DATA_PATH}")
        print(f"  Max users          : {MAX_USERS}")
        print(f"  Top-K cutoff       : {TOP_K}")
        print(f"  Train interactions : {rs.train_set.num_ratings}")
        print(f"  Test interactions  : {rs.test_set.num_ratings}")
        print(f"  Users              : {rs.train_set.num_users}")
        print(f"  Items              : {rs.train_set.num_items}")
        print(f"  Rating threshold   : {RATING_THRESHOLD}")
        print(f"  Exclude unknowns   : False")

        print(f"\n[Model 1 — D_RDW Baseline]")
        print(f"  name               : {drdw_baseline.name}")
        print(f"  diversity_dimension: {drdw_baseline.diversity_dimension}")
        print(f"  targetSize         : {drdw_baseline.targetSize}")
        print(f"  maxHops            : {drdw_baseline.maxHops}")
        print(f"  rankingType        : {drdw_baseline.rankingType}")
        print(f"  rankingObjectives  : {drdw_baseline.rankingObjectives}")
        print(f"  sampleObjective    : {drdw_baseline.sampleObjective}")
        print(f"  filteringCriteria  : {drdw_baseline.filteringCriteria}")
        print(f"  pci_config         : disabled")

        print(f"\n[Model 2 — D_RDW + PCI]")
        print(f"  name               : {drdw_pci.name}")
        print(f"  diversity_dimension: {drdw_pci.diversity_dimension}")
        print(f"  targetSize         : {drdw_pci.targetSize}")
        print(f"  maxHops            : {drdw_pci.maxHops}")
        print(f"  rankingType        : {drdw_pci.rankingType}")
        print(f"  rankingObjectives  : {drdw_pci.rankingObjectives}")
        print(f"  sampleObjective    : {drdw_pci.sampleObjective}")
        print(f"  filteringCriteria  : {drdw_pci.filteringCriteria}")
        print(f"  pci_config:")
        print(f"    enabled          : {pci_config['enabled']}")
        print(f"    threshold (tau)  : {pci_config['threshold']}")
        print(f"    category_column  : {pci_config['category_column']}")
        print(f"    penalty_weight   : {pci_config['penalty_weight']}")
        print(f"    min_history      : {pci_config['min_history']}")

        print(f"\n[Target Distributions]")
        print(f"  sentiment (continuous):")
        for bucket in Target_distribution["sentiment"]["distr"]:
            print(f"    [{bucket['min']:+.1f}, {bucket['max']:+.2f}) -> prob={bucket['prob']}")
        # print(f"  category (discrete): uniform over ... categories ...")  # category distribution disabled

        print(f"\n[Evaluation Metrics]")
        print(f"  Precision@{TOP_K}, Recall@{TOP_K}, NDCG@{TOP_K}, AUC, "
              f"HitRatio@{TOP_K}, ILD@{TOP_K}, PCI@{TOP_K}, GiniCoeff@{TOP_K}")
        print(f"  user_based         : True")

        print(f"\n[Output]")
        print(f"  Results saved to   : {OUTPUT_DIR}")
        print()

        # ================================================================
        # Run ablation experiment: both models side by side
        # ================================================================
        print("="*70)
        print("RUNNING ABLATION EXPERIMENT: D_RDW vs D_RDW+PCI")
        print("="*70 + "\n")

        cornac.Experiment(
            eval_method=rs,
            models=[drdw_baseline, drdw_pci],
            metrics=metrics,
            user_based=True,
            save_dir=OUTPUT_DIR
        ).run()

        elapsed = time.time() - start_time
        end_dt  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print("\n" + "="*70)
        print("ABLATION EXPERIMENT COMPLETED")
        print("="*70)
        print(f"Start time : {start_dt}")
        print(f"End time   : {end_dt}")
        print(f"Elapsed    : {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Results saved to: {OUTPUT_DIR}")
        print()
        print("Expected outcome:")
        print("  - D_RDW+PCI should have LOWER PCI (less viewpoint concentration)")
        print("  - D_RDW+PCI should have HIGHER ILD (more intra-list diversity)")
        print("  - Accuracy metrics may decrease slightly (diversity-accuracy tradeoff)")
        print()

        return True

    except FileNotFoundError as e:
        print(f"\n[FAIL] Error: {e}")
        print("\nTo fix this issue:")
        print("1. Run: python mind_to_drdw_preprocessor.py")
        print("2. This will generate the required data files")
        print("3. Then run this script again")
        return False
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
