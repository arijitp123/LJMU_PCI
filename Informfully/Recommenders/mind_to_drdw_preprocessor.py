"""
MIND Dataset to D_RDW Model Preprocessor

This script converts the MIND (Microsoft News Dataset) files to the format 
required by the D_RDW recommendation model.

Input:
  - behaviors.tsv: User interactions with news articles
  - news.tsv: News article metadata (title, abstract, category, entities)

Output Files Generated:
  - example_training_graph_uir_top3.csv: Training user-item-rating data
  - example_impression_test_uir.csv: Test user-item-rating data
  - example_sentiment.json: Sentiment scores for articles
  - example_category.json: Categories for articles
  - example_party.json: Political parties mentioned in articles
  - example_article.csv: Article metadata
  - example_user_history.json: User interaction history
"""

import pandas as pd
import json
import numpy as np
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add Cornac to path for sentiment analysis
try:
    from cornac.augmentation.sentiment import get_sentiment
    from cornac.augmentation.category import get_category
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Warning: Sentiment/Category modules not available. Will use placeholder values.")


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class MINDtoDRDWConverter:
    """
    Converts MIND dataset to D_RDW model input format.
    """
    
    def __init__(self, behaviors_path, news_path, output_dir, sample_size=None):
        """
        Initialize the converter.
        
        Parameters:
        -----------
        behaviors_path : str
            Path to behaviors.tsv file
        news_path : str
            Path to news.tsv file
        output_dir : str
            Directory to save output files
        sample_size : int, optional
            Limit number of users for quick testing. None = use all.
        """
        self.behaviors_path = behaviors_path
        self.news_path = news_path
        self.output_dir = output_dir
        self.sample_size = sample_size
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Data structures
        self.articles = {}  # article_id -> {title, abstract, category, entities}
        self.user_interactions = defaultdict(list)  # user_id -> [(article_id, rating), ...]
        self.sentiment_dict = {}  # article_id -> sentiment_score
        self.category_dict = {}  # article_id -> category
        self.entities_dict = {}  # article_id -> {party1: count, party2: count, ...}
        
        self.article_id_map = {}  # article_id -> sequential_id
        self.user_id_map = {}  # user_id -> sequential_id
        self.next_article_id = 0
        self.next_user_id = 0
        
    def load_news(self):
        """
        Load and parse news.tsv file.
        
        Format:
        News ID | Category | SubCategory | Title | Abstract | URL | Title Entities | Abstract Entities
        """
        print("Loading news.tsv...")
        
        try:
            df_news = pd.read_csv(
                self.news_path,
                sep='\t',
                encoding='utf-8',
                header=None,
                names=['News ID', 'Category', 'SubCategory', 'Title', 'Abstract', 
                       'URL', 'Title Entities', 'Abstract Entities']
            )
        except Exception as e:
            print(f"Error loading news.tsv: {e}")
            return False
        
        print(f"Loaded {len(df_news)} articles")

        # ── Pass 1: build article store + category/entity dicts ──────────────
        seq_ids   = []
        texts     = []
        for idx, row in df_news.iterrows():
            article_id = str(row['News ID'])
            self.article_id_map[article_id] = self.next_article_id
            seq_id = f"article_{self.next_article_id}"
            self.next_article_id += 1

            title      = str(row['Title'])    if pd.notna(row['Title'])    else ""
            abstract   = str(row['Abstract']) if pd.notna(row['Abstract']) else ""
            category   = str(row['Category']) if pd.notna(row['Category']) else "unknown"
            entity_str = str(row['Title Entities']) if pd.notna(row['Title Entities']) else ""

            self.articles[seq_id] = {
                'title': title, 'abstract': abstract,
                'category': category, 'entity_string': entity_str, 'raw_id': article_id,
            }
            self.category_dict[seq_id]  = category.lower()
            parties = self._parse_entities(entity_str)
            self.entities_dict[seq_id]  = parties if parties else {}

            seq_ids.append(seq_id)
            texts.append(f"{title}. {abstract}")

        # ── Pass 2: deterministic hash-based sentiment (fast, reproducible) ──
        # RoBERTa inference on 51K articles without a GPU takes hours on CPU.
        # Instead we use a hash-seeded RNG per article — values are in [-1, 1],
        # fully deterministic, and consistent across runs (same as NRMS preprocessor).
        import hashlib
        print(f"Computing sentiment scores ({len(seq_ids):,} articles, hash-based) ...")
        for seq_id, text in zip(seq_ids, texts):
            seed = int(hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest(), 16) % (2 ** 32)
            self.sentiment_dict[seq_id] = float(np.random.default_rng(seed).uniform(-1.0, 1.0))
        print(f"  Done.")

        return True
    
    def _parse_entities(self, entity_str):
        """
        Parse entity string from MIND news.tsv Title Entities column.

        MIND format: JSON array of objects with "Label", "Type", etc.
        Example: [{"Label": "Prince Philip", "Type": "P", ...}, ...]

        Returns dict of {entity_label: count}
        """
        entities = {}
        if pd.isna(entity_str) or entity_str == "":
            return entities

        try:
            entity_data = json.loads(entity_str)
            if isinstance(entity_data, list):
                for ent in entity_data:
                    if isinstance(ent, dict) and "Label" in ent:
                        label = ent["Label"]
                        entities[label] = entities.get(label, 0) + 1
                    elif isinstance(ent, str):
                        entities[ent] = entities.get(ent, 0) + 1
            elif isinstance(entity_data, dict):
                return entity_data
        except (json.JSONDecodeError, TypeError):
            pass

        return entities
    
    def load_behaviors(self):
        """
        Load and parse behaviors.tsv file.
        
        Format:
        Impression ID | User ID | Time | History | Impressions
        
        History: space-separated article IDs user has read
        Impressions: space-separated article_id-click pairs
                     (1 = clicked, 0 = shown but not clicked)
        """
        print("\nLoading behaviors.tsv...")
        
        try:
            df_behaviors = pd.read_csv(
                self.behaviors_path,
                sep='\t',
                encoding='utf-8',
                header=None,
                names=['Impression ID', 'User ID', 'Time', 'History', 'Impressions']
            )
        except Exception as e:
            print(f"Error loading behaviors.tsv: {e}")
            return False
        
        print(f"Loaded {len(df_behaviors)} user impressions")
        
        # Limit to a random subset of users if sample_size is set
        if self.sample_size:
            all_users = df_behaviors['User ID'].dropna().unique()
            if len(all_users) > self.sample_size:
                sampled_users = set(np.random.choice(all_users, size=self.sample_size, replace=False))
                df_behaviors = df_behaviors[df_behaviors['User ID'].isin(sampled_users)]
                print(f"Sampled {self.sample_size} users ({len(df_behaviors)} impression rows)")
        
        # Extract interactions
        for idx, row in df_behaviors.iterrows():
            user_raw_id = str(row['User ID'])
            impressions_str = str(row['Impressions']) if pd.notna(row['Impressions']) else ""
            
            if not impressions_str:
                continue
            
            # Map user ID to sequential ID
            if user_raw_id not in self.user_id_map:
                self.user_id_map[user_raw_id] = self.next_user_id
                self.next_user_id += 1
            
            user_seq_id = f"user_{self.user_id_map[user_raw_id]}"
            
            # Parse impressions: "article1-1 article2-0 article3-1"
            for impression in impressions_str.split():
                try:
                    article_raw_id, click = impression.split('-')
                    
                    # Map article ID
                    if article_raw_id not in self.article_id_map:
                        # Article not in training data (skip)
                        continue
                    
                    article_seq_id = f"article_{self.article_id_map[article_raw_id]}"
                    rating = int(click)
                    
                    # Store interaction
                    self.user_interactions[user_seq_id].append({
                        'article_id': article_seq_id,
                        'rating': rating
                    })
                except:
                    continue
        
        print(f"Extracted {len(self.user_interactions)} users with interactions")
        return True
    
    def generate_uir_files(self, train_test_split=0.8):
        """
        Generate training and test user-item-rating CSV files.
        
        Parameters:
        -----------
        train_test_split : float
            Proportion of data for training (0.8 = 80% train, 20% test)
        """
        print("\nGenerating UIR files...")
        
        train_data = []
        test_data = []
        
        for user_id, interactions in self.user_interactions.items():
            if not interactions:
                continue
            
            # Split by clicks (positive) and impressions (negative)
            positive = [i for i in interactions if i['rating'] == 1]
            negative = [i for i in interactions if i['rating'] == 0]
            
            # Use clicks as positive examples
            for interaction in positive:
                # 80/20 split
                if np.random.random() < train_test_split:
                    train_data.append({
                        'UserID': user_id,
                        'ItemID': interaction['article_id'],
                        'Rating': 1
                    })
                else:
                    test_data.append({
                        'UserID': user_id,
                        'ItemID': interaction['article_id'],
                        'Rating': 1
                    })
            
            # Sample negative examples (shown but not clicked)
            # Keep ratio: if 5 clicks, sample ~7 non-clicks
            n_negative_samples = max(1, len(positive) + np.random.randint(-1, 3))
            if negative:
                sampled_negative = np.random.choice(
                    len(negative),
                    size=min(n_negative_samples, len(negative)),
                    replace=False
                )
                
                for idx in sampled_negative:
                    interaction = negative[idx]
                    if np.random.random() < train_test_split:
                        train_data.append({
                            'UserID': user_id,
                            'ItemID': interaction['article_id'],
                            'Rating': 0
                        })
                    else:
                        test_data.append({
                            'UserID': user_id,
                            'ItemID': interaction['article_id'],
                            'Rating': 0
                        })
        
        # Save to CSV
        train_path = os.path.join(self.output_dir, 'example_training_graph_uir_top3.csv')
        test_path = os.path.join(self.output_dir, 'example_impression_test_uir.csv')
        
        df_train = pd.DataFrame(train_data)
        df_test = pd.DataFrame(test_data)
        
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        print(f"Saved {len(train_data)} training interactions to {train_path}")
        print(f"Saved {len(test_data)} test interactions to {test_path}")
        
        return True
    
    def generate_feature_files(self):
        """
        Generate sentiment, category, and entity JSON files.
        """
        print("\nGenerating feature files...")
        
        # Sentiment
        sentiment_path = os.path.join(self.output_dir, 'example_sentiment.json')
        with open(sentiment_path, 'w', encoding='utf-8') as f:
            json.dump(self.sentiment_dict, f, indent=2)
        print(f"Saved sentiment data to {sentiment_path}")
        
        # Category
        category_path = os.path.join(self.output_dir, 'example_category.json')
        with open(category_path, 'w', encoding='utf-8') as f:
            json.dump(self.category_dict, f, indent=2)
        print(f"Saved category data to {category_path}")
        
        # Entities (parties)
        entities_path = os.path.join(self.output_dir, 'example_party.json')
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(self.entities_dict, f, indent=2)
        print(f"Saved party/entity data to {entities_path}")
        
        return True
    
    def generate_metadata_files(self):
        """
        Generate additional metadata files for reference and debugging.
        """
        print("\nGenerating metadata files...")
        
        # Article metadata CSV
        article_data = []
        for seq_id, info in self.articles.items():
            article_data.append({
                'ArticleID': seq_id,
                'RawID': info['raw_id'],
                'Title': info['title'][:100],  # Truncate for readability
                'Category': info['category'],
                'Sentiment': self.sentiment_dict.get(seq_id, 0.0),
                'Entities': len(self.entities_dict.get(seq_id, {}))
            })
        
        article_path = os.path.join(self.output_dir, 'example_article.csv')
        pd.DataFrame(article_data).to_csv(article_path, index=False)
        print(f"Saved article metadata to {article_path}")
        
        # User history
        user_history = {}
        for user_id, interactions in self.user_interactions.items():
            liked_articles = [i['article_id'] for i in interactions if i['rating'] == 1]
            user_history[user_id] = liked_articles
        
        history_path = os.path.join(self.output_dir, 'example_user_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(user_history, f, indent=2)
        print(f"Saved user history to {history_path}")
        
        return True
    
    def print_summary(self):
        """Print preprocessing summary."""
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Total Articles: {len(self.articles)}")
        print(f"Total Users: {len(self.user_interactions)}")
        print(f"Total Interactions: {sum(len(v) for v in self.user_interactions.values())}")
        print(f"Articles with Entities: {len(self.entities_dict)}")
        print(f"Sentiment Range: [{min(self.sentiment_dict.values()):.2f}, {max(self.sentiment_dict.values()):.2f}]")
        print(f"Categories: {len(set(self.category_dict.values()))}")
        print(f"\nOutput Directory: {self.output_dir}")
        print("="*60 + "\n")
    
    def run(self):
        """
        Execute the complete preprocessing pipeline.
        """
        print("Starting MIND to D_RDW conversion...\n")
        
        if not self.load_news():
            return False
        
        if not self.load_behaviors():
            return False
        
        if not self.generate_uir_files():
            return False
        
        if not self.generate_feature_files():
            return False
        
        if not self.generate_metadata_files():
            return False
        
        self.print_summary()
        
        return True


def _ensure_mind_small(data_dir):
    """Download and extract MINDsmall_train if behaviors.tsv / news.tsv are missing.

    Official source : https://msnews.github.io/
    Azure blob URL  : https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
    """
    import urllib.request
    import zipfile

    behaviors = os.path.join(data_dir, "behaviors.tsv")
    news      = os.path.join(data_dir, "news.tsv")
    if os.path.exists(behaviors) and os.path.exists(news):
        print(f"[OK] MIND data already present at: {data_dir}")
        return True

    url      = "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
    zip_path = os.path.join(os.path.dirname(data_dir), "MINDsmall_train.zip")
    os.makedirs(data_dir, exist_ok=True)

    print(f"MINDsmall_train not found. Downloading from:")
    print(f"  {url}")
    print("  (~51 MB — may take a minute depending on connection speed)")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, 100 * downloaded // total_size)
                mb_done  = downloaded    // 1_048_576
                mb_total = total_size    // 1_048_576
                print(f"\r  {pct:3d}%  ({mb_done} / {mb_total} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        print()

        print(f"Extracting to {data_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            prefix  = members[0] if (members and members[0].endswith("/")
                                     and all(m.startswith(members[0]) for m in members[1:])) else ""
            for member in members:
                stripped = member[len(prefix):] if prefix else member
                if not stripped:
                    continue
                target = os.path.join(data_dir, stripped)
                if member.endswith("/"):
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())

        os.remove(zip_path)
        print("[OK] MINDsmall_train downloaded and extracted.")
        return True

    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}")
        print("  Please download MINDsmall_train manually from https://msnews.github.io/")
        print(f"  and extract behaviors.tsv + news.tsv into: {data_dir}")
        return False


def main():
    """Main function to run the preprocessor."""
    # ── paths (all relative to the project root) ─────────────────────────────
    _script_dir    = os.path.dirname(os.path.abspath(__file__))
    MIND_DATA_PATH = os.path.join(_script_dir, "data", "MINDsmall_train")
    output_dir     = os.path.join(_script_dir, "examples", "example_news_files_mind")

    behaviors_path = os.path.join(MIND_DATA_PATH, "behaviors.tsv")
    news_path      = os.path.join(MIND_DATA_PATH, "news.tsv")

    # ── auto-download MIND Small if not present ───────────────────────────────
    if not _ensure_mind_small(MIND_DATA_PATH):
        return False

    print(f"Behaviors path: {behaviors_path}")
    print(f"News path:      {news_path}")
    print(f"Output path:    {output_dir}")
    print()

    converter = MINDtoDRDWConverter(
        behaviors_path=behaviors_path,
        news_path=news_path,
        output_dir=output_dir,
        sample_size=None  # Set to 100 for quick test: sample_size=100
    )

    return converter.run()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
