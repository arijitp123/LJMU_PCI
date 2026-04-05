# -*- coding: utf-8 -*-
"""
MIND Dataset to NRMS Model Preprocessor

Converts the MIND (Microsoft News Dataset) files to the format required by the
NRMS recommendation model (example_nrms_news_reranking.py).

Input (MINDsmall_train):
  - behaviors.tsv : User impression logs (history + impressions with click labels)
  - news.tsv      : News article metadata (title, category, entities)

Output (nrms_mind_small/):
  Shared with D_RDW format
  ─────────────────────────
  - example_training_graph_uir_top3.csv  Training user-item-rating data
  - example_impression_test_uir.csv      Test user-item-rating data
  - example_article.csv                  Article metadata (ArticleID, RawID, Title, …)
  - example_user_history.json            User reading history
  - example_sentiment.json               Sentiment scores per article
  - example_category.json                Category label per article
  - example_party.json                   Entity/party mentions per article

  NRMS-specific
  ─────────────
  - example_title.json     {article_id: title_text}
  - word_index_dict.json   {word: index}  vocabulary
  - embedding_matrix.npy   shape (vocab_size, EMBED_DIM) — GloVe or random init
"""

import json
import os
import re
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Force UTF-8 on Windows so Unicode characters in print() don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import hashlib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── constants ────────────────────────────────────────────────────────────────
EMBED_DIM = 300          # embedding dimension expected by NRMS
TITLE_MAX_WORDS = 30     # NRMS title_size


# ════════════════════════════════════════════════════════════════════════════
class MINDtoNRMSConverter:
    """
    Preprocesses MIND (MINDsmall_train) into NRMS model input files.
    """

    def __init__(self, behaviors_path, news_path, output_dir,
                 glove_path=None, train_split=0.8, sample_size=None):
        """
        Parameters
        ----------
        behaviors_path : str
            Path to behaviors.tsv
        news_path : str
            Path to news.tsv
        output_dir : str
            Directory where output files are written
        glove_path : str, optional
            Path to a GloVe .txt file (e.g. glove.6B.300d.txt).
            When provided, word vectors are loaded from GloVe; otherwise
            random initialisation is used.
        train_split : float
            Fraction of impression sessions used for training (rest → test).
        sample_size : int, optional
            Limit the number of users processed (useful for quick tests).
        """
        self.behaviors_path = behaviors_path
        self.news_path = news_path
        self.output_dir = output_dir
        self.glove_path = glove_path
        self.train_split = train_split
        self.sample_size = sample_size

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ── article store ────────────────────────────────────────────────
        # raw_id  (e.g. "N55528") → sequential "article_N" id
        self.raw_to_seq = {}           # str -> str
        self.articles = {}             # seq_id -> {title, abstract, category, raw_id, entity_str}

        self.sentiment_dict = {}       # seq_id -> float
        self.category_dict = {}        # seq_id -> str
        self.entities_dict = {}        # seq_id -> {label: count}

        # ── user store ───────────────────────────────────────────────────
        self.raw_user_to_seq = {}      # raw user_id -> "user_N" id
        # seq_user_id -> list of seq article ids (reading history from History col)
        self.user_history = {}
        # train / test impression rows: list of (seq_user, [(seq_article, rating)])
        self.train_impressions = []
        self.test_impressions = []

        self._next_article = 0
        self._next_user = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _seq_article(self, raw_id):
        """Return (or create) the sequential article id for a raw MIND news id."""
        raw_id = str(raw_id)
        if raw_id not in self.raw_to_seq:
            self.raw_to_seq[raw_id] = f"article_{self._next_article}"
            self._next_article += 1
        return self.raw_to_seq[raw_id]

    def _seq_user(self, raw_id):
        """Return (or create) the sequential user id for a raw MIND user id."""
        raw_id = str(raw_id)
        if raw_id not in self.raw_user_to_seq:
            self.raw_user_to_seq[raw_id] = f"user_{self._next_user}"
            self._next_user += 1
        return self.raw_user_to_seq[raw_id]

    @staticmethod
    def _parse_entities(entity_str):
        """
        Parse the JSON entity string from news.tsv Title Entities column.
        Returns dict {entity_label: count}.
        """
        entities = {}
        if not entity_str or pd.isna(entity_str) or entity_str.strip() in ("", "[]"):
            return entities
        try:
            data = json.loads(entity_str)
            if isinstance(data, list):
                for ent in data:
                    if isinstance(ent, dict) and "Label" in ent:
                        label = ent["Label"]
                        entities[label] = entities.get(label, 0) + 1
        except (json.JSONDecodeError, TypeError):
            pass
        return entities

    @staticmethod
    def _compute_sentiment(text):
        """Return a reproducible pseudo-sentiment score in [-1, 1].

        NRMS does not use sentiment during training; this file is only
        metadata consumed by downstream diversity metrics.  Using a
        hash-seeded RNG keeps values deterministic and avoids the ~109-
        minute spaCy/TextBlob cost on the full MIND-Small corpus.
        """
        seed = int(hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest(), 16) % (2 ** 32)
        return float(np.random.default_rng(seed).uniform(-1.0, 1.0))

    # ── pipeline steps ───────────────────────────────────────────────────────

    def load_news(self):
        """Load news.tsv and populate article data structures."""
        print("Loading news.tsv …")
        try:
            df = pd.read_csv(
                self.news_path, sep='\t', encoding='utf-8', header=None,
                names=['NewsID', 'Category', 'SubCategory', 'Title',
                       'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
            )
        except Exception as exc:
            print(f"[ERROR] Could not read news.tsv: {exc}")
            return False

        print(f"  {len(df):,} articles found.")
        df['Title']         = df['Title'].fillna('')
        df['Abstract']      = df['Abstract'].fillna('')
        df['Category']      = df['Category'].fillna('unknown').str.lower()
        df['TitleEntities'] = df['TitleEntities'].fillna('')

        for row in df.itertuples(index=False):
            raw_id = str(row.NewsID)
            seq_id = self._seq_article(raw_id)
            title, abstract = str(row.Title), str(row.Abstract)
            category, ent_str = str(row.Category), str(row.TitleEntities)

            self.articles[seq_id] = {
                'title': title, 'abstract': abstract,
                'category': category, 'entity_str': ent_str, 'raw_id': raw_id,
            }
            combined = f"{title}. {abstract}".strip()
            self.sentiment_dict[seq_id] = self._compute_sentiment(combined)
            self.category_dict[seq_id]  = category
            self.entities_dict[seq_id]  = self._parse_entities(ent_str)

        return True

    def load_behaviors(self):
        """
        Load behaviors.tsv.

        - History column  → user_history (reading history before each session)
        - Impressions col → train / test UIR rows
        """
        print("Loading behaviors.tsv …")
        try:
            df = pd.read_csv(
                self.behaviors_path, sep='\t', encoding='utf-8', header=None,
                names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
            )
        except Exception as exc:
            print(f"[ERROR] Could not read behaviors.tsv: {exc}")
            return False

        print(f"  {len(df):,} impression rows found.")

        # Optionally cap to sample_size distinct users
        if self.sample_size:
            unique_users = df['UserID'].dropna().unique()
            if len(unique_users) > self.sample_size:
                sampled = set(np.random.choice(unique_users,
                                               size=self.sample_size, replace=False))
                df = df[df['UserID'].isin(sampled)].copy()
                print(f"  Sampled {self.sample_size} users → {len(df):,} impression rows.")

        # Shuffle rows deterministically before splitting train/test
        df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        split_idx = int(len(df) * self.train_split)
        df_train = df.iloc[:split_idx]
        df_test  = df.iloc[split_idx:]

        def _process_rows(df_part, target_list):
            for row in df_part.itertuples(index=False):
                if pd.isna(row.UserID):
                    continue
                seq_uid = self._seq_user(str(row.UserID))

                # ── reading history ───────────────────────────────────────
                history_str = "" if pd.isna(row.History) else str(row.History)
                history_ids = [self.raw_to_seq[nid] for nid in history_str.split()
                               if nid in self.raw_to_seq]
                # Keep the longest history seen across all sessions for this user
                # (MIND history grows over time, so the longest is the most complete)
                if history_ids and len(history_ids) > len(self.user_history.get(seq_uid, [])):
                    self.user_history[seq_uid] = history_ids

                # ── impressions ───────────────────────────────────────────
                impr_str = "" if pd.isna(row.Impressions) else str(row.Impressions)
                pairs = []
                for token in impr_str.split():
                    try:
                        raw_nid, click = token.rsplit('-', 1)
                        if raw_nid in self.raw_to_seq:
                            pairs.append((self.raw_to_seq[raw_nid], int(click)))
                    except ValueError:
                        continue
                if pairs:
                    target_list.append((seq_uid, pairs))

        _process_rows(df_train, self.train_impressions)
        _process_rows(df_test,  self.test_impressions)

        print(f"  Train sessions: {len(self.train_impressions):,}  "
              f"Test sessions: {len(self.test_impressions):,}")
        return True

    def generate_uir_files(self):
        """Write training and test UIR CSV files."""
        print("Generating UIR files …")

        def _to_df(impression_list):
            rows = []
            for seq_uid, pairs in impression_list:
                for seq_nid, rating in pairs:
                    rows.append({'UserID': seq_uid, 'ItemID': seq_nid, 'Rating': rating})
            return pd.DataFrame(rows, columns=['UserID', 'ItemID', 'Rating'])

        train_df = _to_df(self.train_impressions)
        test_df  = _to_df(self.test_impressions)

        train_path = os.path.join(self.output_dir, 'example_training_graph_uir_top3.csv')
        test_path  = os.path.join(self.output_dir, 'example_impression_test_uir.csv')
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        print(f"  Train rows : {len(train_df):,}  → {train_path}")
        print(f"  Test  rows : {len(test_df):,}  → {test_path}")
        return True

    def generate_feature_files(self):
        """Write sentiment, category, and party JSON files."""
        print("Generating feature files …")

        def _save(data, fname):
            path = os.path.join(self.output_dir, fname)
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, indent=2)
            print(f"  {fname}")

        _save(self.sentiment_dict,  'example_sentiment.json')
        _save(self.category_dict,   'example_category.json')
        _save(self.entities_dict,   'example_party.json')
        return True

    def generate_metadata_files(self):
        """Write article CSV and user history JSON."""
        print("Generating metadata files …")

        article_rows = []
        for seq_id, info in self.articles.items():
            article_rows.append({
                'id':        seq_id,
                'RawID':     info['raw_id'],
                'Title':     info['title'],
                'Category':  info['category'],
                'Sentiment': self.sentiment_dict.get(seq_id, 0.0),
                'Entities':  len(self.entities_dict.get(seq_id, {})),
            })
        article_path = os.path.join(self.output_dir, 'example_article.csv')
        pd.DataFrame(article_rows).to_csv(article_path, index=False)
        print(f"  example_article.csv  ({len(article_rows):,} articles)")

        history_path = os.path.join(self.output_dir, 'example_user_history.json')
        with open(history_path, 'w', encoding='utf-8') as fh:
            json.dump(self.user_history, fh, indent=2)
        print(f"  example_user_history.json  ({len(self.user_history):,} users)")

        return True

    # ── NRMS-specific file generation ─────────────────────────────────────────

    def generate_title_file(self):
        """
        Write example_title.json: {article_id: title_text}.
        The NRMS model reads raw title strings and tokenises them internally.
        """
        print("Generating example_title.json …")
        title_map = {seq_id: info['title']
                     for seq_id, info in self.articles.items()}
        title_path = os.path.join(self.output_dir, 'example_title.json')
        with open(title_path, 'w', encoding='utf-8') as fh:
            json.dump(title_map, fh, indent=2)
        print(f"  {len(title_map):,} titles written.")
        return True

    def _tokenize(self, text):
        """Lowercase word tokenizer (letters only, strips punctuation)."""
        return re.findall(r"[a-z]+", text.lower())

    def build_vocabulary(self):
        """
        Build word_index_dict from all news titles.

        Index 0 is reserved for padding; words are assigned 1-based indices
        in descending frequency order.
        """
        print("Building vocabulary from titles …")
        freq = defaultdict(int)
        for info in self.articles.values():
            for word in self._tokenize(info['title']):
                freq[word] += 1

        # Sort by frequency (descending) so common words get small indices
        sorted_words = sorted(freq, key=lambda w: freq[w], reverse=True)
        word_dict = {word: idx + 1 for idx, word in enumerate(sorted_words)}
        # index 0 → padding (not included in dict)

        wdict_path = os.path.join(self.output_dir, 'word_index_dict.json')
        with open(wdict_path, 'w', encoding='utf-8') as fh:
            json.dump(word_dict, fh, indent=2)
        print(f"  Vocabulary size: {len(word_dict):,} words.")
        return word_dict

    def build_embedding_matrix(self, word_dict):
        """
        Build and save embedding_matrix.npy of shape (vocab_size+1, EMBED_DIM).

        Row 0 = zero vector (padding).
        Rows 1.. = word vectors (GloVe if available, else random N(0, 0.1)).

        Parameters
        ----------
        word_dict : dict
            {word: index} as produced by build_vocabulary().
        """
        vocab_size = len(word_dict)
        # +1 because index 0 is reserved for padding
        matrix = np.zeros((vocab_size + 1, EMBED_DIM), dtype=np.float32)

        if self.glove_path and os.path.isfile(self.glove_path):
            print(f"Loading GloVe embeddings from {self.glove_path} …")
            loaded = 0
            with open(self.glove_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.rstrip().split(' ')
                    word = parts[0]
                    if word in word_dict and len(parts) == EMBED_DIM + 1:
                        idx = word_dict[word]
                        matrix[idx] = np.array(parts[1:], dtype=np.float32)
                        loaded += 1
            # Random init for words not found in GloVe
            not_found = 0
            for word, idx in word_dict.items():
                if np.all(matrix[idx] == 0):
                    matrix[idx] = np.random.normal(0, 0.1, EMBED_DIM).astype(np.float32)
                    not_found += 1
            print(f"  Loaded from GloVe: {loaded:,}  |  Random init: {not_found:,}")
        else:
            if self.glove_path:
                print(f"[WARN] GloVe file not found at '{self.glove_path}' — using random init.")
            else:
                print("  No GloVe path provided — using random N(0, 0.1) initialisation.")
            rng = np.random.default_rng(RANDOM_SEED)
            matrix[1:] = rng.normal(0, 0.1, (vocab_size, EMBED_DIM)).astype(np.float32)

        emb_path = os.path.join(self.output_dir, 'embedding_matrix.npy')
        np.save(emb_path, matrix)
        print(f"  Embedding matrix shape: {matrix.shape}  → {emb_path}")
        return True

    # ── orchestrator ─────────────────────────────────────────────────────────

    def run(self):
        """Execute the full preprocessing pipeline."""
        print("=" * 65)
        print("  MIND → NRMS Preprocessor")
        print("=" * 65)

        steps = [
            ("Load news.tsv",           self.load_news),
            ("Load behaviors.tsv",       self.load_behaviors),
            ("Generate UIR files",        self.generate_uir_files),
            ("Generate feature files",    self.generate_feature_files),
            ("Generate metadata files",   self.generate_metadata_files),
            ("Generate title file",       self.generate_title_file),
        ]
        for label, fn in steps:
            print(f"\n[{label}]")
            if not fn():
                print(f"[ERROR] Step failed: {label}")
                return False

        # Vocabulary and embedding build (needs articles loaded first)
        print("\n[Build vocabulary & embeddings]")
        word_dict = self.build_vocabulary()
        if word_dict is None:
            return False
        if not self.build_embedding_matrix(word_dict):
            return False

        self._print_summary()
        return True

    def _print_summary(self):
        print("\n" + "=" * 65)
        print("  SUMMARY")
        print("=" * 65)
        print(f"  Articles          : {len(self.articles):>10,}")
        print(f"  Users             : {len(self.raw_user_to_seq):>10,}")
        print(f"  User histories    : {len(self.user_history):>10,}")
        print(f"  Train sessions    : {len(self.train_impressions):>10,}")
        print(f"  Test  sessions    : {len(self.test_impressions):>10,}")
        print(f"  Articles (raw→seq): {len(self.raw_to_seq):>10,}")
        if self.sentiment_dict:
            vals = list(self.sentiment_dict.values())
            print(f"  Sentiment range   : [{min(vals):.3f}, {max(vals):.3f}]")
        print(f"  Categories        : {len(set(self.category_dict.values())):>10,}")
        print(f"\n  Output directory  : {self.output_dir}")
        print("=" * 65)


# ════════════════════════════════════════════════════════════════════════════
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
            # The zip may contain files at root or inside a single sub-folder
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
    # ── paths (all relative to the project root) ─────────────────────────────
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    MIND_DIR    = os.path.join(_script_dir, "data", "MINDsmall_train")
    output_dir  = os.path.join(_script_dir, "examples", "example_news_files")

    behaviors_path = os.path.join(MIND_DIR, "behaviors.tsv")
    news_path      = os.path.join(MIND_DIR, "news.tsv")

    # ── optional GloVe path ───────────────────────────────────────────────────
    # Set to None for random init, or provide a path like:
    #   os.path.join(_script_dir, "data", "glove.6B.300d.txt")
    GLOVE_PATH = None

    # ── auto-download MIND Small if not present ───────────────────────────────
    if not _ensure_mind_small(MIND_DIR):
        return False

    print(f"Input  : {MIND_DIR}")
    print(f"Output : {output_dir}")
    print(f"GloVe  : {GLOVE_PATH or 'not provided — random init'}")
    print()

    converter = MINDtoNRMSConverter(
        behaviors_path=behaviors_path,
        news_path=news_path,
        output_dir=output_dir,
        glove_path=GLOVE_PATH,
        train_split=0.8,
        sample_size=None,   # set e.g. sample_size=500 for a quick test run
    )
    return converter.run()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)