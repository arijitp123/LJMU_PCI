import json, os, pandas as pd
from collections import Counter

_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_dir, 'example_category.json')) as f:
    cat_map = json.load(f)  # article_id -> category

df = pd.read_csv(os.path.join(_dir, 'example_training_graph_uir_top3.csv'))
df_pos = df[df['Rating'] == 1]  # only clicked articles

user_pcis = []
for user, grp in df_pos.groupby('UserID'):
    cats = [cat_map[iid] for iid in grp['ItemID'] if iid in cat_map]
    if len(cats) < 3:
        continue
    counts = Counter(cats)
    total = sum(counts.values())
    pci = sum((c/total)**2 for c in counts.values())
    user_pcis.append(pci)

s = pd.Series(user_pcis)
print("PCI distribution for users with at least 3 clicked articles:NRMS\n")
print(s.describe())
print(f"% users above 0.50: {(s > 0.50).mean()*100:.1f}%")
print(f"% users above 0.35: {(s > 0.35).mean()*100:.1f}%")
print(f"% users above 0.25: {(s > 0.25).mean()*100:.1f}%")