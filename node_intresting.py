import pandas as pd
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt

# === Load Data ===
content_df = pd.read_csv("cora.content", sep='\t', header=None)
content_df.columns = ['id'] + [f'word_{i}' for i in range(1, 1434)] + ['category']
cites_df = pd.read_csv("cora.cites", sep='\t', header=None, names=['target', 'source'])

# === Category Mapping ===
CATEGORY_MAPPING = {
    'Neural_Networks': 'AI',
    'Reinforcement_Learning': 'AI',
    'Rule_Learning': 'AI',
    'Theory': 'Theory',
    'Probabilistic_Methods': 'Theory',
    'Case_Based': 'Systems',
    'Genetic_Algorithms': 'Systems'
}

# === Build Graph ===
G = nx.DiGraph()
for _, row in content_df.iterrows():
    super_cat = CATEGORY_MAPPING.get(row['category'], 'Other')
    G.add_node(row['id'], super_category=super_cat)

for _, row in cites_df.iterrows():
    if row['source'] in G and row['target'] in G:
        G.add_edge(row['source'], row['target'])

# === Get Largest Weakly Connected Component ===
largest_wcc = max(nx.weakly_connected_components(G), key=len)
G_sub = G.subgraph(largest_wcc).copy()

# === Centrality Measures ===
deg = nx.degree_centrality(G_sub)
close = nx.closeness_centrality(G_sub)
btw = nx.betweenness_centrality(G_sub)

top_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:10]
top_close = sorted(close.items(), key=lambda x: x[1], reverse=True)[:10]
top_btw = sorted(btw.items(), key=lambda x: x[1], reverse=True)[:10]

# === Count Appearances ===
all_top_ids = [n for n, _ in top_deg + top_close + top_btw]
score_count = Counter(all_top_ids)

# === Build Summary Table ===
rows = []
all_ids = sorted(set(all_top_ids), key=lambda x: score_count[x], reverse=True)[:10]  # << רק 10 צמתים
for node_id in all_ids:
    rows.append({
        'ID': node_id,
        'Degree Centrality': deg.get(node_id, 0),
        'Closeness Centrality': close.get(node_id, 0),
        'Betweenness Centrality': btw.get(node_id, 0),
        'Popularity Score': score_count[node_id],
        'Category': G_sub.nodes[node_id]['super_category']
    })

df = pd.DataFrame(rows)

# === Create Table Image ===
fig, ax = plt.subplots(figsize=(12, 0.6 + 0.5 * len(df)))
ax.axis('off')
table = ax.table(cellText=df.round(4).values,
                 colLabels=df.columns,
                 loc='center',
                 cellLoc='center')
table.scale(1, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.tight_layout()
plt.savefig("centrality_table_top10_only.png", dpi=300)
print("✅ Saved: centrality_table_top10_only.png")

# === Calculate PageRank separately ===
pagerank_scores = nx.pagerank(G_sub)

# === Get top 10 nodes by PageRank
top_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]

# === Build PageRank table
pagerank_rows = []
for node_id, score in top_pagerank:
    pagerank_rows.append({
        'ID': node_id,
        'PageRank': score,
        'Category': G_sub.nodes[node_id].get('super_category', 'Unknown')
    })

pagerank_df = pd.DataFrame(pagerank_rows)

# === Save PageRank Table as Image ===
fig, ax = plt.subplots(figsize=(10, 0.6 + 0.5 * len(pagerank_df)))
ax.axis('off')
table = ax.table(cellText=pagerank_df.round(5).values,
                 colLabels=pagerank_df.columns,
                 loc='center',
                 cellLoc='center')
table.scale(1, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.tight_layout()
plt.savefig("pagerank_top10_table.png", dpi=300)
print("✅ Saved: pagerank_top10_table.png")
