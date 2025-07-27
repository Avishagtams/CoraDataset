import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import community as community_louvain
from networkx.algorithms.community import label_propagation_communities

# === Load Data ===
content_df = pd.read_csv("cora.content", sep='\t', header=None)
content_df.columns = ['id'] + [f'word_{i}' for i in range(1, 1434)] + ['category']
cites_df = pd.read_csv("cora.cites", sep='\t', header=None, names=['target', 'source'])

# === Map fine categories to general ones
CATEGORY_MAPPING = {
    'Neural_Networks': 'AI',
    'Reinforcement_Learning': 'AI',
    'Rule_Learning': 'AI',
    'Theory': 'Theory',
    'Probabilistic_Methods': 'Theory',
    'Case_Based': 'Systems',
    'Genetic_Algorithms': 'Systems'
}

# === Build directed graph
G = nx.DiGraph()
for _, row in content_df.iterrows():
    super_cat = CATEGORY_MAPPING.get(row['category'], 'Other')
    G.add_node(row['id'], super_category=super_cat)

for _, row in cites_df.iterrows():
    if row['source'] in G and row['target'] in G:
        G.add_edge(row['source'], row['target'])

# === Use largest weakly connected component
largest_wcc = max(nx.weakly_connected_components(G), key=len)
G_sub = G.subgraph(largest_wcc).copy()
G_undirected = G_sub.to_undirected()

# === Louvain community detection
louvain_partition = community_louvain.best_partition(G_undirected)
louvain_df = pd.DataFrame.from_dict(louvain_partition, orient='index', columns=['Community'])
louvain_df['id'] = louvain_df.index
louvain_df = louvain_df.merge(content_df[['id', 'category']], on='id')
louvain_df['super_category'] = louvain_df['category'].map(CATEGORY_MAPPING)

# === Label Propagation community detection
labelprop_map = {}
for i, comm in enumerate(label_propagation_communities(G_undirected)):
    for node in comm:
        labelprop_map[node] = i

labelprop_df = pd.DataFrame.from_dict(labelprop_map, orient='index', columns=['Community'])
labelprop_df['id'] = labelprop_df.index
labelprop_df = labelprop_df.merge(content_df[['id', 'category']], on='id')
labelprop_df['super_category'] = labelprop_df['category'].map(CATEGORY_MAPPING)

# === Community composition: Louvain
louvain_counts = louvain_df.groupby(['Community', 'super_category']).size().unstack(fill_value=0)
louvain_counts.to_csv("louvain_community_composition.csv")

# === Community composition: Label Propagation
labelprop_counts = labelprop_df.groupby(['Community', 'super_category']).size().unstack(fill_value=0)
labelprop_counts.to_csv("labelprop_community_composition.csv")

print("✅ Saved: louvain_community_composition.csv and labelprop_community_composition.csv")

# Optional: print a quick summary
print("\nLouvain: Number of communities =", louvain_df['Community'].nunique())
print("Label Propagation: Number of communities =", labelprop_df['Community'].nunique())

# === EXTRA: Generate Louvain community composition plots ===
df = louvain_counts.reset_index()
df['Total'] = df[['AI', 'Systems', 'Theory']].sum(axis=1)
df['Dominant'] = df[['AI', 'Systems', 'Theory']].idxmax(axis=1)

# תקן את חישוב אחוז השליטה בלי להשתמש ב-lookup
df['Dominant_Count'] = df.lookup(df.index, df['Dominant']) if hasattr(df, 'lookup') else df.to_dict(orient='index')
df['Dominant_Count'] = [row[dom] for row, dom in zip(df.to_dict(orient='records'), df['Dominant'])]
df['Dominant_Percent'] = df['Dominant_Count'] / df['Total']

# Count how many communities each domain dominates
dominance_counts = df['Dominant'].value_counts()

# Plot stacked bar chart
plt.figure(figsize=(12, 6))
bottom = [0]*len(df)
for cat, color in zip(['AI', 'Systems', 'Theory'], ['red', 'green', 'blue']):
    plt.bar(df['Community'], df[cat], bottom=bottom, label=cat, color=color)
    bottom = [i + j for i, j in zip(bottom, df[cat])]
plt.title("Community Composition by Category (Louvain)")
plt.xlabel("Community ID")
plt.ylabel("Number of Nodes")
plt.legend()
plt.tight_layout()
plt.savefig("community_composition_stackedbar.png")
plt.close()

# Plot pie chart of dominant categories
plt.figure(figsize=(6, 6))
plt.pie(dominance_counts, labels=dominance_counts.index, autopct='%1.1f%%', colors=['red', 'green', 'blue'])
plt.title("Distribution of Dominant Category per Community (Louvain)")
plt.tight_layout()
plt.savefig("dominant_category_piechart.png")
plt.close()

print("✅ Saved: community_composition_stackedbar.png and dominant_category_piechart.png")
