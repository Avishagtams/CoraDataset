import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import Counter

# === Load Data ===
content_df = pd.read_csv("cora.content", sep='\t', header=None)
content_df.columns = ['id'] + [f'word_{i}' for i in range(1, 1434)] + ['category']
cites_df = pd.read_csv("cora.cites", sep='\t', header=None, names=['target', 'source'])

# === Map categories to super-categories
CATEGORY_MAPPING = {
    'Neural_Networks': 'AI',
    'Reinforcement_Learning': 'AI',
    'Rule_Learning': 'AI',
    'Theory': 'Theory',
    'Probabilistic_Methods': 'Theory',
    'Case_Based': 'Systems',
    'Genetic_Algorithms': 'Systems'
}

# === Build the graph
G = nx.DiGraph()
for _, row in content_df.iterrows():
    super_cat = CATEGORY_MAPPING.get(row['category'], 'Other')
    G.add_node(row['id'], super_category=super_cat)

for _, row in cites_df.iterrows():
    if row['source'] in G and row['target'] in G:
        G.add_edge(row['source'], row['target'])

# === Use only the largest weakly connected component
largest_wcc = max(nx.weakly_connected_components(G), key=len)
G_sub = G.subgraph(largest_wcc).copy()

# === Independent Cascade Model
def independent_cascade(G, seeds, prob=0.3, max_steps=10):
    active = set(seeds)
    newly_active = set(seeds)
    for _ in range(max_steps):
        next_active = set()
        for node in newly_active:
            for neighbor in G.successors(node):
                if neighbor not in active and random.random() < prob:
                    next_active.add(neighbor)
        if not next_active:
            break
        active.update(next_active)
        newly_active = next_active
    return active

# === Run the simulation for each super-category
random.seed(42)
spread_results = {}

for category in ['AI', 'Theory', 'Systems']:
    nodes = [n for n, attr in G_sub.nodes(data=True) if attr['super_category'] == category]
    # Take the 5 nodes with highest out-degree (most influence)
    nodes_sorted = sorted(nodes, key=lambda n: G_sub.out_degree(n), reverse=True)
    seeds = nodes_sorted[:5]
    activated_nodes = independent_cascade(G_sub, seeds, prob=0.3)
    spread_results[category] = len(activated_nodes)
    print(f"{category}: {len(activated_nodes)} nodes activated from 5 top-degree seeds")

# === Plot results
plt.figure(figsize=(7, 5))
plt.bar(spread_results.keys(), spread_results.values(), color=['red', 'blue', 'green'])
plt.title("Information Spread by Category (IC Model)")
plt.ylabel("Number of Activated Nodes")
plt.xlabel("Category of Seed Nodes")
plt.tight_layout()
plt.savefig("spreading_simulation_results.png")
print("✅ Saved: spreading_simulation_results.png")
