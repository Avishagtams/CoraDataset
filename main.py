import pandas as pd

import matplotlib
import warnings
from scipy.stats import powerlaw as scipy_powerlaw
import powerlaw  # זה המודול הנכון ל־Fit של חוק חזקה
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use('Agg')

# Mapping categories to sub-categories
CATEGORY_MAPPING = {
    'Neural_Networks': 'AI',
    'Reinforcement_Learning': 'AI',
    'Rule_Learning': 'AI',
    'Theory': 'Theory',
    'Probabilistic_Methods': 'Theory',
    'Case_Based': 'Systems',
    'Genetic_Algorithms': 'Systems'
}

# Mapping color
COLOR_MAP = {'AI': 'red', 'Theory': 'blue', 'Systems': 'green'}



import random

def check_small_world_property_random_pairs(G_sub, num_samples=1000):
    print("== Small-World Property Check by Random Pairs ==")

    # רק על הרכיב הקשיר היטב
    largest_wcc = max(nx.weakly_connected_components(G_sub), key=len)
    G_connected = G_sub.subgraph(largest_wcc).copy()

    print(f"Number of nodes in the largest weakly connected component: {G_connected.number_of_nodes()}")

    nodes = list(G_connected.nodes())
    path_lengths = []
    attempts = 0

    while len(path_lengths) < num_samples and attempts < num_samples * 2:
        u, v = random.sample(nodes, 2)
        try:
            length = nx.shortest_path_length(G_connected, source=u, target=v)
            path_lengths.append(length)
        except nx.NetworkXNoPath:
            pass  # אין מסלול – מתעלמים
        attempts += 1

    if not path_lengths:
        print("No valid pairs found with paths.")
        return

    avg_path_length = sum(path_lengths) / len(path_lengths)
    clustering_coeff = nx.average_clustering(G_connected.to_undirected())

    print(f"Sample size: {len(path_lengths)}")
    print(f"Average shortest path (from sampled pairs): {avg_path_length:.3f}")
    print(f"Average clustering coefficient: {clustering_coeff:.3f}")

    # בדיקה לפי קריטריון המרצה
    if avg_path_length < 6 and clustering_coeff > 0.1:
        print("✅ The graph supports the Small-World property.")
    else:
        print("❌ The graph does NOT support the Small-World property.")



def plot_normalized_degree_distributions_fixed(G_sub):

    def plot_distribution(degrees, title, filename, color, max_degree=None):
        from matplotlib.ticker import MaxNLocator
        count = Counter(degrees)


        if max_degree:
            count = {k: v for k, v in count.items() if k <= max_degree}

        total = sum(count.values())
        degs = sorted(count.keys())
        freqs = [count[d] / total for d in degs]

        plt.figure(figsize=(10, 6))
        plt.bar(degs, freqs, width=0.8, color='yellow', edgecolor='black', align='center')
        plt.title(title)
        plt.xlabel("Degree")
        plt.ylabel("Relative Frequency")
        plt.xticks(degs if len(degs) < 30 else range(0, max(degs)+1, max(1, max(degs)//15)))  # לא יותר מדי X ticks
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")

    in_degrees = [deg for _, deg in G_sub.in_degree()]
    out_degrees = [deg for _, deg in G_sub.out_degree()]

    plot_distribution(in_degrees, "Normalized In-Degree Distribution", "normalized_in_degree_distribution.png", 'royalblue', max_degree=500)
    plot_distribution(out_degrees, "Normalized Out-Degree Distribution", "normalized_out_degree_distribution.png", 'tomato', max_degree=300)


def load_data():
    content_df = pd.read_csv("cora.content", sep='\t', header=None)
    content_df.columns = ['id'] + [f'word_{i}' for i in range(1, 1434)] + ['category']
    cites_df = pd.read_csv("cora.cites", sep='\t', header=None, names=['target', 'source'])
    return content_df, cites_df


def build_graph(content_df, cites_df):
    G = nx.DiGraph()
    for _, row in content_df.iterrows():
        category = row['category']
        super_category = CATEGORY_MAPPING.get(category, 'Other')
        G.add_node(row['id'], category=category, super_category=super_category)
    for _, row in cites_df.iterrows():
        if row['source'] in G and row['target'] in G:
            G.add_edge(row['source'], row['target'])
    return G
def plot_category_normalized_degrees(G_sub):
    from collections import Counter

    def plot_distribution(degrees, title, filename, color, max_degree=None):
        count = Counter(degrees)
        if max_degree:
            count = {k: v for k, v in count.items() if k <= max_degree}

        total = sum(count.values())
        degs = sorted(count.keys())
        freqs = [count[d] / total for d in degs]

        plt.figure(figsize=(8, 5))
        plt.bar(degs, freqs, width=0.8, color=color, edgecolor='black', align='center')
        plt.title(title)
        plt.xlabel("Degree")
        plt.ylabel("Relative Frequency")
        plt.xticks(degs if len(degs) < 20 else range(0, max(degs)+1, max(1, max(degs)//15)))
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")

    COLORS = {
        'AI': 'red',
        'Theory': 'blue',
        'Systems': 'green'
    }

    for category, color in COLORS.items():
        # Filter nodes by category
        nodes = [n for n, attr in G_sub.nodes(data=True) if attr['super_category'] == category]
        subgraph = G_sub.subgraph(nodes)

        in_degrees = [deg for _, deg in subgraph.in_degree()]
        out_degrees = [deg for _, deg in subgraph.out_degree()]

        plot_distribution(in_degrees,
                          f"{category} - Normalized In-Degree Distribution",
                          f"{category.lower()}_normalized_in_degree.png",
                          color=color,
                          max_degree=500)

        plot_distribution(out_degrees,
                          f"{category} - Normalized Out-Degree Distribution",
                          f"{category.lower()}_normalized_out_degree.png",
                          color=color,
                          max_degree=500)


def analyze_full_graph(G):
    print("\n=== Full graph analysis ===")
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())



def analyze_component(G_sub, total_nodes, total_edges):
    print("\n=== Largest Binding Component Analysis --> our graph")
    num_nodes = G_sub.number_of_nodes()
    num_edges = G_sub.number_of_edges()
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)
    print(f"Percentage of nodes in the total graph: {num_nodes / total_nodes * 100:.2f}%")
    print(f"Percentage of edges in the total graph: {num_edges / total_edges * 100:.2f}%")
    print(f"Number of self-loops: {nx.number_of_selfloops(G_sub)}")


def get_largest_weakly_connected_component(G):
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def compute_graph_metrics(G_sub):
    avg_path = nx.average_shortest_path_length(G_sub.to_undirected())
    clustering = nx.average_clustering(G_sub.to_undirected())
    density = nx.density(G_sub)
    try:
        diameter = nx.diameter(G_sub.to_undirected())
    except nx.NetworkXError:
        diameter = "No diameter (the graph is not connected)"
    print(f"Average path length: {avg_path}")
    print(f"Average clustering coefficient: {clustering}")
    print(f"Density of the largest connected component: {density:.4f}")
    print(f"Diameter of the largest connected component: {diameter}")


def plot_graph(G_sub):
    print("-⏳ Draws the network ⏳-")
    node_colors = [COLOR_MAP.get(G_sub.nodes[n]['super_category'], 'gray') for n in G_sub.nodes]
    pos = nx.spring_layout(G_sub, seed=42)
    plt.figure(figsize=(12, 12))
    nx.draw(G_sub, pos=pos, node_color=node_colors,
            node_size=20, edge_color='gray', arrows=True, with_labels=False)
    plt.title("The big tie-in component - color by general theme (AI / Theory / Systems)")
    plt.savefig("Graph_cora.png")
    print("Saved: cora_graph_colored.png")



def save_node_data(G_sub, content_sub_df):
    data = [{'id': node,
             'category': G_sub.nodes[node]['category'],
             'super_category': G_sub.nodes[node]['super_category'],
             'in_degree': G_sub.in_degree(node),
             'out_degree': G_sub.out_degree(node)} for node in G_sub.nodes()]
    df = pd.DataFrame(data)
    df_with_keywords = df.merge(content_sub_df[['id', 'keyword']], on='id')
    df_with_keywords.rename(columns={'super_category': 'super_category_red'}, inplace=True)
    df_with_keywords.to_csv("cora_nodes_with_keywords.csv", index=False)
    print("Saved: cora_nodes_with_keywords.csv ")


def plot_ego_graph(G_sub):
    print("Calculates the node with the highest in degree...")
    in_degrees = dict(G_sub.in_degree())
    max_in_node = max(in_degrees, key=in_degrees.get)
    print(f"The node with the highest incoming degree is: {max_in_node} (in-degree: {in_degrees[max_in_node]})")
    incoming_neighbors = set(G_sub.predecessors(max_in_node))
    ego_nodes = incoming_neighbors | {max_in_node}
    ego_subgraph = G_sub.subgraph(ego_nodes).copy()
    ego_colors = [COLOR_MAP.get(ego_subgraph.nodes[n]['super_category'], 'gray') for n in ego_subgraph.nodes]
    print("Draws a subgraph around the central node(35)⏳...")
    pos = nx.spring_layout(ego_subgraph, seed=42)
    plt.figure(figsize=(6, 6))
    nx.draw(ego_subgraph, pos=pos, node_color=ego_colors,
            node_size=100, edge_color='gray', arrows=True, with_labels=True,
            font_size=8)
    plt.title(f"Subgraph around the most cited node {max_in_node}")
    plt.savefig("ego_graph_colored.png")
    print("Saved: ego_graph_colored.png")




def count_nodes_by_category(G_sub):
    print("\n=== Node Count by Super Category ===")
    categories = [attr['super_category'] for _, attr in G_sub.nodes(data=True)]
    counts = Counter(categories)

    for category in ['AI', 'Theory', 'Systems']:
        count = counts.get(category, 0)
        print(f"{category}: {count} nodes")


from collections import Counter

def count_edges_between_categories(G_sub):
    print("\n=== Edge Count Between Super Categories ===")
    counts = Counter()
    for u, v in G_sub.edges():
        source_cat = G_sub.nodes[u].get('super_category', 'Unknown')
        target_cat = G_sub.nodes[v].get('super_category', 'Unknown')
        counts[(source_cat, target_cat)] += 1

    categories = ['AI', 'Theory', 'Systems']
    print(f"{'From → To':<20}Count")
    print("-" * 30)
    for src in categories:
        for tgt in categories:
            count = counts.get((src, tgt), 0)
            print(f"{src} → {tgt:<10} {count}")





import matplotlib.pyplot as plt
import numpy as np

def power_law_by_category(G_sub, category, show_fit=True):
    COLORS = {
        'AI': 'red',
        'Theory': 'blue',
        'Systems': 'green'
    }

    # סינון הצמתים לפי הקטגוריה
    nodes = [n for n, attr in G_sub.nodes(data=True) if attr['super_category'] == category]
    degrees = [G_sub.degree(n) for n in nodes if G_sub.degree(n) > 0]

    if not degrees:
        print(f"No degrees to plot for category: {category}")
        return

    # בניית היסטוגרמה
    hist, bin_edges = np.histogram(degrees, bins=range(1, max(degrees)+2), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # ציור
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist, width=0.8, color=COLORS.get(category, 'gray'),
            edgecolor='black', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Vertex Degree")
    plt.ylabel("Probability")
    plt.title(f"Power-law Fit for {category}")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)

    if show_fit:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = powerlaw.Fit(degrees, discrete=True)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        R, p = fit.distribution_compare('power_law', 'lognormal')

        print(f"==== Power Law Fit for {category} ====")
        print(f"  α (exponent): {alpha:.2f}")
        print(f"  xmin: {xmin}")
        print(f"  p-value: {p:.4f} --> {'✅ מתאים' if p > 0.05 else '❌ לא מתאים'}")

        x_fit = np.linspace(xmin, max(degrees), 100)
        y_fit = (x_fit / xmin) ** (-alpha)
        y_fit *= hist[bin_centers >= xmin][0] / y_fit[0]
        plt.plot(x_fit, y_fit, 'r--', label=f'Power-law fit (γ={alpha:.2f})')
        plt.xlim(left=xmin)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{category.lower()}_powerlaw_fit.png")
    plt.close()
    print(f"Saved: {category.lower()}_powerlaw_fit.png")





import networkx as nx

def highlight_extreme_central_nodes(G_sub, content_df=None):
    import networkx as nx
    import numpy as np

    def analyze_centrality(centrality_dict, name):
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
        top_node, top_val = sorted_nodes[0]
        second_val = sorted_nodes[1][1] if len(sorted_nodes) > 1 else 0
        avg_val = np.mean(list(centrality_dict.values()))
        ratio_to_avg = top_val / avg_val if avg_val > 0 else float('inf')
        gap_to_second = top_val - second_val

        node_data = G_sub.nodes[top_node]
        category = node_data.get('category', 'Unknown')
        super_category = node_data.get('super_category', 'Unknown')
        in_deg = G_sub.in_degree(top_node)
        out_deg = G_sub.out_degree(top_node)

        print(f"\n📌 {name} Centrality")
        print(f"🔸 צומת מוביל: {top_node}")
        print(f"   - ערך מדד: {top_val:.4f}")
        print(f"   - מקטגוריה: {category} ({super_category})")
        print(f"   - דרגות: נכנס {in_deg}, יוצא {out_deg}")
        print(f"🔸 ממוצע כלל הצמתים: {avg_val:.4f}")
        print(f"🔸 הצומת השני הכי גבוה: {sorted_nodes[1][0]} (ערך: {second_val:.4f})")
        print(f"🔸 יחס מוביל / ממוצע: x{ratio_to_avg:.1f}")
        print(f"🔸 מוביל ב- {gap_to_second:.4f} לעומת המקום השני")

    print("\n=== 🔍 Highlighted Central Nodes with Comparisons ===")

    degree_centrality = nx.degree_centrality(G_sub)
    closeness_centrality = nx.closeness_centrality(G_sub)
    betweenness_centrality = nx.betweenness_centrality(G_sub)

    analyze_centrality(degree_centrality, "Degree")
    analyze_centrality(closeness_centrality, "Closeness")
    analyze_centrality(betweenness_centrality, "Betweenness")

def main():
    content_df, cites_df = load_data()
    G = build_graph(content_df, cites_df)
    analyze_full_graph(G)
    G_sub = get_largest_weakly_connected_component(G)
    analyze_component(G_sub, G.number_of_nodes(), G.number_of_edges())
    compute_graph_metrics(G_sub)
    count_nodes_by_category(G_sub)
    check_small_world_property_random_pairs(G_sub)
    plot_graph(G_sub)
    plot_ego_graph(G_sub)
    plot_normalized_degree_distributions_fixed(G_sub)
    plot_category_normalized_degrees(G_sub)
    plot_log_degree_distributions_by_super_category(G_sub)
    count_edges_between_categories(G_sub)
    for cat in ['AI', 'Theory', 'Systems']:
        power_law_by_category(G_sub, cat)
    highlight_extreme_central_nodes(G_sub, content_df)



def create_centrality_score_table(G_sub):
    import pandas as pd
    import matplotlib.pyplot as plt

    # Centrality measures
    deg = nx.degree_centrality(G_sub)
    clo = nx.closeness_centrality(G_sub)
    bet = nx.betweenness_centrality(G_sub)

    # Top 10 from each centrality
    top_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:10]
    top_clo = sorted(clo.items(), key=lambda x: x[1], reverse=True)[:10]
    top_bet = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:10]

    # Aggregate appearances
    centrality_counter = {}
    for name, lst in [('Degree Centrality', top_deg), ('Closeness Centrality', top_clo), ('Betweenness Centrality', top_bet)]:
        for node, val in lst:
            if node not in centrality_counter:
                centrality_counter[node] = {'score': 0}
            centrality_counter[node][name] = val
            centrality_counter[node]['score'] += 1

    # Build DataFrame
    rows = []
    for node, data in centrality_counter.items():
        row = {
            'Node ID': node,
            'Degree Centrality': data.get('Degree Centrality', 0),
            'Closeness Centrality': data.get('Closeness Centrality', 0),
            'Betweenness Centrality': data.get('Betweenness Centrality', 0),
            'Centrality Score (0–3)': data['score'],
            'Category': G_sub.nodes[node].get('super_category', 'Unknown')
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by='Centrality Score (0–3)', ascending=False)

    # Save table
    df.to_csv("centrality_top_10_by_metric.csv", index=False)
    print("Saved: centrality_top_10_by_metric.csv")
    print(df.to_string(index=False))

    # Plot by category
    counts = df['Category'].value_counts()
    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar', color=['red', 'blue', 'green'])
    plt.title("Top Central Nodes by Category")
    plt.ylabel("Number of Nodes")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("central_nodes_by_category.png")
    print("Saved: central_nodes_by_category.png")

def plot_log_degree_distributions_by_super_category(G_sub):
    import numpy as np
    import matplotlib.pyplot as plt

    COLORS = {
        'AI': 'red',
        'Theory': 'blue',
        'Systems': 'green'
    }

    def plot_histogram_with_dynamic_scale(degrees, title, filename, color, label):
        degrees = [d for d in degrees if d > 0]
        if not degrees:
            print(f"אין דרגות להציג עבור {title}")
            return

        min_deg = min(degrees)
        max_deg = max(degrees)
        use_log = max_deg >= 10

        plt.figure(figsize=(8, 5))

        if use_log:
            x_min_local = np.log10(min_deg)
            x_max_local = np.log10(max_deg + 1)
            num_bins = min(30, max(10, int(np.sqrt(len(degrees)))))
            bins = np.logspace(x_min_local, x_max_local, num=num_bins)
            plt.hist(degrees, bins=bins, color=color, edgecolor='black')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(f"{label} (log scale)")
            plt.ylabel("Frequency (log scale)")
        else:
            bins = range(1, max_deg + 2)
            plt.hist(degrees, bins=bins, color=color, edgecolor='black', align='left')
            plt.xlabel(f"{label}")
            plt.ylabel("Frequency")

        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")

    for category, color in COLORS.items():
        nodes = [n for n, attr in G_sub.nodes(data=True) if attr['super_category'] == category]
        subgraph = G_sub.subgraph(nodes)

        in_degrees = [deg for _, deg in subgraph.in_degree()]
        out_degrees = [deg for _, deg in subgraph.out_degree()]
        total_degrees = [i + o for i, o in zip(in_degrees, out_degrees)]

        plot_histogram_with_dynamic_scale(in_degrees,
                                          f"In-Degree Distribution - {category}",
                                          f"{category.lower()}_log_in_degree.png",
                                          color,
                                          "In-Degree")

        plot_histogram_with_dynamic_scale(out_degrees,
                                          f"Out-Degree Distribution - {category}",
                                          f"{category.lower()}_log_out_degree.png",
                                          color,
                                          "Out-Degree")

        plot_histogram_with_dynamic_scale(total_degrees,
                                          f"Total Degree Distribution - {category}",
                                          f"{category.lower()}_log_total_degree.png",
                                          color,
                                          "Total Degree")




if __name__ == "__main__":
    main()

