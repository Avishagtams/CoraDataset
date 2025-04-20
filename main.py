import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
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

# צבעים לנושאים
COLOR_MAP = {'AI': 'red', 'Theory': 'blue', 'Systems': 'green'}


def compute_and_save_central_nodes(G_sub, top_n=5):

    print("\n=== מדדי מרכזיות עבור צמתים מעניינים ===")
    degree_centrality = nx.degree_centrality(G_sub)
    closeness_centrality = nx.closeness_centrality(G_sub)
    betweenness_centrality = nx.betweenness_centrality(G_sub)

    df = pd.DataFrame({
        'id': list(G_sub.nodes()),
        'degree_centrality': pd.Series(degree_centrality),
        'closeness_centrality': pd.Series(closeness_centrality),
        'betweenness_centrality': pd.Series(betweenness_centrality)
    })

    # מחשבים ציון ממוצע למדדי המרכזיות
    df['avg_score'] = df[['degree_centrality', 'closeness_centrality', 'betweenness_centrality']].mean(axis=1)

    # שומרים רק את 5 הצמתים הכי מעניינים
    top_df = df.sort_values(by='avg_score', ascending=False).head(top_n).drop(columns='avg_score')

    # מדפיסים למסך
    print(top_df.to_string(index=False))

    # שומרים לקובץ
    top_df.to_csv("centrality_measures_top_nodes.csv", index=False)
    print("נשמר הקובץ: centrality_measures_top_nodes.csv")

    # גרפים לכל צומת מעניין
    for _, row in top_df.iterrows():
        node_id = row['id']
        measures = {
            'Degree Centrality': row['degree_centrality'],
            'Closeness Centrality': row['closeness_centrality'],
            'Betweenness Centrality': row['betweenness_centrality']
        }

        plt.figure(figsize=(6, 4))
        plt.bar(measures.keys(), measures.values(), color=['orange', 'green', 'blue'])
        plt.title(f'Centrality for Node {node_id}')
        plt.ylabel("Centrality Value")
        plt.tight_layout()
        plt.savefig(f'node_{node_id}_centrality.png')
        plt.close()
        print(f"נשמר גרף: node_{node_id}_centrality.png")


def plot_normalized_degree_distributions_fixed(G_sub):

    def plot_distribution(degrees, title, filename, color, max_degree=None):
        from matplotlib.ticker import MaxNLocator
        count = Counter(degrees)

        # חיתוך לפי דרגה מקסימלית אם ביקשו
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
        # סינון צמתים לפי הקטגוריה
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
    print("\n=== ניתוח הגרף המלא ===")
    print("מספר צמתים:", G.number_of_nodes())
    print("מספר קשתות:", G.number_of_edges())
    print(f"מספר רכיבי קשירות חלשים: {nx.number_weakly_connected_components(G)}")
    print(f"מספר רכיבי קשירות חזקים: {nx.number_strongly_connected_components(G)}")


def analyze_component(G_sub, total_nodes, total_edges):
    print("\n=== ניתוח רכיב הקשירות הגדול ביותר ===")
    num_nodes = G_sub.number_of_nodes()
    num_edges = G_sub.number_of_edges()
    print("מספר צמתים:", num_nodes)
    print("מספר קשתות:", num_edges)
    print(f"אחוז הצמתים מכלל הגרף: {num_nodes / total_nodes * 100:.2f}%")
    print(f"אחוז הקשתות מכלל הגרף: {num_edges / total_edges * 100:.2f}%")
    print(f"מספר לולאות עצמיות: {nx.number_of_selfloops(G_sub)}")


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
        diameter = "אין קוטר (הרשת לא קשירה)"
    print(f"\nאורך מסלול ממוצע: {avg_path}")
    print(f"מקדם קיבוץ ממוצע: {clustering}")
    print(f"צפיפות הרכיב הקשיר הגדול: {density:.4f}")
    print(f"קוטר הרכיב הקשיר הגדול: {diameter}")


def plot_graph(G_sub):
    print("מצייר את הרשת לפי נושאים כלליים...")
    node_colors = [COLOR_MAP.get(G_sub.nodes[n]['super_category'], 'gray') for n in G_sub.nodes]
    pos = nx.spring_layout(G_sub, seed=42)
    plt.figure(figsize=(12, 12))
    nx.draw(G_sub, pos=pos, node_color=node_colors,
            node_size=20, edge_color='gray', arrows=True, with_labels=False)
    plt.title("הרכיב הקשיר הגדול - צבע לפי נושא כללי (AI / Theory / Systems)")
    plt.savefig("cora_graph_colored.png")
    print("נשמר הקובץ: cora_graph_colored.png")


def extract_keywords(content_df, G_sub):
    word_columns = [f'word_{i}' for i in range(1, 1434)]
    content_sub_df = content_df[content_df['id'].isin(G_sub.nodes())].copy()

    def extract_keyword(row):
        for col in word_columns:
            if row[col] == 1:
                return col
        return None

    content_sub_df['keyword'] = content_sub_df.apply(extract_keyword, axis=1)
    return content_sub_df


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
    print("נשמר הקובץ: cora_nodes_with_keywords.csv")


def plot_ego_graph(G_sub):
    print("מחשב את הצומת עם הדרגה הנכנסת הכי גבוהה...")
    in_degrees = dict(G_sub.in_degree())
    max_in_node = max(in_degrees, key=in_degrees.get)
    print(f"הצומת עם הדרגה הנכנסת הגבוהה ביותר הוא: {max_in_node} (in-degree: {in_degrees[max_in_node]})")
    incoming_neighbors = set(G_sub.predecessors(max_in_node))
    ego_nodes = incoming_neighbors | {max_in_node}
    ego_subgraph = G_sub.subgraph(ego_nodes).copy()
    ego_colors = [COLOR_MAP.get(ego_subgraph.nodes[n]['super_category'], 'gray') for n in ego_subgraph.nodes]
    print("מצייר תת-גרף סביב הצומת המרכזי...")
    pos = nx.spring_layout(ego_subgraph, seed=42)
    plt.figure(figsize=(6, 6))
    nx.draw(ego_subgraph, pos=pos, node_color=ego_colors,
            node_size=100, edge_color='gray', arrows=True, with_labels=True,
            font_size=8)
    plt.title(f"תת-גרף סביב הצומת המצוטט ביותר {max_in_node}")
    plt.savefig("ego_graph_colored.png")
    print("נשמר הקובץ: ego_graph_colored.png")


def main():
    content_df, cites_df = load_data()
    G = build_graph(content_df, cites_df)
    analyze_full_graph(G)
    G_sub = get_largest_weakly_connected_component(G)
    analyze_component(G_sub, G.number_of_nodes(), G.number_of_edges())
    compute_graph_metrics(G_sub)
    plot_graph(G_sub)
    content_sub_df = extract_keywords(content_df, G_sub)
    save_node_data(G_sub, content_sub_df)
    plot_ego_graph(G_sub)
    plot_normalized_degree_distributions_fixed(G_sub)
    plot_category_normalized_degrees(G_sub)
    compute_and_save_central_nodes(G_sub)



if __name__ == "__main__":
    main()
