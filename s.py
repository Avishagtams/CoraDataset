import pandas as pd
import networkx as nx

# --- 1. טעינת הנתונים ---
print("טוען את הקבצים...")
content_df = pd.read_csv("cora.content", sep='\t', header=None)
content_df.columns = ['id'] + [f'word_{i}' for i in range(1, 1434)] + ['category']
cites_df = pd.read_csv("cora.cites", sep='\t', header=None, names=['target', 'source'])

# --- 2. בניית גרף מכוון ---
G = nx.DiGraph()
for _, row in content_df.iterrows():
    # שומרים גם את המילים הבינאריות כ־"keywords" לצורך עיבוד בהמשך
    G.add_node(row['id'], category=row['category'], keywords=row[1:-1].tolist())

for _, row in cites_df.iterrows():
    if row['source'] in G and row['target'] in G:
        G.add_edge(row['source'], row['target'])

# --- 3. שליפת הרכיב הקשיר הגדול ביותר (חלש) ---
largest_cc = max(nx.weakly_connected_components(G), key=len)
G_sub = G.subgraph(largest_cc).copy()
print(f"מספר צמתים ברכיב הקשיר הגדול: {G_sub.number_of_nodes()}")

# --- 4. שמירת מילות מפתח אמיתיות בלבד לקובץ ---
print("מכין קובץ עם מילות מפתח בפועל (ללא אפסים)...")
word_columns = [f'word_{i}' for i in range(1, 1434)]  # מיפוי בין אינדקסים לעמודות

keyword_data = []
for node in G_sub.nodes(data=True):
    word_presence = node[1]['keywords']
    actual_words = [word_columns[i] for i, val in enumerate(word_presence) if val == 1]
    keyword_data.append({
        'id': node[0],
        'category': node[1]['category'],
        'keywords': actual_words
    })

keyword_df = pd.DataFrame(keyword_data)
keyword_df.to_csv("keywords_in_largest_component_clean.csv", index=False)
print("נשמר הקובץ: keywords_in_largest_component_clean.csv")

# --- 5. הדפסת מספר הקטגוריות ברכיב ---
unique_categories = set(nx.get_node_attributes(G_sub, 'category').values())
print(f"מספר הקטגוריות הייחודיות ברכיב הקשיר הגדול ביותר: {len(unique_categories)}")
print(f"הקטגוריות הן: {sorted(unique_categories)}")
