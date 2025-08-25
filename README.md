CORA Citation Network Analysis
📌 Project Overview

This project analyzes the CORA citation network, where each node represents an academic paper, and directed edges represent citations between papers. The papers are categorized into three research domains: AI, Theory, and Systems.
To view the project's summary report in Hebrew,click on link:
https://github.com/Avishagtams/CoraDataset/blob/main/Summary%20report.pdf

🎯 Objectives

Analyze the structure of the CORA citation network.

Identify relationships between research domains and their centrality and influence.

Explore dynamic influence using Independent Cascade (ICM) simulations.

Detect communities with Louvain and Label Propagation algorithms.

🛠 Tools & Libraries

Language: Python 3

Libraries: NetworkX, Pandas, Matplotlib, Powerlaw, Seaborn

🔍 Key Insights

The network contains 2,485 nodes and 5,209 edges, forming a directed graph.

Small-world property observed: short paths between nodes and high clustering.

Power-law distribution: A few nodes have very high degrees, confirming a scale-free structure.

Centrality: Systems papers are the most central and well-connected; Theory shows strong dynamic influence; AI is common across communities but less central.

Community detection: 29 strong communities (Louvain) showing strong overlaps between domains.

🚀 How to Run
git clone LINK_GITHUB
cd cora-citation-analysis
pip install -r requirements.txt
python main.py

📌 Research Question

Is there a connection between a paper's research domain (AI, Theory, Systems) and its position and influence in the citation network?


