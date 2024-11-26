import pandas as pd
import networkx as nx
import json
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from matplotlib import cm
import matplotlib.pyplot as plt

data = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')
sampled_data = data.sample(frac=0.005, random_state=42)
grouped_data = sampled_data[['Customer ID', 'Product Category', 'Quantity']].groupby(['Customer ID', 'Product Category']).sum().reset_index()
category_mapping = {"Books": 1, "Electronics": 2, "Home": 3, "Clothing": 4}
grouped_data['Product Category'] = grouped_data['Product Category'].map(category_mapping)

# Create graph
G = nx.Graph()
G.add_nodes_from(grouped_data['Customer ID'].unique(), type='customer')
G.add_nodes_from(grouped_data['Product Category'].unique(), type='product')
for _, row in grouped_data.iterrows():
    G.add_edge(row['Customer ID'], row['Product Category'], weight=row['Quantity'])

# Detect communities using greedy modularity optimization
communities = list(greedy_modularity_communities(G))
pos = nx.spring_layout(G)
color_map = cm.rainbow(np.linspace(0, 1, len(communities)))

# Draw the graph
for i, community in enumerate(communities):
    nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=[color_map[i]], node_size=30)
nx.draw_networkx_edges(G, pos, alpha=0.5)

plt.show(block=True)

# Save communities to a JSON file
clusters = {f"Cluster_{i}": list(map(int, community)) for i, community in enumerate(communities)}
with open("clusters.json", "w") as f:
    json.dump(clusters, f)

# Modularity score
modularity = nx.algorithms.community.modularity(G, communities)
print(f"Modularity: {modularity}")

# Additional Network Metrics
# Average Degree
avg_degree = np.mean([deg for node, deg in G.degree()])
print(f"Average Degree: {avg_degree}")

# Diameter (longest shortest path)
try:
    diameter = nx.diameter(G)
    print(f"Diameter: {diameter}")
except nx.NetworkXError:
    diameter = "Inf"  # In case the graph is disconnected
    print(f"Diameter: {diameter}")

# Average Path Length (average number of steps along shortest paths for all pairs of nodes)
try:
    avg_path_length = nx.average_shortest_path_length(G)
    print(f"Average Path Length: {avg_path_length}")
except nx.NetworkXError:
    avg_path_length = "Inf"  # In case the graph is disconnected
    print(f"Average Path Length: {avg_path_length}")

# Clustering Coefficient (for the entire graph, not per node)
clustering_coefficient = nx.average_clustering(G)
print(f"Clustering Coefficient: {clustering_coefficient}")

# Graph Density
density = nx.density(G)
print(f"Graph Density: {density}")

# Assortativity (measuring degree correlation)
assortativity = nx.degree_assortativity_coefficient(G)
print(f"Assortativity: {assortativity}")
