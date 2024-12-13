#Some functions are pulled from wrong libraries occasionally,
#Running this set of uninstalls and installs seems to set it straight

#!pip uninstall networkx python-louvain community -y
#!pip install python-louvain
#!pip install networkx
#!pip install python-igraph
#!pip install leidenalg


import leidenalg as la
import igraph
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.algorithms import community
import random
import pandas as pd
import community as community_louvain

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")


def simple_Louvain(G):
    random.seed()
    """ Louvain method github basic example"""
    partition = community_louvain.best_partition(G)
    pos = graphviz_layout(G)

    # Calculate modularity
    modularity = community_louvain.modularity(partition, G)
    print(f"Louvain Modularity: {modularity}") # Print the modularity value

    max_k_w = []
    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        max_k_w = max_k_w + [list_nodes]

    node_mapping = {}
    map_v = 0
    for node in G.nodes():
        node_mapping[node] = map_v
        map_v += 1

    community_num_group = len(max_k_w)
    color_list_community = [[] for i in range(len(G.nodes()))]

    # color
    for i in G.nodes():
        for j in range(community_num_group):
            if i in max_k_w[j]:
                color_list_community[node_mapping[i]] = j

    return G, pos, color_list_community, community_num_group, max_k_w, modularity, partition # Return modularity


def simple_Leiden(G):
    """ Leiden method similar to Louvain example"""
    random.seed()
    # Convert the NetworkX graph to igraph using the igraph package
    G_igraph = igraph.Graph.from_networkx(G)

    # Apply the Leiden algorithm
    partition = la.find_partition(G_igraph, la.ModularityVertexPartition)

    # Get community structure
    communities = {}
    for i, community_id in enumerate(partition.membership):
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(list(G.nodes())[i])

    # Create color list for communities
    community_num_group = len(communities)
    color_list_community = [0] * len(G.nodes())
    for node, community_id in enumerate(partition.membership):
        color_list_community[node] = community_id

    # Assuming you have 'pos' from your Louvain code (or generate it)
    pos = nx.spring_layout(G)  # Or use your existing 'pos'

    # Calculate modularity using igraph's modularity function
    modularity = G_igraph.modularity(partition.membership)

    return G, pos, color_list_community, community_num_group, communities, modularity, partition # Return modularity

if __name__ == "__main__":
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    df_nodes = pd.read_csv('stack_network_nodes.csv')
    df_edges = pd.read_csv('stack_network_links.csv')

    # get edges and weight
    edges = df_edges[['source', 'target']].values.tolist()
    weights = [float(l) for l in df_edges.value.values.tolist()]

    # Make Graph and apply weight
    G = nx.Graph(directed=True)
    G.add_edges_from(edges)
    for cnt, a in enumerate(G.edges(data=True)):
        G.edges[(a[0], a[1])]['weight'] = weights[cnt]

    G, pos, color_list_community, community_num_group, max_k_w, modularity, partition = simple_Louvain(G)

    # Create a list to store the edge data
    edge_data = []

    # Iterate over the edges in the graph
    for u, v, data in G.edges(data=True):
        # Get the community assignments for the source and target nodes
        source_community = partition.get(u)  # Use the appropriate partition variable (e.g., partition from Louvain)
        target_community = partition.get(v)

        # Append the edge data to the list
        edge_data.append([u, v, data['weight'], source_community, target_community])

    # Create a pandas DataFrame from the edge data
    df_edges_with_communities = pd.DataFrame(edge_data, columns=['source', 'target', 'weight', 'source_community', 'target_community'])

    # Save the DataFrame to a CSV file
    df_edges_with_communities.to_csv('results/louvain/edge_list_with_communities_Louvain.csv', index=False)

    G1, pos1, color_list_community1, community_num_group1, max_k_w1, modularity1, partition1 = simple_Leiden(G)
    print(f"Leiden Modularity: {modularity1}")  # Print the modularity

    # Create a list to store the edge data
    edge_data = []

    # Iterate over the edges in the graph
    for u, v, data in G1.edges(data=True):
        # Get the community assignments for the source and target nodes
        u_index = list(G1.nodes()).index(u)
        v_index = list(G1.nodes()).index(v)

        # Access community assignments using membership attribute and node indices
        source_community = partition1.membership[u_index] # Use partition1 here
        target_community = partition1.membership[v_index] # Use partition1 here

        # Append the edge data to the list
        edge_data.append([u, v, data['weight'], source_community, target_community])

    # Create a pandas DataFrame from the edge data
    df_edges_with_communities = pd.DataFrame(edge_data, columns=['source', 'target', 'weight', 'source_community', 'target_community'])

    # Save the DataFrame to a CSV file
    df_edges_with_communities.to_csv('results/leiden/edge_list_with_communities_Leiden.csv', index=False)

    #Visualize Louvain
    edges = G.edges()
    Feature_color_sub = color_list_community
    node_size = 350

    fig = plt.figure(figsize=(200, 100))
    im = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=Feature_color_sub, cmap='jet', vmin=0, vmax=community_num_group)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=50, font_color="black")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im)
    plt.savefig('results/louvain/louvain_image.png')
    plt.show(block=False)

    #Visualize Leiden

    edges = G1.edges()
    Feature_color_sub = color_list_community1
    node_size = 350

    fig = plt.figure(figsize=(400, 200))
    im = nx.draw_networkx_nodes(G1, pos1, node_size=node_size, node_color=Feature_color_sub, cmap='jet', vmin=0, vmax=community_num_group1)
    nx.draw_networkx_edges(G1, pos1)
    nx.draw_networkx_labels(G1, pos1, font_size=50, font_color="black")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im)
    plt.savefig('results/leiden/leiden_image.png')
    plt.show(block=False)
