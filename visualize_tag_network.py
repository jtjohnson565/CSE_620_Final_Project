import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx

if __name__ == "__main__":
    # Create plots directory
    if os.path.exists('plots') == False:
        os.mkdir('plots')
    
    # Extract DataFrames
    tags_df = pd.read_csv('stack_network_nodes.csv')
    links_df = pd.read_csv('stack_network_links.csv')
    print(tags_df)
    print(links_df)

    # Define Dictionaries of Tag IDs
    tag_to_id_dict = {tags_df.at[i, 'name']: str(i) for i in tags_df.index}
    id_to_tag_dict = {i: tags_df.at[i, 'name'] for i in tags_df.index}

    # Replace Tags with IDs
    tags_df = tags_df.replace(tag_to_id_dict)
    tags_df['name'] = tags_df['name'].astype(int)
    links_df = links_df.replace(tag_to_id_dict)
    links_df['source'] = links_df['source'].astype(int)
    links_df['target'] = links_df['target'].astype(int)

    print(tags_df)
    print(links_df)
    
    # Define graph from edgelist
    tag_network = nx.from_pandas_edgelist(links_df, 'source', 'target', create_using = nx.Graph())

    # Add Tag and Link Attributes
    nx.set_node_attributes(tag_network, {i: {'name': id_to_tag_dict[i], 'walk_trap_group': tags_df.at[i, 'group']} for i in tag_network.nodes})
    nx.set_edge_attributes(tag_network, {(links_df.at[i, 'source'], links_df.at[i, 'target']): {"weight": links_df.at[i, 'value']} for i in links_df.index})
    
    color_dict = {group: list(colors.CSS4_COLORS.values())[i*9] for i, group in enumerate(np.unique(tags_df['group']))}
    pos = nx.spring_layout(tag_network, weight = 'weight', k = 6.5, iterations = 550, seed = 389)

    # Plot Graph
    plt.figure(figsize=(75,75))
    nx.draw_networkx(tag_network, pos, labels = id_to_tag_dict, node_size = [len(tag)**2 * 55 for tag in nx.get_node_attributes(tag_network, 'name').values()],
                     linewidths = 1, edgecolors = 'black')
    plt.savefig('plots/tag_network.png')

    # Plot Graph with Baseline Walktrap Communities 
    plt.figure(figsize=(75,75))
    nx.draw_networkx(tag_network, pos, labels = id_to_tag_dict, node_size = [len(tag)**2 * 55 for tag in nx.get_node_attributes(tag_network, 'name').values()],
                     node_color = [color_dict[tag_network.nodes[tag]['walk_trap_group']] for tag in tag_network.nodes],
                     linewidths = 1, edgecolors = 'black')
    plt.savefig('plots/tag_network_walktrap_groups.png')
    
    # Get Modularity of Walk-Trap
    sum_mod = 0
    resolution = 1
    total_edge_weights = np.sum(list(nx.get_edge_attributes(tag_network, 'weight').values()))
    
    for i, j in tag_network.edges:
        if tag_network.nodes[i]['walk_trap_group'] == tag_network.nodes[j]['walk_trap_group']:
            a_ij = tag_network.get_edge_data(i, j)['weight']
            k_i = np.sum([tag_network.get_edge_data(n1, n2)['weight'] for n1, n2 in tag_network.edges(i)])
            k_j = np.sum([tag_network.get_edge_data(n1, n2)['weight'] for n1, n2 in tag_network.edges(j)])

            sum_mod += a_ij - resolution * ((k_i * k_j) / (2 * total_edge_weights))

    modularity = (1 /(2 * total_edge_weights)) * sum_mod

    print("\nWalk-Trap Modularity:", modularity)
    print("Number of Walk-Trap Communities:", len(np.unique(list(nx.get_node_attributes(tag_network, 'walk_trap_group').values()))))
