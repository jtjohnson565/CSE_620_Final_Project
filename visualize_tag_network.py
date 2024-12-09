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

    # Define graph from edgelist
    tag_network = nx.from_pandas_edgelist(links_df, 'source', 'target')

    # Add Tag and Link Attributes
    nx.set_node_attributes(tag_network, {tags_df.at[i, 'name']: {'walk_trap_group': tags_df.at[i, 'group']} for i in tags_df.index})
    nx.set_edge_attributes(tag_network, {(links_df.at[i, 'source'], links_df.at[i, 'target']): {"weight": links_df.at[i, 'value']} for i in links_df.index})
    
    color_dict = {group: list(colors.CSS4_COLORS.values())[i*9] for i, group in enumerate(np.unique(tags_df['group']))}
    
    pos = nx.spring_layout(tag_network, weight = 'weight', k = 6.5, iterations = 550, seed = 389)
    plt.figure(figsize=(75,75))
    nx.draw_networkx(tag_network, pos, node_size = [len(tag)**2 * 55 for tag in tag_network.nodes], linewidths = 1, edgecolors = 'black')
    plt.savefig('plots/tag_network')

    plt.figure(figsize=(75,75))
    nx.draw_networkx(tag_network, pos, node_size = [len(tag)**2 * 55 for tag in tag_network.nodes], node_color = [color_dict[tag_network.nodes[tag]['walk_trap_group']] for tag in tag_network.nodes],
                     linewidths = 1, edgecolors = 'black')
    plt.savefig('plots/tag_network_walktrap_groups')
