import os, copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx

############################################################################################################################
# Create Class Which Represent an Instance of the Modularity Maximization Problem
class ModularityMaximization():
    def __init__(self, mod_max = None):
        if mod_max == None:
            # Extract DataFrames
            tags_df = pd.read_csv('stack_network_nodes.csv')
            links_df = pd.read_csv('stack_network_links.csv')

            # Define Dictionaries of Tag IDs
            tag_to_id_dict = {tags_df.at[i, 'name']: str(i) for i in tags_df.index}
            id_to_tag_dict = {i: tags_df.at[i, 'name'] for i in tags_df.index}

            # Replace Tags with IDs
            tags_df = tags_df.replace(tag_to_id_dict)
            tags_df['name'] = tags_df['name'].astype(int)
            links_df = links_df.replace(tag_to_id_dict)
            links_df['source'] = links_df['source'].astype(int)
            links_df['target'] = links_df['target'].astype(int)

            # Define Graph Object from Edgelist
            self.tag_network = nx.from_pandas_edgelist(links_df, 'source', 'target', create_using = nx.Graph())

            # Add Tag and Link Attributes
            nx.set_node_attributes(self.tag_network, {i: {'name': id_to_tag_dict[i], 'community': i, 'color': None} for i in self.tag_network.nodes}) # Initially set Vertex to its own Community
            nx.set_edge_attributes(self.tag_network, {(links_df.at[i, 'source'], links_df.at[i, 'target']): {"weight": links_df.at[i, 'value']} for i in links_df.index})

            
            
        else:
            self.tag_network = mod_max.tag_network
            self.color_dict = mod_max.color_dict

        # Set Other Graph Attributes
        self.total_edge_weights = np.sum(list(nx.get_edge_attributes(self.tag_network, 'weight').values()))

        
    ############################################################################################################################   
    # Get the Fitness for the Current Instance of the Problem
    def get_modularity(self, resolution = 1.0):
        sum_mod = 0
        
        for i, j in self.tag_network.edges:
            if self.tag_network.nodes[i]['community'] == self.tag_network.nodes[j]['community']:
                a_ij = self.tag_network.get_edge_data(i, j)['weight']
                k_i = np.sum([self.tag_network.get_edge_data(n1, n2)['weight'] for n1, n2 in self.tag_network.edges(i)])
                k_j = np.sum([self.tag_network.get_edge_data(n1, n2)['weight'] for n1, n2 in self.tag_network.edges(j)])

                sum_mod += a_ij - resolution * ((k_i * k_j) / (2 * self.total_edge_weights))

        return (1 /(2 * self.total_edge_weights)) * sum_mod

              
    ############################################################################################################################
    # Get the Neighbor of the Current Graph that is Obtained through Transfering a Given Node to a Given Community
    def get_neighbor_from_node_transfer(self, node_id, community_id):
        neighbor_network = copy.deepcopy(self)
        neighbor_network.tag_network.nodes[node_id]['community'] = community_id
        neighbor_network.tag_network.nodes[node_id]['color'] = self.color_dict[community_id]

        return ModularityMaximization(mod_max = neighbor_network)


    ############################################################################################################################
    # Generate Random Colors for Each Community While Making Sure They're All Not Too Close to Each Other
    def set_community_colors(self):
        self.color_dict = {i: None for i in np.unique(list(nx.get_node_attributes(self.tag_network, 'community').values()))}

        for i in self.color_dict.keys():
            while True:
                color = np.random.choice(range(50, 256), 3)

                if len([col for col in [self.color_dict[k] for k in self.color_dict.keys() if self.color_dict[k] is not None] if math.dist(color, col) < 40.0]) == 0:
                    self.color_dict[i] = color
                    break

        self.color_dict = {i: colors.to_hex(self.color_dict[i] / 255) for i in self.color_dict.keys()}
        nx.set_node_attributes(self.tag_network, self.color_dict, 'color')
    
    
    ############################################################################################################################
    # Plot the Current Instance of the Problem
    def plot_communities(self, plot_path, k = 6.5, iterations = 550, seed = 389):
        plt.figure(figsize=(75,75))
        
        pos = nx.spring_layout(self.tag_network, weight = 'weight', k = k, iterations = iterations, seed = seed)
        nx.draw_networkx(self.tag_network, pos, labels = nx.get_node_attributes(self.tag_network, 'name'),
                         node_size = [len(tag)**2 * 65 for tag in nx.get_node_attributes(self.tag_network, 'name').values()],
                         node_color = nx.get_node_attributes(self.tag_network, 'color').values(), linewidths = 1, edgecolors = 'black')
        
        plt.savefig(plot_path)

        
    ############################################################################################################################
    # Plot the Current Instance of the Problem
    def output_edgelist(self, edgelist_output_path):
        edgelist_dict = {i: {'source': u, 'target': v, 'weight': self.tag_network.get_edge_data(u, v)['weight'], 'source_community': self.tag_network.nodes[u]['community'], 'target_community': self.tag_network.nodes[v]['community']} \
                         for i, (u, v) in enumerate(self.tag_network.edges)}
        edgelist_df = pd.DataFrame.from_dict(edgelist_dict, orient = 'index', columns = ['source', 'target', 'weight', 'source_community', 'target_community'])
        edgelist_df.to_csv(edgelist_output_path, index= False)
