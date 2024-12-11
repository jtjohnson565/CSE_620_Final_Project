import random, math, copy
import numpy as np
import networkx as nx
import matplotlib.colors as colors
from problem import ModularityMaximization


##########################################################################################################################################################
# Implement Simulated Annealing for Modularity Maximization without Recursively Merging Community Nodes into One Node
def Simulated_Annealing_No_Merge(mod_max, T_max, cooling_method, cooling_alpha, T_f = 0.01, k_max = np.inf, resolution = 1.0, n_min_moves = 1, n_max_moves = 5, random_state = None):
    assert cooling_method == "lin" or cooling_method == "log" or cooling_method == "exp"
    assert k_max > 0
    assert T_max > T_f and T_f >= 0
    assert n_min_moves >= 1 and n_max_moves >= n_min_moves

    k = 0
    T = T_max
    cur_mod_max = copy.deepcopy(mod_max)

    if random_state is not None:
        random.seed(random_state)

    # Randomly Assign Each Node to Between 2 and |V| - 1 communities
    communities = np.arange(1, np.random.choice(np.arange(2, len(cur_mod_max.tag_network.nodes))))
    for n in mod_max.tag_network.nodes:
        cur_mod_max.tag_network.nodes[n]['community'] = np.random.choice(communities)
        
    best_mod_max = cur_mod_max

    while T > T_f and k < k_max:
        neighbor_mod_max = generate_neighbor(cur_mod_max, n_min_moves, n_max_moves, T, T_f)

        if neighbor_mod_max.get_modularity(resolution) > cur_mod_max.get_modularity(resolution):
            cur_mod_max = neighbor_mod_max

            if cur_mod_max.get_modularity(resolution) > best_mod_max.get_modularity(resolution):
                best_mod_max = cur_mod_max

        elif np.random.uniform(0, 1) < (np.exp((cur_mod_max.get_modularity(resolution) - neighbor_mod_max.get_modularity(resolution)) / T)):
            cur_mod_max = neighbor_mod_max
            
        k += 1
        T = cooling(cooling_method, k, cooling_alpha, T_max)
        
        if (k+1) % 20 == 0:
            print("Iteration #{}:".format(k+1))
            print("\tTemperature: {}".format(T))
            print("\tCurrent Modularity: {}".format(cur_mod_max.get_modularity(resolution)))
            print("\tBest Modularity: {}".format(best_mod_max.get_modularity(resolution)))
            print("\tNumber of Communities Remaining in Current Modularity: {}".format(len(np.unique(list(nx.get_node_attributes(cur_mod_max.tag_network, 'community').values())))))
            print("\tNumber of Communities Remaining in Best Modularity: {}\n".format(len(np.unique(list(nx.get_node_attributes(best_mod_max.tag_network, 'community').values())))))

    return best_mod_max


def generate_neighbor(mod_max, n_min_moves, n_max_moves, T, T_f):
    neighbor_mod_max = copy.deepcopy(mod_max)

    if T > T_f + 1:
        n_moves = np.random.choice(np.arange(n_min_moves, n_max_moves + 1))
    else:
        n_moves = 1
    
    # Get Community Information.
    communities_dict = nx.get_node_attributes(neighbor_mod_max.tag_network, 'community')
    communities = np.unique(list(communities_dict.values()))
            
    # Iterate n times to avoid getting stuck in local extrema
    for i in range(0, n_moves):
        while True:
            n = random.choice(list(neighbor_mod_max.tag_network.nodes))
            c = random.choice(communities)

            if len([x for x in list(communities_dict.values()) if x == c]) == 0 or \
               len([x for x in neighbor_mod_max.tag_network.edges if (x[0] == n and neighbor_mod_max.tag_network.nodes[x[1]]['community'] == c) or (x[1] == n and neighbor_mod_max.tag_network.nodes[x[0]]['community'] == c)]) > 0:
                neighbor_mod_max = neighbor_mod_max.get_neighbor_from_node_transfer(n, c)
                break
            
    return neighbor_mod_max




##########################################################################################################################################################
# Implement Cooling
def cooling(cooling_method, k, alpha, T_max):
    assert cooling_method == "lin" or cooling_method == "log" or cooling_method == "exp"
    
    if cooling_method == "lin":
        return T_max / (1 + alpha * k)

    elif cooling_method == "log":
        return T_max / (1 + alpha * np.log(1 + k))

    else:
        return T_max * (alpha ** k)


if __name__ == "__main__":
    init_mm = ModularityMaximization()
    opt_mm = Simulated_Annealing_No_Merge(init_mm, T_max = 1000, cooling_method = 'log', cooling_alpha = 0.99, T_f = 100, n_min_moves = 1, n_max_moves = 1)
    print("Modularity:", opt_mm.get_modularity())
    print("Number of Communities Remaining:", len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, 'community').values()))))
    opt_mm.plot_communities('plots/SA_Results_1')
    with open("plots/Results_1.txt", "w") as text_file:
        text_file.write("Modularity: {}\nNumber of Communities: {}\n\nT_max: {}\nCooling Method: {}\nCooling Alpha: {}\nT_f: {}\nMin Moves: {}\nMax Moves{}".format(opt_mm.get_modularity(),
                                                                                                                                                            len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, \
                                                                                                                                                                                                               'community').values()))),
                                                                                                                                                                     1000, 'log', 0.99, 100, 1, 1))
        
    init_mm = ModularityMaximization()
    opt_mm = Simulated_Annealing_No_Merge(init_mm, T_max = 1000, cooling_method = 'log', cooling_alpha = 0.99, T_f = 140, n_min_moves = 100, n_max_moves = 250)
    print("Modularity:", opt_mm.get_modularity())
    print("Number of Communities Remaining:", len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, 'community').values()))))
    opt_mm.plot_communities('plots/SA_Results_2')
    with open("plots/Results_2.txt", "w") as text_file:
        text_file.write("Modularity: {}\nNumber of Communities: {}\n\nT_max: {}\nCooling Method: {}\nCooling Alpha: {}\nT_f: {}\nMin Moves: {}\nMax Moves{}".format(opt_mm.get_modularity(),
                                                                                                                                                            len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, \
                                                                                                                                                                                                               'community').values()))),
                                                                                                                                                                     1000, 'log', 0.99, 140, 100, 250))
    
    init_mm = ModularityMaximization()
    opt_mm = Simulated_Annealing_No_Merge(init_mm, T_max = 1000, cooling_method = 'log', cooling_alpha = 0.1, T_f = 600, n_min_moves = 100, n_max_moves = 250)
    print("Modularity:", opt_mm.get_modularity())
    print("Number of Communities Remaining:", len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, 'community').values()))))
    opt_mm.plot_communities('plots/SA_Results_3')
    with open("plots/Results_3.txt", "w") as text_file:
        text_file.write("Modularity: {}\nNumber of Communities: {}\n\nT_max: {}\nCooling Method: {}\nCooling Alpha: {}\nT_f: {}\nMin Moves: {}\nMax Moves{}".format(opt_mm.get_modularity(),
                                                                                                                                                                     len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, \
                                                                                                                                                                                                               'community').values()))),
                                                                                                                                                                     1000, 'log', 0.1, 600, 100, 250))
                                                                                                                                                                     
    init_mm = ModularityMaximization()
    opt_mm = Simulated_Annealing_No_Merge(init_mm, T_max = 1000, cooling_method = 'log', cooling_alpha = 0.99, T_f = 140, n_min_moves = 100, n_max_moves = 250, resolution = 0.5)
    print("Modularity:", opt_mm.get_modularity())
    print("Number of Communities Remaining:", len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, 'community').values()))))
    opt_mm.plot_communities('plots/SA_Results_4')
    with open("plots/Results_4.txt", "w") as text_file:
        text_file.write("Modularity: {}\nNumber of Communities: {}\n\nT_max: {}\nCooling Method: {}\nCooling Alpha: {}\nT_f: {}\nMin Moves: {}\nMax Moves{}\nResolution: {}".format(opt_mm.get_modularity(0.5),
                                                                                                                                                                     len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, \
                                                                                                                                                                                                               'community').values()))),
                                                                                                                                                                     1000, 'log', 0.99, 140, 100, 250, 0.5))

    init_mm = ModularityMaximization()
    opt_mm = Simulated_Annealing_No_Merge(init_mm, T_max = 1000, cooling_method = 'log', cooling_alpha = 0.99, T_f = 140, n_min_moves = 100, n_max_moves = 250, resolution = 1.5)
    print("Modularity:", opt_mm.get_modularity())
    print("Number of Communities Remaining:", len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, 'community').values()))))
    opt_mm.plot_communities('plots/SA_Results_5')
    with open("plots/Results_5.txt", "w") as text_file:
        text_file.write("Modularity: {}\nNumber of Communities: {}\n\nT_max: {}\nCooling Method: {}\nCooling Alpha: {}\nT_f: {}\nMin Moves: {}\nMax Moves{}\nResolution: {}".format(opt_mm.get_modularity(1.5),
                                                                                                                                                                     len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, \
                                                                                                                                                                                                               'community').values()))),
                                                                                                                                                                     1000, 'log', 0.99, 140, 100, 250, 1.5))
