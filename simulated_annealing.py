import random
import numpy as np
import networkx as nx
from problem import ModularityMaximization

# Implement Simulated Annealing for Modularity Maximization without Recursively Merging Communities into One Node
def Simulated_Annealing_No_Merge(mod_max, T_max, cooling_method, cooling_alpha, T_f = 0.01, k_max = np.inf, resolution = 1.0, random_state = 42):
    assert cooling_method == "lin" or cooling_method == "log" or cooling_method == "exp"
    assert k_max > 0
    assert T_max > T_f and T_f >= 0

    k = 0
    T = T_max
    cur_mod_max = mod_max
    best_mod_max = mod_max

    while T > T_f and k < k_max:
        # Generate Random Node and Join with Random Connected Community
        neighbor_mod_max = None
        
        node = random.choice(list(cur_mod_max.tag_network.nodes))
        #node_neighbors = list(tag_network.neighbors(node))
        community = random.choice(np.unique(list(nx.get_node_attributes(cur_mod_max.tag_network, 'community').values())))

        
        neighbor_mod_max = cur_mod_max.get_neighbor_from_node_transfer(node, community)

        

        if neighbor_mod_max.get_modularity(resolution) > cur_mod_max.get_modularity(resolution):
            cur_mod_max = neighbor_mod_max

            if cur_mod_max.get_modularity(resolution) > best_mod_max.get_modularity(resolution):
                best_mod_max = cur_mod_max
            
        elif np.random.uniform(0, 1) < (np.exp((cur_mod_max.get_modularity(resolution) - neighbor_mod_max.get_modularity(resolution)) / T)):
            cur_mod_max = neighbor_mod_max
            
        k += 1
        T = cooling(cooling_method, k, cooling_alpha, T_max)

        if (k+1) % 50 == 0:
            print("Iteration #{}:".format(k+1))
            print("\tTemperature: {}".format(T))
            print("\tCurrent Modularity: {}".format(cur_mod_max.get_modularity(resolution)))
            print("\tBest Modularity: {}".format(best_mod_max.get_modularity(resolution)))
            print("\tNeighbor Modularity: {}\n".format(neighbor_mod_max.get_modularity(resolution)))

    return best_mod_max


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
    
    opt_mm = Simulated_Annealing_No_Merge(init_mm, T_max = 500, cooling_method = 'lin', cooling_alpha = 0.95, T_f = 0.1)
    print("Modularity:", opt_mm.get_modularity())
    print("Number of Communities Remaining:", len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, 'community').values()))))
    opt_mm.plot_communities('plots/sa_results')
