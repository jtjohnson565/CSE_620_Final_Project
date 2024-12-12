import math, copy, argparse
import numpy as np
import networkx as nx
import matplotlib.colors as colors
from problem import ModularityMaximization

# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--plot_output_filepath', type = str, default = "results/simulated_annealing/plots/sa_plot_0.png", help = " Output File Path for the Generated Graph Image")
parser.add_argument('--edgelist_output_filepath', type = str, default = "results/simulated_annealing/edge_lists/sa_edges_0.csv", help = "Output File Path for the Edge List with Corresponding Community Assignments")
parser.add_argument('--log_path_filepath', type = str, default = "results/simulated_annealing/logs/sa_log_0.txt", help = "Output File Path for the Generating Log for Program Execution")
parser.add_argument('--t_max', type = float, default = 1000.0, help = "Maximum (Initial) Temperature")
parser.add_argument('--t_final', type = float, default = 0.01, help = "Final Temperature")
parser.add_argument('--cooling_method', type = str, choices = ['lin', 'log', 'exp'], default = 'log', help = "Method of Cooling Temperature for the Algorithm")
parser.add_argument('--cooling_rate', type = float, default = 0.99, help = "Rate at Which the Temperature Cools for the Algorithm")
parser.add_argument('--k_max', type = int, default = np.inf, help = "Maximum Number of Iterations to Allow the Simulated Annealing to Run")
parser.add_argument('--n_min_moves', type = int, default = 10, help = "Minimum Number of Times to Perform the Neighborhood Operator in Each Iteration to Help Escape Local Minima")
parser.add_argument('--n_max_moves', type = int, default = 25, help = "Maximum Number of Times to Perform the Neighborhood Operator in Each Iteration to Help Escape Local Minima")
parser.add_argument('--resolution', type = float, default = 1.0, help = "Resolution Parameter for Calculating the Modularity of the Network")
parser.add_argument('--sa_random_state', type = int, default = None, help = "Random Seed to Provide Consistent Output for Simulated Annealing")
parser.add_argument('--plot_random_state', type = int, default = 389, help = "Random Seed to Provide Consistent Plot Layouts")


##########################################################################################################################################################
# Implement Simulated Annealing for Modularity Maximization without Recursively Merging Community Nodes into One Node
def Simulated_Annealing_No_Merge(mod_max, T_max, T_f, cooling_method, cooling_rate, k_max, n_min_moves, n_max_moves, resolution, random_state):
    assert cooling_method == "lin" or cooling_method == "log" or cooling_method == "exp"
    assert k_max > 0
    assert T_max > T_f and T_f >= 0
    assert n_min_moves >= 1 and n_max_moves >= n_min_moves

    k = 0
    T = T_max
    cur_mod_max = copy.deepcopy(mod_max)

    if random_state is not None:
        np.random.seed(random_state)

    # Assign Each Node to It's Own Community
    for i, n in enumerate(mod_max.tag_network.nodes):
        cur_mod_max.tag_network.nodes[n]['community'] = i

    cur_mod_max.set_community_colors()
    
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
        T = cooling(cooling_method, k, cooling_rate, T_max)
        
        if (k+1) % 50 == 0:
            print("Iteration #{}:".format(k+1))
            print("\tTemperature: {}".format(T))
            print("\tCurrent Modularity: {}".format(cur_mod_max.get_modularity()))
            print("\tBest Modularity: {}".format(best_mod_max.get_modularity()))
            print("\tNumber of Communities Remaining in Current Modularity: {}".format(len(np.unique(list(nx.get_node_attributes(cur_mod_max.tag_network, 'community').values())))))
            print("\tNumber of Communities Remaining in Best Modularity: {}\n".format(len(np.unique(list(nx.get_node_attributes(best_mod_max.tag_network, 'community').values())))))
    
    return best_mod_max


##########################################################################################################################################################
# Generate Between a Certain Range of Neighborhood Moves, While Making sure They Don't Cause Split Communities with the Same ID
def generate_neighbor(mod_max, n_min_moves, n_max_moves, T, T_f):
    neighbor_mod_max = copy.deepcopy(mod_max)

    if T > T_f + 1:
        n_moves = np.random.choice(np.arange(n_min_moves, n_max_moves + 1))
    else:
        n_moves = 1
    
    # Get Community Information
    communities_dict = nx.get_node_attributes(neighbor_mod_max.tag_network, 'community')
    communities = np.unique(list(communities_dict.values()))
            
    # Iterate n times to avoid getting stuck in local extrema
    for i in range(0, n_moves):
        while True:
            n = np.random.choice(list(neighbor_mod_max.tag_network.nodes))
            c = np.random.choice(communities)

            if len([x for x in list(communities_dict.values()) if x == c]) == 0 or \
               len([x for x in neighbor_mod_max.tag_network.edges if (x[0] == n and neighbor_mod_max.tag_network.nodes[x[1]]['community'] == c) or (x[1] == n and neighbor_mod_max.tag_network.nodes[x[0]]['community'] == c)]) > 0:
                neighbor_mod_max = neighbor_mod_max.get_neighbor_from_node_transfer(n, c)
                break
            
    return neighbor_mod_max


##########################################################################################################################################################
# Implement Cooling for Simulated Annealing
def cooling(cooling_method, k, cooling_rate, T_max):
    assert cooling_method == "lin" or cooling_method == "log" or cooling_method == "exp"
    
    if cooling_method == "lin":
        return T_max / (1 + cooling_rate * k)

    elif cooling_method == "log":
        return T_max / (1 + cooling_rate * np.log(1 + k))

    else:
        return T_max * (cooling_rate ** k)


##########################################################################################################################################################
if __name__ == "__main__":
    args = parser.parse_args()

    # Run Simulated Annealing Implementation
    init_mm = ModularityMaximization()
    opt_mm = Simulated_Annealing_No_Merge(init_mm,
                                          T_max = args.t_max,
                                          T_f = args.t_final,
                                          cooling_method = args.cooling_method,
                                          cooling_rate = args.cooling_rate,
                                          k_max = args.k_max,
                                          n_min_moves = args.n_min_moves,
                                          n_max_moves = args.n_max_moves,
                                          resolution = args.resolution,
                                          random_state = args.sa_random_state)

    # Print Modularity Results and Output Plot and Edge List to Desired Paths
    print("Modularity: {}".format(opt_mm.get_modularity()))
    print("Number of Communities Remaining: {}".format(len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, 'community').values())))))
    opt_mm.plot_communities(args.plot_output_filepath)
    opt_mm.output_edgelist(args.edgelist_output_filepath)

    # Write Information to Log File
    with open(args.log_path_filepath, "w") as text_file:
        text_file.write("plot_output_filepath: {}\n".format(args.plot_output_filepath))
        text_file.write("edgelist_output_filepath: {}\n\n".format(args.edgelist_output_filepath))

        text_file.write("Modularity: {}\n".format(opt_mm.get_modularity()))
        text_file.write("Number of Communities Remaining: {}\n\n".format(len(np.unique(list(nx.get_node_attributes(opt_mm.tag_network, 'community').values())))))
        
        text_file.write("t_max: {}\n".format(args.t_max))
        text_file.write("t_final: {}\n".format(args.t_final))
        text_file.write("cooling_method: {}\n".format(args.cooling_method))
        text_file.write("cooling_rate: {}\n".format(args.cooling_rate))
        text_file.write("k_max: {}\n".format(args.k_max))
        text_file.write("n_min_moves: {}\n".format(args.n_min_moves))
        text_file.write("n_max_moves: {}\n".format(args.n_max_moves))
        text_file.write("resolution: {}\n".format(args.resolution))
        text_file.write("sa_random_state: {}\n".format(args.sa_random_state))
        text_file.write("plot_random_state: {}\n".format(args.plot_random_state))

    text_file.close()
