import igraph as ig
import networkx as nx
import numpy as np
np.random.seed(1)
import pdb
import string
import random
from itertools import combinations
import sys
import pickle

def unique_tuples(tuples_list):
    """
    Returns a list of unique tuples from the given list of tuples,
    preserving their order and ensuring uniqueness.
    """
    seen = set()
    unique_list = []
    for t in tuples_list:
        if t not in seen:
            seen.add(t)
            unique_list.append(t)
    return unique_list


def topological_ordering(GRAPH):

    """
    Performs a topological ordering of the nodes in a Directed Acyclic Graph (DAG).
    Args:
        graph (list): The DAG represented as a list of edges.
    Returns:
        list: The topologically ordered list of nodes.
    """
    # Create an empty directed graph
    G = nx.DiGraph()
    # Add edges from the adjacency matrix
    edges = GRAPH
    G.add_edges_from(edges)
    # Return the topological ordering of the nodes

    return list(nx.topological_sort(G))


def subsample_dict(input_dict, sample_size=2):
    """
    Randomly subsample key-value pairs from a given dictionary.

    Parameters:
    - input_dict: dict, the dictionary to subsample from
    - sample_size: int, the number of key-value pairs to subsample

    Returns:
    - dict, a new dictionary with the subsampled key-value pairs
    """
    if sample_size > len(input_dict):
        raise ValueError("Sample size cannot be greater than the number of available items in the dictionary.")

    # Randomly sample keys
    sampled_keys = random.sample(list(input_dict.keys()), sample_size)

    # Create a new dictionary with the sampled key-value pairs
    sampled_dict = {key: input_dict[key] for key in sampled_keys}

    return sampled_dict

import random

def subsample_dict_equal_classes(input_dict, sample_size):
    """
    Subsample a dictionary to get an equal number of key-value pairs from both value classes ('Yes' and 'No').

    Parameters:
    - input_dict: dict, the dictionary to subsample from
    - sample_size: int, the total number of key-value pairs to subsample

    Returns:
    - dict, a new dictionary with the subsampled key-value pairs
    """
    if sample_size % 2 != 0:
        raise ValueError("Sample size must be even to ensure equal classes.")
    
    half_size = sample_size // 2
    
    yes_items = [(k, v) for k, v in input_dict.items() if v == 'Yes']
    no_items = [(k, v) for k, v in input_dict.items() if v == 'No']
    
    if len(yes_items) < half_size or len(no_items) < half_size:
        raise ValueError("Not enough elements to sample from each class.")

    # Randomly sample from each class
    sampled_yes = random.sample(yes_items, half_size)
    sampled_no = random.sample(no_items, half_size)
    
    # Combine the samples into a new dictionary
    sampled_dict = {k: v for k, v in sampled_yes + sampled_no}
    
    return sampled_dict




def sample_graph(num_nodes,num_edges,graph_type):
    '''
    We will sample a DAG from from the Erdos-Reneyi Model with given
    number of node and edge.
    '''
    if graph_type=="ER":
        #Erdos-Renayi
        G_und = ig.Graph.Erdos_Renyi(n=num_nodes,m=num_edges)
        B_und = _graph_to_adj_matrix(G_und)
        B = _dagify_randomly(B_und)
    elif graph_type=="SF":
        #Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=num_nodes,
                                m=int(round(num_edges/num_nodes)),
                                directed=True)
        B = _graph_to_adj_matrix(G)
    else:
        raise ValueError("unknown graph type")

    #Now we have a adj mat of DAG, just permute it
    B_perm = _permute_adj_matrix(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag(),"AdjMat not DAG"

    return B_perm

def _permute_adj_matrix(adj_mat):
    P = np.random.permutation(np.eye(adj_mat.shape[0]))
    return P.T @ adj_mat @ P

def _graph_to_adj_matrix(G):
    return np.array(G.get_adjacency().data)

def _dagify_randomly(adj_mat):
    return np.tril(_permute_adj_matrix(adj_mat),k=-1)

def get_graph_metrics(adj_mat):
    num_edges=np.sum(adj_mat)
    exp_indegree=np.mean(np.sum(adj_mat,axis=1))

    return (num_edges,exp_indegree)


def is_reachable(graph, start, target, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    
    if start == target:
        return 'Yes'
    
    for edge in graph:
        if edge[0] == start and edge[1] not in visited:
            if is_reachable(graph, edge[1], target, visited) == 'Yes':
                return 'Yes'
    
    return 'No'


def generate_statements_w_all_pairs(tuple_list, node_lst):
    #This function will generate a statement along with a question made of all possible node pairs.
    # print("Tuple List:", tuple_list)
    ans_list = []
    branch_list = {}
    # Initialize an empty set to store unique elements
    unique_elements = set()

    # Convert the set of unique elements back to a list if needed
    unique_elements_list = node_lst
    # print("Number of nodes: ", len(node_lst))
    all_pairs = list(combinations(unique_elements_list, 2))
    # Create a list of reverse combinations
    reverse_pairs = [(pair[1], pair[0]) for pair in all_pairs]
    # Combine original and reverse combinations to form all possible pairs
    all_combinations = all_pairs + reverse_pairs
    all_combinations_final = unique_tuples(all_combinations)

    relationships = []
    for group in tuple_list:
        cause = group[0]
        effect = group[1]
        # print(f"'{cause}' causes '{effect}'.", end=' ')
        # print('\n')
        relationships.append(f"{cause} causes {effect}.")

    for pairs in all_combinations_final:
        merged_structure = ' '.join(relationships)
        branch_list[merged_structure + f" Does {pairs[0]} cause {pairs[-1]}?"] = is_reachable(tuple_list, pairs[0], pairs[-1])

    return branch_list

def dict_to_tuples (graph_dict):
     # Initialize an empty list to store directed edges
    directed_edges = []

    # Iterate through each key-value pair in the graph dictionary
    for key, values in graph_dict.items():
        # Add each directed edge to the list of tuples
        for value in values:
            directed_edges.append((key, value))

    # Print the list of directed edges
    return(directed_edges)



if __name__=="__main__":
    #Now lets test the DAG creation with expected number of edges etc.
    node_lst = [5,8,10,12]

    for num_nodes in node_lst:
        sys.stdout = open('/home/t-aniketva/Desktop/CausalAxioms/branching_transitivity/'+str(num_nodes)+'_node_branching_14_bfactor.txt', 'w')

        graph_metrics=[]
        num_graphs=300

        #Creating the arguments for generator
        # generator_args={}
        # generator_args["scale_alpha"]=5

        
        final_merged_dict = {}
        for idx in range(num_graphs):
            #Generating the graph
            graph_type = "ER" #if idx%2==0 else "ER"
            adj_mat = sample_graph(num_nodes=num_nodes,
                                        num_edges=int(num_nodes*1.4),
                                        graph_type=graph_type)
            #Getting the graph metrics
            graph_metrics.append(get_graph_metrics(adj_mat))
            #pdb.set_trace()
            # print("Adjacency Matrix: ", adj_mat)
        # print("Adjacency Metrics: ", graph_metrics)

            characters = string.ascii_letters + string.digits  # alphabets (both lowercase and uppercase) + digits
            alphanumeric_names = []
            final_dag_list = []
            # Define the characters to use for generating alphanumeric names
            characters = string.ascii_letters + string.digits
            for i in range(num_nodes):
                rand_len = random.randint(1, 3)
                # Generate a random alphanumeric name of length 5
                name = ''.join(random.choice(characters) for _ in range(rand_len))
                alphanumeric_names.append(name)

            node_labels = alphanumeric_names

            if len(node_labels) != num_nodes:
                breakpoint()
            
            edge_list = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj_mat[i, j] == 1:
                        edge_list.append((node_labels[i], node_labels[j]))
            
            # print(edge_list)
            # cd = dict_to_tuples(edge_list)
            try:
                order = topological_ordering(edge_list)
                final_dag_list.append(edge_list)

            except ValueError as e:
                print(f"Skipping example {i} due to error: {e}")
            except Exception as e:
            # Catch any other unexpected exceptions
                print(f"Skipping example {i} due to unexpected error: {e}")
            
            
            for dag in final_dag_list:
                branch_graph = generate_statements_w_all_pairs(dag, node_labels)
                # sampled_dict = subsample_dict(branch_graph, 2)
                sampled_dict = subsample_dict_equal_classes(branch_graph, 2)
                final_merged_dict.update(sampled_dict)

        result_dict = {}
        final_dict = subsample_dict(final_merged_dict, 500)
        result_dict.update(final_dict)
        print("Final graph dict: ",result_dict)
        print("Total test instances: ",len(result_dict))

        with open('/home/t-aniketva/Desktop/CausalAxioms/branching_transitivity/'+str(num_nodes)+'_node_branching_14_bfactor.pkl', 'wb') as f:
            pickle.dump(result_dict, f)

    
    

   