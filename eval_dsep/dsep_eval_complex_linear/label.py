from collections import defaultdict, deque
import json
import os
import re
import random

class Graph:
    def __init__(self):
        self.edges = defaultdict(list)
        self.parents = defaultdict(list)
        
    def add_edge(self, from_node, to_node):
        self.edges[from_node].append(to_node)
        self.parents[to_node].append(from_node)
    
    def get_paths(self, start, end):
        paths = []
        
        def dfs(current, target, path, visited):
            if current == target:
                paths.append(path[:])
                return
            
            neighbors = set(self.edges[current])
            neighbors.update(self.parents[current])
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor, target, path + [neighbor], visited)
                    visited.remove(neighbor)
        
        dfs(start, end, [start], {start})
        return paths
    
    def is_d_separated(self, x, y, z=None):
        if z is None:
            z = set()
        else:
            z = set(z)
            
        paths = self.get_paths(x, y)
        for path in paths:
            if not self._is_path_blocked(path, z):
                return False
        return True
    
    def _is_path_blocked(self, path, z):
        n = len(path)
        for i in range(1, n-1):
            prev_node = path[i-1]
            curr_node = path[i]
            next_node = path[i+1]

            if curr_node in z and ((prev_node in self.parents[curr_node] and next_node in self.edges[curr_node]) or 
                                 (prev_node in self.edges[curr_node] and next_node in self.parents[curr_node])): 
                return True
                
            if curr_node in z and prev_node in self.edges[curr_node] and next_node in self.edges[curr_node]:
                return True
                
            if curr_node not in z and prev_node in self.parents[curr_node] and next_node in self.parents[curr_node]:
                if not self._has_active_descendant(curr_node, z):
                    return True
        return False
    
    def _has_active_descendant(self, node, z):
        visited = {node}
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            if current in z and current != node:
                return True
                
            for neighbor in self.edges[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

def parse_graph_and_query(line):
    """Parse a line containing graph description and d-separation query"""
    # Split on the d-separation query
    parts = line.split("Are")
    if len(parts) != 2:
        return None, None, None, None
        
    graph_desc = parts[0].strip()
    query = "Are" + parts[1].strip()
    
    # Parse graph edges
    g = Graph()
    edges = graph_desc.split(".")
    for edge in edges:
        edge = edge.strip()
        if edge:
            parts = edge.split("causes")
            if len(parts) == 2:
                from_node = parts[0].strip()
                to_node = parts[1].strip()
                g.add_edge(from_node, to_node)
    
    # Parse query
    query_match = re.search(r"Are\s*(.*?)\s+and\s+(.*?)\s+d-separated\s+given\s+{(.*?)}\?", query)
    if not query_match:
        # Try without conditioning set
        query_match = re.search(r"Are\s*(.*?)\s+and\s+(.*?)\s+d-separated\?", query)
        if query_match:
            node1, node2 = query_match.groups()
            evidence = []
            return g, node1.strip(), node2.strip(), evidence
        return None, None, None, None
        
    node1, node2, evidence = query_match.groups()
    evidence = [e.strip() for e in evidence.split(",")]
    return g, node1.strip(), node2.strip(), evidence

def process_files():
    base_dir = "/data1/aniket/CausalAxiomsV2/data/complex_linear_subset_test/dsep_instances"
    output_dir = os.path.join(base_dir, "final")
    os.makedirs(output_dir, exist_ok=True)
    
    lengths = range(7, 16)  # 7 to 15 inclusive
    
    for length in lengths:
        input_file = os.path.join(base_dir, f"dsep_instances_{length}_len.txt")
        print(f"\nProcessing length {length} file: {input_file}")
        
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            continue
        
        # Initialize lists for yes/no instances
        yes_instances = []
        no_instances = []
        
        with open(input_file, 'r') as f:
            for line in f:
                # Skip if we already have enough instances
                if len(yes_instances) >= 250 and len(no_instances) >= 250:
                    break
                    
                line = line.strip()
                g, node1, node2, evidence = parse_graph_and_query(line)
                if g is None:
                    continue
                
                is_dseparated = g.is_d_separated(node1, node2, evidence)
                label = "Yes" if is_dseparated else "No"
                
                instance_data = {
                    "line": line,
                    "jsonl": {
                        "prompt": f"Answer the following question with ONLY 'Yes' or 'No'.\n\nPremise: {line}",
                        "completion": label
                    }
                }
                
                # Only add if we need more instances of this label
                if label == "Yes" and len(yes_instances) < 250:
                    yes_instances.append(instance_data)
                elif label == "No" and len(no_instances) < 250:
                    no_instances.append(instance_data)
        
        print(f"Found {len(yes_instances)} Yes instances and {len(no_instances)} No instances")
        
        # Check if we have enough instances
        if len(yes_instances) < 250 or len(no_instances) < 250:
            print(f"Warning: Not enough instances for balanced dataset!")
            continue
        
        # Take exactly 250 from each category
        yes_instances = yes_instances[:250]
        no_instances = no_instances[:250]
        
        # Combine and shuffle
        random.seed(42)
        all_instances = yes_instances + no_instances
        random.shuffle(all_instances)
        
        # Write output files
        jsonl_output = os.path.join(output_dir, f"dsep_instances_{length}_len_labeled.jsonl")
        txt_output = os.path.join(output_dir, f"dsep_instances_{length}_len_labeled.txt")
        
        with open(jsonl_output, 'w') as jsonl_out, open(txt_output, 'w') as txt_out:
            for instance in all_instances:
                jsonl_out.write(json.dumps(instance["jsonl"]) + '\n')
                txt_out.write(f"'{instance['line']}' '{instance['jsonl']['completion']}'\n")
        
        print(f"\nCreated balanced dataset for length {length}")
        print(f"Total instances: {len(all_instances)}")
        print(f"Yes instances: {len(yes_instances)}")
        print(f"No instances: {len(no_instances)}")
        print(f"Created {jsonl_output}")
        print(f"Created {txt_output}")

if __name__ == "__main__":
    process_files()
