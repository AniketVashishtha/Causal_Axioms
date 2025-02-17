import os

def update_txt_file(input_file):
    output_file = input_file.replace('.txt', '_updated.txt')
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            # Split by tab to separate the instance from the label
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print(f"Skipping malformed line: {line}")
                continue
                
            instance, label = parts
            
            # Format the output line - keep premise and hypothesis together
            output_line = f"'{instance}' '{label}'\n"
            fout.write(output_line)
    
    print(f"Updated file saved as: {output_file}")
    return output_file

def main():
    # Directory containing the data files
    base_dir = "/data1/aniket/CausalAxiomsV2/data/branching_dsep_bfactor_14/labeled_instances"
    
    # Process files for nodes 5, 8, 10, 12
    node_counts = [5, 8, 10, 12]
    
    for nodes in node_counts:
        input_file = os.path.join(base_dir, f"dsep_instances_{nodes}_nodes.txt")
        if os.path.exists(input_file):
            print(f"Processing file for {nodes} nodes...")
            updated_file = update_txt_file(input_file)
            print(f"Completed processing {nodes}-node file")
        else:
            print(f"File not found: {input_file}")

if __name__ == "__main__":
    main()