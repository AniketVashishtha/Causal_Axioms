import json
import glob
import os

def update_jsonl_file(input_file):
    output_file = input_file.replace('.jsonl', '_updated.jsonl')
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            data = json.loads(line.strip())
            
            # Format the prompt with the specific instruction
            prompt = f"Answer the following question with ONLY 'Yes' or 'No'.\n\nPremise: {data['premise']} {data['hypothesis']}"
            
            # Create new data structure
            updated_data = {
                "prompt": prompt,
                "completion": data["label"]
            }
            
            # Write the updated data
            fout.write(json.dumps(updated_data) + '\n')
    
    print(f"Updated file saved as: {output_file}")
    return output_file

def main():
    # Directory containing the data files
    base_dir = "/data1/aniket/CausalAxiomsV2/data/branching_dsep_bfactor_14/labeled_instances"
    
    # Process files for nodes 5, 8, 10, 12
    node_counts = [5, 8, 10, 12]
    
    for nodes in node_counts:
        input_file = os.path.join(base_dir, f"dsep_instances_{nodes}_nodes.jsonl")
        if os.path.exists(input_file):
            print(f"Processing file for {nodes} nodes...")
            updated_file = update_jsonl_file(input_file)
            print(f"Completed processing {nodes}-node file")
        else:
            print(f"File not found: {input_file}")

if __name__ == "__main__":
    main()