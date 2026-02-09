
import os
import sys
import numpy as np
from collections import defaultdict

# Add parent directory to path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from TopologyDataLoader import TopologyDataLoader

def run_inspection():
    # Define paths
    data_folder = os.path.join(parent_dir, 'data')
    
    print(f"Loading data from: {data_folder}")
    
    # Initialize loader
    loader = TopologyDataLoader(data_folder=data_folder)
    
    try:
        data = loader.load()
    except Exception as e:
        print(f"FAILED to load data: {e}")
        return

    # Group data by number of links
    grouped_data = defaultdict(list)
    
    for i, item in enumerate(data):
        n_binary = item.get('number_of_binary_links', 0)
        n_ternary = item.get('number_of_ternary_links', 0)
        n_quaternary = item.get('number_of_quaternary_links', 0)
        
        total_links = n_binary + n_ternary + n_quaternary
        grouped_data[total_links].append((i, item))

    # Helper to format numpy arrays
    def format_val(v):
        if isinstance(v, np.ndarray):
            return str(v).replace('\n', '\n' + ' '*22) # Indent array content
        return str(v)

    # Process each group
    for link_count, items in grouped_data.items():
        filename = f"topology_visualization_{link_count}bar.txt"
        output_file = os.path.join(current_dir, filename)
        
        lines = []
        lines.append("="*100)
        lines.append(f" METHEUS DYNAMICS - {link_count}-BAR TOPOLOGY INSPECTION")
        lines.append("="*100)
        lines.append(f"Total {link_count}-bar Topologies: {len(items)}")
        lines.append(f"Source: {loader.filepath}")
        lines.append("-" * 100)
        lines.append("")
        
        for idx, item in items:
            lines.append(f"TOPOLOGY ID: {idx}")
            lines.append("~"*30)
            
            keys_order = [
                'number_of_binary_links', 'number_of_ternary_links', 'number_of_quaternary_links',
                'index_of_ground_link', 'number_of_joints_of_ground_link', 'input_link_index',
                'joint_type_list', 'rocker_list', 'end_effector_link_list',
                'array_of_adjacency_matrices'
            ]
            
            for key in keys_order:
                if key in item:
                    val = item[key]
                    lines.append(f"  {key:<35} : {format_val(val)}")
            
            lines.append("")
            lines.append("-" * 100)
            lines.append("")
            
        # Save file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"[{link_count}-bar] Saved {len(items)} items to: {filename}")
        except Exception as e:
            print(f"Error writing {filename}: {e}")

if __name__ == "__main__":
    run_inspection()
