import os
from pathlib import Path

def smart_tree(directory, indent="", ignore_list=None, shallow_list=None):
    """
    Scans directory and prints a tree.
    - ignore_list: Folders/files to hide completely.
    - shallow_list: Folders to show, but NOT list their contents.
    """
    if ignore_list is None:
        ignore_list = {'.git', 'venv', '__pycache__', '.ipynb_checkpoints', 'node_modules'}
    
    if shallow_list is None:
        # We acknowledge these exist, but we don't open them.
        shallow_list = {'numpy files', 'Engines', 
                        'Dataset++', 'Dataset++_No_Aug', 
                        'Dataset', 'Dataset_SMOTE' }

    root = Path(directory)
    
    # 1. Filter items based on ignore_list
    items = [item for item in root.iterdir() if item.name not in ignore_list]
    
    # 2. Sort: Folders first, then files
    items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        # 3. Logic for Shallow Folders
        if item.is_dir() and item.name in shallow_list:
            # Count files inside for a better summary (optional but helpful)
            file_count = len(list(item.glob('*')))
            print(f"{indent}{connector}{item.name}/ ({file_count} items hidden)")
            continue # Skip the recursive call!
            
        print(f"{indent}{connector}{item.name}")
        
        # 4. Standard recursion for non-shallow directories
        if item.is_dir():
            new_indent = indent + ("    " if is_last else "│   ")
            smart_tree(item, new_indent, ignore_list, shallow_list)

# --- EXECUTION ---
if __name__ == "__main__":
    # Point this to your project root
    target_project = Path.cwd() 
    print(f"📁 {target_project.name}/")
    smart_tree(target_project)
    
    