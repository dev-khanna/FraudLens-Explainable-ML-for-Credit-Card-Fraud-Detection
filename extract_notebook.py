import json
import sys

def extract_notebook(filepath, output_path):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb.get('cells', [])):
            cell_type = cell.get('cell_type', '')
            source = "".join(cell.get('source', []))
            
            if cell_type == 'markdown':
                out.write(f"### [Markdown Cell {i+1}]\n{source}\n\n")
            elif cell_type == 'code':
                out.write(f"### [Code Cell {i+1}]\n```python\n{source}\n```\n\n")

if __name__ == '__main__':
    extract_notebook('eda.ipynb', 'eda_summary.md')
