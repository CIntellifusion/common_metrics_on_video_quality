import argparse
import json
import os
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Aggregate video quality metrics from JSON files to CSV')
    parser.add_argument('--metrics-root', required=True, help='Directory containing JSON metric files')
    parser.add_argument('--output-file', required=True, help='Output CSV file path')
    args = parser.parse_args()

    metrics_root = Path(args.metrics_root)
    if not metrics_root.exists():
        raise FileNotFoundError(f"Metrics root not found: {metrics_root}")

    # Collect all records
    records = []
    all_metric_keys = set()

    # Find all JSON files
    json_files = list(metrics_root.glob('*.json'))
    print(f"Found {len(json_files)} JSON files in {metrics_root}")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            record = data['result']
                
            records.append(record)
            
            # Track all metric keys (excluding exp_name)
            all_metric_keys.update(k for k in record.keys() if k != 'exp_name')
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue

    if not records:
        print("No valid records found!")
        return

    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Ensure exp_name is first column, then sort other columns alphabetically
    sorted_cols = ['exp_name'] + sorted([c for c in df.columns if c != 'exp_name'])
    
    # Reorder columns (handle missing columns by filling NaN)
    for col in sorted_cols:
        if col not in df.columns:
            df[col] = None
    
    df = df[sorted_cols]
    
    # Sort by exp_name
    df = df.sort_values('exp_name').reset_index(drop=True)
    
    # Save to CSV
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Saved summary to: {output_path}")
    print(f"Total experiments: {len(df)}")
    print(f"Metrics columns: {list(df.columns[1:])}")

if __name__ == "__main__":
    main()