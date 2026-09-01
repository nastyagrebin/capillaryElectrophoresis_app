import pandas as pd
import argparse
import re
import sys

def rename_column(col_name):
    if col_name.lower() == "time":
        return col_name
    
    # Search for the 6 or 7 digit sample name that follows an underscore and precedes a dash
    # Based on actual format: #-#_#-#_[6 or 7 digits]-[date and other stuff]
    match = re.search(r'_(\d{6,7})-', col_name)
    
    if match:
        sample_id = match.group(1)
        if col_name.endswith('_pt'):
            return f"{sample_id}_pt"
        else:
            return sample_id
            
    # Fallback if the pattern doesn't match
    return col_name

def main():
    parser = argparse.ArgumentParser(description="Rename columns in a pseudotimes or alignment CSV file.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("-o", "--output", help="Path to save the renamed CSV file. If not provided, it will save as [input_file]_renamed.csv", default=None)
    
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output
    
    if not output_file:
        if input_file.lower().endswith(".csv"):
            output_file = input_file[:-4] + "_renamed.csv"
        else:
            output_file = input_file + "_renamed.csv"
            
    print(f"Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
        
    original_cols = list(df.columns)
    new_cols = [rename_column(c) for c in original_cols]
    
    df.columns = new_cols
    
    # Print a quick preview of the renames
    print("\nColumn Rename Summary:")
    for orig, new in zip(original_cols, new_cols):
        if orig != new:
            print(f"  {orig}  -->  {new}")
            
    try:
        df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved renamed file to: {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
