#!/usr/bin/env python3
"""
Script to preprocess datasets for llm-embed project.
Uses functions from data_utils.py to process raw data files.
"""

import argparse
import os
import sys
from data_utils import process_raw_data

def main():
    """
    Main function to parse command line arguments and run preprocessing.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for the llm-embed project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--data-types", type=str, default="bw,fw,math",
                        help="Comma-separated list of data types to process (bw,fw,math)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root directory for data files")
    
    args = parser.parse_args()
    
    # Split the data types to process
    data_types = [dt.strip() for dt in args.data_types.split(",")]
    valid_types = ["bw", "fw", "math"]
    
    # Validate data types
    invalid_types = [dt for dt in data_types if dt not in valid_types]
    if invalid_types:
        print(f"Error: Invalid data type(s): {', '.join(invalid_types)}")
        print(f"Valid data types are: {', '.join(valid_types)}")
        sys.exit(1)
    
    # Check if data directory exists
    if not os.path.isdir(args.data_dir):
        print(f"Error: Root directory '{args.data_dir}' does not exist")
        sys.exit(1)
    
    # Process each data type
    for data_type in data_types:
        print(f"Processing {data_type} data...")
        process_raw_data(data_type, args.data_dir)
        print(f"Finished processing {data_type} data")
    
    print("All preprocessing completed!")

if __name__ == "__main__":
    main()
