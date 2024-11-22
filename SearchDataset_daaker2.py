import os
import csv
import pandas as pd
import sys

def search_files(input_file, output_csv, keywords):
    results = []

    with open(input_file, 'r') as file:
        filenames = file.readlines()

    for filename in filenames:
        filename = filename.strip()
        try:
            with open(filename, 'r', errors='ignore', encoding='latin-1') as f:
                content = f.read().lower()
                label = None
                for keyword, label_name in keywords.items():
                    if keyword in content:
                        label = label_name
                        break
                if label:
                    results.append([filename, label])
        except FileNotFoundError:
            print(f"File not found: {filename}")

    df = pd.DataFrame(results, columns=['Filename', 'Label'])
    
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # Input text file containing text files of filenamesHERE 
    input_file = sys.argv[1]
    output_csv = 'TUHEEG_labels.csv'
    keywords = {'epilep': 'epilepsy', 'stroke': 'stroke', 'concus': 'concussion'}

    search_files(input_file, output_csv, keywords)
