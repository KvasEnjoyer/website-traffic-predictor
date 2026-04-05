import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def clean_data():
    try:
        main_csv_path = './datasets/big-5-traffic.csv'
        precise_files_folder = './datasets/precision-enhancement-ds'
        output_file = './datasets/big-5-scaled.csv'

        scaled_dfs = []

        df_main = pd.read_csv(main_csv_path, index_col='Time', parse_dates=True) # Combined data for the websites

        column_maxes = df_main.max()

        for filename in os.listdir(precise_files_folder):
            if filename.endswith(".csv") and filename != os.path.basename(main_csv_path):
                file_path = os.path.join(precise_files_folder, filename) # Safety measures for .csv file management
                
                df_precise = pd.read_csv(file_path, index_col='Time', parse_dates=True)
                column_name = df_precise.columns[0]
        
                if column_name in column_maxes:
                    peak_value = column_maxes[column_name]

                    if peak_value <= 0: # Shouldn't happen, but added anyway.
                        print(f"Skipped {column_name} due to undefined behavior.")
                        continue
                    
                    # Compress the full range of df_precise to range [0; peak_value] and replace original.
                    df_precise[column_name] = df_precise[column_name] * (peak_value / 100)
                    
                    scaled_dfs.append(df_precise)
                    print(f"Scaled {column_name} to {peak_value}")
            else:
                print(f"Skipping {filename}: '{column_name}' not in Main CSV.")
        if scaled_dfs:
            df = pd.concat(scaled_dfs, axis=1) # Merge based on the 'Time' index

            df = df / 100

            df.index = pd.to_datetime(df.index).to_period('M')
            df["Year"]  = df.index.year
            df["Month"] = df.index.month

            cols_to_front = ['Year', 'Month']
            other_cols = [c for c in df.columns if c not in cols_to_front]
            df = df[cols_to_front + other_cols]

            df.to_csv(output_file)
            print(f"All files processed and saved to {output_file} successfully.")
        else:
            print("No matching files were found to process.")
    except Exception as e:
        print(e)

clean_data()
