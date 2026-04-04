import pandas as pd
import os

def add_precision():
    try:
        main_csv_path = './datasets/big-5-traffic.csv'
        precise_files_folder = './datasets/precision-enhancement-ds'
        output_folder = './datasets/'

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder, exist_ok=True)

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
                    
                    output_path = os.path.join(output_folder, f"scaled_{filename}")
                    df_precise.to_csv(output_path)

                    print(f"Processed {column_name}: precision improved successfully.")
                else:
                    print(f"Skipping {filename}: Column '{column_name}' not found in main CSV.")

        print("Task complete. Files are in the 'processed_output' folder.")
    except Exception as e:
        print(e)

add_precision()