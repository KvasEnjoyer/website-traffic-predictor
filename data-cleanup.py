import pandas as pd
import os

main_csv_path = 'big-5-traffic.csv'
precise_files_folder = './datasets/precision-enhancement-ds'
output_folder = './datasets/big-5-traffic.csv'
os.makedirs(output_folder, exist_ok=True)

df_main = pd.read_csv(main_csv_path, index_col='Time', parse_dates=True)
global_max = df_main.max().max() 

print(f"Global maximum found: {global_max}")

for filename in os.listdir(precise_files_folder):
    if filename.endswith(".csv") and filename != main_csv_path:
        file_path = os.path.join(precise_files_folder, filename)
        
        # Read the precise data
        df_precise = pd.read_csv(file_path, index_col='Time', parse_dates=True)
        
        # Identify which column from the main CSV matches the precise file
        # (Assuming the precise file column name exists in the main CSV)
        column_name = df_precise.columns[0]
        
        if column_name in df_main.columns:
            # Calculation: Precise_Value * (Normalized_Value / Global_Max)
            # Alignment happens automatically on the 'Time' index
            scaling_series = df_main[column_name] / global_max
            df_precise[column_name] = df_precise[column_name] * scaling_series
            
            # 3. Save the result
            output_path = os.path.join(output_folder, f"scaled_{filename}")
            df_precise.to_csv(output_path)
            print(f"Processed and saved: {filename}")
        else:
            print(f"Skipping {filename}: Column '{column_name}' not found in main CSV.")

print("Task complete.")