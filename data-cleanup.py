import pandas as pd
import os

def add_precision():
    try:
        main_csv_path = './datasets/big-5-traffic.csv'
        precise_files_folder = './datasets/precision-enhancement-ds'
        output_folder = './datasets/'
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        df_main = pd.read_csv(main_csv_path, index_col='Time', parse_dates=True)

        column_maxes = df_main.max()

        # 3. Process each "precise" file
        for filename in os.listdir(precise_files_folder):
            if filename.endswith(".csv") and filename != main_csv_path:
                file_path = os.path.join(precise_files_folder, filename)
                
                # Read precise data (e.g., the OLX file)
                df_precise = pd.read_csv(file_path, index_col='Time', parse_dates=True)
                
                # Identify the column name (e.g., "OLX")
                column_name = df_precise.columns[0]
                
                if column_name in df_main.columns:
                    # Get the specific maximum for THIS column
                    local_max = column_maxes[column_name]
                    
                    if local_max == 0:
                        print(f"Skipping {column_name} to avoid division by zero.")
                        continue
                        
                    # Calculation: Precise_Value * (Normalized_Value / Column_Specific_Max)
                    scaling_series = df_main[column_name] / local_max
                    df_precise[column_name] = df_precise[column_name] * scaling_series
                    
                    # Save the result
                    output_path = os.path.join(output_folder, f"scaled_{filename}")
                    df_precise.to_csv(output_path)
                    print(f"Processed {filename} using max value: {local_max}")
                else:
                    print(f"Column '{column_name}' not found in main CSV.")

        print("Task complete.")
    except Exception as e:
        print(e)

add_precision()