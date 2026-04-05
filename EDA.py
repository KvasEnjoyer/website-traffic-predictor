import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

custom_palette = {
    "Розетка": "#05bc52",
    "OLX": "#23e5db", 
    "Фокстрот": "#e95d2a",
    "Епіцентр К": "#1060c1",
    "Prom.ua": "#8b08fa",
}

def set_custom_style():
    sns.set_theme(style="whitegrid", palette="viridis")
    sns.set_context("notebook", font_scale=1.1)
    
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (10, 6)

def initial_analysis(moving_average=0):
    csv_path = "datasets/big-5-scaled.csv"

    x_axis_label = "Час (роки та місяці)"
    y_axis_label = "Відносна популярність запитів"
    season_label = "Сезонний всплеск запитів (листоп.-груд.)"
    chart_title = f"Порівняння пошукових запитів на сайти українських онлайн-магазинів (середнє знач. за {moving_average} міс.)"

    df = pd.read_csv(csv_path)
    retailer_cols = df.columns[3:]

    if moving_average > 1:
        df[retailer_cols] = df[retailer_cols].rolling(window=moving_average, center=True, min_periods=1).mean()

    # Melt the data for Seaborn
    df_long = df.melt(id_vars=["Time"], 
                    value_vars=retailer_cols, 
                    var_name="Retailer", 
                    value_name="Scaled Value")

    set_custom_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_long, x="Time", y="Scaled Value", hue="Retailer", ax=ax, palette=custom_palette)

    # Draw the seasonal surge in demand
    years = pd.to_datetime(df['Time']).dt.year.unique()
    
    data_start = pd.to_datetime(df['Time']).min()
    data_end = pd.to_datetime(df['Time']).max()

    for year in years:
        s_start = pd.to_datetime(f"{year}-11-01")
        s_end = pd.to_datetime(f"{year}-12-31")
        
        if s_start <= data_end and s_end >= data_start:
            actual_end = min(s_end, data_end)
            
            ax.axvspan(s_start.strftime('%Y-%m'), 
                    actual_end.strftime('%Y-%m'), 
                    color='gray', alpha=0.2, 
                    label=season_label if year == years[0] else "")

    # Custom X- and Y- axis

    ticks_pos = list(np.arange(0.0, 1.01, 0.1)) 
    tick_labels = [f"{p*100:.0f}%" for p in ticks_pos]

    plt.xticks(ticks=df['Time'][::12], rotation=45)
    plt.yticks(ticks=ticks_pos, labels=tick_labels)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    # Draw 3-month lines

    ax.vlines(x=np.arange(0, 270, 3), ymin=0, ymax=1, 
          colors='lightgray', linestyle=':', linewidth=1)


    plt.title(chart_title, loc='left', fontsize=16)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

initial_analysis(moving_average=1)
initial_analysis(moving_average=12)