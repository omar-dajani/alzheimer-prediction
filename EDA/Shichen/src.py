import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats


def run_batch_anova(dataframe, target_col):
    # 1. Select only numerical columns to test against the target
    numerical_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
    
    # Remove the target column from the list if it happens to be numerically encoded
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
        
    results = []

    # 2. Iterate through each numerical feature
    for col in numerical_cols:
        # Create a temporary dataframe without NaNs for these two specific columns
        clean_df = dataframe[[col, target_col]].dropna()
        
        # Group the continuous variable by the unique categories in the target column
        # This creates a list of arrays, one for each diagnostic group
        groups = [group[col].values for name, group in clean_df.groupby(target_col)]
        
        # 3. Perform One-Way ANOVA 
        # (Check that we have at least 2 groups with data to compare)
        if len(groups) > 1:
            # The * operator unpacks the list of arrays as separate arguments
            f_stat, p_val = stats.f_oneway(*groups) 
            results.append({'Feature': col, 'F-Statistic': f_stat, 'P-Value': p_val})
        else:
            results.append({'Feature': col, 'F-Statistic': None, 'P-Value': None})

    # 4. Format and return the results
    results_df = pd.DataFrame(results)
    
    # Sort by P-Value to bring the most statistically significant features to the top
    results_df = results_df.sort_values(by='P-Value').reset_index(drop=True)
    
    return results_df

def plot_diagnosis_distribution(df, variable_name, target_col='DIAGNOSIS'):
    # 1. Drop missing values for the specific columns we are plotting
    plot_data = df[[variable_name, target_col]].dropna()
    
    # 2. Set up a figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 3. Plot 1: KDE Plot (Smooth Histogram)
    # The 'hue' parameter automatically separates and colors the data by DIAGNOSIS
    sns.kdeplot(
        data=plot_data, 
        x=variable_name, 
        hue=target_col, 
        fill=True, 
        alpha=0.4, 
        common_norm=False,
        ax=axes[0]
    )
    axes[0].set_title(f'Density Distribution of {variable_name} by {target_col}')
    
    # 4. Plot 2: Boxplot (Summary statistics and outliers)
    sns.boxplot(
        data=plot_data, 
        x=target_col, 
        y=variable_name, 
        ax=axes[1]
    )
    axes[1].set_title(f'Boxplot of {variable_name} by {target_col}')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming your dataframe is 'df' and you want to plot 'FA_FXST_L'
# plot_diagnosis_distribution(df, 'FA_FXST_L')


def plot_specific_patients(df, variable_name, patient_list):
    # 1. Create a copy to avoid altering the original dataframe
    plot_df = df.copy()
    
    # 2. Convert EXAMDATE to datetime for proper time-scale plotting
    plot_df['EXAMDATE'] = pd.to_datetime(plot_df['EXAMDATE'])
    
    # 3. Filter the dataframe to KEEP ONLY the patients in your provided list
    filtered_df = plot_df[plot_df['PTID'].isin(patient_list)]
    
    # Safety check: ensure the resulting dataframe isn't empty
    if filtered_df.empty:
        print("None of the provided patient IDs were found in the dataset.")
        return
    
    # 4. Build a color map: one color per unique DIAGNOSIS
    unique_diagnoses = filtered_df['DIAGNOSIS'].unique()
    palette = sns.color_palette('tab10', n_colors=len(unique_diagnoses))
    diagnosis_color_map = dict(zip(unique_diagnoses, palette))
    
    # 5. Set up and create the plot
    plt.figure(figsize=(10, 6))
    
    # Draw one line per PTID, colored by their DIAGNOSIS
    for ptid, patient_data in filtered_df.groupby('PTID'):
        # Use the DIAGNOSIS of the first row for this patient to pick the color
        diagnosis = patient_data['DIAGNOSIS'].iloc[0]
        color = diagnosis_color_map[diagnosis]
        
        plt.plot(
            patient_data['EXAMDATE'],
            patient_data[variable_name],
            color=color,
            marker='o',
            linewidth=2,
            label=diagnosis  # Will be deduplicated in legend below
        )
    
    # 6. Build a clean legend with one entry per DIAGNOSIS (no duplicates)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))  # dict keys are unique, deduplicates by diagnosis label
    plt.legend(unique_legend.values(), unique_legend.keys(), title='Diagnosis')
    
    # 7. Formatting
    plt.title(f'Longitudinal Progression of {variable_name} for Selected Patients')
    plt.xlabel('Exam Date')
    plt.ylabel(variable_name)
    
    plt.tight_layout()
    plt.show()

# Example usage:
# my_patients = ['011_S_0002', '011_S_0003', '022_S_0004']
# plot_specific_patients(df, 'FA_FXST_L', my_patients)

def plot_patient_progression(df, variable_name, n_patients=10):
    # 1. Create a copy to avoid SettingWithCopy warnings
    plot_df = df.copy()
    
    # 2. Convert EXAMDATE to proper datetime objects so the x-axis scales correctly
    plot_df['EXAMDATE'] = pd.to_datetime(plot_df['EXAMDATE'])
    
    # 3. Drop missing values only for the columns we need
    plot_df = plot_df.dropna(subset=['PTID', 'EXAMDATE', variable_name])
    
    # 4. Get all unique patient IDs
    all_ptids = plot_df['PTID'].unique()
    
    # Safety check: ensure we don't try to select more patients than exist
    n_select = min(n_patients, len(all_ptids))
    
    # 5. Randomly select 10 unique PTIDs
    selected_ptids = np.random.choice(all_ptids, size=n_select, replace=False)
    
    # 6. Filter the dataframe to keep only the selected patients
    filtered_df = plot_df[plot_df['PTID'].isin(selected_ptids)]
    
    # 7. Create the plot
    plt.figure(figsize=(12, 6))
    
    # sns.lineplot connects the points for each patient (hue='PTID')
    # marker='o' places a visible dot at each actual exam date
    sns.lineplot(
        data=filtered_df, 
        x='EXAMDATE', 
        y=variable_name, 
        hue='PTID', 
        marker='o',
        linewidth=2
    )
    
    # 8. Formatting
    plt.title(f'Longitudinal Progression of {variable_name} for {n_select} Random Patients')
    plt.xlabel('Exam Date')
    plt.ylabel(variable_name)
    
    # Move the legend outside the plot area so it doesn't cover the data lines
    plt.legend(title='Patient ID (PTID)', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming your dataframe is named 'df' and you want to plot 'FA_FXST_L'
# plot_patient_progression(df, 'FA_FXST_L')