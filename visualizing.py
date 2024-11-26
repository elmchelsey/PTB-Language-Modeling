import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob

def visualize_perplexities(file_paths, test_perplexities):
    # Create a list to store results
    results = []
    
    # Find all CSV files
    for file_path in file_paths:        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        if 'ppl' not in df.columns:
            print(f"Warning: {file_path} does not contain a 'ppl' column. Skipping.")
            continue
            
        # Calculate average training perplexity - take the last value instead of mean
        train_perplexity = df['ppl'].iloc[-1]  # Changed from mean() to iloc[-1]
        
        # Get model name (extract just the filename without extension)
        model_name = file_path.split('/')[-1].replace('.csv', '')
        
        # Get test perplexity for this model
        test_ppl = test_perplexities.get(model_name)
        
        # Store results
        test_perplexities = {
            'default': 422.94,
            '2decs': 459.81,
            '3decs': 414.03,
            '4decs': 446.78
    }

        # Store results
        results.append({
            'Model': model_name,
            'Validation Perplexity': train_perplexity,  # Now using final training perplexity
            'Test Perplexity': test_perplexities[model_name]
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Melt the DataFrame to create a format suitable for grouped bars
    melted_df = pd.melt(results_df, 
                        id_vars=['Model'], 
                        value_vars=['Validation Perplexity', 'Test Perplexity'],
                        var_name='Metric',
                        value_name='Perplexity')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted_df, x='Model', y='Perplexity', hue='Metric')
    
    # Customize the plot
    plt.title('Validation and Test Perplexity by Model')
    plt.xlabel('Model')
    plt.ylabel('Perplexity')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    # Add value labels on top of each bar
    for i in plt.gca().containers:
        plt.gca().bar_label(i, fmt='%.2f')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Display the plot
    plt.show()

# Example usage:
if __name__ == '__main__':
    file_paths = ['default.csv', '2decs.csv', '3decs.csv', '4decs.csv']
    test_perplexities = {
        'default': 422.94,
        '2decs': 459.81,
        '3decs': 414.03,
        '4decs': 446.78
    }
    visualize_perplexities(file_paths, test_perplexities)