import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def plot_kde_model(kde_model, filename, save_plots=True, show_plots=False):
    """
    Plot a single KDE model with appropriate formatting based on the filename
    """
    # Extract original dataset
    original_data = kde_model.dataset.flatten()
    
    # Get data statistics
    data_min = original_data.min()
    data_max = original_data.max()
    data_mean = original_data.mean()
    data_std = original_data.std()
    
    # Determine plot parameters based on filename
    if 'dwelltime' in filename.lower():
        xlabel = 'Dwell Time (hours)'
        title_prefix = 'Dwell Time Distribution'
        x_min = max(0, data_min - (data_max - data_min) * 0.1)
    elif 'arrival' in filename.lower():
        xlabel = 'Arrival Time (hours of day)'
        title_prefix = 'Arrival Time Distribution'
        x_min = max(0, data_min - 1)  # Small padding for arrival times
    elif 'weight' in filename.lower():
        xlabel = 'Container Weight (units)'
        title_prefix = 'Container Weight Distribution'
        x_min = max(0, data_min - (data_max - data_min) * 0.05)
    else:
        xlabel = 'Value'
        title_prefix = 'Distribution'
        x_min = data_min - (data_max - data_min) * 0.1
    
    # Create title from filename
    clean_name = filename.replace('.pkl', '').replace('kde_', '').replace('_', ' ').title()
    title = f'{title_prefix}: {clean_name}'
    
    # Create x-axis range for plotting
    padding = (data_max - data_min) * 0.1
    x_max = data_max + padding
    
    # Create dense x values for smooth curve
    x_plot = np.linspace(x_min, x_max, 1000)
    
    # Evaluate the KDE at these points
    pdf_values = kde_model.pdf(x_plot)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Main KDE plot
    ax1.plot(x_plot, pdf_values, 'b-', linewidth=2, label='KDE')
    ax1.hist(original_data, bins=min(50, int(np.sqrt(len(original_data)))), 
             density=True, alpha=0.3, color='gray', label='Data Histogram')
    
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'n = {kde_model.n:,}\nMean = {data_mean:.2f}\nStd = {data_std:.2f}'
    ax1.text(0.75, 0.75, stats_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Focused plot (central 95%)
    q025 = np.percentile(original_data, 2.5)
    q975 = np.percentile(original_data, 97.5)
    x_focused = np.linspace(q025, q975, 1000)
    pdf_focused = kde_model.pdf(x_focused)
    
    ax2.plot(x_focused, pdf_focused, 'r-', linewidth=2)
    ax2.fill_between(x_focused, pdf_focused, alpha=0.3, color='red')
    
    # Add percentile markers
    ax2.axvline(data_mean, color='black', linestyle='--', alpha=0.7, 
                label=f'Mean ({data_mean:.2f})')
    ax2.axvline(np.median(original_data), color='green', linestyle='--', alpha=0.7, 
                label=f'Median ({np.median(original_data):.2f})')
    
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title(f'{title} (Central 95%)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        output_dir = 'plots'
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = filename.replace('.pkl', '_plot.png')
        plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
        print(f"Saved plot: {os.path.join(output_dir, plot_filename)}")
    
    # Show plot if requested
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return {
        'filename': filename,
        'n_samples': kde_model.n,
        'mean': data_mean,
        'std': data_std,
        'min': data_min,
        'max': data_max,
        'bandwidth_factor': kde_model.factor
    }

def process_all_kde_models(models_dir='models', save_plots=True, show_plots=False):
    """
    Process all KDE models in the specified directory
    """
    # Find all .pkl files
    pkl_files = list(Path(models_dir).glob('*.pkl'))
    
    if not pkl_files:
        print(f"No .pkl files found in {models_dir} directory")
        return
    
    print(f"Found {len(pkl_files)} KDE model files:")
    for file in pkl_files:
        print(f"  - {file.name}")
    print()
    
    # Process each model
    results = []
    
    for pkl_file in pkl_files:
        try:
            print(f"Processing: {pkl_file.name}")
            
            # Load the KDE model
            with open(pkl_file, 'rb') as f:
                kde_model = pickle.load(f)
            
            # Plot the model
            result = plot_kde_model(kde_model, pkl_file.name, save_plots, show_plots)
            results.append(result)
            
            print(f"  - Samples: {result['n_samples']:,}")
            print(f"  - Range: {result['min']:.2f} to {result['max']:.2f}")
            print(f"  - Mean: {result['mean']:.2f}, Std: {result['std']:.2f}")
            print()
            
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {str(e)}")
            continue
    
    # Create summary report
    if results:
        print("="*60)
        print("SUMMARY REPORT")
        print("="*60)
        
        for result in results:
            print(f"{result['filename']:<35} | n={result['n_samples']:<8,} | μ={result['mean']:<8.2f} | σ={result['std']:<8.2f}")
    
    return results

# Main execution
if __name__ == "__main__":
    # Configuration
    MODELS_DIR = 'models'  # Directory containing your .pkl files
    SAVE_PLOTS = True      # Save plots as PNG files
    SHOW_PLOTS = False     # Display plots (set to True if you want to see them)
    
    # Process all models
    results = process_all_kde_models(
        models_dir=MODELS_DIR, 
        save_plots=SAVE_PLOTS, 
        show_plots=SHOW_PLOTS
    )
    
    print(f"\nProcessing complete! Check the 'plots' directory for saved images.")

# Alternative: Process specific models only
def process_specific_models(model_names, models_dir='models'):
    """
    Process only specific models by name
    """
    for model_name in model_names:
        pkl_path = os.path.join(models_dir, model_name)
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    kde_model = pickle.load(f)
                plot_kde_model(kde_model, model_name, save_plots=True, show_plots=True)
            except Exception as e:
                print(f"Error processing {model_name}: {str(e)}")
        else:
            print(f"File not found: {pkl_path}")

# Uncomment and modify to process specific models:
# process_specific_models([
#     'kde_model_dwelltime_hours.pkl',
#     'container_weight_kde.pkl',
#     'kde_arrival_weekday_pickups.pkl'
# ])

