import os
import re
import matplotlib.pyplot as plt
import glob
import numpy as np

def parse_logs(log_dir="log"):
    """
    Parses log files to extract fitness (stress) values.
    Returns a dictionary: {iteration: [stress_values]}
    """
    data = {}
    
    # Pattern to match filenames: iter_0_part_0.log
    file_pattern = os.path.join(log_dir, "iter_*_part_*.log")
    files = glob.glob(file_pattern)
    
    print(f"Found {len(files)} log files in {log_dir}...")
    
    filename_re = re.compile(r"iter_(\d+)_part_(\d+)\.log")
    stress_re = re.compile(r"MAX_STRESS:\s*([0-9\.eE\+\-]+)")
    
    for filepath in files:
        filename = os.path.basename(filepath)
        match = filename_re.match(filename)
        if not match:
            continue
            
        iteration = int(match.group(1))
        
        try:
            with open(filepath, "r") as f:
                content = f.read()
                
            stress_match = stress_re.search(content)
            if stress_match:
                stress = float(stress_match.group(1))
                
                if iteration not in data:
                    data[iteration] = []
                data[iteration].append(stress)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    return data

def plot_fitness(data, output_file="fitness_vs_iteration.png"):
    """
    Generates a plot of Best Fitness vs Iteration.
    """
    if not data:
        print("No data found to plot.")
        return

    iterations = sorted(data.keys())
    best_fitness_per_iter = []
    global_best_history = []
    current_global_best = float('inf')
    
    for it in iterations:
        stresses = data[it]
        min_stress = min(stresses)
        best_fitness_per_iter.append(min_stress)
        
        if min_stress < current_global_best:
            current_global_best = min_stress
        global_best_history.append(current_global_best)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, best_fitness_per_iter, 'o--', label='Iteration Best', alpha=0.6)
    plt.plot(iterations, global_best_history, 'r-', linewidth=2, label='Global Best')
    
    plt.xlabel('Iteration')
    plt.ylabel('Max von Mises Stress (Pa)')
    plt.title('PSO Convergence: Fitness vs Iteration')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.yscale('log') # Log scale often helps with convergence plots
    
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    # Print stats
    print("\nConvergence Stats:")
    for it, best, global_best in zip(iterations, best_fitness_per_iter, global_best_history):
        print(f"  Iter {it}: Best = {best:.2e}, Global Best = {global_best:.2e}")

if __name__ == "__main__":
    log_dir = "log"
    if not os.path.exists(log_dir):
        print(f"Directory '{log_dir}' not found.")
    else:
        data = parse_logs(log_dir)
        plot_fitness(data)
