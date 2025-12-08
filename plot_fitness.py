import os
import re
import matplotlib.pyplot as plt
import glob
import numpy as np

def parse_logs(log_dir="log"):
    """
    Parses CSV log file to extract fitness (stress) values.
    Returns a dictionary: {iteration: [stress_values]}
    """
    data = {}
    csv_file = os.path.join(log_dir, "pso_history.csv")
    
    if not os.path.exists(csv_file):
        print(f"CSV log file not found: {csv_file}")
        return data
        
    print(f"Reading logs from {csv_file}...")
    
    try:
        with open(csv_file, "r") as f:
            header = f.readline() # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                    
                iteration = int(parts[0])
                # instance = int(parts[1])
                stress = float(parts[2])
                
                if iteration not in data:
                    data[iteration] = []
                data[iteration].append(stress)
                
    except Exception as e:
        print(f"Error reading CSV: {e}")
            
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
