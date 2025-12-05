

import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def run_openmp_test(array_size, num_threads, num_runs=3):
    """Run OpenMP implementation multiple times and return the best (minimum) execution time"""
    cmd = [f"../openmp/minmax_openmp.exe", str(array_size), str(num_threads)]
    
    best_time = None
    
    for run in range(num_runs):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            output = result.stdout
            
            # Parse execution time
            for line in output.split('\n'):
                if "Time taken:" in line:
                    time_str = line.split(":")[1].strip().split()[0]
                    exec_time = float(time_str)
                    
                    # Keep the best (minimum) time
                    if best_time is None or exec_time < best_time:
                        best_time = exec_time
                    break
        except Exception as e:
            if run == 0:  # Only print error on first run
                print(f"Error running OpenMP with {num_threads} threads: {e}")
            continue
    
    return best_time

def evaluate_openmp():
    """Perform OpenMP performance evaluation"""
    array_size = 100000000  # Fixed array size for testing
    thread_counts = [1, 2, 4, 8, 16]
    
    execution_times = []
    speedups = []
    
    print("OpenMP Performance Evaluation")
    print("=" * 50)
    print(f"Array size: {array_size}")
    print(f"Thread counts: {thread_counts}")
    print()
    
    # Run tests for each thread count
    for threads in thread_counts:
        print(f"Testing with {threads} thread(s) (running 3 times, taking best)...")
        exec_time = run_openmp_test(array_size, threads)
        
        if exec_time is not None:
            execution_times.append(exec_time)
            print(f"  Best execution time: {exec_time:.6f} seconds")
    
    # Calculate speedup (relative to single thread)
    if execution_times:
        single_thread_time = execution_times[0]
        speedups = [single_thread_time / t for t in execution_times]
        
        # Display results
        print("\n" + "=" * 50)
        print("Performance Results:")
        print("-" * 50)
        for i, threads in enumerate(thread_counts):
            print(f"Threads: {threads:2d} | Time: {execution_times[i]:.6f}s | "
                  f"Speedup: {speedups[i]:.2f}x")
        
        # Create plots
        create_plots(thread_counts, execution_times, speedups)
        
        # Save results to file
        save_results(thread_counts, execution_times, speedups)

def create_plots(thread_counts, execution_times, speedups):
    """Create performance plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Execution time vs Threads
    ax1.plot(thread_counts, execution_times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('OpenMP: Threads vs Execution Time')
    ax1.grid(True, alpha=0.3)
    
    # Add values on plot points
    for i, (x, y) in enumerate(zip(thread_counts, execution_times)):
        ax1.text(x, y, f'{y:.4f}s', ha='center', va='bottom')
    
    # Plot 2: Speedup vs Threads
    ax2.plot(thread_counts, speedups, 'ro-', linewidth=2, markersize=8, label='Actual Speedup')
    ax2.plot(thread_counts, thread_counts, 'g--', linewidth=2, label='Ideal Speedup')
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Speedup')
    ax2.set_title('OpenMP: Threads vs Speedup')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add values on plot points
    for i, (x, y) in enumerate(zip(thread_counts, speedups)):
        ax2.text(x, y, f'{y:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    if not os.path.exists('../results'):
        os.makedirs('../results')
    
    plt.savefig('../results/openmp_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlots saved to: ../results/openmp_performance.png")

def save_results(thread_counts, execution_times, speedups):
    """Save results to text file"""
    with open('../results/openmp_results.txt', 'w') as f:
        f.write("OpenMP Performance Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Array Size: 100000000\n")
        f.write("\n" + "-" * 50 + "\n")
        f.write("Threads | Time (s) | Speedup\n")
        f.write("-" * 50 + "\n")
        
        for i in range(len(thread_counts)):
            f.write(f"{thread_counts[i]:7d} | {execution_times[i]:8.6f} | {speedups[i]:.2f}x\n")
        
        f.write("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    evaluate_openmp()