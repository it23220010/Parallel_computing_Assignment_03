#!/usr/bin/env python3
"""
CUDA Evaluation - Two Graphs in One Sheet
Generates both on the same figure:
1. Configuration parameters vs Execution time
2. Configuration parameters vs Speedup
"""

import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def run_cuda_test(array_size, block_size, num_blocks, num_runs=3):
    """Run CUDA program multiple times and return the best (minimum) execution time"""
    cuda_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cuda", "minmax_cuda.exe"))
    cmd = [cuda_path, str(array_size), str(block_size), str(num_blocks)]
    
    best_time = None
    
    for run in range(num_runs):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Parse kernel execution time
            for line in result.stdout.split('\n'):
                if "Kernel execution time:" in line:
                    exec_time = float(line.split(":")[1].strip().split()[0])
                    
                    # Keep the best (minimum) time
                    if best_time is None or exec_time < best_time:
                        best_time = exec_time
                    break
        except:
            continue
    
    return best_time

def get_serial_time(array_size):
    """Get serial execution time for speedup calculation"""
    serial_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "serial", "minmax_serial.exe"))
    cmd = [serial_path, str(array_size)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if "Time taken:" in line:
                time_str = line.split(":")[1].strip().split()[0]
                return float(time_str) * 1000  # Convert to ms
    except:
        return None
    
    return None

def main():
    print("=" * 60)
    print("CUDA EVALUATION - TWO GRAPHS IN ONE SHEET")
    print("=" * 60)
    
    # Configuration
    ARRAY_SIZE = 100000000  # 10^8
    BLOCK_SIZES = [32, 64, 128, 256, 512]
    NUM_BLOCKS = 40000  # Adjusted for larger array (100M / 256 threads per block)
    
    print(f"Array Size: {ARRAY_SIZE:,}")
    print(f"Number of Blocks (fixed): {NUM_BLOCKS}")
    print(f"Block Sizes to test: {BLOCK_SIZES}")
    print("-" * 40)
    
    # Get serial time for speedup
    print("\nGetting serial baseline time...")
    serial_time = get_serial_time(ARRAY_SIZE)
    
    if serial_time:
        print(f"Serial execution time: {serial_time:.2f} ms")
    else:
        print("Warning: Could not get serial time. Using estimated value.")
        serial_time = 150  # Estimated value in ms
    
    # Run CUDA tests
    execution_times = []
    valid_block_sizes = []
    
    print("\nRunning CUDA tests (each config runs 3 times, taking best)...")
    for block_size in BLOCK_SIZES:
        print(f"  Testing block size {block_size}...", end=" ", flush=True)
        
        exec_time = run_cuda_test(ARRAY_SIZE, block_size, NUM_BLOCKS)
        
        if exec_time:
            execution_times.append(exec_time)
            valid_block_sizes.append(block_size)
            print(f"{exec_time:.2f} ms")
        else:
            print("Failed")
    
    if not execution_times:
        print("\n❌ No successful CUDA runs!")
        sys.exit(1)
    
    # Calculate speedup
    speedups = [serial_time / t for t in execution_times]
    
    # Create results directory
    os.makedirs("../results", exist_ok=True)
    
    # ==============================================
    # CREATE SINGLE FIGURE WITH TWO SUBPLOTS
    # ==============================================
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set main title for the entire figure
    fig.suptitle('CUDA Performance Analysis - Min-Max Normalization\n'
                 f'Array Size: {ARRAY_SIZE:,}, Number of Blocks: {NUM_BLOCKS}',
                 fontsize=16, fontweight='bold', y=1.02)
    
    # ==============================================
    # SUBPLOT 1: Execution Time vs Block Size
    # ==============================================
    
    # Plot execution time
    line1, = ax1.plot(valid_block_sizes, execution_times, 
                      marker='o', color='blue', 
                      linewidth=3, markersize=10,
                      markerfacecolor='white', markeredgewidth=2,
                      label='CUDA Execution Time')
    
    # Add value labels on points
    for i, (block, time_val) in enumerate(zip(valid_block_sizes, execution_times)):
        ax1.text(block, time_val, f'{time_val:.1f}ms', 
                 ha='center', va='bottom',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='blue', alpha=0.9))
    
    # Add horizontal line for serial time
    line2 = ax1.axhline(y=serial_time, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                        label=f'Serial Time: {serial_time:.1f}ms')
    
    # Configure subplot 1
    ax1.set_xlabel('Block Size (Threads per Block)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Execution Time (milliseconds)', fontsize=13, fontweight='bold')
    ax1.set_title('Block Size vs Execution Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(valid_block_sizes)
    ax1.set_xticklabels([f'{bs}\nthreads' for bs in valid_block_sizes], fontsize=11)
    
    # Add legend for subplot 1
    ax1.legend(handles=[line1, line2], fontsize=11, loc='upper right')
    
    # Add annotation for best time
    best_time_idx = execution_times.index(min(execution_times))
    best_block = valid_block_sizes[best_time_idx]
    best_time = execution_times[best_time_idx]
    
    ax1.annotate(f'OPTIMAL\n{best_block} threads\n{best_time:.1f} ms',
                 xy=(best_block, best_time),
                 xytext=(best_block, best_time * 1.3),
                 arrowprops=dict(arrowstyle='->', color='green', linewidth=2, connectionstyle="arc3,rad=0.2"),
                 fontsize=11, color='green', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", edgecolor='green', alpha=0.8),
                 ha='center', va='bottom')
    
    # ==============================================
    # SUBPLOT 2: Speedup vs Block Size
    # ==============================================
    
    # Plot speedup
    line3, = ax2.plot(valid_block_sizes, speedups,
                      marker='s', color='green',
                      linewidth=3, markersize=10,
                      markerfacecolor='white', markeredgewidth=2,
                      label='CUDA Speedup')
    
    # Add value labels on points
    for i, (block, speedup) in enumerate(zip(valid_block_sizes, speedups)):
        ax2.text(block, speedup, f'{speedup:.1f}x', 
                 ha='center', va='bottom',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='green', alpha=0.9))
    
    # Add horizontal line for baseline (speedup = 1)
    line4 = ax2.axhline(y=1.0, color='black', linestyle=':', linewidth=2, alpha=0.5, 
                        label='Baseline (Serial)')
    
    # Configure subplot 2
    ax2.set_xlabel('Block Size (Threads per Block)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Speedup (vs Serial Implementation)', fontsize=13, fontweight='bold')
    ax2.set_title('Block Size vs Speedup', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(valid_block_sizes)
    ax2.set_xticklabels([f'{bs}\nthreads' for bs in valid_block_sizes], fontsize=11)
    
    # Add legend for subplot 2
    ax2.legend(handles=[line3, line4], fontsize=11, loc='upper left')
    
    # Add annotation for best speedup
    best_speedup_idx = speedups.index(max(speedups))
    best_block_s = valid_block_sizes[best_speedup_idx]
    best_speedup = speedups[best_speedup_idx]
    
    ax2.annotate(f'BEST SPEEDUP\n{best_block_s} threads\n{best_speedup:.1f}x',
                 xy=(best_block_s, best_speedup),
                 xytext=(best_block_s, best_speedup * 0.8),
                 arrowprops=dict(arrowstyle='->', color='blue', linewidth=2, connectionstyle="arc3,rad=0.2"),
                 fontsize=11, color='blue', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor='blue', alpha=0.8),
                 ha='center', va='top')
    
    # ==============================================
    # ADJUST LAYOUT AND SAVE
    # ==============================================
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save the figure
    output_file = "../results/cuda_performance_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    # ==============================================
    # DISPLAY RESULTS
    # ==============================================
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Block Size':>12} | {'Time (ms)':>12} | {'Speedup':>12}")
    print("-" * 60)
    
    for block, time_val, speedup in zip(valid_block_sizes, execution_times, speedups):
        print(f"{block:12d} | {time_val:12.2f} | {speedup:12.2f}x")
    
    print("-" * 60)
    
    # Find best configuration
    best_time_idx = execution_times.index(min(execution_times))
    best_speedup_idx = speedups.index(max(speedups))
    
    print(f"\nOptimal Configuration (Fastest Execution):")
    print(f"  Block Size: {valid_block_sizes[best_time_idx]} threads")
    print(f"  Execution Time: {execution_times[best_time_idx]:.2f} ms")
    print(f"  Speedup: {speedups[best_time_idx]:.2f}x")
    
    print(f"\nBest Speedup Configuration:")
    print(f"  Block Size: {valid_block_sizes[best_speedup_idx]} threads")
    print(f"  Execution Time: {execution_times[best_speedup_idx]:.2f} ms")
    print(f"  Speedup: {speedups[best_speedup_idx]:.2f}x")
    
    # Save results to file
    with open("../results/cuda_evaluation_results.txt", "w") as f:
        f.write("CUDA EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Array Size: {ARRAY_SIZE:,}\n")
        f.write(f"Number of Blocks: {NUM_BLOCKS}\n")
        f.write(f"Serial Time: {serial_time:.2f} ms\n\n")
        
        f.write("Configuration Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Block':>6} | {'Time (ms)':>10} | {'Speedup':>10}\n")
        f.write("-" * 50 + "\n")
        
        for block, time_val, speedup in zip(valid_block_sizes, execution_times, speedups):
            f.write(f"{block:6d} | {time_val:10.2f} | {speedup:10.2f}x\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Best Block Size for Speed: {valid_block_sizes[best_speedup_idx]} threads\n")
        f.write(f"Maximum Speedup Achieved: {speedups[best_speedup_idx]:.2f}x\n")
        f.write(f"Most Efficient Block Size: {valid_block_sizes[best_time_idx]} threads\n")
        f.write(f"Minimum Execution Time: {execution_times[best_time_idx]:.2f} ms\n")
        f.write(f"Performance Improvement: {((serial_time/execution_times[best_time_idx])-1)*100:.1f}% faster than serial\n")
    
    print("\n" + "=" * 60)
    print("GRAPH GENERATED:")
    print(f"  ✓ {output_file}")
    print("\nGraph shows:")
    print("  • LEFT: Block Size vs Execution Time")
    print("  • RIGHT: Block Size vs Speedup")
    print("=" * 60)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Check if CUDA program exists
    if not os.path.exists("../cuda/minmax_cuda.exe"):
        print("❌ CUDA executable not found!")
        print("Please compile the CUDA program first:")
        print("  cd ../cuda && make")
        sys.exit(1)
    
    # Check if serial program exists
    if not os.path.exists("../serial/minmax_serial.exe"):
        print("❌ Serial executable not found!")
        print("Please compile the serial program first:")
        print("  cd ../serial && make")
        sys.exit(1)
    
    main()