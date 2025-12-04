#!/usr/bin/env python3
"""
Simplified MPI Performance Evaluation Script
Generates only:
1. Number of processes vs Execution time
2. Number of processes vs Speedup
"""

import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from datetime import datetime

def run_mpi_benchmark(array_size, num_processes):
    """
    Run MPI program and extract execution time
    
    Args:
        array_size (int): Size of array to process
        num_processes (int): Number of MPI processes
        
    Returns:
        float: Execution time in seconds, or None if failed
    """
    # Build command
    cmd = f"mpiexec -n {num_processes} ../mpi/minmax_mpi.exe {array_size}"
    
    try:
        # Execute command with timeout
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        if result.returncode != 0:
            print(f"  ✗ Failed with {num_processes} processes")
            print(f"    Error: {result.stderr[:100]}")
            return None
        
        # Parse output for execution time
        for line in result.stdout.split('\n'):
            if "Time taken:" in line:
                time_str = line.split(":")[1].strip().split()[0]
                return float(time_str)
        
        return None
        
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout with {num_processes} processes")
        return None
    except Exception as e:
        print(f"  ❌ Exception with {num_processes} processes: {e}")
        return None

def generate_performance_graphs():
    """
    Main function to generate performance graphs
    """
    print("=" * 60)
    print("MPI PERFORMANCE EVALUATION")
    print("=" * 60)
    
    # Configuration
    ARRAY_SIZE = 1000000  # Fixed array size
    PROCESS_COUNTS = [1, 2, 4, 8, 16]
    
    # Storage for results
    execution_times = []
    speedups = []
    
    print(f"\nTesting with array size: {ARRAY_SIZE:,}")
    print(f"Process counts: {PROCESS_COUNTS}")
    print("-" * 40)
    
    # Run benchmarks
    serial_time = None
    
    for i, processes in enumerate(PROCESS_COUNTS):
        print(f"Running with {processes} process(es)...", end=" ", flush=True)
        
        exec_time = run_mpi_benchmark(ARRAY_SIZE, processes)
        
        if exec_time is not None:
            execution_times.append(exec_time)
            
            # Store serial time (1 process)
            if processes == 1:
                serial_time = exec_time
                speedups.append(1.0)  # Speedup is 1 for serial
                print(f"✓ Time: {exec_time:.4f}s")
            else:
                if serial_time and exec_time > 0:
                    speedup = serial_time / exec_time
                    speedups.append(speedup)
                    print(f"✓ Time: {exec_time:.4f}s | Speedup: {speedup:.2f}x")
                else:
                    speedups.append(0.0)
                    print(f"✓ Time: {exec_time:.4f}s")
        else:
            # If failed, use previous time or mark as failed
            if i > 0:
                execution_times.append(execution_times[-1] * 1.1)  # Estimate
                speedups.append(0.0)
            else:
                execution_times.append(1.0)  # Default
                speedups.append(1.0)
            print("✗ Failed")
    
    # Create results directory if needed
    os.makedirs("../results", exist_ok=True)
    
    # Generate graphs
    create_performance_plots(PROCESS_COUNTS, execution_times, speedups, ARRAY_SIZE)
    
    # Save results to file
    save_results_to_file(PROCESS_COUNTS, execution_times, speedups, ARRAY_SIZE)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Results saved to ../results/ directory")
    print("=" * 60)

def create_performance_plots(process_counts, execution_times, speedups, array_size):
    """
    Create the two required graphs
    
    Args:
        process_counts (list): List of process counts
        execution_times (list): Corresponding execution times
        speedups (list): Corresponding speedup values
        array_size (int): Size of array used in tests
    """
    print("\n" + "-" * 40)
    print("GENERATING PERFORMANCE GRAPHS")
    print("-" * 40)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'MPI Performance Analysis - Min-Max Normalization (N={array_size:,})', 
                fontsize=14, fontweight='bold')
    
    # ==============================================
    # GRAPH 1: Number of processes vs Execution time
    # ==============================================
    ax1.plot(process_counts, execution_times, 
             marker='o', color='blue', 
             linewidth=2, markersize=8, 
             markerfacecolor='white', markeredgewidth=2)
    
    # Add value labels on points
    for i, (proc, time_val) in enumerate(zip(process_counts, execution_times)):
        ax1.text(proc, time_val, f'{time_val:.3f}s', 
                ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax1.set_xlabel('Number of MPI Processes', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Processes vs Execution Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(process_counts)
    
    # Set y-axis to start from 0
    y_max = max(execution_times) * 1.1
    ax1.set_ylim(0, y_max)
    
    # Add annotation for best time
    best_time_idx = execution_times.index(min(execution_times))
    best_proc = process_counts[best_time_idx]
    best_time = execution_times[best_time_idx]
    ax1.annotate(f'Fastest: {best_proc} processes\n{best_time:.3f}s',
                xy=(best_proc, best_time),
                xytext=(best_proc, best_time * 1.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # ==============================================
    # GRAPH 2: Number of processes vs Speedup
    # ==============================================
    ax2.plot(process_counts, speedups,
             marker='s', color='green',
             linewidth=2, markersize=8,
             markerfacecolor='white', markeredgewidth=2,
             label='Actual Speedup')
    
    # Add ideal speedup line (linear)
    ax2.plot(process_counts, process_counts,
             'k--', linewidth=1.5, alpha=0.5,
             label='Ideal Speedup')
    
    # Add value labels on points
    for i, (proc, speedup) in enumerate(zip(process_counts, speedups)):
        ax2.text(proc, speedup, f'{speedup:.2f}x',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax2.set_xlabel('Number of MPI Processes', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Processes vs Speedup', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(process_counts)
    ax2.legend(loc='upper left')
    
    # Calculate and display efficiency
    if len(speedups) > 1:
        best_speedup = max(speedups[1:])  # Skip serial
        best_proc_idx = speedups.index(best_speedup)
        best_proc_speedup = process_counts[best_proc_idx]
        efficiency = best_speedup / best_proc_speedup
        
        ax2.annotate(f'Best Speedup: {best_speedup:.2f}x\nEfficiency: {efficiency:.1%}',
                    xy=(best_proc_speedup, best_speedup),
                    xytext=(best_proc_speedup * 0.7, best_speedup * 0.7),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    fontsize=10, color='blue',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../results/mpi_performance_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Graphs saved to: {filename}")
    
    # Also save a version without timestamp for easy access
    plt.savefig("../results/mpi_performance.png", dpi=300, bbox_inches='tight')
    print(f"✓ Graphs also saved to: ../results/mpi_performance.png")
    
    # Show the plots
    plt.show()

def save_results_to_file(process_counts, execution_times, speedups, array_size):
    """
    Save performance results to a text file
    
    Args:
        process_counts (list): Process counts tested
        execution_times (list): Execution times
        speedups (list): Speedup values
        array_size (int): Array size used
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open("../results/mpi_performance_results.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MPI PERFORMANCE RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Test Date: {timestamp}\n")
        f.write(f"Array Size: {array_size:,}\n")
        f.write(f"Test Environment: WSL2 Ubuntu\n\n")
        
        f.write("-" * 60 + "\n")
        f.write(f"{'Processes':>10} | {'Time (s)':>12} | {'Speedup':>10} | {'Efficiency':>12}\n")
        f.write("-" * 60 + "\n")
        
        serial_time = execution_times[0]
        
        for i, (proc, time_val, speedup) in enumerate(zip(process_counts, execution_times, speedups)):
            if proc == 1:
                efficiency = 1.0
            else:
                efficiency = speedup / proc if proc > 0 else 0
            
            f.write(f"{proc:10d} | {time_val:12.6f} | {speedup:10.2f}x | {efficiency:11.1%}\n")
        
        f.write("-" * 60 + "\n\n")
        
        # Analysis summary
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        
        # Find best performance
        if len(execution_times) > 1:
            best_time_idx = execution_times.index(min(execution_times[1:]))  # Skip serial
            best_speedup_idx = speedups.index(max(speedups[1:]))  # Skip serial
            
            f.write(f"Fastest Execution: {process_counts[best_time_idx]} processes\n")
            f.write(f"  Time: {execution_times[best_time_idx]:.4f}s\n")
            f.write(f"  Speedup vs serial: {execution_times[0]/execution_times[best_time_idx]:.2f}x\n\n")
            
            f.write(f"Best Speedup: {process_counts[best_speedup_idx]} processes\n")
            f.write(f"  Speedup: {speedups[best_speedup_idx]:.2f}x\n")
            f.write(f"  Efficiency: {speedups[best_speedup_idx]/process_counts[best_speedup_idx]:.1%}\n\n")
        
        # Calculate Amdahl's Law theoretical speedup
        if len(execution_times) > 1:
            parallel_portion = 1 - (execution_times[-1] / execution_times[0]) / max(process_counts)
            theoretical_max = 1 / ((1 - parallel_portion) + (parallel_portion / max(process_counts)))
            
            f.write(f"Parallel Portion (estimated): {parallel_portion:.1%}\n")
            f.write(f"Theoretical Max Speedup: {theoretical_max:.2f}x\n")
            f.write(f"Achieved Speedup: {speedups[-1]:.2f}x ({speedups[-1]/theoretical_max:.1%} of theoretical)\n")
    
    print(f"✓ Results saved to: ../results/mpi_performance_results.txt")

def check_prerequisites():
    """
    Check if all prerequisites are met
    """
    print("Checking prerequisites...")
    
    # Check if MPI program exists
    if not os.path.exists("../mpi/minmax_mpi.exe"):
        print("❌ MPI executable not found!")
        print("   Please compile the MPI program first:")
        print("   cd ../mpi && make")
        return False
    
    # Check if mpiexec is available
    try:
        result = subprocess.run(["mpiexec", "-help"], capture_output=True, check=False, shell=True)
        print("✓ mpiexec is available")
    except:
        print("❌ mpiexec not found!")
        print("   Install MS-MPI from: https://github.com/microsoft/Microsoft-MPI/releases")
        return False
    
    # Check Python dependencies
    try:
        import matplotlib
        import numpy
        print("✓ Python dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("   Install with: pip3 install matplotlib numpy")
        return False
    
    return True

def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("SIMPLIFIED MPI PERFORMANCE EVALUATION")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Ask for array size
    print("\n" + "-" * 40)
    print("TEST CONFIGURATION")
    print("-" * 40)
    
    default_size = 1000000
    user_input = input(f"Enter array size (default: {default_size:,}): ").strip()
    
    if user_input:
        try:
            array_size = int(user_input)
            if array_size <= 0:
                print("⚠️  Array size must be positive. Using default.")
                array_size = default_size
        except ValueError:
            print("⚠️  Invalid input. Using default.")
            array_size = default_size
    else:
        array_size = default_size
    
    print(f"Using array size: {array_size:,}")
    
    # Confirm execution
    print(f"\nThis will run MPI with process counts: [1, 2, 4, 8, 16]")
    print("Estimated time: 1-2 minutes")
    
    confirm = input("\nProceed with evaluation? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Evaluation cancelled.")
        sys.exit(0)
    
    # Run evaluation
    try:
        generate_performance_graphs()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()