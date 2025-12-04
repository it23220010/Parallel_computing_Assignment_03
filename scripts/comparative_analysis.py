#!/usr/bin/env python3
"""
Comprehensive Comparative Analysis of Parallel Implementations
Compares Serial, OpenMP, MPI, and CUDA implementations
Generates side-by-side bar charts matching the specified format
"""

import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def run_implementation(executable_path, *args):
    """Run an implementation and extract execution time"""
    try:
        result = subprocess.run(
            [executable_path] + [str(arg) for arg in args],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        for line in result.stdout.split('\n'):
            if "Time taken:" in line:
                time_str = line.split(":")[1].strip().split()[0]
                return float(time_str)
    except Exception as e:
        print(f"Error running {executable_path}: {e}")
        return None
    
    return None

def collect_performance_data(array_size):
    """Collect performance data from all implementations"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..")
    
    results = {}
    
    # Serial baseline
    print("\n" + "="*70)
    print("COLLECTING PERFORMANCE DATA")
    print("="*70)
    print("\nRunning Serial implementation...")
    serial_path = os.path.join(base_dir, "serial", "minmax_serial.exe")
    serial_time = run_implementation(serial_path, array_size)
    
    if serial_time:
        results['Serial'] = {
            'time': serial_time,
            'speedup': 1.0,
            'config': ''
        }
        print(f"  ✓ Serial: {serial_time:.4f}s")
    else:
        print("  ✗ Serial failed!")
        return None
    
    # OpenMP - test multiple thread counts
    print("\nRunning OpenMP implementation...")
    openmp_path = os.path.join(base_dir, "openmp", "minmax_openmp.exe")
    thread_counts = [1, 2, 4, 8, 16]
    best_openmp = None
    best_openmp_time = float('inf')
    best_threads = 0
    
    for threads in thread_counts:
        time = run_implementation(openmp_path, array_size, threads)
        if time and time < best_openmp_time:
            best_openmp_time = time
            best_threads = threads
    
    if best_openmp_time != float('inf'):
        results['OpenMP'] = {
            'time': best_openmp_time,
            'speedup': serial_time / best_openmp_time,
            'config': f'{best_threads} threads'
        }
        print(f"  ✓ OpenMP (best with {best_threads} threads): {best_openmp_time:.4f}s, Speedup: {results['OpenMP']['speedup']:.2f}x")
    else:
        print("  ✗ OpenMP failed!")
    
    # MPI - test multiple process counts
    print("\nRunning MPI implementation...")
    mpi_path = os.path.join(base_dir, "mpi", "minmax_mpi.exe")
    process_counts = [1, 2, 4, 8]
    best_mpi_time = float('inf')
    best_processes = 0
    
    for procs in process_counts:
        try:
            result = subprocess.run(
                ["mpiexec", "-n", str(procs), mpi_path, str(array_size)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            for line in result.stdout.split('\n'):
                if "Time taken:" in line:
                    time_str = line.split(":")[1].strip().split()[0]
                    time = float(time_str)
                    if time < best_mpi_time:
                        best_mpi_time = time
                        best_processes = procs
        except Exception as e:
            continue
    
    if best_mpi_time != float('inf'):
        results['MPI'] = {
            'time': best_mpi_time,
            'speedup': serial_time / best_mpi_time,
            'config': f'{best_processes} processes'
        }
        print(f"  ✓ MPI (best with {best_processes} processes): {best_mpi_time:.4f}s, Speedup: {results['MPI']['speedup']:.2f}x")
    else:
        print("  ✗ MPI failed!")
    
    # CUDA - test multiple block sizes
    print("\nRunning CUDA implementation...")
    cuda_path = os.path.join(base_dir, "cuda", "minmax_cuda.exe")
    block_sizes = [128, 256]
    best_cuda_time = float('inf')
    best_blocks = 0
    
    for block_size in block_sizes:
        num_blocks = min(65535, (array_size + block_size - 1) // block_size)
        try:
            result = subprocess.run(
                [cuda_path, str(array_size), str(block_size), str(num_blocks)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            for line in result.stdout.split('\n'):
                if "Total GPU time" in line:
                    # Parse "Total GPU time (incl. memcpy): 22.000 ms"
                    time_str = line.split(":")[1].strip().split()[0]
                    time = float(time_str) / 1000.0  # Convert ms to seconds
                    if time < best_cuda_time:
                        best_cuda_time = time
                        best_blocks = block_size
                    break
        except Exception as e:
            print(f"  Warning: CUDA block size {block_size} failed: {e}")
            continue
    
    if best_cuda_time != float('inf'):
        results['CUDA'] = {
            'time': best_cuda_time,
            'speedup': serial_time / best_cuda_time,
            'config': f'{best_blocks} blocks'
        }
        print(f"  ✓ CUDA (best with {best_blocks} block size): {best_cuda_time:.4f}s, Speedup: {results['CUDA']['speedup']:.2f}x")
    else:
        print("  ✗ CUDA failed!")
    
    return results

def create_comparative_charts(results, array_size):
    """Create side-by-side bar charts matching the specified format"""
    
    # Prepare data
    implementations = list(results.keys())
    times = [results[impl]['time'] for impl in implementations]
    speedups = [results[impl]['speedup'] for impl in implementations]
    configs = [results[impl]['config'] for impl in implementations]
    
    # Colors matching the image
    colors = ['#808080', '#4A90E2', '#50C878', '#E74C3C']  # Gray, Blue, Green, Red
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Comparative Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Chart 1: Execution Time
    bars1 = ax1.bar(range(len(implementations)), times, color=colors[:len(implementations)], 
                    edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Comparative Execution Time', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(implementations)))
    
    # Create labels with configuration
    labels = []
    for impl, config in zip(implementations, configs):
        if config:
            labels.append(f"{impl} ({config})")
        else:
            labels.append(impl)
    
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Chart 2: Speedup vs Serial
    bars2 = ax2.bar(range(len(implementations)), speedups, color=colors[:len(implementations)], 
                    edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Speedup', fontsize=11, fontweight='bold')
    ax2.set_title('Comparative Speedup vs Serial', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(implementations)))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comparative_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparative charts saved to: {output_path}")
    
    return output_path

def generate_detailed_report(results, array_size):
    """Generate comprehensive analysis report for assignment"""
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "comparative_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE COMPARATIVE ANALYSIS REPORT\n")
        f.write("Min-Max Normalization: Parallel Computing Implementations\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Array Size: {array_size:,} elements\n")
        f.write(f"Total Data: {array_size * 4 / (1024**2):.2f} MB (float32)\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 80 + "\n\n")
        
        # Sort by speedup
        sorted_results = sorted(results.items(), key=lambda x: x[1]['speedup'], reverse=True)
        
        for impl, data in sorted_results:
            f.write(f"{impl:15} | Time: {data['time']:8.4f}s | Speedup: {data['speedup']:6.2f}x")
            if data['config']:
                f.write(f" | Config: {data['config']}")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("4. COMPARATIVE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Find best performer
        best_impl = max(results.items(), key=lambda x: x[1]['speedup'])
        
        f.write("4.1 PERFORMANCE COMPARISON\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Best Performer: {best_impl[0]} with {best_impl[1]['speedup']:.2f}x speedup\n\n")
        
        # Execution time comparison
        f.write("Execution Time Ranking:\n")
        sorted_by_time = sorted(results.items(), key=lambda x: x[1]['time'])
        for i, (impl, data) in enumerate(sorted_by_time, 1):
            f.write(f"  {i}. {impl:12} - {data['time']:.4f}s")
            if data['config']:
                f.write(f" ({data['config']})")
            f.write("\n")
        
        f.write("\nSpeedup Comparison:\n")
        for i, (impl, data) in enumerate(sorted_results, 1):
            percentage = (data['speedup'] - 1) * 100
            f.write(f"  {i}. {impl:12} - {data['speedup']:.2f}x ({percentage:+.1f}% vs serial)")
            if data['config']:
                f.write(f" ({data['config']})")
            f.write("\n")
        
        f.write("\n\n4.2 MOST APPROPRIATE IMPLEMENTATION\n")
        f.write("-" * 80 + "\n\n")
        
        if 'CUDA' in results and results['CUDA']['speedup'] > 5:
            f.write("RECOMMENDATION: CUDA GPU Implementation\n\n")
            f.write("JUSTIFICATION:\n")
            f.write(f"• Achieves highest speedup of {results['CUDA']['speedup']:.2f}x over serial\n")
            f.write("• Ideal for data-parallel operations like min-max normalization\n")
            f.write("• Can process millions of elements simultaneously\n")
            f.write("• Best performance when sufficient GPU resources are available\n")
            f.write("• Minimal code complexity for data-parallel algorithms\n\n")
            
            f.write("WHEN TO USE:\n")
            f.write("• Large datasets (>10M elements)\n")
            f.write("• Repeated computations on similar data\n")
            f.write("• Systems with dedicated GPU hardware\n")
            f.write("• Real-time processing requirements\n")
        elif 'OpenMP' in results:
            f.write("RECOMMENDATION: OpenMP Shared-Memory Parallelization\n\n")
            f.write("JUSTIFICATION:\n")
            f.write(f"• Good speedup of {results['OpenMP']['speedup']:.2f}x with minimal overhead\n")
            f.write("• Easy to implement with simple pragma directives\n")
            f.write("• Excellent for multi-core CPU systems\n")
            f.write("• Lower memory requirements than MPI\n")
            f.write("• Best balance of performance and ease of use\n")
        
        f.write("\n\n4.3 STRENGTHS AND WEAKNESSES\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("SERIAL IMPLEMENTATION:\n")
        f.write("Strengths:\n")
        f.write("  • Simplest to understand and debug\n")
        f.write("  • No parallel overhead\n")
        f.write("  • Portable across all systems\n")
        f.write("  • Predictable execution time\n")
        f.write("Weaknesses:\n")
        f.write("  • Cannot utilize multiple cores/processors\n")
        f.write("  • Slowest execution for large datasets\n")
        f.write("  • Not scalable\n\n")
        
        if 'OpenMP' in results:
            f.write("OPENMP IMPLEMENTATION:\n")
            f.write("Strengths:\n")
            f.write("  • Easy to implement with compiler directives\n")
            f.write("  • Automatic load balancing\n")
            f.write(f"  • Good speedup ({results['OpenMP']['speedup']:.2f}x) for shared-memory systems\n")
            f.write("  • Low programming complexity\n")
            f.write("  • Efficient memory usage (shared address space)\n")
            f.write("Weaknesses:\n")
            f.write("  • Limited to single node (shared memory)\n")
            f.write("  • Scalability limited by number of CPU cores\n")
            f.write("  • Thread synchronization overhead\n")
            f.write("  • Potential race conditions if not careful\n\n")
        
        if 'MPI' in results:
            f.write("MPI IMPLEMENTATION:\n")
            f.write("Strengths:\n")
            f.write("  • Can scale across multiple nodes/machines\n")
            f.write("  • Suitable for distributed systems\n")
            f.write("  • No shared memory constraints\n")
            f.write("  • Industry standard for HPC\n")
            f.write("Weaknesses:\n")
            f.write("  • Higher communication overhead\n")
            f.write(f"  • Moderate speedup ({results['MPI']['speedup']:.2f}x) for this algorithm\n")
            f.write("  • More complex programming model\n")
            f.write("  • Data distribution and gathering overhead\n")
            f.write("  • Not efficient for small datasets\n\n")
        
        if 'CUDA' in results:
            f.write("CUDA IMPLEMENTATION:\n")
            f.write("Strengths:\n")
            f.write(f"  • Exceptional speedup ({results['CUDA']['speedup']:.2f}x)\n")
            f.write("  • Massive parallelism (thousands of threads)\n")
            f.write("  • Perfect for data-parallel operations\n")
            f.write("  • High memory bandwidth\n")
            f.write("  • Excellent for repetitive computations\n")
            f.write("Weaknesses:\n")
            f.write("  • Requires NVIDIA GPU hardware\n")
            f.write("  • Higher programming complexity\n")
            f.write("  • Memory transfer overhead (CPU↔GPU)\n")
            f.write("  • Limited by GPU memory size\n")
            f.write("  • Not portable to non-NVIDIA systems\n\n")
        
        f.write("\n4.4 ALGORITHM-SPECIFIC ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        f.write("Min-Max Normalization Characteristics:\n")
        f.write("• Highly data-parallel operation\n")
        f.write("• Two phases: 1) Find min/max, 2) Normalize\n")
        f.write("• Independent computations per element in normalization phase\n")
        f.write("• Reduction operation for min/max finding\n\n")
        
        f.write("Why This Algorithm Benefits from Parallelization:\n")
        f.write("• Each element can be normalized independently\n")
        f.write("• Reduction operations are efficiently parallelizable\n")
        f.write("• High compute-to-communication ratio\n")
        f.write("• Memory access patterns are regular and predictable\n")
        f.write("• Minimal data dependencies\n\n")
        
        f.write("\n4.5 SCALABILITY ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        
        if 'OpenMP' in results:
            efficiency = (results['OpenMP']['speedup'] / int(results['OpenMP']['config'].split()[0])) * 100
            f.write(f"OpenMP Parallel Efficiency: {efficiency:.1f}%\n")
            if efficiency > 80:
                f.write("  → Excellent scaling efficiency\n")
            elif efficiency > 60:
                f.write("  → Good scaling efficiency\n")
            else:
                f.write("  → Moderate scaling efficiency, overhead visible\n")
        
        f.write("\nKey Observations:\n")
        if 'CUDA' in results and results['CUDA']['speedup'] > results.get('OpenMP', {}).get('speedup', 0):
            f.write("• CUDA shows superior performance for large-scale parallelism\n")
        if 'OpenMP' in results:
            f.write("• OpenMP provides best ease-of-use to performance ratio\n")
        if 'MPI' in results:
            f.write("• MPI overhead is significant for this algorithm on single node\n")
        
        f.write("\n\n4.6 RECOMMENDATIONS FOR DIFFERENT SCENARIOS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Small Datasets (<1M elements):\n")
        f.write("  → Use Serial or OpenMP (parallel overhead not justified)\n\n")
        
        f.write("Medium Datasets (1M-10M elements):\n")
        f.write("  → Use OpenMP for best balance of performance and simplicity\n\n")
        
        f.write("Large Datasets (>10M elements):\n")
        f.write("  → Use CUDA if GPU available, otherwise OpenMP\n\n")
        
        f.write("Distributed Systems:\n")
        f.write("  → Use MPI when data must be distributed across nodes\n\n")
        
        f.write("Production Systems:\n")
        if 'CUDA' in results and results['CUDA']['speedup'] > 50:
            f.write("  → CUDA for maximum throughput with GPU hardware\n")
        else:
            f.write("  → OpenMP for reliability and portability\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF COMPARATIVE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Detailed report saved to: {report_path}")
    return report_path

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARATIVE ANALYSIS")
    print("Min-Max Normalization: Serial vs OpenMP vs MPI vs CUDA")
    print("=" * 80)
    
    # Use command-line argument or default to 10^8
    array_size = int(sys.argv[1]) if len(sys.argv) > 1 else 100000000
    
    print(f"\nArray Size: {array_size:,} elements")
    print(f"Data Size: {array_size * 4 / (1024**2):.2f} MB\n")
    
    # Collect performance data
    results = collect_performance_data(array_size)
    
    if not results or len(results) < 2:
        print("\n❌ Insufficient data collected. Please ensure implementations are compiled.")
        sys.exit(1)
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING COMPARATIVE CHARTS")
    print("=" * 70)
    create_comparative_charts(results, array_size)
    
    # Generate detailed report
    print("\n" + "=" * 70)
    print("GENERATING ANALYSIS REPORT")
    print("=" * 70)
    generate_detailed_report(results, array_size)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • results/comparative_analysis.png - Side-by-side comparison charts")
    print("  • results/comparative_analysis_report.txt - Comprehensive analysis report")
    print("\nThis analysis addresses all requirements for Section 4:")
    print("  ✓ Comparison on same dataset/problem size")
    print("  ✓ Comparative graphs (execution time and speedup)")
    print("  ✓ Justification for most appropriate implementation")
    print("  ✓ Strengths and weaknesses discussion")
    print("\n")

if __name__ == "__main__":
    main()
