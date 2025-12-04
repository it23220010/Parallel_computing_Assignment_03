# Build and Run Script for Parallel Computing Assignment 03
# PowerShell alternative to Makefile

param(
    [string]$Target = "help",
    [int]$ArraySize = 100000000
)

# Add MSYS2 to PATH for GCC
$env:Path = "C:\msys64\mingw64\bin;C:\msys64\ucrt64\bin;$env:Path"

# Compiler commands (now in PATH)
$GCC = "gcc"
$NVCC = "nvcc"
$VCVARS = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# Directories
$SERIAL_DIR = "serial"
$OPENMP_DIR = "openmp"
$MPI_DIR = "mpi"
$CUDA_DIR = "cuda"
$SCRIPTS_DIR = "scripts"
$RESULTS_DIR = "results"

function Show-Help {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Parallel Computing Assignment 03" -ForegroundColor Cyan
    Write-Host "Build and Run Script" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\build.ps1 [-Target <target>] [-ArraySize <size>]"
    Write-Host ""
    Write-Host "Available targets:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  all              - Build all implementations (default)"
    Write-Host "  serial           - Build serial implementation only"
    Write-Host "  openmp           - Build OpenMP implementation only"
    Write-Host "  mpi              - Build MPI implementation only"
    Write-Host "  cuda             - Build CUDA implementation only"
    Write-Host ""
    Write-Host "  run-all          - Build and run all implementations"
    Write-Host "  comparative      - Generate comprehensive comparative analysis"
    Write-Host "  quick-compare    - Run quick performance comparison"
    Write-Host ""
    Write-Host "  clean            - Remove all executables and results"
    Write-Host "  clean-build      - Remove only executables (keep results)"
    Write-Host "  check            - Check which executables exist"
    Write-Host "  help             - Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\build.ps1 -Target all"
    Write-Host "  .\build.ps1 -Target comparative"
    Write-Host "  .\build.ps1 -Target run-all -ArraySize 10000000"
    Write-Host ""
}

function Build-Serial {
    Write-Host "Building Serial implementation..." -ForegroundColor Yellow
    Push-Location $SERIAL_DIR
    & $GCC -O3 -Wall -o minmax_serial.exe minmax_serial.c -lm
    Pop-Location
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Serial executable created" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] Serial build failed" -ForegroundColor Red
    }
}

function Build-OpenMP {
    Write-Host "Building OpenMP implementation..." -ForegroundColor Yellow
    Push-Location $OPENMP_DIR
    & $GCC -O3 -Wall -fopenmp -o minmax_openmp.exe minmax_openmp.c -lm
    Pop-Location
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] OpenMP executable created" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] OpenMP build failed" -ForegroundColor Red
    }
}

function Build-MPI {
    Write-Host "Building MPI implementation..." -ForegroundColor Yellow
    Push-Location $MPI_DIR
    & $GCC -O3 -Wall -o minmax_mpi.exe minmax_mpi.c -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi -lm
    Pop-Location
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] MPI executable created" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] MPI build failed" -ForegroundColor Red
    }
}

function Build-CUDA {
    Write-Host "Building CUDA implementation..." -ForegroundColor Yellow
    Push-Location $CUDA_DIR
    cmd /c "`"$VCVARS`" && nvcc -O3 -arch=sm_75 -o minmax_cuda.exe minmax_cuda.cu"
    Pop-Location
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] CUDA executable created" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] CUDA build failed" -ForegroundColor Red
    }
}

function Build-All {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Building All Implementations" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Build-Serial
    Build-OpenMP
    Build-MPI
    Build-CUDA
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "All implementations compiled!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
}

function Run-All {
    Build-All
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Running All Implementations" -ForegroundColor Cyan
    Write-Host "Array size: $ArraySize" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "[Serial Implementation]" -ForegroundColor Yellow
    Push-Location $SERIAL_DIR
    & .\minmax_serial.exe $ArraySize
    Pop-Location
    
    Write-Host ""
    Write-Host "[OpenMP Implementation - 8 threads]" -ForegroundColor Yellow
    Push-Location $OPENMP_DIR
    & .\minmax_openmp.exe $ArraySize 8
    Pop-Location
    
    Write-Host ""
    Write-Host "[MPI Implementation - 4 processes]" -ForegroundColor Yellow
    Push-Location $MPI_DIR
    mpiexec -n 4 .\minmax_mpi.exe $ArraySize
    Pop-Location
    
    Write-Host ""
    Write-Host "[CUDA Implementation]" -ForegroundColor Yellow
    Push-Location $CUDA_DIR
    & .\minmax_cuda.exe $ArraySize 128 781250
    Pop-Location
    
    Write-Host ""
}

function Run-Comparative {
    Build-All
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Running Comprehensive Comparative Analysis" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "This will generate:" -ForegroundColor Yellow
    Write-Host "  - Comparative charts (execution time and speedup)" -ForegroundColor Yellow
    Write-Host "  - Detailed analysis report" -ForegroundColor Yellow
    Write-Host ""
    
    Push-Location $SCRIPTS_DIR
    & "C:\Program Files\Python313\python.exe" comparative_analysis.py $ArraySize
    Pop-Location
    
    Write-Host ""
    Write-Host "Results saved in $RESULTS_DIR/ directory" -ForegroundColor Green
    Write-Host ""
}

function Run-QuickCompare {
    Build-All
    
    Write-Host ""
    Write-Host "Running quick comparison..." -ForegroundColor Yellow
    Push-Location $SCRIPTS_DIR
    python quick_compare.py
    Pop-Location
}

function Clean-All {
    Write-Host "Cleaning all build artifacts and results..." -ForegroundColor Yellow
    
    if (Test-Path "$SERIAL_DIR\minmax_serial.exe") { Remove-Item "$SERIAL_DIR\minmax_serial.exe" }
    if (Test-Path "$OPENMP_DIR\minmax_openmp.exe") { Remove-Item "$OPENMP_DIR\minmax_openmp.exe" }
    if (Test-Path "$MPI_DIR\minmax_mpi.exe") { Remove-Item "$MPI_DIR\minmax_mpi.exe" }
    if (Test-Path "$CUDA_DIR\minmax_cuda.exe") { Remove-Item "$CUDA_DIR\minmax_cuda.exe" }
    
    if (Test-Path $RESULTS_DIR) {
        Get-ChildItem $RESULTS_DIR -Include *.png,*.txt,*.csv -Recurse | Remove-Item
    }
    
    Write-Host "Clean complete!" -ForegroundColor Green
}

function Clean-Build {
    Write-Host "Cleaning executables only..." -ForegroundColor Yellow
    
    if (Test-Path "$SERIAL_DIR\minmax_serial.exe") { Remove-Item "$SERIAL_DIR\minmax_serial.exe" }
    if (Test-Path "$OPENMP_DIR\minmax_openmp.exe") { Remove-Item "$OPENMP_DIR\minmax_openmp.exe" }
    if (Test-Path "$MPI_DIR\minmax_mpi.exe") { Remove-Item "$MPI_DIR\minmax_mpi.exe" }
    if (Test-Path "$CUDA_DIR\minmax_cuda.exe") { Remove-Item "$CUDA_DIR\minmax_cuda.exe" }
    
    Write-Host "Executables cleaned!" -ForegroundColor Green
}

function Check-Executables {
    Write-Host "Checking compiled executables..." -ForegroundColor Yellow
    
    if (Test-Path "$SERIAL_DIR\minmax_serial.exe") {
        Write-Host "  [OK] Serial" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] Serial" -ForegroundColor Red
    }
    
    if (Test-Path "$OPENMP_DIR\minmax_openmp.exe") {
        Write-Host "  [OK] OpenMP" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] OpenMP" -ForegroundColor Red
    }
    
    if (Test-Path "$MPI_DIR\minmax_mpi.exe") {
        Write-Host "  [OK] MPI" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] MPI" -ForegroundColor Red
    }
    
    if (Test-Path "$CUDA_DIR\minmax_cuda.exe") {
        Write-Host "  [OK] CUDA" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] CUDA" -ForegroundColor Red
    }
}

# Main execution
switch ($Target.ToLower()) {
    "all" { Build-All }
    "serial" { Build-Serial }
    "openmp" { Build-OpenMP }
    "mpi" { Build-MPI }
    "cuda" { Build-CUDA }
    "run-all" { Run-All }
    "comparative" { Run-Comparative }
    "quick-compare" { Run-QuickCompare }
    "clean" { Clean-All }
    "clean-build" { Clean-Build }
    "check" { Check-Executables }
    "help" { Show-Help }
    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Show-Help
    }
}
