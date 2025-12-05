# OpenMP Run Script
# Compile and run the OpenMP min-max normalization

Write-Host "Compiling OpenMP program..." -ForegroundColor Green
gcc -O2 -Wall -Wextra -fopenmp -o minmax_openmp.exe minmax_openmp.c

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilation successful!" -ForegroundColor Green
    Write-Host "Running with 100M elements and 8 threads..." -ForegroundColor Cyan
    .\minmax_openmp.exe 100000000 8
} else {
    Write-Host "Compilation failed!" -ForegroundColor Red
}
