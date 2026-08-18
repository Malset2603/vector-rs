<#
.SYNOPSIS
    Starts a local VectorRS distributed search cluster (1 Coordinator + 3 Workers).

.PARAMETER IndexType
    Index backend type: "hnsw" or "ivf-pq" (default: "hnsw").

.PARAMETER Dimension
    Vector dimension (default: 128).

.PARAMETER Metric
    Distance metric: "l2", "dot", "cosine", "manhattan", "minkowski", "chebyshev", "hamming", "mahalanobis", "jaccard", "hellinger" (default: "l2").

.PARAMETER DataDir
    Optional directory containing shard_0.bin, shard_1.bin, shard_2.bin.

.PARAMETER Release
    Runs binaries compiled in release mode.
#>

param(
    [string]$IndexType = "hnsw",
    [int]$Dimension = 128,
    [string]$Metric = "l2",
    [string]$DataDir = "",
    [switch]$Release
)

$ErrorActionPreference = "Stop"

$BinMode = if ($Release) { "release" } else { "debug" }
$CargoFlag = if ($Release) { "--release" } else { "" }

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   VectorRS Distributed Cluster Orchestrator      " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Index Type:  $IndexType"
Write-Host "Dimension:   $Dimension"
Write-Host "Metric:      $Metric"
Write-Host "Build Mode:  $BinMode"
if ($DataDir) { Write-Host "Data Dir:    $DataDir" }
Write-Host "--------------------------------------------------"

# Build worker and coordinator binaries first
Write-Host "[1/5] Compiling cluster binaries ($BinMode)..." -ForegroundColor Yellow
if ($Release) {
    cargo build --release --bin vector-worker --bin vector-coordinator
} else {
    cargo build --bin vector-worker --bin vector-coordinator
}

$WorkerBin = "target\$BinMode\vector-worker.exe"
$CoordBin = "target\$BinMode\vector-coordinator.exe"

$Processes = @()

try {
    # 1. Start Worker 0 (Port 50051)
    Write-Host "[2/5] Starting Worker 0 on port 50051 (Shard 0, Metric: $Metric)..." -ForegroundColor Green
    $W0Args = @("--port", "50051", "--worker-id", "worker-0", "--shard-id", "0", "--index-type", $IndexType, "--dimension", $Dimension, "--metric", $Metric)
    if ($DataDir -and (Test-Path "$DataDir\shard_0.bin")) {
        $W0Args += @("--storage-file", "$DataDir\shard_0.bin")
    }
    $p0 = Start-Process -FilePath $WorkerBin -ArgumentList $W0Args -PassThru -NoNewWindow
    $Processes += $p0

    # 2. Start Worker 1 (Port 50052)
    Write-Host "[3/5] Starting Worker 1 on port 50052 (Shard 1, Metric: $Metric)..." -ForegroundColor Green
    $W1Args = @("--port", "50052", "--worker-id", "worker-1", "--shard-id", "1", "--index-type", $IndexType, "--dimension", $Dimension, "--metric", $Metric)
    if ($DataDir -and (Test-Path "$DataDir\shard_1.bin")) {
        $W1Args += @("--storage-file", "$DataDir\shard_1.bin")
    }
    $p1 = Start-Process -FilePath $WorkerBin -ArgumentList $W1Args -PassThru -NoNewWindow
    $Processes += $p1

    # 3. Start Worker 2 (Port 50053)
    Write-Host "[4/5] Starting Worker 2 on port 50053 (Shard 2, Metric: $Metric)..." -ForegroundColor Green
    $W2Args = @("--port", "50053", "--worker-id", "worker-2", "--shard-id", "2", "--index-type", $IndexType, "--dimension", $Dimension, "--metric", $Metric)
    if ($DataDir -and (Test-Path "$DataDir\shard_2.bin")) {
        $W2Args += @("--storage-file", "$DataDir\shard_2.bin")
    }
    $p2 = Start-Process -FilePath $WorkerBin -ArgumentList $W2Args -PassThru -NoNewWindow
    $Processes += $p2

    Start-Sleep -Seconds 2

    # 4. Start Coordinator (Port 50050)
    Write-Host "[5/5] Starting Coordinator on port 50050..." -ForegroundColor Green
    $CoordArgs = @(
        "--port", "50050",
        "--workers", "http://127.0.0.1:50051,http://127.0.0.1:50052,http://127.0.0.1:50053"
    )
    $pCoord = Start-Process -FilePath $CoordBin -ArgumentList $CoordArgs -PassThru -NoNewWindow
    $Processes += $pCoord

    Write-Host "`n>>> VectorRS Cluster is ONLINE and HEALTHY! <<<" -ForegroundColor Cyan
    Write-Host "Coordinator endpoint: http://127.0.0.1:50050"
    Write-Host "Press Ctrl+C to terminate all cluster nodes.`n"

    # Wait for processes
    while ($true) {
        Start-Sleep -Seconds 1
        foreach ($proc in $Processes) {
            if ($proc.HasExited) {
                Write-Host "Process $($proc.Id) has stopped. Shutting down cluster." -ForegroundColor Red
                break
            }
        }
    }
} finally {
    Write-Host "`nTerminating all VectorRS cluster processes..." -ForegroundColor Yellow
    foreach ($proc in $Processes) {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "All cluster processes stopped cleanly." -ForegroundColor Green
}
