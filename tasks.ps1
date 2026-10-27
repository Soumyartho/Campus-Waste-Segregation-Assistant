<#
.SYNOPSIS
  Windows equivalent of the Makefile. Usage: ./tasks.ps1 <target>
  Targets: setup, data, train-baseline, train, ablations, eval, robustness, app, test, lint, export, benchmark
#>
param([Parameter(Position = 0)][string]$Target = "help")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Py = if (Test-Path ".venv/Scripts/python.exe") { ".venv/Scripts/python.exe" } else { "python" }

function Run([string[]]$Cmd) {
    Write-Host ">> $Py $($Cmd -join ' ')" -ForegroundColor Cyan
    & $Py @Cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

switch ($Target) {
    "setup" {
        Run @("-m", "pip", "install", "-r", "requirements.txt")
        Run @("-m", "pip", "install", "-e", ".")
    }
    "data" {
        Run @("scripts/download_data.py")
        Run @("-m", "waste.data.prepare", "--config", "configs/data.yaml")
    }
    "train-baseline" { Run @("-m", "waste.train", "--config", "configs/train_custom_cnn.yaml") }
    "train" { Run @("-m", "waste.train", "--config", "configs/train_mobilenet.yaml") }
    "ablations" {
        Run @("-m", "waste.train", "--config", "configs/train_mobilenet_noaug.yaml")
        Run @("-m", "waste.train", "--config", "configs/train_mobilenet_headonly.yaml")
        Run @("-m", "waste.train", "--config", "configs/train_resnet18.yaml")
    }
    "eval" {
        Run @("-m", "waste.evaluate", "all", "--split", "test")
        Run @("-m", "waste.evaluate", "coverage", "--run", "mobilenet_v2", "--write-config")
        Run @("-m", "waste.evaluate", "compare")
    }
    "robustness" {
        Run @("-m", "waste.evaluate", "external", "--run", "all", "--dataset", "trashnet")
        Run @("-m", "waste.evaluate", "external", "--run", "mobilenet_v2", "--dataset", "campus")
        Run @("-m", "waste.evaluate", "compare")
    }
    "app" { Run @("-m", "streamlit", "run", "app/streamlit_app.py") }
    "test" { Run @("-m", "pytest") }
    "lint" { Run @("-m", "ruff", "check", ".") }
    "export" { Run @("scripts/export_onnx.py", "--run", "mobilenet_v2") }
    "benchmark" { Run @("scripts/benchmark.py") }
    default {
        Write-Host "Targets: setup, data, train-baseline, train, ablations, eval, robustness, app, test, lint, export, benchmark"
    }
}
