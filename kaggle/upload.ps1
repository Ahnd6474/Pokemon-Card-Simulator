param(
    [Parameter(Mandatory = $true)]
    [string]$Username,
    [string]$ConfigDir = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Kaggle = Join-Path $RepoRoot ".venv\Scripts\kaggle.exe"
$KernelPath = Join-Path $PSScriptRoot "kernel"

if ($ConfigDir) {
    $env:KAGGLE_CONFIG_DIR = $ConfigDir
}

& $Python (Join-Path $PSScriptRoot "prepare_kaggle.py") --username $Username
& $Kaggle kernels push -p $KernelPath --accelerator NvidiaTeslaT4
