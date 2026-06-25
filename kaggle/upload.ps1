param(
    [Parameter(Mandatory = $true)]
    [string]$Username,
    [string]$ConfigDir = "",
    [switch]$VersionDataset
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Kaggle = Join-Path $RepoRoot ".venv\Scripts\kaggle.exe"
$RuntimePath = Join-Path $PSScriptRoot "staging\runtime"
$KernelPath = Join-Path $PSScriptRoot "kernel"

if ($ConfigDir) {
    $env:KAGGLE_CONFIG_DIR = $ConfigDir
}

& $Python (Join-Path $PSScriptRoot "prepare_kaggle.py") --username $Username

if ($VersionDataset) {
    & $Kaggle datasets version -p $RuntimePath -m "Update Q-DVN runtime"
} else {
    & $Kaggle datasets create -p $RuntimePath -r zip
}

& $Kaggle kernels push -p $KernelPath --accelerator gpu
