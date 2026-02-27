param(
    [string]$EnvName = "prtreid_train",
    [string]$PrtreidCommit = "30617a7"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

Write-Host "[setup] root: $root"
Write-Host "[setup] env: $EnvName"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda not found in PATH. Install Miniconda/Anaconda first."
}

# Create/update core environment.
conda create -n $EnvName python=3.9 pip -y
conda install -n $EnvName pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Install Python dependencies.
conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install -r requirements/requirements.txt

# Clone PRTReid source (pinned commit for reproducibility).
if (-not (Test-Path "third_party/prtreid/.git")) {
    git clone https://github.com/VlSomers/prtreid third_party/prtreid
}

git -C third_party/prtreid fetch --all --tags
git -C third_party/prtreid checkout $PrtreidCommit

# Install official prtreid python requirements (some are not in local requirements.txt).
conda run -n $EnvName python -m pip install -r third_party/prtreid/requirements.txt
conda run -n $EnvName python -m pip install scikit-learn

# Try editable install; on Windows it may fail if MSVC build tools are missing.
try {
    conda run -n $EnvName python -m pip install -e third_party/prtreid
    Write-Host "[setup] editable prtreid install succeeded"
}
catch {
    Write-Warning "[setup] editable install failed (likely missing MSVC build tools for Cython extension)."
    Write-Warning "[setup] fallback is supported: main_train.py/main_eval.py will import from third_party/prtreid source directly."
}

# Quick validation.
conda run -n $EnvName python scripts/check_install.py

Write-Host "[setup] done"
Write-Host "Activate with: conda activate $EnvName"
