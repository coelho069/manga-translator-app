$ErrorActionPreference = "Stop"

Write-Host "Configurando Manga Translator App..."

$pythonCmd = "py -3.10"
try {
    Invoke-Expression "$pythonCmd --version"
} catch {
    Write-Error "Python 3.10 nao encontrado. Instale o Python 3.10 e tente novamente."
    exit 1
}

if (!(Test-Path ".venv")) {
    Invoke-Expression "$pythonCmd -m venv .venv"
}

& ".\.venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

New-Item -ItemType Directory -Force -Path "models" | Out-Null
New-Item -ItemType Directory -Force -Path "output" | Out-Null

Write-Host ""
Write-Host "Instalacao concluida."
Write-Host "Coloque o modelo YOLO em: models/bubble_seg.pt"
Write-Host "Depois execute: .\run_app.ps1"
