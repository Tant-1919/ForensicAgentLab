param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$runner = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
& $runner -m pip install --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$envFile = @()
if (Test-Path -LiteralPath '.env') { $envFile = @('--env-file', '.env') }
Write-Host "Open http://127.0.0.1:$Port/"
& $runner -m uvicorn app.main:app --host 127.0.0.1 --port $Port @envFile

