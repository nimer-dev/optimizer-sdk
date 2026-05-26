# PyPI release script for nimer (reads version from nimer/_version.py)
# Run after generating a token at https://pypi.org/manage/account/token/
#
# Usage:
#   $env:PYPI_TOKEN = "pypi-AgEIcHlwaS5vcmc..."   # token starts with `pypi-`
#   .\publish_pypi.ps1
#
# To dry-run on TestPyPI first, set $env:NIMER_PUBLISH_TEST = "1".

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not $env:PYPI_TOKEN) {
    Write-Host "ERROR: set PYPI_TOKEN env var first" -ForegroundColor Red
    Write-Host "  Generate at: https://pypi.org/manage/account/token/" -ForegroundColor Yellow
    Write-Host "  Recommended scope: 'Project: nimer' (or whole-account if first upload)" -ForegroundColor Yellow
    exit 1
}

# Sanity: confirm the dist files exist and match the version we expect
$expected_version = (Get-Content nimer/_version.py | Select-String -Pattern '"([0-9]+\.[0-9]+\.[0-9]+)"').Matches.Groups[1].Value
$whl = "dist/nimer-$expected_version-py3-none-any.whl"
$sdist = "dist/nimer-$expected_version.tar.gz"

if (-not (Test-Path $whl) -or -not (Test-Path $sdist)) {
    Write-Host "Building dist files for v$expected_version ..." -ForegroundColor Cyan
    if (Test-Path dist) { Remove-Item -Recurse -Force dist }
    python -m build
}

Write-Host "Verifying with twine check ..." -ForegroundColor Cyan
python -m twine check dist/*
if ($LASTEXITCODE -ne 0) { Write-Host "twine check failed" -ForegroundColor Red; exit 1 }

if ($env:NIMER_PUBLISH_TEST -eq "1") {
    Write-Host "Uploading to TestPyPI ..." -ForegroundColor Cyan
    python -m twine upload --repository testpypi --username __token__ --password $env:PYPI_TOKEN dist/*
    Write-Host "Test install: pip install -i https://test.pypi.org/simple/ nimer==$expected_version" -ForegroundColor Green
} else {
    Write-Host "Uploading to PyPI ..." -ForegroundColor Cyan
    python -m twine upload --username __token__ --password $env:PYPI_TOKEN dist/*
    Write-Host "Install with: pip install nimer==$expected_version" -ForegroundColor Green
    Write-Host "Released: https://pypi.org/project/nimer/$expected_version/" -ForegroundColor Green
}
