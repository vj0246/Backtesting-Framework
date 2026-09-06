<#
.SYNOPSIS
    Build the package and test what users will actually install.

.DESCRIPTION
    A passing pytest run says the source tree works. It cannot see a module
    left out of the wheel, a missing runtime dependency, a lost py.typed
    marker, or an "optional" extra that is really required. This builds the
    distribution, validates its metadata, installs the wheel into an empty
    virtual environment, and exercises it from a directory where the source
    tree is not importable.

    Run this before every release.

.EXAMPLE
    .\scripts\clean_room_test.ps1
#>
[CmdletBinding()]
param(
    # Python version for the clean environment. Defaults to the package floor,
    # which is the version most likely to expose a version-specific mistake.
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$cleanEnv = Join-Path $env:TEMP "qg-cleanroom"

function Step($number, $text) {
    Write-Host ""
    Write-Host "[$number/4] $text" -ForegroundColor Cyan
}

Push-Location $project
try {
    Step 1 "Building sdist and wheel"
    uv build
    if ($LASTEXITCODE -ne 0) { throw "build failed" }

    Step 2 "Validating distribution metadata"
    & (Join-Path $project ".venv\Scripts\python.exe") -m twine check "$project\dist\*"
    if ($LASTEXITCODE -ne 0) { throw "twine check failed" }

    Step 3 "Installing the wheel into an empty Python $PythonVersion environment"
    if (Test-Path $cleanEnv) { Remove-Item -Recurse -Force $cleanEnv }
    uv venv $cleanEnv --python $PythonVersion
    if ($LASTEXITCODE -ne 0) { throw "could not create the clean environment" }

    $wheel = Get-ChildItem (Join-Path $project "dist\*.whl") |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $wheel) { throw "no wheel found in dist/" }
    Write-Host "      $($wheel.Name)"

    $cleanPython = Join-Path $cleanEnv "Scripts\python.exe"
    uv pip install --python $cleanPython $wheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "the wheel failed to install" }

    Step 4 "Exercising the installed package with the source tree off sys.path"
    # Run from the temp directory: if an import only works because src/ happens
    # to be next to it, that is exactly the bug this is here to catch.
    Push-Location $env:TEMP
    try {
        & $cleanPython (Join-Path $project "scripts\verify_install.py")
        $verifyCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($verifyCode -ne 0) { throw "verify_install.py reported failures" }

    Write-Host ""
    Write-Host "The built package is sound. Safe to publish." -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "CLEAN-ROOM TEST FAILED: $_" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
