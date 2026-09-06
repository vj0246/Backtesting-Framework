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
$cleanEnv = Join-Path $env:TEMP "fbt-cleanroom"

function Step($number, $text) {
    Write-Host ""
    Write-Host "[$number/4] $text" -ForegroundColor Cyan
}

# Windows PowerShell turns a native command's stderr into a terminating error
# while $ErrorActionPreference is "Stop". Tools like uv and twine write ordinary
# progress to stderr, so success would be reported as failure. Run native
# commands with that preference relaxed and judge them by their exit code.
function Invoke-Native {
    param(
        [Parameter(Mandatory)][scriptblock]$Command,
        [Parameter(Mandatory)][string]$What
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Command } finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

Push-Location $project
try {
    $venvPython = Join-Path $project ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) { throw "no development environment at .venv" }

    Step 1 "Building sdist and wheel"
    # Clear dist/ first. Artifacts from an earlier name or version linger there,
    # and `twine upload dist/*` would happily publish every one of them.
    $dist = Join-Path $project "dist"
    if (Test-Path $dist) {
        Get-ChildItem $dist -Include *.whl, *.tar.gz -Recurse | Remove-Item -Force
    }
    Invoke-Native { uv build } "uv build"

    Step 2 "Validating distribution metadata"
    Invoke-Native { & $venvPython -m twine check "$project\dist\*" } "twine check"

    Step 3 "Installing the wheel into an empty Python $PythonVersion environment"
    if (Test-Path $cleanEnv) { Remove-Item -Recurse -Force $cleanEnv }
    Invoke-Native { uv venv $cleanEnv --python $PythonVersion } "creating the clean environment"

    $wheel = Get-ChildItem (Join-Path $project "dist\*.whl") |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $wheel) { throw "no wheel found in dist/" }
    Write-Host "      $($wheel.Name)"

    $cleanPython = Join-Path $cleanEnv "Scripts\python.exe"
    Invoke-Native { uv pip install --python $cleanPython $wheel.FullName } "installing the wheel"

    Step 4 "Exercising the installed package with the source tree off sys.path"
    # Run from the temp directory: if an import only works because src/ happens
    # to be next to it, that is exactly the bug this is here to catch.
    Push-Location $env:TEMP
    try {
        Invoke-Native { & $cleanPython (Join-Path $project "scripts\verify_install.py") } "verify_install.py"
    }
    finally {
        Pop-Location
    }

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
