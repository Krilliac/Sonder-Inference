<#
.SYNOPSIS
  Configure, build, and (optionally) test Sonder Inference with MSVC + Ninja.

.DESCRIPTION
  Enters a Visual Studio x64 developer environment (located with vswhere) for
  this process only, clears poisoned CC/CXX overrides, then runs the CMake
  preset. Requires Visual Studio 2022+ (or Build Tools) with the C++ workload;
  CMake and Ninja from that installation are used when not already on PATH.

.EXAMPLE
  powershell -NoProfile -File scripts\build.ps1 -Preset msvc-debug -Test
#>
[CmdletBinding()]
param(
    [string]$Preset = 'msvc-debug',
    [switch]$Test,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Enter-MsvcEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { throw 'vswhere.exe not found; install Visual Studio 2022+ with the C++ workload.' }
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) { throw 'No Visual Studio installation with the MSVC x64 toolset was found.' }
    $devShell = Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
    Import-Module $devShell
    Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
}

Enter-MsvcEnvironment
# A stray CC/CXX (for example pointing at a non-compiler) breaks CMake's
# compiler detection; presets pin cl explicitly.
Remove-Item Env:CC -ErrorAction SilentlyContinue
Remove-Item Env:CXX -ErrorAction SilentlyContinue

Push-Location $repoRoot
try {
    $binaryDir = Join-Path $repoRoot "build\$Preset"
    if ($Clean -and (Test-Path $binaryDir)) { Remove-Item -Recurse -Force $binaryDir }
    cmake --preset $Preset
    if ($LASTEXITCODE -ne 0) { throw "configure failed ($LASTEXITCODE)" }
    cmake --build --preset $Preset
    if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }
    if ($Test) {
        ctest --preset $Preset
        if ($LASTEXITCODE -ne 0) { throw "tests failed ($LASTEXITCODE)" }
    }
}
finally {
    Pop-Location
}
