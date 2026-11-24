<#
.SYNOPSIS
  Configure, build, and (optionally) test Sonder Inference with MSVC + Ninja.

.DESCRIPTION
  Enters a Visual Studio x64 developer environment (located with vswhere) for
  this process only, clears poisoned CC/CXX overrides, then runs the CMake
  preset. Requires Visual Studio 2022+ (or Build Tools) with the C++ workload;
  CMake and Ninja from that installation are used when not already on PATH.

  Pass -LlamaCppSourceDir to build direct llama.cpp presets (such as
  windows-cuda-blackwell, windows-cuda-rtx, windows-rocm-xt, or
  windows-vulkan-xt) against a local PrismML or mainline llama.cpp checkout,
  and -ExtraCMakeArgs for any additional -D cache overrides.

.EXAMPLE
  powershell -NoProfile -File scripts\build.ps1 -Preset msvc-debug -Test

.EXAMPLE
  powershell -NoProfile -File scripts\build.ps1 -Preset windows-cuda-blackwell -LlamaCppSourceDir D:\ollama\runtimes\PrismML-Eng\llama.cpp -Test
#>
[CmdletBinding()]
param(
    [string]$Preset = 'msvc-debug',
    [switch]$Test,
    [switch]$Clean,
    [string]$LlamaCppSourceDir = '',
    [ValidateRange(0, 1024)]
    [int]$Jobs = 0,
    [string[]]$ExtraCMakeArgs = @()
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

function Show-GpuToolchainSummary([string]$PresetName) {
    if ($PresetName -notmatch '(?i)(cuda|rocm|vulkan)') { return }
    Write-Host "[sonder-build] GPU preset '$PresetName' toolchain probe:"
    $nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($nvcc) {
        Write-Host "  CUDA   : $($nvcc.Source)"
    } elseif ($env:CUDA_PATH) {
        Write-Host "  CUDA   : CUDA_PATH=$($env:CUDA_PATH) (nvcc.exe not on PATH)"
    } else {
        Write-Host "  CUDA   : not detected"
    }

    $hipcc = Get-Command hipcc.exe -ErrorAction SilentlyContinue
    if (-not $hipcc) { $hipcc = Get-Command hipcc.bat -ErrorAction SilentlyContinue }
    if ($hipcc) {
        Write-Host "  ROCm   : $($hipcc.Source)"
    } elseif ($env:HIP_PATH) {
        Write-Host "  ROCm   : HIP_PATH=$($env:HIP_PATH)"
    } else {
        Write-Host "  ROCm   : not detected"
    }

    $glslc = Get-Command glslc.exe -ErrorAction SilentlyContinue
    if ($glslc) {
        Write-Host "  Vulkan : $($glslc.Source)"
    } elseif ($env:VULKAN_SDK) {
        Write-Host "  Vulkan : VULKAN_SDK=$($env:VULKAN_SDK)"
    } else {
        Write-Host "  Vulkan : not detected"
    }
}

Enter-MsvcEnvironment
# A stray CC/CXX (for example pointing at a non-compiler) breaks CMake's
# compiler detection; presets pin cl explicitly.
Remove-Item Env:CC -ErrorAction SilentlyContinue
Remove-Item Env:CXX -ErrorAction SilentlyContinue

Show-GpuToolchainSummary -PresetName $Preset

$configureArgs = @('--preset', $Preset)
if (-not [string]::IsNullOrWhiteSpace($LlamaCppSourceDir)) {
    if (-not (Test-Path -LiteralPath $LlamaCppSourceDir -PathType Container)) {
        throw "LlamaCppSourceDir does not exist or is not a directory: $LlamaCppSourceDir"
    }
    $normalizedLlamaSrc = (Resolve-Path -LiteralPath $LlamaCppSourceDir).ProviderPath.Replace('\', '/')
    $configureArgs += "-DFETCHCONTENT_SOURCE_DIR_LLAMACPP=$normalizedLlamaSrc"
}
if ($ExtraCMakeArgs -and $ExtraCMakeArgs.Count -gt 0) {
    $configureArgs += $ExtraCMakeArgs
}

$buildArgs = @('--build', '--preset', $Preset)
$testArgs = @('--preset', $Preset)
if ($Jobs -gt 0) {
    $buildArgs += @('--parallel', "$Jobs")
    $testArgs += @('--parallel', "$Jobs")
}

Push-Location $repoRoot
try {
    $binaryDir = Join-Path $repoRoot "build\$Preset"
    if ($Clean -and (Test-Path -LiteralPath $binaryDir)) {
        Remove-Item -Recurse -Force -LiteralPath $binaryDir
    }
    & cmake @configureArgs
    if ($LASTEXITCODE -ne 0) { throw "configure failed ($LASTEXITCODE)" }
    & cmake @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }
    if ($Test) {
        & ctest @testArgs
        if ($LASTEXITCODE -ne 0) { throw "tests failed ($LASTEXITCODE)" }
    }
}
finally {
    Pop-Location
}
