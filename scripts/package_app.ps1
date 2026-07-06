$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$distRoot = Join-Path $root "dist"
$packageDir = Join-Path $distRoot "eLSFG"
$zipPath = Join-Path $distRoot "eLSFG-portable.zip"

if (Test-Path $packageDir) {
    Remove-Item -LiteralPath $packageDir -Recurse -Force
}
New-Item -ItemType Directory -Path $packageDir | Out-Null

$items = @(
    "app",
    "core",
    "models",
    "profiles",
    "checkpoints",
    "requirements.txt",
    "environment.yml",
    "run_app.ps1",
    "README.md"
)

foreach ($item in $items) {
    $source = Join-Path $root $item
    if (Test-Path $source) {
        Copy-Item -LiteralPath $source -Destination $packageDir -Recurse -Force
    }
}

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $packageDir "*") -DestinationPath $zipPath
Write-Host "Packaged app: $zipPath"
