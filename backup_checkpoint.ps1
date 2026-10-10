param(
    [string]$DestinationParent = (Split-Path $PSScriptRoot -Parent),
    [string]$ChatStorage = "$env:APPDATA\Code\User\workspaceStorage\a0a735211a6f6a98767ace593aa890ce"
)

$ErrorActionPreference = 'Stop'
$destination = Join-Path $DestinationParent ('MMM-recovery-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $destination | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archivePath = Join-Path $destination 'project.zip'
$archive = [System.IO.Compression.ZipFile]::Open($archivePath, 'Create')
$fileCount = 0
try {
    $projectFiles = Get-ChildItem -LiteralPath $PSScriptRoot -File -Recurse -Force |
        Where-Object { $_.FullName -notmatch '\\(\.venv|__pycache__)\\' }
    foreach ($projectFile in $projectFiles) {
        $relative = $projectFile.FullName.Substring($PSScriptRoot.Length + 1).Replace('\', '/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive, $projectFile.FullName, $relative, [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
        $fileCount++
    }
} finally {
    $archive.Dispose()
}
$verification = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    if ($verification.Entries.Count -ne $fileCount) { throw 'Archive entry count mismatch.' }
    foreach ($entry in $verification.Entries) {
        $original = Join-Path $PSScriptRoot $entry.FullName
        $stream = $entry.Open()
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $archivedHash = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '')
            if ($archivedHash -ne (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash) {
                throw "Archive content mismatch: $($entry.FullName)"
            }
        } finally {
            $stream.Dispose()
            $sha.Dispose()
        }
    }
} finally {
    $verification.Dispose()
}
$chatDestination = Join-Path $destination 'chat-raw'
New-Item -ItemType Directory -Path $chatDestination | Out-Null
foreach ($name in @('chatSessions', 'chatEditingSessions', 'GitHub.copilot-chat', 'workspace.json')) {
    $source = Join-Path $ChatStorage $name
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination $chatDestination -Recurse
    }
}
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $python) {
    & $python --version 2>&1 | Set-Content (Join-Path $destination 'python-version.txt')
    & $python -m pip freeze | Set-Content (Join-Path $destination 'installed-packages.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Package inventory failed.' }
}
git -C $PSScriptRoot rev-parse HEAD | Set-Content (Join-Path $destination 'git-commit.txt')
git -C $PSScriptRoot status --short | Set-Content (Join-Path $destination 'git-status.txt')
Get-ChildItem -LiteralPath $destination -File -Recurse | ForEach-Object {
    [pscustomobject]@{
        RelativePath = $_.FullName.Substring($destination.Length + 1)
        Bytes = $_.Length
        SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }
} | Export-Csv -NoTypeInformation -Path (Join-Path $destination 'checksums.csv')
Write-Output "Verified project archive: $fileCount files, each SHA-256 matched."
Write-Output "Recovery folder: $destination"
Write-Output 'Chat records are raw storage snapshots, not supported chat exports. Copy this recovery folder off-machine.'