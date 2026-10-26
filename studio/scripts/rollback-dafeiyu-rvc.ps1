[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^backup/pre-dafeiyu-rvc-\d{8}-\d{6}$')]
    [string]$BackupBranch
)

$ErrorActionPreference = "Stop"

$repoRoot = (& git rev-parse --show-toplevel 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Run this script from inside the Unsloth Git checkout."
}
Push-Location -LiteralPath $repoRoot.Trim()
try {
    $status = (& git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw "Could not read Git working-tree status." }
    if ($status) {
        throw "Working tree is not clean. Save or review those changes, then run rollback again."
    }

    & git show-ref --verify --quiet "refs/heads/$BackupBranch"
    if ($LASTEXITCODE -ne 0) { throw "Backup branch $BackupBranch does not exist." }

    & git switch $BackupBranch
    if ($LASTEXITCODE -ne 0) { throw "Could not switch to $BackupBranch." }
    Write-Output "Now on $BackupBranch. Restart Unsloth Studio manually if needed."
}
finally {
    Pop-Location
}
