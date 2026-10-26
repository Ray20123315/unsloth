[CmdletBinding()]
param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$Remote = "origin",

    [Parameter()]
    [ValidatePattern('^feature/dafeiyu-rvc-unsloth$')]
    [string]$Branch = "feature/dafeiyu-rvc-unsloth"
)

$ErrorActionPreference = "Stop"

$repoRoot = (& git rev-parse --show-toplevel 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Run this script from inside the Unsloth Git checkout."
}
$repoRoot = $repoRoot.Trim()
Push-Location -LiteralPath $repoRoot
try {
    $status = (& git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw "Could not read Git working-tree status." }
    if ($status) {
        throw "Working tree is not clean. Save or review those changes, then run this script again."
    }

    $current = (& git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $current) {
        throw "Detached HEAD is not supported by this deployment script."
    }
    if ($current -eq $Branch) {
        throw "Already on $Branch; review the branch and restart Studio manually if needed."
    }

    $existing = (& git branch --list $Branch)
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect local branches." }
    if ($existing) {
        throw "Local branch $Branch already exists. Review it manually; the script will not overwrite it."
    }

    $refspec = "refs/heads/$Branch" + ":" + "refs/remotes/$Remote/$Branch"
    & git fetch --no-tags $Remote $refspec
    if ($LASTEXITCODE -ne 0) {
        throw "Could not fetch $Branch from remote $Remote."
    }

    $remoteRef = "refs/remotes/$Remote/$Branch"
    & git show-ref --verify --quiet $remoteRef
    if ($LASTEXITCODE -ne 0) {
        throw "Remote-tracking branch $remoteRef was not created."
    }

    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backup = "backup/pre-dafeiyu-rvc-$stamp"
    & git branch $backup
    if ($LASTEXITCODE -ne 0) { throw "Could not create backup branch $backup." }

    & git switch --track -c $Branch "$Remote/$Branch"
    if ($LASTEXITCODE -ne 0) {
        throw "Branch switch failed. Your original commit remains available as $backup."
    }

    Write-Output "Now on $Branch. Backup branch: $backup"
    Write-Output "Review the branch, then restart Unsloth Studio manually."
}
finally {
    Pop-Location
}
