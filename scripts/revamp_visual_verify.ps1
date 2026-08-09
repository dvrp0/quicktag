[CmdletBinding()]
param(
    [ValidateSet("Capture", "Verify")]
    [string] $Mode = "Verify",

    [string] $Packages = "D:\SteamLibrary\steamapps\common\Marathon\packages",

    [string] $BaselineRoot = "",

    [string[]] $Passes = @(
        "final", "base-color", "diffuse", "ao", "specular", "hdr", "normal",
        "mrt-albedo", "mrt-normal", "mrt-properties", "mrt-emissive", "mrt-flags"
    )
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($BaselineRoot)) {
    $BaselineRoot = Join-Path $repoRoot "tests\visual\baselines\80A9FF17"
}
$baselineRoot = [System.IO.Path]::GetFullPath($BaselineRoot)
$outputRoot = Join-Path $repoRoot "target\quicktag-model-probe"
$case = "revamp-br33-vibrant-sport-deluxe"
$test = "gui::model_renderer::tests::renders_weapon_mod_skin_visual_comparisons"

if (-not (Test-Path -LiteralPath $Packages -PathType Container)) {
    throw "Marathon package directory missing: $Packages"
}

if ($Mode -eq "Capture") {
    New-Item -ItemType Directory -Force -Path $baselineRoot | Out-Null
} elseif (-not (Test-Path -LiteralPath $baselineRoot -PathType Container)) {
    throw "Baseline directory missing: $baselineRoot. Run with -Mode Capture first."
}

Push-Location $repoRoot
try {
    $env:QUICKTAG_MARATHON_PACKAGES = $Packages
    $env:QUICKTAG_MODEL_PROBE_CASE = $case
    $env:QUICKTAG_PROBE_FIDELITY = "strict"
    $env:RUST_TEST_THREADS = "1"

    cargo test --no-default-features --features wordlist $test --no-run
    if ($LASTEXITCODE -ne 0) {
        throw "Visual probe build failed: exit=$LASTEXITCODE"
    }
    $testExe = Get-ChildItem -LiteralPath (Join-Path $repoRoot "target\debug\deps") -Filter "quicktag-*.exe" |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $testExe) {
        throw "Quicktag test executable not found"
    }

    foreach ($pass in $Passes) {
        $env:QUICKTAG_PROBE_PASS = $pass
        $fileName = if ($pass -eq "final") { "$case.png" } else { "$case-$pass.png" }
        $baselinePath = Join-Path $baselineRoot $fileName

        if ($Mode -eq "Verify") {
            if (-not (Test-Path -LiteralPath $baselinePath -PathType Leaf)) {
                throw "Missing baseline: $baselinePath"
            }
            $env:QUICKTAG_PROBE_BASELINE = $baselinePath
        } else {
            Remove-Item Env:\QUICKTAG_PROBE_BASELINE -ErrorAction SilentlyContinue
        }

        Write-Host "[$Mode] 80A9FF17 / $pass"
        & $testExe.FullName $test --ignored --exact --nocapture
        if ($LASTEXITCODE -ne 0) {
            throw "Visual probe failed: pass=$pass exit=$LASTEXITCODE"
        }

        $outputPath = Join-Path $outputRoot $fileName
        $captureName = "$case-$pass.capture.json"
        $capturePath = Join-Path $outputRoot $captureName
        if (-not (Test-Path -LiteralPath $capturePath -PathType Leaf)) {
            throw "Missing capture metadata: $capturePath"
        }
        if ($Mode -eq "Capture") {
            Copy-Item -LiteralPath $outputPath -Destination $baselinePath -Force
            Copy-Item -LiteralPath $capturePath -Destination (Join-Path $baselineRoot $captureName) -Force
        }
    }

    if ($Mode -eq "Capture") {
        $parentCommit = (git rev-parse HEAD).Trim()
        $worktreeDirty = [bool](git status --porcelain)
        $packageFiles = Get-ChildItem -LiteralPath $Packages -File
        $latestPackage = $packageFiles | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
        $manifest = [ordered]@{
            schema = 2
            renderer_schema_version = 2
            adapter_version = "goliath-render-v1"
            asset = "80A9FF17"
            case = $case
            owner = "80A7AA89"
            attachments = @(
                [ordered]@{ name = "Cold Vigilance Scope"; tag = "80A60FED"; rarity = "Deluxe" },
                [ordered]@{ name = "Impulse Brake"; tag = "80A60608"; rarity = "Deluxe" }
            )
            game = "Marathon"
            package_variant = "goliath"
            output_size = @(1024, 640)
            yaw_degrees = -24.0
            pitch_degrees = 14.0
            scale = 3.1
            passes = $Passes
            quicktag_parent_commit = $parentCommit
            quicktag_worktree_dirty = $worktreeDirty
            captured_utc = [DateTime]::UtcNow.ToString("o")
            package_path = (Resolve-Path -LiteralPath $Packages).Path
            package_file_count = $packageFiles.Count
            package_latest_file = if ($null -ne $latestPackage) { $latestPackage.Name } else { $null }
            package_latest_write_utc = if ($null -ne $latestPackage) { $latestPackage.LastWriteTimeUtc.ToString("o") } else { $null }
            cargo_features = @("wordlist")
            xg_enabled = $false
            fidelity_mode = "StrictTiger"
            capture_metadata = @($Passes | ForEach-Object { "$case-$_.capture.json" })
        }
        $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $baselineRoot "manifest.json") -Encoding utf8
    }
} finally {
    Remove-Item Env:\QUICKTAG_PROBE_BASELINE -ErrorAction SilentlyContinue
    Remove-Item Env:\QUICKTAG_PROBE_PASS -ErrorAction SilentlyContinue
    Remove-Item Env:\QUICKTAG_MODEL_PROBE_CASE -ErrorAction SilentlyContinue
    Remove-Item Env:\QUICKTAG_MARATHON_PACKAGES -ErrorAction SilentlyContinue
    Remove-Item Env:\QUICKTAG_PROBE_FIDELITY -ErrorAction SilentlyContinue
    Pop-Location
}

Write-Host "$Mode complete. Baselines: $baselineRoot; reports: $outputRoot"
