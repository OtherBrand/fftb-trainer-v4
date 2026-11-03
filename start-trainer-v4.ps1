# Launches a detached V4 training run (family ops doctrine: trainer runs
# NEVER live inside a Claude session — background tasks get reaped; runs
# belong in their own process with logs on disk).
#
# Usage from the repo root:  powershell -File start-trainer-v4.ps1 [-Hours 6] [-Population 0]
param(
    [double]$Hours = 6,
    [int]$Population = 0,          # 0 = auto-size to free memory (fraction 0.7)
    [string]$Dataset = "deep",     # deep = dedup'd dual-cache union (gate-comparable)
    [int]$Threads = 0              # 0 = all cores; N = cap rayon (good-citizen
                                   # mode when another family run saturates)
)

if ($Threads -gt 0) { $env:RAYON_NUM_THREADS = "$Threads" }

$ts = Get-Date -Format "yyyyMMdd-HHmm"
$outDir = "data\runs\$Dataset-$ts"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$exe = "target\release\fftb-trainer-v4.exe"
$argList = @("train", "--dataset", $Dataset, "--hours", $Hours, "--population", $Population,
             "--seed", "42", "--memory-fraction", "0.7", "--out", $outDir)

$proc = Start-Process -FilePath $exe -ArgumentList $argList `
    -RedirectStandardOutput "$outDir\run.log" `
    -RedirectStandardError "$outDir\run.err.log" `
    -WorkingDirectory (Get-Location) -WindowStyle Hidden -PassThru

"launched pid $($proc.Id): $exe $($argList -join ' ')"
"run dir: $outDir"
