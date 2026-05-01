# =============================================================
# push_to_github.ps1 - Push a8-hbt Datathon project to GitHub
# Run this from PowerShell in the project root: C:\Users\znigh\Datathon\Datathon
#
# Usage:
#   1. Open PowerShell
#   2. cd C:\Users\znigh\Datathon\Datathon
#   3. .\push_to_github.ps1
#
# You'll be prompted for GitHub credentials when pushing.
# Use a Personal Access Token (PAT) as password:
#   https://github.com/settings/tokens (scope: repo)
# =============================================================

$ErrorActionPreference = "Stop"

# --- Config (EDIT THESE) ---
$REPO_URL    = "https://github.com/BhunZ/a8-hbt-datathon-the_gridbreaker.git"
$BRANCH      = "main"
$YOUR_NAME   = "BhunZ"                       # <- thay bang GitHub username cua ban
$YOUR_EMAIL  = "nd.baohung3105@gmail.com"    # <- thay bang email GitHub cua ban

# --- 0. Sanity check ---
Write-Host ""
Write-Host "===== a8-hbt Datathon - Push to GitHub =====" -ForegroundColor Cyan
Write-Host "Repo: $REPO_URL"
Write-Host "Author: $YOUR_NAME ($YOUR_EMAIL)"
Write-Host ""

if (!(Test-Path "src") -or !(Test-Path "submissions")) {
    Write-Host "ERROR: Run this script from C:\Users\znigh\Datathon\Datathon" -ForegroundColor Red
    Write-Host "       (must be in the project root with src/ and submissions/)" -ForegroundColor Red
    exit 1
}

# --- 1. Init git if not already ---
if (!(Test-Path ".git")) {
    Write-Host "[1/6] Initializing git repository..." -ForegroundColor Yellow
    git init -b $BRANCH
} else {
    Write-Host "[1/6] Git already initialized (skip)." -ForegroundColor Gray
}

# --- 2. Configure user ---
Write-Host "[2/6] Configuring git user..." -ForegroundColor Yellow
git config user.name  $YOUR_NAME
git config user.email $YOUR_EMAIL

# --- 3. Stage all (respects .gitignore) ---
Write-Host "[3/6] Staging files (per .gitignore)..." -ForegroundColor Yellow
git add .
$tracked = (git ls-files | Measure-Object).Count
Write-Host "      -> $tracked files staged" -ForegroundColor Gray

# --- 4. Commit ---
Write-Host "[4/6] Creating commit..." -ForegroundColor Yellow
$commitMsg = "Initial commit - a8-hbt Datathon 2026 final submission`n" +
    "`nTeam: Tung Lam Nguyen (VKU), Quoc Hung Le (VNU-HCM),`n" +
    "      Bao Hung Nguyen Duc (HCMOU), Thanh Dat Hoang Ngoc (UEH).`n" +
    "`nFinal submission: submissions/submission_v10c_scaled_105.csv`n" +
    "Kaggle public MAE: 772,912`n" +
    "`nIncludes: src/ pipeline, notebooks/, reports/ (NeurIPS PDF + business PDF),`n" +
    "docs/, plans/, all submissions, figures/.`n" +
    "Excludes: data/raw/ and data/processed/ (competition data, not redistributed)."

# Use staged check to avoid re-committing
$staged = git diff --cached --name-only
if ($staged.Count -eq 0) {
    Write-Host "      -> nothing to commit (working tree clean)" -ForegroundColor Gray
} else {
    git commit -m $commitMsg
}

# --- 5. Add remote ---
Write-Host "[5/6] Configuring remote 'origin'..." -ForegroundColor Yellow
$existingRemote = git remote 2>$null
if ($existingRemote -contains "origin") {
    git remote set-url origin $REPO_URL
} else {
    git remote add origin $REPO_URL
}
git remote -v

# --- 6. Push ---
Write-Host ""
Write-Host "[6/6] Pushing to GitHub..." -ForegroundColor Yellow
Write-Host '      You will be asked for GitHub username and password.' -ForegroundColor Cyan
Write-Host '      Use a Personal Access Token as password (https://github.com/settings/tokens).' -ForegroundColor Cyan
Write-Host ""
git push -u origin $BRANCH
