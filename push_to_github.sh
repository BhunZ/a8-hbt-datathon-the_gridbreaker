#!/usr/bin/env bash
# =============================================================
# push_to_github.sh — Push a8-hbt Datathon project to GitHub
# Run from project root (Linux / Mac / WSL):
#   chmod +x push_to_github.sh
#   ./push_to_github.sh
# =============================================================

set -e

# --- Config (EDIT THESE) ---
REPO_URL="https://github.com/BhunZ/a8-hbt-datathon-the_gridbreaker.git"
BRANCH="main"
YOUR_NAME="BhunZ"                       # ← thay bằng GitHub username của bạn
YOUR_EMAIL="rich.nguyenduc@gmail.com"   # ← thay bằng email GitHub của bạn

# --- Sanity check ---
echo
echo "===== a8-hbt Datathon — Push to GitHub ====="
echo "Repo: $REPO_URL"
echo "Author: $YOUR_NAME <$YOUR_EMAIL>"
echo

if [ ! -d "src" ] || [ ! -d "submissions" ]; then
    echo "ERROR: Run from project root (must have src/ and submissions/)" >&2
    exit 1
fi

# --- 1. Init git if needed ---
if [ ! -d ".git" ]; then
    echo "[1/6] Initializing git..."
    git init -b "$BRANCH"
else
    echo "[1/6] Git already initialized (skip)."
fi

# --- 2. Configure user ---
echo "[2/6] Configuring git user..."
git config user.name  "$YOUR_NAME"
git config user.email "$YOUR_EMAIL"

# --- 3. Stage ---
echo "[3/6] Staging files (per .gitignore)..."
git add .
echo "      → $(git ls-files | wc -l) files staged"

# --- 4. Commit ---
echo "[4/6] Creating commit..."
if [ -z "$(git diff --cached --name-only)" ]; then
    echo "      → nothing to commit (working tree clean)"
else
    git commit -m "Initial commit — a8-hbt Datathon 2026 final submission

Team: Tung Lam Nguyen (VKU), Quoc Hung Le (VNU-HCM),
      Bao Hung Nguyen Duc (HCMOU), Thanh Dat Hoang Ngoc (UEH).

Final submission: submissions/submission_v10c_scaled_105.csv
Kaggle public MAE: 772,912 (best of 17 versions tried).

Includes: src/ pipeline, notebooks/, reports/ (NeurIPS PDF + business PDF),
docs/, plans/, all submissions, figures/.
Excludes: data/raw/ and data/processed/ (competition data, not redistributed)."
fi

# --- 5. Remote ---
echo "[5/6] Configuring remote 'origin'..."
if git remote | grep -q '^origin$'; then
    git remote set-url origin "$REPO_URL"
else
    git remote add origin "$REPO_URL"
fi
git remote -v

# --- 6. Push ---
echo
echo "[6/6] Pushing to GitHub..."
echo "      You'll be asked for GitHub username and password."
echo "      Use Personal Access Token as password (https://github.com/settings/tokens)."
echo
git push -u origin "$BRANCH"

echo
echo "===== DONE ====="
echo "View at: https://github.com/BhunZ/a8-hbt-datathon-the_gridbreaker"
