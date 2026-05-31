#!/bin/bash
# Push to both GitHub and Hugging Face Spaces.
# HF rejects binary images, so we sync only backend files on top of HF's own history.
set -e

echo "→ Pushing to GitHub..."
git push origin main

echo "→ Syncing backend to Hugging Face..."
git fetch hf --quiet

git checkout -b _hf_tmp hf/main --quiet 2>/dev/null

# Copy only the files HF needs (no images)
git checkout main -- \
  agents/ \
  knowledge/ \
  app.py \
  requirements.txt \
  Dockerfile \
  agent-chat.html \
  2>/dev/null || true

git add -A

if git diff --cached --quiet; then
  echo "   HF already up to date."
else
  MSG=$(git log origin/main -1 --format="%s")
  git commit -m "hf-sync: $MSG" --quiet
  git push hf _hf_tmp:main --quiet
  echo "   HF updated."
fi

git checkout main --quiet
git branch -D _hf_tmp --quiet

echo "✓ Done — both remotes updated."
