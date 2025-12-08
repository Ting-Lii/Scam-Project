#!/bin/bash

# Simple script to merge from upstream with README.md protection
# This uses git merge with the merge strategy configured in .gitattributes

set -e

UPSTREAM_REMOTE="upstream"
UPSTREAM_BRANCH="main"
CURRENT_BRANCH=$(git branch --show-current)

echo "=== Simple Upstream Merge ==="
echo "Current branch: $CURRENT_BRANCH"
echo ""

# Fetch latest
echo "Fetching from upstream..."
git fetch $UPSTREAM_REMOTE

# Show what will be merged
echo ""
echo "Changes to be merged:"
git log --oneline HEAD..$UPSTREAM_REMOTE/$UPSTREAM_BRANCH | head -10
echo ""

# Show file changes
echo "Files changed:"
git diff --name-status HEAD $UPSTREAM_REMOTE/$UPSTREAM_BRANCH
echo ""

# Confirm merge
read -p "Proceed with merge? (README.md will be protected) (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Merge cancelled."
    exit 0
fi

# Perform merge (README.md will be automatically protected by .gitattributes)
echo "Merging..."
git merge --no-edit $UPSTREAM_REMOTE/$UPSTREAM_BRANCH

echo ""
echo "✅ Merge complete!"
echo "README.md has been protected (your local version kept)."
echo ""
echo "Review any modified files and commit if needed."

