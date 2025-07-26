#!/bin/bash
# Setup git rerere for automated merge conflict resolution

set -e

echo "Setting up git rerere for autonomous backlog management..."

# Enable rerere globally
git config --global rerere.enabled true
git config --global rerere.autoupdate true

echo "✓ Enabled git rerere globally"

# Setup merge drivers for common file types
git config --global merge.theirs.name "Prefer incoming changes"
git config --global merge.theirs.driver "cp -f '%B' '%A'"

git config --global merge.union.name "Line union merge"
git config --global merge.union.driver "git merge-file -p %A %O %B > %A"

git config --global merge.lock.name "Lock file - no merge"
git config --global merge.lock.driver "echo 'Lock file conflict - manual resolution required' && false"

echo "✓ Configured merge drivers"

# Create .gitattributes for file-aware merging
cat > .gitattributes << 'EOF'
# Merge strategies for different file types

# Lock files - prefer incoming
package-lock.json merge=theirs
poetry.lock      merge=theirs
yarn.lock        merge=theirs
Pipfile.lock     merge=theirs

# Snapshots - prefer incoming  
*.snap           merge=theirs

# Documentation - union merge
*.md             merge=union
CHANGELOG.md     merge=union
README.md        merge=union

# Binary files - prevent merge
*.png            merge=lock
*.jpg            merge=lock
*.jpeg           merge=lock
*.gif            merge=lock
*.svg            merge=lock
*.zip            merge=lock
*.tar.gz         merge=lock

# Configuration - manual merge (default)
*.yml            
*.yaml           
*.json           
*.toml
EOF

echo "✓ Created .gitattributes with merge strategies"

# Create git hooks for automated rebase
mkdir -p .git/hooks

# Prepare commit message hook to enable rerere
cat > .git/hooks/prepare-commit-msg << 'EOF'
#!/bin/bash
# Enable rerere for this commit

git config rerere.enabled true
git config rerere.autoupdate true
EOF

# Pre-push hook for automatic rebase
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
# Automatic rebase before push

set -e

# Get the current branch
current_branch=$(git branch --show-current)

# Skip if on main branch
if [ "$current_branch" = "main" ]; then
    exit 0
fi

echo "Attempting automatic rebase onto main..."

# Fetch latest changes
git fetch origin main

# Attempt rebase
if git rebase origin/main; then
    echo "✓ Automatic rebase successful"
else
    echo "❌ Rebase conflicts detected - manual resolution required"
    echo "Run: git rebase --abort && git rebase origin/main"
    exit 1
fi
EOF

# Make hooks executable
chmod +x .git/hooks/prepare-commit-msg
chmod +x .git/hooks/pre-push

echo "✓ Created git hooks for automated rebase"

# Create rerere cache sharing directory
mkdir -p tools/rerere-cache

echo "✓ Created rerere cache directory"

# Test configuration
echo ""
echo "Testing git rerere configuration..."
if git config --get rerere.enabled | grep -q "true"; then
    echo "✓ Git rerere is properly configured"
else
    echo "❌ Git rerere configuration failed"
    exit 1
fi

echo ""
echo "🎉 Git rerere setup complete!"
echo ""
echo "The following merge strategies are now configured:"
echo "  • Lock files (package-lock.json, poetry.lock): prefer incoming"
echo "  • Documentation (*.md): union merge"  
echo "  • Binary files: prevent automatic merge"
echo "  • Configuration files: manual merge (default)"
echo ""
echo "Hooks installed:"
echo "  • prepare-commit-msg: enables rerere per commit"
echo "  • pre-push: automatic rebase onto main"
echo ""