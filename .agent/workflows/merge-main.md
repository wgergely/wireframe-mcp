---
description: Merge the main branch into the current feature branch
---

// turbo
1. Check if on a feature branch and no uncommitted changes:
   ```powershell
   $branch = git rev-parse --abbrev-ref HEAD
   if ($branch -notmatch '^feature-') { throw "Not on a feature branch: $branch" }
   $status = git status --porcelain
   if ($status) { throw "Uncommitted changes detected. Please commit or stash before merging." }
   ```

2. Merge the main branch:
   ```powershell
   git merge main
   ```

3. If there are collisions, attempt to resolve them.
   > [!NOTE]
   > Only notify the user if collision resolution results in feature loss or if manual intervention is strictly required.

4. Complete the merge by running the `ruff-and-push` workflow:
   ```powershell
   /ruff-and-push
   ```
