---
description: Create a new feature branch and add it as a git worktree
---

1. Ensure you are in the `main` branch worktree.
2. Create and add a new worktree for the feature:
   ```bash
   git worktree add -b feature-{name} ../feature-{name}
   ```
3. Initialize the remote tracking branch:
   ```bash
   cd ../feature-{name}
   git push -u origin feature-{name}
   ```
