---
description: Format, lint, commit, and push changes
---

// turbo-all
1. Format the code:
   ```powershell
   ruff format .
   ```

2. Run lint checks and fix automatically:
   ```powershell
   ruff check --fix .
   ```

3. If there are remaining lint issues, fix them manually.

4. Stage all changes:
   ```powershell
   git add .
   ```

5. Commit changes (ensure you follow the project's commit message rules: `{type}: {description}`):
   ```powershell
   git commit
   ```

6. Push changes to the remote:
   ```powershell
   git push
   ```
