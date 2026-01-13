---
trigger: always_on
---

- Ensure .claude/CLAUDE.md contains all md file imports found .agent/rules/*.md dir, e.g.:
    `@../.agent/rules/repo-rules.md`
- Project uses git worktrees using the following structure:
    {container}/main              <-- git branch main
    {container}/feature-{name1}   <-- git branch feature-{name1}
    {container}/feature-{name2}   <-- git branch feature-{name2}
    {container}/venv-{arch}       <-- python venv, `venv-win` on Windows
- Repository secrets are stored in the main branch's .env file