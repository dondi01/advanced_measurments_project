# Git Workflow Guidelines

This document outlines best practices for using Git to maintain an orderly repository and minimize conflicts.

## Branching Strategy

1. **Main Branch**:
   - The `main` branch should always contain production-ready code.
   - Direct commits to `main` should be avoided. Use pull requests instead.

2. **Feature Branches**:
   - Create a new branch for each feature or bug fix.
   - Use descriptive names for branches, such as `feature/user-authentication` or `bugfix/login-error`.


## Pushing Changes

1. **Commit Messages**:
   - Write clear and concise commit messages.
   - Use the imperative mood, such as "Add user authentication" instead of "Added user authentication".

2. **Frequent Commits**:
   - Commit changes frequently to avoid large, unwieldy commits.
   - Use `git add` to stage changes and `git commit` to commit them with a meaningful message.
