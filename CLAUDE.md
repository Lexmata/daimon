# Project instructions — daimon

## Give the user a window before merging

Before merging a branch/PR (whether via subagent-driven-development, finishing-a-development-branch, or any other workflow), stop and give the user the chance to run `/simplify`, `/code-review`, `/optimize`, or similar audit commands themselves first — even if the branch already went through its own review/fix loop. Present the branch as ready (tests green, reviews passed) and ask whether to proceed with push/PR/merge, or wait while the user runs additional audits. Don't merge automatically just because internal review passed.
