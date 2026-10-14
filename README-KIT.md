# FlipCheck build kit

Copy these into your new project folder:

- CLAUDE.md                project rules Claude Code loads every session
- docs/SPEC.md             what to build
- docs/PLAN.md             build order and ready-to-paste prompts
- docs/PROGRESS.md         status tracker
- .claude/agents/          six project subagents
- .claude/settings.json    turns off Claude Code commit and PR attribution
- .githooks/commit-msg     strips AI attribution and blocks em dashes in commit messages

Then follow docs/PLAN.md from Phase 0.1. Start every Claude Code session with the prompt under "How to start a session".
