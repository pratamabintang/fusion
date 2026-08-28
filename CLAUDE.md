# CLAUDE: Fusion Project

See [AGENTS.md](AGENTS.md) for full project conventions, domain model, and agent workflows.

## Agent skills

### Issue tracker
Issues and specs live as markdown files under `.scratch/<feature>/` (and GitHub Issues when remote is active). See `docs/agents/issue-tracker.md`.

### Triage labels
Standard 5-label triage vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs
Single-context layout using `CONTEXT.md` and `docs/adr/`. See `docs/agents/domain.md`.

## Key Workflows
- **Interview & Spec**: `/grill-with-docs` -> `/to-spec` -> `/to-tickets`
- **Building**: `/implement` -> `/tdd` -> `/code-review`
- **Debugging**: `/diagnosing-bugs`
- **Navigation**: `/ask-matt`, `/wayfinder`
