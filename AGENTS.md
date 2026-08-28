# AGENTS: Fusion Project Instructions

This file guides AI agents working in this repository. Follow the conventions, workflows, and domain rules defined here.

---

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature>/` (and GitHub Issues when remote is active). See [`docs/agents/issue-tracker.md`](file:///D:/fusion/docs/agents/issue-tracker.md).

### Triage labels

The standard 5-label triage vocabulary is used (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See [`docs/agents/triage-labels.md`](file:///D:/fusion/docs/agents/triage-labels.md).

### Domain docs

Single-context layout using [`CONTEXT.md`](file:///D:/fusion/CONTEXT.md) and [`docs/adr/`](file:///D:/fusion/docs/adr). See [`docs/agents/domain.md`](file:///D:/fusion/docs/agents/domain.md).

---

## Available Skills & Workflows

All skills from [Matt Pocock's Skills](https://github.com/mattpocock/skills) are installed under [`.agents/skills/`](file:///D:/fusion/.agents/skills):

### 1. Main Development Flow: Idea → Ship
1. **`/grill-with-docs`**: Run interactive grilling to clarify requirements, build domain model, and update [`CONTEXT.md`](file:///D:/fusion/CONTEXT.md) and [`docs/adr/`](file:///D:/fusion/docs/adr).
2. **`/to-spec`**: Synthesize the conversation into a comprehensive technical specification.
3. **`/to-tickets`**: Break down the spec into modular, isolated tickets with explicit blocker dependencies in `.scratch/<feature>/issues/`.
4. **`/implement`**: Implement tickets one at a time. Drives **`/tdd`** (red-green-refactor) and concludes with **`/code-review`** (Standards + Spec axes) before committing.

### 2. On-Ramps & Specialized Skills
- **`/ask-matt`**: Ask which skill or workflow fits your current situation.
- **`/diagnosing-bugs`**: Disciplined debugging loop (tight feedback loop → minimize → hypothesize → instrument → fix → regression test).
- **`/wayfinder`**: Navigate complex, multi-session initiatives by building a decision map.
- **`/triage`**: Move incoming requests through triage states.
- **`/improve-codebase-architecture`**: Survey codebase for architectural deepening opportunities.
- **`/prototype`**: Fast throwaway prototype to test UX or technical feasibility.
- **`/research`**: Background investigation with cited findings.
- **`/resolving-merge-conflicts`**: Conflict resolution by intent.
- **`/handoff`**: Compact and export context for multi-agent or session transitions.

---

## Core Engineering Rules

- **Strict TDD**: Write failing tests before writing implementation code. Run test suite to verify failure and success.
- **Ubiquitous Language**: Never invent terms; use the exact glossary in [`CONTEXT.md`](file:///D:/fusion/CONTEXT.md).
- **ADR Recording**: Any non-trivial architectural decision must be recorded as an ADR in [`docs/adr/`](file:///D:/fusion/docs/adr).
- **Clean Seams**: Maintain deep modules and test through public interfaces.
