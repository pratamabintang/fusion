# 1. Adopt Matt Pocock Skills & Engineering Agent Architecture

Date: 2026-08-28

## Status

Accepted

## Context

We need a disciplined, composable, and reproducible software engineering process for AI agents (Antigravity, Claude Code, OpenAI Codex, etc.) working on this repository to avoid unstructured "vibe coding" and maintain high code quality over time.

## Decision

We adopt the engineering and productivity skills ecosystem from [mattpocock/skills](https://github.com/mattpocock/skills):

1. **Skills Placement**: Skills are mounted under `.agents/skills/` and `skills/` for seamless discovery across Antigravity, Claude Code, and other agent platforms.
2. **Issue Tracking**: Local markdown tickets under `.scratch/<feature>/issues/<NN>-<slug>.md` as default tracker, with GitHub issues available when remote is attached.
3. **Core Development Lifecycle**:
   - `/grill-with-docs` for sharpening ideas and updating glossary/ADRs upfront.
   - `/to-spec` & `/to-tickets` for breaking complex tasks into isolated, testable tickets.
   - `/implement` driving `/tdd` for vertical-slice implementation and `/code-review` before committing.
   - `/diagnosing-bugs` for disciplined hypothesis-and-test defect resolution.
   - `/wayfinder` for large multi-session efforts.
4. **Domain Documentation**: Single-context model using root `CONTEXT.md` for ubiquitous language and `docs/adr/` for immutable architectural decisions.

## Consequences

- The codebase maintains high test coverage through mandatory TDD cycles.
- Misunderstandings are eliminated early via interactive grilling sessions.
- Documentation, architecture decisions, and code remain synchronized automatically.
