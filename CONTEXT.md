# CONTEXT: Fusion

This document establishes the ubiquitous language, architectural principles, and core definitions for the Fusion project. All AI agents and human contributors should read and strictly adhere to the terms and constraints documented here.

---

## 1. Domain Vocabulary & Glossary

Use these precise terms consistently across code, tests, documentation, and issues:

- **Fusion**: The overarching project codebase and system.
- **Seam**: A clean boundary between components where tests or alternate implementations can be attached without reaching into private internals.
- **Deep Module**: A module that exposes a simple interface hiding substantial logic, validation, and domain rules behind it.
- **Tracer Bullet**: An end-to-end slice through all architectural layers that demonstrates viability before fleshing out full functionality.
- **Decision Ticket**: A unit of work designed to resolve architectural ambiguity, answer a question, or produce a specification before code implementation.

---

## 2. Engineering Principles

1. **Disciplined Workflow**: Idea/Task → `/grill-with-docs` → `/to-spec` → `/to-tickets` → `/implement` (with `/tdd` & `/code-review`).
2. **Test-Driven Development**: Work in strict Red-Green-Refactor cycles. Write the failing test asserting desired behavior, verify it fails, make it pass with minimal code, then refactor cleanly.
3. **Keep Context Clean**: Separate architectural design and specification phases from individual ticket implementations.
4. **Preserve Documentation Integrity**: Always update `CONTEXT.md` and record significant decisions as Architecture Decision Records in `docs/adr/`.
