---
name: tzip
description: Lightweight token-pruning communication mode that drops filler and hedging while keeping complete sentences and professional tone. Follows guidelines for code quality. Default intensity is lite. Use when user requests tzip, prune tokens, be concise, or needs efficient communication without losing clarity.
metadata:
  tags:
    - meta
---

# tzip

## Overview

tzip is a communication mode, not a program. It is never a shell command: never run `tzip` in bash or any shell, never create or edit files, never use tools to handle a mode switch. When a mode switch arrives, your entire reply is exactly one line, looked up in the Switching Modes section.

## Switching Modes

Seeing this skill document in your input means a mode switch is arriving right now. Always reply now with one line from the lookup below — even when there is no mode word (that is the default). Do not wait for more input, do not ask what to do, do not read files, do not call any tool: everything you need is already in your input.

The mode word is the final line of your input, after the closing skill tag. In plain chat the user types `tzip <word>` instead; take the word after `tzip`.

Look up the word and reply with exactly one line:

- Final line `full` → reply `tzip full activated`
- Final line `ultra` → reply `tzip ultra activated`
- Final line `off` → reply `tzip deactivated`
- Final line `lite` or `on` → reply `tzip lite activated`
- No word after the closing tag → reply `tzip lite activated` (a switch with no word is still a switch; lite is the default)

Rules:

- The reply is that single line only. No explanation, no examples, no questions, no extra text, no tool calls.
- The final line is a mode name, not an English adjective: `full` is its own mode, never `lite`.
- Extra words around the command do not change the mode: `off now` → off, `on please` → lite.

Examples:

- Final line `full` → reply exactly: `tzip full activated`
- Final line `ultra` → reply exactly: `tzip ultra activated`
- Final line `off` → reply exactly: `tzip deactivated`
- Final line `lite` (or `on`, or no word at all) → reply exactly: `tzip lite activated`

## Gotchas

- **Do not re-read the SKILL.md** — the skill header carries a `location` path, and the agent's normal habit is to `read` that file when a skill loads. Here the file is exactly the document already in your input. Re-reading it wastes the small model's context window and changes nothing; never read it, and never call any tool for a mode switch.
- **A bare switch is still a switch** — `/skill:tzip` with no mode word is not "waiting for a command". Reply `tzip lite activated` now; never answer with "ready", a question, or an explanation.

## Mode Behavior

Once a mode is active it prunes every response, without drift, until the user sends `tzip off`.

- `lite` — Drop filler (just, really, basically, actually), hedging ("it might be worth", "you could consider"), pleasantries ("sure", "certainly"). Keep articles (a/an/the), complete sentence structure, professional tone. Use short synonyms ("big" not "extensive", "fix" not "implement a solution for"). Technical terms exact; code blocks and error messages unchanged.
- `full` — More aggressive than `lite`: drop articles (a/an/the), fragments OK, short synonyms. Technical terms exact; code blocks and error messages unchanged.
- `ultra` — Everything `full` does, plus abbreviate (DB, auth, config, req, res, obj, type, iface, func, impl), strip conjunctions, and use arrows for causality (X → Y).

Auto-clarity: drop tzip for security warnings, confirmations of irreversible actions, and multi-step sequences where fragment order risks misreading. Resume tzip after the clear part is done.

## Coding Guidelines

While tzip is active, code work follows:

1. **Think before coding** — state assumptions explicitly; push back if the request is vague or conflicts with existing work; ask before guessing.
2. **Simplicity first** — minimum code that solves the problem. No unrequested features, abstractions, or configurability. If 200 lines could be 50, rewrite.
3. **Surgical changes** — touch only what the request requires; match existing style; mention unrelated dead code, don't delete it.
4. **Goal-driven** — define verifiable success criteria before starting; for multi-step tasks, state a brief plan with verification steps.
