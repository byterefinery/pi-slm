# SLM Extension Specification

Known issues of reliability of small language models (<14B params) and known feature list how to improve it:

## Feature List

- Blocks destructive actions (`write` tool interception, `bash` bypass interception) - Use `edit` tool instead of `write`, and force `edit` to use correct `path`, `edits[].oldText` and `edits[].newText`. `edits[].oldText` is the exact text for one targeted replacement.
- High (50%) `edit` failure rate, force/pressure model not to make. Failed edit repeat up to `PI_EDIT_THRESHOLD` times (min 2). After N times, if it keeps failing, report that tool call has failed and change strategy. Error is usually in `path` or most likely `edits[].oldText` argument. `edits[].oldText` is the exact text for one targeted replacement.
- Skills listing. Replace hallucinated skills listings with factual ones. Detect via text (`available skill`, `installed skill`, `list of skill`, etc.) + list structure. Replace with factual data as a simple markdown list. Try to us LLM to recognize if it was asked for list of tools if unsure (non-English prompt).
- Tools listing. Replace hallucinated skills listings with factual ones. Detect via header (`available tool`, `i have access to`, `tools available`, etc.) + list/table structure. Try to us LLM to recognize if it was asked for list of tools if unsure (non-English prompt).
- Skill invocation.
- Skill loading, references and scripts. References paths need to be converted to absolute paths and loaded on demand. Script paths are also absolute needs to be executed relative to working directory.
- Tool invocation. Inject plain-language hints when tool calls fail parameter validation.
- Detects loops with `PI_LOOP_THRESHOLD` (min 2)
- Corrects hallucinations
- Helps models recover from errors
- Convert `read` errors on directories into directory listings. `EISDIR` to Directory Listing.
- If `read` tool call of the same file path (often the same offset/limit) happens without any modifications in between, remove all unnecessary `read` tool calls/results, and just keep the beginning one (first in the loop) because there were not changes afterwards. And try something else. You can put synthetic reasoning content and synthetic content to steer model into new direction to explore.
- If you detect loop of the same series of messages, remove later ones, and keep initial messages just before looping started. And try something else. You can put synthetic reasoning content and synthetic content to steer model into new direction to explore.

- `edit` intercept and check if all `edits[].oldText` exist. keep one that exist, remove which don't exist. if non exist remove last message that caused `edit` tool call in the first place.
- on new session, insert first two synthetic (compact in terms of tokens) messages with `Available skills:` and `Available tools:` as YAML, with name, description, absolute paths, etc.

---

Detects loops in Tool invocation. Varied messages, randomly picked on each breach:

```
Warning: Doom looping detected, let's try another approach.
Warning: Heads up, you're stuck in a doom loop. Try something different.
Warning: Repeated doom loop. Pivot to a new strategy.
Warning: Tou keep doom looping. Explore a different path.
Warning: Spinning in circles. Try a fresh angle.
Warning: Repeated doom looping. Shift your approach.
Warning: Loop detected. Move on.
Warning: You've looped and repeated messages. Consider the next step.
Warning: Looping on the same messages. Let's try another route.
```

---

## Blocks destructive actions

### Problem
...

### Solution
...

### Validation
...

### Notes
...


---

...
