# SLM Extension Specification

Reliability extension for small language models (<14B params):

## Feature List

- Blocks destructive actions (`write` tool interception, `bash` bypass interception) - Use `edit` tool instead of `write`, and force `edit` to use correct `path`, `oldText` and `newText`.
- Skills listing. Replace hallucinated skills listings with factual ones. Detect via text (`available skill`, `installed skill`, `list of skill`, etc.) + list structure. Replace with factual data as a simple markdown list.
- Tools listing. Replace hallucinated skills listings with factual ones. Detect via header (`available tool`, `i have access to`, `tools available`, etc.) + list/table structure.
- Skill invocation.
- Skill loading, references and scripts. References paths need to be converted to absolute paths and loaded on demand. Script paths are also absolute needs to be executed relative to working directory.
- Tool invocation. Inject plain-language hints when tool calls fail parameter validation.
- Detects loops (`PI_LOOP_THRESHOLD` min 2)
- Corrects hallucinations
- Helps models recover from errors
- Convert `read` errors on directories into directory listings. `EISDIR` to Directory Listing.

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
