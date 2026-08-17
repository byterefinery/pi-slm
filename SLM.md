# SLM Extension Specification

Reliability extension for Small Language Models.

## Feature 1:
  - On new session, insert first two synthetic (compact in terms of tokens) messages.
  - First message is with `Available skills:` as valid YAML, and should have name (single line text), description (single line text), list of reference paths for reference files as absolute paths, and list of scripts with  script files as absolute paths. If model supports reasoning insert both short synthetic reasoning.
  - Second message is `Available tools:` as valid YAML.
  - After these two synthetic messages are inserted insert user message/request.
  - Reason why we do this is that small language models forget from system message mentioned skills and tools, so we remind them what they can use.

---

## Feature 1: Available skills and tools

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
