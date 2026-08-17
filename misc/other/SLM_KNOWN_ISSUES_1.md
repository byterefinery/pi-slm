# SLM Extension Specification

Known issues of reliability of small language models and known feature list how to improve it:

## Feature List

- Blocks destructive actions (`write` tool interception; also perform `bash` bypass interception because models can figure out that `write` was blocked, so they try to use `bash` as workaround). Use `edit` tool instead of `write`, and force `edit` to use correct `path`, `edits[].oldText` and `edits[].newText`. `edits[].oldText` is the exact text for one targeted replacement.
- Intercept `edit` and check if all `edits[].oldText` exist. Keep one that exist in file with `path`, remove which don't exist. If non exist remove last message that caused `edit` tool call in the first place.
- On new session, insert first two synthetic (compact in terms of tokens) messages with `Available skills:` and `Available tools:` as YAML, with name, description, absolute paths, etc. If model supports reasoning insert both short synthetic reasoning.

  Example of available skills YAML:
  ```
  Available skills:
  - name: git
    description: Git version control. Use when the user mentions git, commits, branches, pushing, pulling, merging, rebasing, stashing, worktrees, submodules, or any version control task. Covers straightforward workflows (add/commit/push) and advanced topics.
  - name: webfetch
    description: Fetches web pages as LLM-ready markdown. Use when the user wants to fetch, scrape, download, retrieve, grab, pull, or access any URL or webpage content. Defaults to Safari impersonation and AI-targeted markdown output optimized for LLM consumption. Falls back through browser → requests if needed. Supports --html, --file, --tool, --impersonate, --no-ai-targeted. Use this whenever the user asks to read a website, get page content, or fetch a URL.
  ...
  ```
  
  Example of available tools YAML:
  
  ```
  Available tools:
  - name: read
    description: Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.
    parameters:
      required:
        - path
      properties:
        path:
          description: Path to the file to read (relative or absolute)
        offset:
          type: number
          description: Line number to start reading from (1-indexed)
        limit:
          type: number
          description: Maximum number of lines to read
    strict: false
  ...
  ```
  
  Default tool property type is `type: string`, so only use `type: ...` where it differs like `type: number`.
  Default tool is `strict: false`, so only include `strict: true` where otherwise stated.

- When specific skill is used (skill loading), if skill was not used before, add synthetic hints as YAML (include reasoning content) of that skill like name, description, reference/script files (absolute paths). This way model can learn structure of skill, but loading will anyway happen on the next message whatever user typed.

  Example of a skill:
  ```
  - name: git
    description: Git version control. Use when the user mentions git, commits, branches, pushing, pulling, merging, rebasing, stashing, worktrees, submodules, or any version control task. Covers straightforward workflows (add/commit/push) and advanced topics.
    references:
      - 01-command-reference.md
      - 02-worktrees.md
      - ...
    scripts:
      - script1.sh
      - script_2.py
      - ...
  ```
  
  If not references are used in skill do not output it in YAML.
  If not scripts are used in skill do not output it in YAML.

- Tool `bash` should always have `timeout` argument set.
- Tool invocation error, find tool in list of available tools that failed, inject synthetic hints (include reasoning content) when tool call fails, so model can recall what was tool definition, so agent can retry on next iteration/message. This way it learns how to recover from tool invocation errors.

  Example of a tool:
  
  ```
  - name: read
    description: Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.
    parameters:
      required:
        - path
      properties:
        path:
          description: Path to the file to read (relative or absolute)
        offset:
          type: number
          description: Line number to start reading from (1-indexed)
        limit:
          type: number
          description: Maximum number of lines to read
    strict: false
  ...
  ```
  
  Default tool property type is `type: string`, so only use `type: ...` where it differs like `type: number`.
  Default tool is `strict: false`, so only include `strict: true` where otherwise stated.

- Skill invocation error, find skill used, inject synthetic hints (include reasoning content) for that skill how it should be used, so model learns how to recover from similar errors.

  Example of a skill:
  ```
  - name: git
    description: Git version control. Use when the user mentions git, commits, branches, pushing, pulling, merging, rebasing, stashing, worktrees, submodules, or any version control task. Covers straightforward workflows (add/commit/push) and advanced topics.
    references:
      - 01-command-reference.md
      - 02-worktrees.md
      - ...
    scripts:
      - script1.sh
      - script_2.py
      - ...
  ```
  
  If not references are used in skill do not output it in YAML.
  If not scripts are used in skill do not output it in YAML.

- Convert `read` errors on directories into directory listings. `EISDIR` to Directory Listing.
- Interrupted reasoning content needs to close \`\`\` code blocks properly by altering reasoning content of message.

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
