# Usage of LiquidAI/LFM2.5-2.6B in pi coding agent

## Issue 1: pi System Message Template

### 1.1: Default pi System Message

```bash
pi --model llamacpp/LiquidAI/LFM2.5-2.6B --no-tools --tools 'read,write,edit,bash'
```

Request:

```json
{
  "model": "LiquidAI/LFM2.5-2.6B",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.\n\nAvailable tools:\n- read: Read file contents\n- write: Create or overwrite files\n- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call\n- bash: Execute bash commands (ls, grep, find, etc.)\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.\n\nGuidelines:\n- Use bash for file operations like ls, rg, find\n- Use read to examine files instead of cat or sed.\n- Use write only for new files or complete rewrites.\n- Use edit for precise changes (edits[].oldText must match exactly)\n- When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls\n- Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.\n- Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.\n- You can inspect PI_* environment variables for current model and session details.\n- Be concise in your responses\n- Show file paths clearly when working with files\n\nPi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):\n- Main documentation: /opt/pi-coding-agent/README.md\n- Additional docs: /opt/pi-coding-agent/docs\n- Examples: /opt/pi-coding-agent/examples (extensions, custom tools, SDK)\n- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory\n- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md), environment variables (docs/environment-variables.md)\n- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing\n- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n<project_instructions path=\"/home/mtasic/projects-b/pi-slm/AGENTS.md\">\n# AGENTS.md\n\nThis is a project for working with the `pi` coding agent harness internals, implementing and testing new features.\n\n</project_instructions>\n\n</project_context>\n\n\nThe following skills provide specialized instructions for specific tasks.\nUse the read tool to load a skill's file when the task matches its description.\nWhen a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.\n\n<available_skills>\n  <skill>\n    <name>git</name>\n    <description>Git version control. Use when the user mentions git, commits, branches, pushing, pulling, merging, rebasing, stashing, worktrees, submodules, or any version control task. Covers straightforward workflows (add/commit/push) and advanced topics.</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/git/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>skman</name>\n    <description>Introduces the Agent Skills System — a standardized, lightweight, open format for extending AI agent capabilities with specialized knowledge and workflows. Use for scaffolding, validating, and inspecting agent skills (SKILL.md files and other skills&apos; files and directories).</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/skman/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>webfetch</name>\n    <description>Fetches web pages as LLM-ready markdown. Use when the user wants to fetch, scrape, download, retrieve, grab, pull, or access any URL or webpage content. Defaults to Safari impersonation and AI-targeted markdown output optimized for LLM consumption. Falls back through browser → requests if needed. Supports --html, --file, --tool, --impersonate, --no-ai-targeted. Use this whenever the user asks to read a website, get page content, or fetch a URL.</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/webfetch/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>websearch</name>\n    <description>Searches DuckDuckGo and returns LLM-optimized markdown, JSON, or YAML. Use this skill whenever the user wants to search the web, look up information online, find URLs, do web search, research a topic, or needs current/web-based answers. Uses Safari TLS impersonation and AI-targeted sanitization. Output can be markdown (default), --json, --yaml.</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/websearch/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>find-skills</name>\n    <description>Helps users discover and install agent skills when they ask questions like &quot;how do I do X&quot;, &quot;find a skill for X&quot;, &quot;is there a skill that can...&quot;, or express interest in extending capabilities. This skill should be used when the user is looking for functionality that might exist as an installable skill.</description>\n    <location>/home/mtasic/.pi/agent/skills/find-skills/SKILL.md</location>\n  </skill>\n</available_skills>\nCurrent working directory: /home/mtasic/projects-b/pi-slm"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "hi"
        }
      ]
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  },
  "max_tokens": 49152,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read",
        "description": "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
        "parameters": {
          "type": "object",
          "required": [
            "path"
          ],
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file to read (relative or absolute)"
            },
            "offset": {
              "type": "number",
              "description": "Line number to start reading from (1-indexed)"
            },
            "limit": {
              "type": "number",
              "description": "Maximum number of lines to read"
            }
          }
        },
        "strict": false
      }
    },
    {
      "type": "function",
      "function": {
        "name": "write",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
        "parameters": {
          "type": "object",
          "required": [
            "path",
            "content"
          ],
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file to write (relative or absolute)"
            },
            "content": {
              "type": "string",
              "description": "Content to write to the file"
            }
          }
        },
        "strict": false
      }
    },
    {
      "type": "function",
      "function": {
        "name": "edit",
        "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
        "parameters": {
          "type": "object",
          "required": [
            "path",
            "edits"
          ],
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file to edit (relative or absolute)"
            },
            "edits": {
              "type": "array",
              "items": {
                "type": "object",
                "required": [
                  "oldText",
                  "newText"
                ],
                "properties": {
                  "oldText": {
                    "type": "string",
                    "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."
                  },
                  "newText": {
                    "type": "string",
                    "description": "Replacement text for this targeted edit."
                  }
                }
              },
              "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead."
            }
          }
        },
        "strict": false
      }
    },
    {
      "type": "function",
      "function": {
        "name": "bash",
        "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
        "parameters": {
          "type": "object",
          "required": [
            "command"
          ],
          "properties": {
            "command": {
              "type": "string",
              "description": "Bash command to execute"
            },
            "timeout": {
              "type": "number",
              "description": "Timeout in seconds (optional, no default timeout)"
            }
          }
        },
        "strict": false
      }
    }
  ],
  "reasoning_effort": "high",
  "temperature": 0.1,
  "top_k": 50,
  "repeat_penalty": 1.1
}
```

Rendered:

```
<|im_start|>system
You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- write: Create or overwrite files
- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call
- bash: Execute bash commands (ls, grep, find, etc.)

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
- Use bash for file operations like ls, rg, find
- Use read to examine files instead of cat or sed.
- Use write only for new files or complete rewrites.
- Use edit for precise changes (edits[].oldText must match exactly)
- When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls
- Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.
- Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.
- You can inspect PI_* environment variables for current model and session details.
- Be concise in your responses
- Show file paths clearly when working with files

Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):
- Main documentation: /opt/pi-coding-agent/README.md
- Additional docs: /opt/pi-coding-agent/docs
- Examples: /opt/pi-coding-agent/examples (extensions, custom tools, SDK)
- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md), environment variables (docs/environment-variables.md)
- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing
- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)

The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.

<available_skills>
  <skill>
    <name>pi-0-84-2</name>
    <description>Pi coding agent CLI (v0.84.2) — minimal, extensible terminal coding agent. Use when running the `pi` CLI — interactive TUI sessions, one-shot `pi -p` prompts, JSON/RPC modes, continuing or forking sessions, choosing models/providers/thinking levels, tool allowlists, installing pi packages, or configuring settings, context files, and environment variables. Covers install, authentication, and the full CLI flag reference.</description>
    <location>/home/mtasic/projects-b/pi-slm/.agents/skills-general/pi-0-84-2/SKILL.md</location>
  </skill>
</available_skills>
Current working directory: /home/mtasic/projects-b/pi-slm
List of tools: [{"type": "function", "function": {"name": "read", "description": "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.", "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string", "description": "Path to the file to read (relative or absolute)"}, "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"}, "limit": {"type": "number", "description": "Maximum number of lines to read"}}}, "strict": false}}, {"type": "function", "function": {"name": "write", "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.", "parameters": {"type": "object", "required": ["path", "content"], "properties": {"path": {"type": "string", "description": "Path to the file to write (relative or absolute)"}, "content": {"type": "string", "description": "Content to write to the file"}}}, "strict": false}}, {"type": "function", "function": {"name": "edit", "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.", "parameters": {"type": "object", "required": ["path", "edits"], "properties": {"path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"}, "edits": {"type": "array", "items": {"type": "object", "required": ["oldText", "newText"], "properties": {"oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."}, "newText": {"type": "string", "description": "Replacement text for this targeted edit."}}}, "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead."}}}, "strict": false}}, {"type": "function", "function": {"name": "bash", "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.", "parameters": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string", "description": "Bash command to execute"}, "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}}}, "strict": false}}]<|im_end|>
<|im_start|>user
hi<|im_end|>
```

### 1.2: Custom pi System Message

```bash
pi --model llamacpp/LiquidAI/LFM2.5-2.6B --no-tools --tools 'read,write,edit,bash' --system-prompt "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files."
```

Request:

```json
{
  "model": "LiquidAI/LFM2.5-2.6B",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n<project_instructions path=\"/home/mtasic/projects-b/pi-slm/AGENTS.md\">\n# AGENTS.md\n\nThis is a project for working with the `pi` coding agent harness internals, implementing and testing new features.\n\n</project_instructions>\n\n</project_context>\n\n\nThe following skills provide specialized instructions for specific tasks.\nUse the read tool to load a skill's file when the task matches its description.\nWhen a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.\n\n<available_skills>\n  <skill>\n    <name>git</name>\n    <description>Git version control. Use when the user mentions git, commits, branches, pushing, pulling, merging, rebasing, stashing, worktrees, submodules, or any version control task. Covers straightforward workflows (add/commit/push) and advanced topics.</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/git/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>skman</name>\n    <description>Introduces the Agent Skills System — a standardized, lightweight, open format for extending AI agent capabilities with specialized knowledge and workflows. Use for scaffolding, validating, and inspecting agent skills (SKILL.md files and other skills&apos; files and directories).</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/skman/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>webfetch</name>\n    <description>Fetches web pages as LLM-ready markdown. Use when the user wants to fetch, scrape, download, retrieve, grab, pull, or access any URL or webpage content. Defaults to Safari impersonation and AI-targeted markdown output optimized for LLM consumption. Falls back through browser → requests if needed. Supports --html, --file, --tool, --impersonate, --no-ai-targeted. Use this whenever the user asks to read a website, get page content, or fetch a URL.</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/webfetch/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>websearch</name>\n    <description>Searches DuckDuckGo and returns LLM-optimized markdown, JSON, or YAML. Use this skill whenever the user wants to search the web, look up information online, find URLs, do web search, research a topic, or needs current/web-based answers. Uses Safari TLS impersonation and AI-targeted sanitization. Output can be markdown (default), --json, --yaml.</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/websearch/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>find-skills</name>\n    <description>Helps users discover and install agent skills when they ask questions like &quot;how do I do X&quot;, &quot;find a skill for X&quot;, &quot;is there a skill that can...&quot;, or express interest in extending capabilities. This skill should be used when the user is looking for functionality that might exist as an installable skill.</description>\n    <location>/home/mtasic/.pi/agent/skills/find-skills/SKILL.md</location>\n  </skill>\n</available_skills>\nCurrent working directory: /home/mtasic/projects-b/pi-slm\n"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "hi"
        }
      ]
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  },
  "max_tokens": 49152,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read",
        "description": "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
        "parameters": {
          "type": "object",
          "required": [
            "path"
          ],
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file to read (relative or absolute)"
            },
            "offset": {
              "type": "number",
              "description": "Line number to start reading from (1-indexed)"
            },
            "limit": {
              "type": "number",
              "description": "Maximum number of lines to read"
            }
          }
        },
        "strict": false
      }
    },
    {
      "type": "function",
      "function": {
        "name": "write",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
        "parameters": {
          "type": "object",
          "required": [
            "path",
            "content"
          ],
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file to write (relative or absolute)"
            },
            "content": {
              "type": "string",
              "description": "Content to write to the file"
            }
          }
        },
        "strict": false
      }
    },
    {
      "type": "function",
      "function": {
        "name": "edit",
        "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
        "parameters": {
          "type": "object",
          "required": [
            "path",
            "edits"
          ],
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file to edit (relative or absolute)"
            },
            "edits": {
              "type": "array",
              "items": {
                "type": "object",
                "required": [
                  "oldText",
                  "newText"
                ],
                "properties": {
                  "oldText": {
                    "type": "string",
                    "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."
                  },
                  "newText": {
                    "type": "string",
                    "description": "Replacement text for this targeted edit."
                  }
                }
              },
              "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead."
            }
          }
        },
        "strict": false
      }
    },
    {
      "type": "function",
      "function": {
        "name": "bash",
        "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
        "parameters": {
          "type": "object",
          "required": [
            "command"
          ],
          "properties": {
            "command": {
              "type": "string",
              "description": "Bash command to execute"
            },
            "timeout": {
              "type": "number",
              "description": "Timeout in seconds (optional, no default timeout)"
            }
          }
        },
        "strict": false
      }
    }
  ],
  "reasoning_effort": "high",
  "temperature": 0.1,
  "top_k": 50,
  "repeat_penalty": 1.1
}
```

Rendered:

```
<|im_start|>system
You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

<project_context>

Project-specific instructions and guidelines:

<project_instructions path="/home/mtasic/projects-b/pi-slm/AGENTS.md">
# AGENTS.md

This is a project for working with the `pi` coding agent harness internals, implementing and testing new features.

</project_instructions>

</project_context>


The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.

<available_skills>
  <skill>
    <name>git</name>
    <description>Git version control. Use when the user mentions git, commits, branches, pushing, pulling, merging, rebasing, stashing, worktrees, submodules, or any version control task. Covers straightforward workflows (add/commit/push) and advanced topics.</description>
    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/git/SKILL.md</location>
  </skill>
  <skill>
    <name>skman</name>
    <description>Introduces the Agent Skills System — a standardized, lightweight, open format for extending AI agent capabilities with specialized knowledge and workflows. Use for scaffolding, validating, and inspecting agent skills (SKILL.md files and other skills&apos; files and directories).</description>
    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/skman/SKILL.md</location>
  </skill>
  <skill>
    <name>webfetch</name>
    <description>Fetches web pages as LLM-ready markdown. Use when the user wants to fetch, scrape, download, retrieve, grab, pull, or access any URL or webpage content. Defaults to Safari impersonation and AI-targeted markdown output optimized for LLM consumption. Falls back through browser → requests if needed. Supports --html, --file, --tool, --impersonate, --no-ai-targeted. Use this whenever the user asks to read a website, get page content, or fetch a URL.</description>
    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/webfetch/SKILL.md</location>
  </skill>
  <skill>
    <name>websearch</name>
    <description>Searches DuckDuckGo and returns LLM-optimized markdown, JSON, or YAML. Use this skill whenever the user wants to search the web, look up information online, find URLs, do web search, research a topic, or needs current/web-based answers. Uses Safari TLS impersonation and AI-targeted sanitization. Output can be markdown (default), --json, --yaml.</description>
    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/websearch/SKILL.md</location>
  </skill>
  <skill>
    <name>find-skills</name>
    <description>Helps users discover and install agent skills when they ask questions like &quot;how do I do X&quot;, &quot;find a skill for X&quot;, &quot;is there a skill that can...&quot;, or express interest in extending capabilities. This skill should be used when the user is looking for functionality that might exist as an installable skill.</description>
    <location>/home/mtasic/.pi/agent/skills/find-skills/SKILL.md</location>
  </skill>
</available_skills>
Current working directory: /home/mtasic/projects-b/pi-slm

List of tools: [{"type": "function", "function": {"name": "read", "description": "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.", "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string", "description": "Path to the file to read (relative or absolute)"}, "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"}, "limit": {"type": "number", "description": "Maximum number of lines to read"}}}, "strict": false}}, {"type": "function", "function": {"name": "write", "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.", "parameters": {"type": "object", "required": ["path", "content"], "properties": {"path": {"type": "string", "description": "Path to the file to write (relative or absolute)"}, "content": {"type": "string", "description": "Content to write to the file"}}}, "strict": false}}, {"type": "function", "function": {"name": "edit", "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.", "parameters": {"type": "object", "required": ["path", "edits"], "properties": {"path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"}, "edits": {"type": "array", "items": {"type": "object", "required": ["oldText", "newText"], "properties": {"oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."}, "newText": {"type": "string", "description": "Replacement text for this targeted edit."}}}, "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead."}}}, "strict": false}}, {"type": "function", "function": {"name": "bash", "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.", "parameters": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string", "description": "Bash command to execute"}, "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}}}, "strict": false}}]<|im_end|>
<|im_start|>user
hi<|im_end|>
```

### 1.3: Minimal custom pi System Message (no AGENTS.md context in system message)

```bash
pi --model llamacpp/LiquidAI/LFM2.5-2.6B --no-tools --tools 'read,write,edit,bash' --system-prompt "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files." --no-context-files
```

Request:

```json
{
  "model": "LiquidAI/LFM2.5-2.6B",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.\n\nThe following skills provide specialized instructions for specific tasks.\nUse the read tool to load a skill's file when the task matches its description.\nWhen a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.\n\n<available_skills>\n  <skill>\n    <name>git</name>\n    <description>Git version control. Use when the user mentions git, commits, branches, pushing, pulling, merging, rebasing, stashing, worktrees, submodules, or any version control task. Covers straightforward workflows (add/commit/push) and advanced topics.</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/git/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>skman</name>\n    <description>Introduces the Agent Skills System — a standardized, lightweight, open format for extending AI agent capabilities with specialized knowledge and workflows. Use for scaffolding, validating, and inspecting agent skills (SKILL.md files and other skills&apos; files and directories).</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/skman/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>webfetch</name>\n    <description>Fetches web pages as LLM-ready markdown. Use when the user wants to fetch, scrape, download, retrieve, grab, pull, or access any URL or webpage content. Defaults to Safari impersonation and AI-targeted markdown output optimized for LLM consumption. Falls back through browser → requests if needed. Supports --html, --file, --tool, --impersonate, --no-ai-targeted. Use this whenever the user asks to read a website, get page content, or fetch a URL.</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/webfetch/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>websearch</name>\n    <description>Searches DuckDuckGo and returns LLM-optimized markdown, JSON, or YAML. Use this skill whenever the user wants to search the web, look up information online, find URLs, do web search, research a topic, or needs current/web-based answers. Uses Safari TLS impersonation and AI-targeted sanitization. Output can be markdown (default), --json, --yaml.</description>\n    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/websearch/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>find-skills</name>\n    <description>Helps users discover and install agent skills when they ask questions like &quot;how do I do X&quot;, &quot;find a skill for X&quot;, &quot;is there a skill that can...&quot;, or express interest in extending capabilities. This skill should be used when the user is looking for functionality that might exist as an installable skill.</description>\n    <location>/home/mtasic/.pi/agent/skills/find-skills/SKILL.md</location>\n  </skill>\n</available_skills>\nCurrent working directory: /home/mtasic/projects-b/pi-slm\n"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "hi"
        }
      ]
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  },
  "max_tokens": 49152,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read",
        "description": "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
        "parameters": {
          "type": "object",
          "required": [
            "path"
          ],
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file to read (relative or absolute)"
            },
            "offset": {
              "type": "number",
              "description": "Line number to start reading from (1-indexed)"
            },
            "limit": {
              "type": "number",
              "description": "Maximum number of lines to read"
            }
          }
        },
        "strict": false
      }
    },
    {
      "type": "function",
      "function": {
        "name": "write",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
        "parameters": {
          "type": "object",
          "required": [
            "path",
            "content"
          ],
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file to write (relative or absolute)"
            },
            "content": {
              "type": "string",
              "description": "Content to write to the file"
            }
          }
        },
        "strict": false
      }
    },
    {
      "type": "function",
      "function": {
        "name": "edit",
        "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
        "parameters": {
          "type": "object",
          "required": [
            "path",
            "edits"
          ],
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file to edit (relative or absolute)"
            },
            "edits": {
              "type": "array",
              "items": {
                "type": "object",
                "required": [
                  "oldText",
                  "newText"
                ],
                "properties": {
                  "oldText": {
                    "type": "string",
                    "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."
                  },
                  "newText": {
                    "type": "string",
                    "description": "Replacement text for this targeted edit."
                  }
                }
              },
              "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead."
            }
          }
        },
        "strict": false
      }
    },
    {
      "type": "function",
      "function": {
        "name": "bash",
        "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
        "parameters": {
          "type": "object",
          "required": [
            "command"
          ],
          "properties": {
            "command": {
              "type": "string",
              "description": "Bash command to execute"
            },
            "timeout": {
              "type": "number",
              "description": "Timeout in seconds (optional, no default timeout)"
            }
          }
        },
        "strict": false
      }
    }
  ],
  "reasoning_effort": "high",
  "temperature": 0.1,
  "top_k": 50,
  "repeat_penalty": 1.1
}
```

Rendered:

```
<|im_start|>system
You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.

<available_skills>
  <skill>
    <name>git</name>
    <description>Git version control. Use when the user mentions git, commits, branches, pushing, pulling, merging, rebasing, stashing, worktrees, submodules, or any version control task. Covers straightforward workflows (add/commit/push) and advanced topics.</description>
    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/git/SKILL.md</location>
  </skill>
  <skill>
    <name>skman</name>
    <description>Introduces the Agent Skills System — a standardized, lightweight, open format for extending AI agent capabilities with specialized knowledge and workflows. Use for scaffolding, validating, and inspecting agent skills (SKILL.md files and other skills&apos; files and directories).</description>
    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/skman/SKILL.md</location>
  </skill>
  <skill>
    <name>webfetch</name>
    <description>Fetches web pages as LLM-ready markdown. Use when the user wants to fetch, scrape, download, retrieve, grab, pull, or access any URL or webpage content. Defaults to Safari impersonation and AI-targeted markdown output optimized for LLM consumption. Falls back through browser → requests if needed. Supports --html, --file, --tool, --impersonate, --no-ai-targeted. Use this whenever the user asks to read a website, get page content, or fetch a URL.</description>
    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/webfetch/SKILL.md</location>
  </skill>
  <skill>
    <name>websearch</name>
    <description>Searches DuckDuckGo and returns LLM-optimized markdown, JSON, or YAML. Use this skill whenever the user wants to search the web, look up information online, find URLs, do web search, research a topic, or needs current/web-based answers. Uses Safari TLS impersonation and AI-targeted sanitization. Output can be markdown (default), --json, --yaml.</description>
    <location>/home/mtasic/projects-b/pi-slm/.agents/skills/websearch/SKILL.md</location>
  </skill>
  <skill>
    <name>find-skills</name>
    <description>Helps users discover and install agent skills when they ask questions like &quot;how do I do X&quot;, &quot;find a skill for X&quot;, &quot;is there a skill that can...&quot;, or express interest in extending capabilities. This skill should be used when the user is looking for functionality that might exist as an installable skill.</description>
    <location>/home/mtasic/.pi/agent/skills/find-skills/SKILL.md</location>
  </skill>
</available_skills>
Current working directory: /home/mtasic/projects-b/pi-slm

List of tools: [{"type": "function", "function": {"name": "read", "description": "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.", "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string", "description": "Path to the file to read (relative or absolute)"}, "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"}, "limit": {"type": "number", "description": "Maximum number of lines to read"}}}, "strict": false}}, {"type": "function", "function": {"name": "write", "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.", "parameters": {"type": "object", "required": ["path", "content"], "properties": {"path": {"type": "string", "description": "Path to the file to write (relative or absolute)"}, "content": {"type": "string", "description": "Content to write to the file"}}}, "strict": false}}, {"type": "function", "function": {"name": "edit", "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.", "parameters": {"type": "object", "required": ["path", "edits"], "properties": {"path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"}, "edits": {"type": "array", "items": {"type": "object", "required": ["oldText", "newText"], "properties": {"oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."}, "newText": {"type": "string", "description": "Replacement text for this targeted edit."}}}, "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead."}}}, "strict": false}}, {"type": "function", "function": {"name": "bash", "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.", "parameters": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string", "description": "Bash command to execute"}, "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}}}, "strict": false}}]<|im_end|>
<|im_start|>user
hi<|im_end|>
```

### 1.4: Observation

- Default (1.1): the system message is the default preamble (identity line, one-line "Available tools" summary, bullet guidelines, pi-docs pointers) + `<available_skills>` block + cwd line + trailing `List of tools` JSON. Notably it contains AGENTS.md `<project_context>` block.
- `--system-prompt` (1.2) only replaces the default preamble — identity, tool guidelines, and the pi-docs block are gone. The harness-appended sections (AGENTS.md `<project_context>`, `<available_skills>`, cwd line, trailing `List of tools` JSON) are identical to 1.1.
- `--no-context-files` (1.3) strips only the AGENTS.md `<project_context>` block; skills, cwd, and the tools JSON remain. It does not make the message minimal.
- All three variants end in a raw JSON dump of the 4 tool schemas (~3k chars, ~1k tokens) that duplicates what the API `tools` parameter already provides — pure overhead for a 2.6B model.
- Even the "minimal" 1.3 is dominated by boilerplate: the custom prompt is one line, while skills metadata + the tools JSON make up the bulk of the system message.
