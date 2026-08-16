# Providers & Models

Pi supports subscription providers (OAuth via `/login`) and API-key providers (env vars or `~/.pi/agent/auth.json`). Built-in model catalogs ship with pi; configured providers may refresh newer catalogs, cached in `~/.pi/agent/models-store.json` for offline use. Run `pi update --models` to force a refresh.

## Authentication

- **Subscription:** `/login` in interactive mode, select a provider. Tokens store in `~/.pi/agent/auth.json` (0600 perms) and auto-refresh when expired. `/logout` clears them.
- **API key:** `export ANTHROPIC_API_KEY=sk-ant-...` before launching, or `/login` → pick an API-key provider to store it in `auth.json`. Auth file credentials take priority over environment variables.

### Subscriptions

Claude Pro/Max, ChatGPT Plus/Pro (Codex), GitHub Copilot, xAI (Grok/X), OpenRouter (mints a user-controlled API key billed from credits), Radius. Notes:

- **Claude Pro/Max:** third-party harness usage draws from Anthropic "extra usage" (billed per token), not plan limits.
- **OpenRouter over SSH:** the browser can't reach the loopback callback — paste the final redirect URL or authorization code into the login prompt instead.
- **GitHub Copilot:** "model not supported" usually means enabling the model in VS Code (Copilot Chat → model selector → Enable).

### API keys (common)

| Provider | Env var | `auth.json` key |
|----------|---------|-----------------|
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic` |
| OpenAI | `OPENAI_API_KEY` | `openai` |
| Google Gemini | `GEMINI_API_KEY` | `google` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | `azure-openai-responses` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek` |
| xAI | `XAI_API_KEY` | `xai` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter` |
| Mistral | `MISTRAL_API_KEY` | `mistral` |
| Groq | `GROQ_API_KEY` | `groq` |
| Cerebras | `CEREBRAS_API_KEY` | `cerebras` |
| Hugging Face | `HF_TOKEN` | `huggingface` |
| Fireworks | `FIREWORKS_API_KEY` | `fireworks` |
| Together AI | `TOGETHER_API_KEY` | `together` |
| Baseten | `BASETEN_API_KEY` | `baseten` |
| Amazon Bedrock | `AWS_BEARER_TOKEN_BEDROCK` | `amazon-bedrock` |

Also supported: Ant Ling, NVIDIA NIM, Cloudflare AI Gateway/Workers AI, Vercel AI Gateway, ZAI (Global/China), OpenCode Zen/Go, Kimi For Coding, MiniMax, Qwen Token Plan, Xiaomi MiMo. Full list in upstream `docs/providers.md`.

`auth.json` shape:

```json
{
  "anthropic": { "type": "api_key", "key": "sk-ant-..." },
  "openai": { "type": "api_key", "key": "sk-..." }
}
```

## Selecting Models

- `--provider <name> --model <id>` — explicit
- `--model provider/id` — provider prefix, no `--provider` needed
- `--model <pattern>:<thinking>` — thinking-level shorthand (e.g. `sonnet:high`)
- `--list-models [search]` — browse the catalog
- Interactive: `/model` or Ctrl+L; Shift+Tab cycles thinking level; Ctrl+P cycles scoped models (`--models` / `/scoped-models`)
- `--thinking <level>`: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`
- `--api-key <key>` overrides env vars for one run

## Custom Providers & Models

- **Supported APIs** (OpenAI, Anthropic, Google): declare in `~/.pi/agent/models.json`
- **Custom APIs or OAuth:** write an extension (`pi.registerProvider()`)
- **llama.cpp:** `/login llama.cpp`, manage downloads/loaded models with `/llama`, pick with `/model`

See upstream `docs/models.md` and `docs/custom-provider.md` for schemas.
