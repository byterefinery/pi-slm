# Optimize for LFM2.5 2.6B

```bash
firejail --noprofile --quiet --deterministic-exit-code --whitelist=~/.pi --whitelist=$(pwd)
uv run optim.py
```
