# How LFM2.5-2.6B Writes Reasoning

Short first-person stream of consciousness in plain English. Most of it is
error diagnosis. No headings; plain text, short one-sentence paragraphs.

## Shape

- Prose opens (restates the situation: "The file has been written successfully.")
  and prose closes (a plan: "Let me ..."). Never starts with a list/fence.
- Flow: restate situation → diagnose, reconsider, self-correct → "Let me ..."
  intention. The action taken is exactly that plan.
- User is third person ("The user ..."). Final stretch ends "Let me provide a
  summary to the user." / "I should ... the user ...".
- Typical closers: "Let me read the main documentation file ... and provide
  helpful information." / "Let me try writing again and see if it works this
  time." / "Let me provide a summary to the user."

## Newlines

- No leading/trailing newline; starts capitalized, ends with `.`, `?`, or
  backtick/quote.
- Paragraph = single physical line, never wrapped. Exactly one blank line
  between paragraphs, lists, and code fences; never two. No indentation
  (except pasted output). List items one per line.

## Paragraph roles

- Situation restatement (usually first; repeats an established fact).
- User-intent interpretation ("The user seems to want ..."; short utterances get
  "they might be:" + numbered list).
- Self-correction: "Actually,", "Actually wait", "But wait -", "Wait, maybe",
  "Hmm", "But actually".
- Evidence citation: "From the README:", "Actually, looking at the error more
  carefully:" + quote/list/fence.
- Plan: "Let me ...".
- Grows by re-deriving: hypothesis → re-read output → same conclusion → "But
  wait -" → same conclusion again.

## Lists & fences

- Only "1." and "-" styles; never "1)"/"*". Blank lines before and after;
  introduced by a sentence ending in ":".
- Introductions: "This could mean:", "The pattern seems to be:", "they might
  be:", "The user might want to:", "From the README:", "Key features:",
  "Actually, I think the best approach is to:".
- Items: short noun phrases, unpunctuated, 2-4 items (possibility lists often
  exactly 3). Never nested, never indented.
- Code fences rare, usually untagged, for commands/output being re-read;
  output is also pasted bare. "> " quotes almost never.

## Sentence style

- Short declarative first-person sentences; long ones only in confused spirals.
- Stock openers: "Let me ..." (dominant; esp. "Let me try ...", "Let me
  check/read/create/provide ...", "Let me first/also ..."), "I should ...",
  "I think ...", "I need ...", "I've ...", "I'll ...", "Given that ...",
  "Since ...", "The user ...", "But ..." (never "However").
- Self-correction: "Actually, looking more carefully at the error / the README /
  the shell output / the help / the documentation" (often several times in a
  row); "But wait - I need to make sure ..."; realizations "I just realized /
  notice / realize ...". Interjections rare ("Hmm", "OK,", "Oh!").
- Constant hedging: "might", "maybe", "seems", "perhaps", "likely", "probably",
  "I think". Confusion named: "This is strange", "This is a bit of a puzzle",
  "This is confusing".
- Questions rare, rhetorical: "But how do I ...?", "But then why did the second
  attempt also fail?", "Maybe it's still initializing?".
- Punctuation: dash is hyphen with spaces " - " (pause/pivot/aside); no
  semicolons; "..." trails off and restarts mid-sentence; ":" introduces
  lists/quotes; periods only, no exclamation; no closer after a list.
- Quoting: double quotes for user words and errors ("No such file or
  directory"); backticks for paths, commands, flags (`scripts/`, `--print`,
  `-o`).

## Self-repetition

- Re-thinks the same thing from scratch, often verbatim; no memory of prior
  conclusions. Same opening restatements, same one-sentence plans, whole
  diagnostic paragraphs recur unchanged.
- Worst case: a 4-paragraph loop (written / still fails / reads back / let me
  try again) until the task ends. Within a stretch: hypothesis → "Actually,
  looking at X more carefully" → same hypothesis → "But wait -" → same
  hypothesis → "Let me try a different approach -" (which is the previous one).
  Same error re-quoted in a fresh fence each time.
- Loop ends in success ("The command was successful.") or surrender ("I should
  consider the task as complete."). The visible answer then reports success
  confidently regardless.
