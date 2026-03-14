"""Launch options parsing prompt for LLM."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a parser for Steam game launch options strings.
You break down launch option strings into structured components: \
environment variables, wrapper tools, game arguments, and unparsed remainder.
Return valid JSON only, no explanations."""


def format_single_prompt(raw_launch: str) -> str:
    """Format prompt for a single launch options string."""
    return f"""\
Parse this Steam launch options string into structured components.
Return JSON only.

Input: "{raw_launch}"

{{
  "env_vars": [{{"name": "KEY", "value": "VALUE"}}],
  "wrappers": [{{"tool": "gamescope|mangohud|gamemoderun|prime-run|other", "args": "..."}}],
  "game_args": ["-dx11", "-skipintro"],
  "unparsed": "anything that doesn't fit above"
}}

Rules:
- Everything before %command% that matches KEY=VALUE is an env var.
- Known wrapper tools: gamescope, mangohud, gamemoderun, prime-run, taskset, obs-gamecapture.
- Everything after %command% (or after -- for gamescope) is a game argument.
- If no %command%, infer structure from known patterns.
- Wrappers have their own args (gamescope -W 1920 -H 1080 -f) -- separate from game_args.
- Empty arrays for missing categories, empty string for unparsed if all parsed.
"""


def format_batch_prompt(raw_launches: list[str]) -> str:
    """Format prompt for a batch of launch options strings."""
    numbered = "\n".join(f'{i+1}. "{lo}"' for i, lo in enumerate(raw_launches))
    return f"""\
Parse these Steam launch options strings into structured components.
Return a JSON array with one result per input, in the same order.

Inputs:
{numbered}

For each input, return:
{{
  "env_vars": [{{"name": "KEY", "value": "VALUE"}}],
  "wrappers": [{{"tool": "gamescope|mangohud|gamemoderun|prime-run|other", "args": "..."}}],
  "game_args": ["-dx11", "-skipintro"],
  "unparsed": "anything that doesn't fit above"
}}

Return format: {{"results": [...]}}

Rules:
- Everything before %command% that matches KEY=VALUE is an env var.
- Known wrapper tools: gamescope, mangohud, gamemoderun, prime-run, taskset, obs-gamecapture.
- Everything after %command% (or after -- for gamescope) is a game argument.
- If no %command%, infer structure from known patterns.
- Wrappers have their own args (gamescope -W 1920 -H 1080 -f) -- separate from game_args.
- Empty arrays for missing categories, empty string for unparsed if all parsed.
"""
