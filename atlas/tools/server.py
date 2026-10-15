"""Anthropic-hosted server tools.

CONCEPT: server tools are declared like any other tool but run on Anthropic's
infrastructure - there is no function for you to implement and no loop
iteration for you to drive. Results come back as content blocks in the same
response (`web_search_tool_result`, `bash_code_execution_tool_result`, ...).

Two design rules worth knowing:
* `web_search_20260209` / `web_fetch_20260209` already use code execution
  internally to filter results ("dynamic filtering"), so don't give the same
  agent a separate `code_execution` tool - two sandboxes confuse the model.
  Atlas gives web tools to the *researcher* and code execution to the *analyst*.
* A long server-tool turn can end with `stop_reason == "pause_turn"`; the agent
  loop simply re-sends to let it continue.
"""

WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search", "max_uses": 8}
WEB_FETCH = {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 8}
CODE_EXECUTION = {"type": "code_execution_20260521", "name": "code_execution"}
