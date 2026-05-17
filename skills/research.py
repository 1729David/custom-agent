NAME = "research"
DESCRIPTION = "Deep-research a topic: multi-source search, fetch, synthesize, save"

SYSTEM_PROMPT_ADDITION = (
    "You are in deep-research mode. For the given topic: "
    "(1) search at least 3 distinct sources with web_search, "
    "(2) fetch full page content for the most relevant results, "
    "(3) cross-reference findings and note contradictions, "
    "(4) produce a structured report with Executive Summary, Key Findings, and Sources. "
    "Save important findings to memory."
)

WORKFLOW_PROMPT = (
    "Conduct a comprehensive research report on:\n\n{query}\n\n"
    "Follow the deep-research workflow: search multiple sources, fetch pages, "
    "synthesize findings, produce a structured report."
)

# No new tools — reuses existing web_search, fetch_webpage, save_memory
TOOL_FUNCTIONS = {}
TOOL_DEFINITIONS = []
