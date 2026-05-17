import argparse

from agent import Agent
from memory import MemoryStore
from skills import load_skills
from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS, make_memory_tools

MODEL = "qwen3.5:latest"
EMBED_MODEL = "qwen3-embedding:latest"  # set to e.g. "nomic-embed-text" after: ollama pull nomic-embed-text

_BASE_SYSTEM_PROMPT = (
    "You are a research assistant with access to tools. "
    "Before using web_search, always call search_memories to check prior knowledge — "
    "only use web_search if memory returns no results or the information is insufficient. "
    "When you do research the web: search for relevant sources, fetch the most promising "
    "pages, and synthesize a clear, well-organized answer with key findings. "
    "Cite your sources (title + URL) at the end of your response. "
    "For local documents, use parse_pdf or read_file as appropriate. "
    "Use save_memory to store important findings for future sessions."
)


def main():
    parser = argparse.ArgumentParser(description="Ollama research agent")
    parser.add_argument("-m", "--model", default=MODEL, help="Ollama model to use (default: %(default)s)")
    parser.add_argument("-e", "--embed-model", default=EMBED_MODEL, help="Ollama embedding model (default: %(default)s)")
    args = parser.parse_args()

    store = MemoryStore(embed_model=args.embed_model)

    recent = store.recent(limit=5)
    if recent:
        lines = [f"- [{r['created_at']}] {r['content']}" for r in recent]
        memory_preamble = "\n\nRecent memories from past sessions (use search_memories for more):\n" + "\n".join(lines)
    else:
        memory_preamble = ""

    system_prompt = _BASE_SYSTEM_PROMPT + memory_preamble

    all_tool_functions = {**TOOL_FUNCTIONS, **make_memory_tools(store)}

    agent = Agent(
        model=args.model,
        tool_definitions=TOOL_DEFINITIONS,
        tool_functions=all_tool_functions,
        system_prompt=system_prompt,
    )

    skills = load_skills("skills")
    for skill in skills.values():
        if skill.tool_functions:
            agent.tool_functions.update(skill.tool_functions)
        if skill.tool_definitions:
            agent.tool_definitions.extend(skill.tool_definitions)

    print(f"Agent ready (model: {args.model})")
    slash_cmds = ", ".join(f"/{s}" for s in sorted(skills)) if skills else "none"
    print(f"Skills loaded: {slash_cmds}")
    print("Commands: 'reset', 'memory [query]', '/help', 'quit'\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            if not skills:
                print("No skills loaded.\n")
            else:
                print("Available skills:")
                for sname, skill in sorted(skills.items()):
                    print(f"  /{sname:<16} {skill.description}")
            print()
            continue

        if user_input.startswith("/"):
            parts = user_input[1:].split(None, 1)
            skill_name = parts[0].lower()
            query = parts[1] if len(parts) > 1 else ""
            if skill_name not in skills:
                print(f"Unknown skill '/{skill_name}'. Type /help for available skills.\n")
                continue
            skill = skills[skill_name]
            original_prompt = agent.system_prompt
            agent.system_prompt = original_prompt + "\n\n" + skill.system_prompt_addition
            reply = agent.chat(skill.workflow_prompt.format(query=query))
            print(f"Agent: {reply}\n")
            agent.system_prompt = original_prompt
            continue

        if user_input.lower() == "reset":
            agent.reset()
            print("History cleared.\n")
            continue

        if user_input.lower().startswith("memory"):
            parts = user_input.split(None, 1)
            if len(parts) == 1:
                rows = store.recent(10)
            else:
                rows = store.search(parts[1], limit=10)
            if not rows:
                print("No memories found.\n")
            else:
                for row in rows:
                    print(f"  [{row['id']}] {row['created_at']} | tags: {row['tags']}")
                    print(f"  {row['content']}")
                    print()
            continue

        # Auto-inject relevant memories so the LLM sees them without needing to call a tool
        relevant = store.search(user_input, limit=5)
        if relevant:
            mem_lines = []
            for row in relevant:
                mem_lines.append(f"[id={row['id']} | {row['created_at']} | tags: {row['tags']}]\n{row['content']}")
            memory_block = "[Relevant memories from past sessions:]\n" + "\n---\n".join(mem_lines) + "\n\nUser question: "
            augmented_input = memory_block + user_input
        else:
            augmented_input = user_input

        reply = agent.chat(augmented_input)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
