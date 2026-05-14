import argparse

from agent import Agent
from memory import MemoryStore
from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS, make_memory_tools

MODEL = "qwen3.5:latest"

_BASE_SYSTEM_PROMPT = (
    "You are a research assistant with access to tools. "
    "When asked to research a topic: search the web for relevant sources, "
    "fetch the most promising pages to read the full content, and synthesize "
    "a clear, well-organized answer with key findings. "
    "Cite your sources (title + URL) at the end of your response. "
    "For local documents, use parse_pdf or read_file as appropriate. "
    "Use save_memory to store important findings for future sessions, and "
    "search_memories at the start of research tasks to check prior knowledge."
)


def main():
    parser = argparse.ArgumentParser(description="Ollama research agent")
    parser.add_argument("-m", "--model", default=MODEL, help="Ollama model to use (default: %(default)s)")
    args = parser.parse_args()

    store = MemoryStore(embed_model=args.model)

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

    print(f"Agent ready (model: {args.model})")
    print("Commands: 'reset' to clear history, 'memory [query]' to inspect memory, 'quit' to exit\n")

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
                    print(f"  {row['content'][:120]}")
                    print()
            continue

        reply = agent.chat(user_input)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
