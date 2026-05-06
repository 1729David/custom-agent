import argparse

from agent import Agent
from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS

MODEL = "qwen3.5:latest"

SYSTEM_PROMPT = (
    "You are a research assistant with access to tools. "
    "When asked to research a topic: search the web for relevant sources, "
    "fetch the most promising pages to read the full content, and synthesize "
    "a clear, well-organized answer with key findings. "
    "Cite your sources (title + URL) at the end of your response. "
    "For local documents, use parse_pdf or read_file as appropriate."
)


def main():
    parser = argparse.ArgumentParser(description="Ollama research agent")
    parser.add_argument("-m", "--model", default=MODEL, help="Ollama model to use (default: %(default)s)")
    args = parser.parse_args()

    agent = Agent(
        model=args.model,
        tool_definitions=TOOL_DEFINITIONS,
        tool_functions=TOOL_FUNCTIONS,
        system_prompt=SYSTEM_PROMPT,
    )

    print(f"Agent ready (model: {args.model})")
    print("Commands: 'reset' to clear history, 'quit' to exit\n")

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

        reply = agent.chat(user_input)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
