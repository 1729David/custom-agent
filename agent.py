import ollama


class Agent:
    def __init__(
        self,
        model: str,
        tool_definitions: list,
        tool_functions: dict,
        system_prompt: str = "",
    ):
        self.model = model
        self.tool_definitions = tool_definitions
        self.tool_functions = tool_functions
        self.system_prompt = system_prompt
        self.history: list[dict] = []

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.history)

        while True:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=self.tool_definitions,
            )

            msg = response.message

            # Append assistant turn (with or without tool calls) as a plain dict
            assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_entry)

            if not msg.tool_calls:
                reply = msg.content or ""
                self.history.append({"role": "assistant", "content": reply})
                return reply

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = tool_call.function.arguments or {}

                print(f"  \033[90m[tool] {name}({args})\033[0m")

                if name in self.tool_functions:
                    try:
                        result = self.tool_functions[name](**args)
                    except Exception as e:
                        result = f"Error calling {name}: {e}"
                else:
                    result = f"Unknown tool: {name}"

                messages.append({"role": "tool", "content": str(result)})

    def reset(self):
        self.history.clear()
