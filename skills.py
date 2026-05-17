import importlib.util
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    system_prompt_addition: str = ""
    workflow_prompt: str = "{query}"
    tool_functions: dict = field(default_factory=dict)
    tool_definitions: list = field(default_factory=list)


def load_skills(skills_dir: str = "skills") -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    path = Path(skills_dir)
    if not path.is_dir():
        return skills
    for py_file in sorted(path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        name = getattr(mod, "NAME", py_file.stem)
        skills[name] = Skill(
            name=name,
            description=getattr(mod, "DESCRIPTION", ""),
            system_prompt_addition=getattr(mod, "SYSTEM_PROMPT_ADDITION", ""),
            workflow_prompt=getattr(mod, "WORKFLOW_PROMPT", "{query}"),
            tool_functions=getattr(mod, "TOOL_FUNCTIONS", {}),
            tool_definitions=getattr(mod, "TOOL_DEFINITIONS", []),
        )
    return skills
