"""Run one purchase request through the agent with the real model and print the result."""

import sys

from procurement_agent.agent import ProcurementAgent
from procurement_agent.knowledge import KnowledgeBase
from procurement_agent.llm import GroqClient
from procurement_agent.tools import ToolBox


def main() -> None:
    agent = ProcurementAgent(GroqClient(), ToolBox(), KnowledgeBase())
    for text in sys.argv[1:]:
        print(f"\n### {text}")
        print(agent.run(text).model_dump_json(indent=1, exclude_none=True))


if __name__ == "__main__":
    main()
