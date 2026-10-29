import asyncio
import json
import sys
from dataclasses import asdict

from agent.graph import build_graph, build_llm, run_agent
from agent.tools import agent_tools


async def main(question: str) -> None:

    async with agent_tools() as tools:
        graph = build_graph(build_llm(), tools)
        result = await run_agent(graph, question)

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit('uso: python -m agent "pregunta"')

    asyncio.run(main(" ".join(sys.argv[1:])))
