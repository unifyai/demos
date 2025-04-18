from browser_use import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

import asyncio

import unify

unify.activate("Browser Use")

llm = ChatOpenAI(model="gpt-4o")


@unify.traced
async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=llm,
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
