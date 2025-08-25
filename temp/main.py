from lite_agent import Agent, Runner

agent = Agent(model="gpt-4.1-nano", name="ParentAgent", instructions="You are a helpful parent agent. You lucky number is 2.")
sub = Agent(model="gpt-4.1-nano", name="ChildAgent", instructions="You are a helpful child agent. Your lucky number is 47.")

agent.add_handoff(sub)


async def main():
    print("=== Testing completion API ===")
    runner = Runner(agent, api="completion")
    await runner.run_until_complete("Handoff to ChildAgent")

    new_runner = Runner(agent, api="completion")
    messages = runner.get_messages()
    print(f"Messages type: {[type(msg) for msg in messages]}")
    new_runner.set_chat_history(messages)

    await new_runner.run_until_complete("What's your lucky number?")
    print("Completion API result:", new_runner.messages[-1].content[0].text)

    print("\n=== Testing responses API ===")
    runner2 = Runner(agent, api="responses")
    await runner2.run_until_complete("Handoff to ChildAgent")

    new_runner2 = Runner(agent, api="responses")
    new_runner2.set_chat_history(runner2.get_messages())

    await new_runner2.run_until_complete("What's your lucky number?")
    print("Responses API result:", new_runner2.messages[-1].content[0].text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
