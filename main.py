import chainlit as cl # type: ignore
from agents import Agent,Runner,AsyncOpenAI , OpenAIChatCompletionsModel #type: ignore
from agents.run import RunConfig #type: ignore
import os
from dotenv import load_dotenv , find_dotenv# type: ignore

load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")


provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,

)

run_config = RunConfig(
    model = model,
    model_provider = provider,
    tracing_disabled = True,
)

agent = Agent(
    name="Student Assistant",
    instructions='''You are a helpful student assistant agent that can:

Answer academic questions
Provide study tips
Summarize small text passages''',
        
)




# print(result.final_output)
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Hello! I am your student assistant. How can I help you today?").send()
@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({"role": "user", "content": message.content})
    result = await Runner.run(
        agent,
        input=history,
        run_config=run_config,
    )
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()

