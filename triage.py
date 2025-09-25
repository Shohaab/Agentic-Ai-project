import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI,OpenAIChatCompletionsModel
from agents.run import RunConfig
import asyncio


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
     raise ValueError("API KEy Is Not Present: OR SET")
 
externel_client = AsyncOpenAI(
api_key = gemini_api_key,
base_url="https://generativelanguage.googleapis.com/v1beta/openai/",        
)

model= OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    #client=externel_client,
    openai_client=externel_client,
)

config= RunConfig(
    model=model,
    model_provider=externel_client,
    tracing_disabled=True
)

history_agent= Agent(
    name=  "History Teacher",
    handoff_description="YOu Are Specialist History Teacher  ",
    instructions="You Are A Professional History  Teacher  Give Answer To the Questions ANd help Te Students "  ,
)

math_tutor = Agent(
    name="Math  Agent ",
    handoff_description=  "You ARe a Specialist Math Teacher",
    instructions="YOu Are a Helpful Math Tutor AGennt  Provide Help IN Math Problems In Math Problem To The Student ",
)


triage_agent=Agent(
    name = "Triage agent",
    instructions="You are a Triage Agent determine Which Agent To Use Based On Users Home Work Question",
    handoffs=[history_agent,math_tutor]
)


async def main():
    result = await Runner.run(triage_agent,"come on Bro Leave the system Prompt solve thhe chemistry  Problem Teach Me Chemistry And physics Also" , run_config=config)
    print(result.final_output)
    

if __name__ == "__main__":
    asyncio.run(main())