import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI,OpenAIChatCompletionsModel,function_tool,ModelSettings
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


@function_tool
def  calculate_area(length: float, width:float) ->str:
     """Calculate the area of a rectangle."""
     area = length*width
     return f"Area = {length} × {width} = {area} square Units"
 


def main():
     """Learn Model Settings with simple examples."""
    # 🎯 Example 1: Temperature (Creativity Control)
print("\n❄️🔥 Temperature Settings")
print("-" * 30)


agent_cold =Agent(
    name= "Cold Agent",
    instructions=" You are A HelpFul Assisstant",
    model_settings= ModelSettings(temprature=0.1),
    model=model
)
agent_hot =Agent(
    name="Hot Agent",
    instructions=" You Are a HelpFull Assisstant",
    model_settings=ModelSettings(temprature=1.9),
    model=model
)

question= " Tell ME About AI In 2 Sentence"
print("cold Agent (Temprature =0.1): ")
result_cold =Runner.run_sync(agent_cold,question)

print("Hot Agent (Temprature =1.9): ")
result_hot =Runner.run_sync(agent_hot,question)


print("\n💡 Notice: Cold = focused, Hot = creative")
print("📝 Note: Gemini temperature range extends to 2.0")
    
print("\n🔧 Tool Choice Settings")
print("-" * 30)


agent_auto =Agent(
    name="Auto",
    tools=[calculate_area],
    model_settings=ModelSettings(tool_choice="auto"),
    model=model,
)

agent_required =Agent(
    name= "Required",
    tools=[calculate_area],
    model_settings=ModelSettings(tool_choice="required"),
    model=model,
)

agent_none = Agent(
        name="None",
        tools=[calculate_area],
        model_settings=ModelSettings(tool_choice="none"),
        model=model
    )

question = "What's the area of a 5x3 rectangle?"



print("Auto Tool Choice:")
result_auto = Runner.run_sync(agent_auto, question)
print(result_auto.final_output)
    
print("\nRequired Tool Choice:")
result_required = Runner.run_sync(agent_required, question)
print(result_required.final_output)

print("\nNone Tool Choice:")
result_none = Runner.run_sync(agent_none, question)
print(result_none.final_output)
    
print("\n💡 Notice: Auto = decides, Required = must use tool")






    

if __name__ == "__main__":
    asyncio.run(main())