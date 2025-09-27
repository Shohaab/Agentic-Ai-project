import os
import asyncio
import requests
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, Tool, function_tool
from agents.run import RunConfig
from pydantic import BaseModel
from typing import Type


# Pydantic model for structured weather info
class WeatherInfo(BaseModel):
    city: str
    temp: float
    feels_like: int
    humidity: int
    description: str
    pressure: int


# Load environment variables
load_dotenv()
weather_api_key = os.getenv("weather_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing. Please set it in your .env file.")


# External Gemini client
externel_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model configuration
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=externel_client,
)

config = RunConfig(
    model=model,
    model_provider=externel_client,
    tracing_disabled=True,
)


# Weather fetching tool
@function_tool
def get_weather(city: str) -> WeatherInfo:
    if not weather_api_key:
        return WeatherInfo(
            city=city,
            temp=0.0,
            feels_like=0,
            humidity=0,
            description="❌ Weather API Key is not set. Please configure it in your .env file.",
            pressure=0,
        )

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}"
    response = requests.get(url)

    if response.status_code != 200:
        return WeatherInfo(
            city=city,
            temp=0.0,
            feels_like=0,
            humidity=0,
            description=f"❌ Error: Unable to fetch weather data ({response.status_code})",
            pressure=0,
        )

    data = response.json()
    return WeatherInfo(
        city=city,
        temp=data["main"]["temp"],
        feels_like=data["main"]["feels_like"],
        humidity=data["main"]["humidity"],
        description=data["weather"][0]["description"],
        pressure=data["main"]["pressure"],
    )
   
    



# Define agent
weather_assistant = Agent(
    name="WeatherForecasting",
    instructions="You are a weather assistant. Use the weather tool to fetch data and describe it in a friendly way.",
    tools=[get_weather],
    output_type=str,  # ✅ output as natural language
)


# Main runner
async def main():
    result = await Runner.run(
        weather_assistant, "What's the current weather in Paris?  if not retreive information identify the error ", run_config=config
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
