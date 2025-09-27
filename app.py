# app.py
import streamlit as st
from dotenv import load_dotenv
import os
import asyncio
import nest_asyncio
# Agents SDK imports (same as your script)
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

st.set_page_config(page_title="Agentic Homework Helper", page_icon="🤖", layout="centered")

@st.cache_resource
def init_agents():
    """
    Initialize client, model, runconfig and agents once and cache them.
    """
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env. Please set it before running the app.")

    # AsyncOpenAI client pointed at Gemini-compatible endpoint
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url=base_url,
    )

    # Model object (this class requires openai_client)
    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client,
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True,
    )

    # Your agents (same logic as your main.py)
    history_agent = Agent(
        name="History Teacher",
        handoff_description="You are a specialist History Teacher.",
        instructions="You are a professional history teacher. Answer history questions clearly.",
    )

    math_tutor = Agent(
        name="Math Agent",
        handoff_description="You are a specialist Math Teacher.",
        instructions="You are a professional math tutor. Help with math problems step-by-step.",
    )

    triage_agent = Agent(
        name="Triage Agent",
        instructions="You are a triage agent. Route the question to the appropriate specialist agent.",
        handoffs=[history_agent, math_tutor],
    )

    return external_client, model, config, triage_agent, history_agent, math_tutor

def run_agent_sync(agent, prompt, run_config):
    """
    Helper to run the async Runner.run from sync Streamlit callbacks.
    Handles the common "asyncio.run() cannot be called from a running event loop" problem.
    """
    coro = Runner.run(agent, prompt, run_config=run_config)
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # If Streamlit already has an event loop, use nest_asyncio fallback.
        # nest_asyncio allows nested loops; it's in requirements as optional.
        #import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

# ---------- UI ----------
st.title("🤖 Agentic Homework Helper")
st.write("Ask your homework question — a triage agent will route it to the right specialist (History or Math).")

# Initialize cached clients/agents
try:
    external_client, model, config, triage_agent, history_agent, math_tutor = init_agents()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# session history
if "history" not in st.session_state:
    st.session_state.history = []

col_a, col_b = st.columns([4,1])
with col_a:
    prompt = st.text_area("Enter your question", height=160, placeholder="e.g. When was the Ottoman Empire formed?")
with col_b:
    run_mode = st.selectbox("Run via", ["Triage (auto)", "History Agent", "Math Agent"])
    ask = st.button("Ask")

if ask and prompt.strip():
    # choose which agent to run
    if run_mode == "Triage (auto)":
        agent_to_run = triage_agent
    elif run_mode == "History Agent":
        agent_to_run = history_agent
    else:
        agent_to_run = math_tutor

    with st.spinner("Thinking..."):
        try:
            run_result = run_agent_sync(agent_to_run, prompt, run_config=config)
            # Runner.run returns an object; final_output is what you used previously
            output = getattr(run_result, "final_output", str(run_result))
            st.session_state.history.append({"prompt": prompt, "response": output, "agent": agent_to_run.name})
            st.success("Answer received")
        except Exception as e:
            st.error(f"Error while running the agent: {e}")

# show history (most recent first)
if st.session_state.history:
    st.markdown("---")
    st.header("Conversation history")
    for item in reversed(st.session_state.history):
        st.markdown(f"**Q ({item['agent']}):** {item['prompt']}")
        st.markdown(f"**A:** {item['response']}")
        st.markdown("----")

# control buttons
st.sidebar.header("Controls")
if st.sidebar.button("Clear history"):
    st.session_state.history = []
    st.sidebar.success("History cleared")
