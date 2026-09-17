import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Single AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Single AI Agent")
st.write("AI Agent using Groq, Tavily Search and WeatherStack")


# ============================================================
# CHECK API KEYS
# ============================================================

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is not set.")
    st.stop()

if not WEATHERSTACK_API_KEY:
    st.error("WEATHERSTACK_API_KEY is not set.")
    st.stop()


# ============================================================
# WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    api_key = WEATHERSTACK_API_KEY

    url = "http://api.weatherstack.com/current"

    params = {
        "access_key": api_key,
        "query": city
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

    except Exception as e:

        return f"Weather API error: {e}"


    if "current" not in data:

        return f"Could not fetch weather data for {city}: {data}"


    current = data["current"]

    temperature = current.get(
        "temperature",
        "N/A"
    )

    description = current.get(
        "weather_descriptions",
        ["N/A"]
    )[0]

    humidity = current.get(
        "humidity",
        "N/A"
    )


    return (
        f"City: {city}\n"
        f"Temperature: {temperature}°C\n"
        f"Weather: {description}\n"
        f"Humidity: {humidity}%"
    )


# ============================================================
# TAVILY SEARCH TOOL
# ============================================================

search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=TAVILY_API_KEY
)


# ============================================================
# GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)


# ============================================================
# CREATE AI AGENT
# ============================================================

tools = [
    search_tool,
    get_weather
]


agent = create_agent(
    model=llm,
    tools=tools
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Ask your AI Agent something..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if user_input:

    # Display user message

    with st.chat_message("user"):

        st.markdown(user_input)


    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # AI response

    with st.chat_message("assistant"):

        with st.spinner("🤔 Agent is thinking..."):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_input
                            }
                        ]
                    }
                )


                # Get final AI message

                final_message = response["messages"][-1]


                if hasattr(
                    final_message,
                    "content"
                ):

                    answer = final_message.content

                else:

                    answer = str(final_message)


            except Exception as e:

                answer = f"❌ Error: {e}"


        st.markdown(answer)


    # Save AI response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )