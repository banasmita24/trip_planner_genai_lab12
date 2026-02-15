from datetime import date, timedelta
import os
from dotenv import load_dotenv
import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load local .env variables
load_dotenv()


# ---------------- API KEY HELPERS ----------------
def get_openweather_api_key() -> str | None:
    """Get OpenWeather API key from .env or Streamlit secrets."""
    return os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY")


def get_groq_api_key() -> str | None:
    """Get Groq API key from .env or Streamlit secrets."""
    return os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")


# ---------------- LLM SETUP ----------------
def get_llm():
    api_key = get_groq_api_key()

    if not api_key:
        st.error("Groq API key not found. Add it to .env or Streamlit secrets.")
        return None

    return ChatGroq(
        groq_api_key=api_key,
        model_name="moonshotai/kimi-k2-instruct-0905",
        temperature=0.7,
    )


# ---------------- WEATHER ----------------
def fetch_weather(city: str, api_key: str) -> dict | None:
    """Fetch current weather for a city using OpenWeather API in metric units."""
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None


def build_weather_summary(weather_data: dict | None) -> str:
    """Create a concise natural-language summary of the weather data."""
    if not weather_data:
        return "Weather information is currently unavailable."

    try:
        main = weather_data.get("weather", [{}])[0].get("description", "N/A").title()
        temp = weather_data.get("main", {}).get("temp")
        feels_like = weather_data.get("main", {}).get("feels_like")
        humidity = weather_data.get("main", {}).get("humidity")
        wind = weather_data.get("wind", {}).get("speed")

        parts = [f"Conditions: {main}"]
        if temp is not None:
            parts.append(f"Temperature: {temp:.1f}°C")
        if feels_like is not None:
            parts.append(f"Feels like: {feels_like:.1f}°C")
        if humidity is not None:
            parts.append(f"Humidity: {humidity}%")
        if wind is not None:
            parts.append(f"Wind speed: {wind} m/s")

        return " | ".join(parts)
    except Exception:
        return "Weather information is currently unavailable."


# ---------------- ITINERARY ----------------
def build_itinerary_chain(llm):
    template = """
You are a professional travel planner.

Destination: {destination}
Number of days: {days}
Weather summary: {weather_summary}
Preferences: {preferences}

1. Give a short cultural overview.
2. Provide a day-by-day itinerary.
3. Add 3 travel tips.

Keep response under 500 words.
"""
    prompt = PromptTemplate.from_template(template.strip())
    return prompt | llm


# ---------------- MOCK DATA ----------------
def generate_mock_flights(destination: str, start: date, days: int) -> list[dict]:
    return [
        {
            "airline": "SkyWays",
            "route": f"Home → {destination}",
            "depart": start.strftime("%Y-%m-%d"),
            "return": (start + timedelta(days=days)).strftime("%Y-%m-%d"),
            "price": "$650",
        },
        {
            "airline": "GlobeAir",
            "route": f"Home → {destination}",
            "depart": (start + timedelta(days=1)).strftime("%Y-%m-%d"),
            "return": (start + timedelta(days=days + 1)).strftime("%Y-%m-%d"),
            "price": "$720",
        },
    ]


def generate_mock_hotels(destination: str, start: date, days: int) -> list[dict]:
    return [
        {
            "name": f"{destination} Central Hotel",
            "check_in": start.strftime("%Y-%m-%d"),
            "check_out": (start + timedelta(days=days)).strftime("%Y-%m-%d"),
            "price_per_night": "$140",
            "rating": "4.3 / 5",
        },
        {
            "name": f"{destination} Boutique Stay",
            "check_in": start.strftime("%Y-%m-%d"),
            "check_out": (start + timedelta(days=days)).strftime("%Y-%m-%d"),
            "price_per_night": "$190",
            "rating": "4.7 / 5",
        },
    ]


# ---------------- MAIN APP ----------------
def main() -> None:
    st.set_page_config(page_title="AI Travel Planner", page_icon="✈️")

    st.title("✈️ AI Travel Planner")
    st.write(
        "Plan your next trip with live weather data, an AI-powered itinerary, "
        "and sample flight and hotel ideas."
    )

    with st.sidebar:
        st.header("Settings")
        st.markdown(
            "This app uses **OpenWeather** for current weather and an **AI model via Groq** "
            "with LangChain for itinerary generation."
        )

    col1, col2 = st.columns(2)

    with col1:
        destination = st.text_input("Destination city", placeholder="e.g., Paris")
    with col2:
        days = st.number_input("Number of days", min_value=1, max_value=30, value=5, step=1)

    preferences = st.text_area(
        "Preferences (optional)",
        placeholder="Tell the planner what you enjoy: museums, nightlife, food, outdoors, budget level, etc.",
    )

    start_date = st.date_input("Approximate departure date", value=date.today())

    if st.button("Plan my trip"):
        if not destination:
            st.warning("Please enter a destination city.")
            return

        with st.spinner("Fetching weather information..."):
            ow_api_key = get_openweather_api_key()
            if not ow_api_key:
                st.warning("OpenWeather API key not configured.")
                weather_data = None
            else:
                weather_data = fetch_weather(destination, ow_api_key)

        weather_summary = build_weather_summary(weather_data)

        st.subheader("Current Weather")
        st.write(weather_summary)

        st.subheader("AI-Generated Cultural Overview & Itinerary")
        with st.spinner("Generating your itinerary..."):
            llm = get_llm()
            if not llm:
                return
            chain = build_itinerary_chain(llm)
            try:
                response = chain.invoke({
                    "destination": destination,
                    "days": int(days),
                    "weather_summary": weather_summary,
                    "preferences": preferences or "No special preferences provided.",
                })
                st.markdown(response.content)
            except Exception as e:
                st.error(f"Error generating itinerary: {e}")

        st.subheader("Sample Flight Options (Mock Data)")
        flights = generate_mock_flights(destination, start_date, int(days))
        for f in flights:
            with st.expander(f"{f['airline']} – {f['route']} ({f['price']})"):
                st.write(f"**Depart:** {f['depart']}")
                st.write(f"**Return:** {f['return']}")

        st.subheader("Sample Hotel Options (Mock Data)")
        hotels = generate_mock_hotels(destination, start_date, int(days))
        for h in hotels:
            with st.expander(f"{h['name']} – {h['price_per_night']}/night"):
                st.write(f"**Check-in:** {h['check_in']}")
                st.write(f"**Check-out:** {h['check_out']}")
                st.write(f"**Rating:** {h['rating']}")


if __name__ == "__main__":
    main()