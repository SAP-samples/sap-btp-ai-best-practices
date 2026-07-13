"""Shared tools for the code-based agent examples.

These are simple, self-contained tools that require no external APIs.
They are intentionally trivial so the focus stays on the agent loop, not the tools.
"""


def add(a: float, b: float) -> float:
    """Add two numbers together. Use this for addition operations."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Use this for multiplication operations."""
    return a * b


def get_weather(city: str) -> str:
    """Get the current weather for a given city (mock implementation for demonstration)."""
    mock_weather = {
        "berlin": "Cloudy, 8 degrees Celsius",
        "new york": "Sunny, 22 degrees Celsius",
        "tokyo": "Rainy, 18 degrees Celsius",
        "london": "Foggy, 12 degrees Celsius",
        "paris": "Partly cloudy, 15 degrees Celsius",
    }
    return mock_weather.get(city.lower(), f"Weather data not available for {city}")
