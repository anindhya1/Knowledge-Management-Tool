import requests


def generate_insight_mistral(prompt):
    """Generate insights using Mistral through Ollama API."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False,  # Set to False for simpler handling
            "options": {
                "num_predict": 400,
                "temperature": 0.7,
                "top_p": 0.9
            }
        },
        timeout=1200
    )

    if response.status_code == 200:
        result = response.json()
        return result.get("response", "No response generated").strip()
    else:
        return f"Error: HTTP {response.status_code}"
