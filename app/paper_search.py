# Handles Semantic Scholar API interaction
import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def search_papers(query: str, year: int = 2020, limit: int = 5):
    fields = "title,abstract,authors,year,citationCount,url"

    params = {
        "query": query,
        "year": year,
        "limit": limit,
        "fields": fields
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"Semantic Scholar API Error: {response.status_code} {response.text}")

    return response.json().get("data", [])
