# app/paper_search.py
import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def search_papers(query: str, year: int = 2020, limit: int = 5):
    fields = "title,abstract,authors,year,citationCount,url"

    # Yıl filtrelemesini doğrudan query'ye ekliyoruz
    query_with_year = f"{query} year:{year}"

    params = {
        "query": query_with_year,
        "limit": limit,
        "fields": fields
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print("❌ Semantic Scholar API Error:", response.status_code, response.text)
        raise Exception(f"Semantic Scholar API Error: {response.status_code} {response.text}")

    return response.json().get("data", [])
