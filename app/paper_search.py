# app/paper_search.py
import time
import requests
from typing import Any, Dict, List, Optional

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = "paperId,title,abstract,authors,year,citationCount,url"


def search_papers(
    query: str,
    year: Optional[int] = 2020,
    limit: int = 5,
    retries: int = 3,
    timeout: int = 20,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search papers from Semantic Scholar.

    Args:
        query: Search query string.
        year: Minimum publication year filter. If None, no year filter is applied.
        limit: Number of results to return.
        retries: Number of retry attempts for rate limit / temporary failures.
        timeout: Request timeout in seconds.
        api_key: Optional Semantic Scholar API key.

    Returns:
        A normalized list of paper dictionaries.
    """
    headers = {}
    if api_key:
        headers["7rEDZUDjtI1rUp3c5iZNy7GtFqPWXerD82Kg2ryy"] = api_key

    params = {
        "query": query,
        "limit": limit,
        "fields": DEFAULT_FIELDS,
    }

    if year is not None:
        params["filter"] = f"year>{year}"

    for attempt in range(retries):
        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code == 200:
                data = response.json().get("data", [])
                return [_normalize_paper(paper) for paper in data]

            if response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"⏳ Rate limited by Semantic Scholar. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            if 500 <= response.status_code < 600:
                wait_time = 2 ** attempt
                print(f"⚠️ Server error {response.status_code}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            print("❌ Semantic Scholar API Error:", response.status_code, response.text)
            return []

        except requests.RequestException as e:
            wait_time = 2 ** attempt
            print(f"❌ Request failed: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    return []


def _normalize_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Semantic Scholar paper response into a consistent structure."""
    return {
        "paper_id": paper.get("paperId"),
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "authors": [author.get("name") for author in paper.get("authors", []) if author.get("name")],
        "year": paper.get("year"),
        "citations": paper.get("citationCount"),
        "url": paper.get("url"),
        "source": "semantic_scholar",
    }
