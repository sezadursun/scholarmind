# arXiv API integration for preprints
# 🧪 arxiv.py – arXiv API ile preprint makale arama

import requests
import feedparser

def search_arxiv(query: str, max_results: int = 5):
    """
    arXiv API üzerinden verilen anahtar kelimeye göre makale arar.
    
    Returns:
        List of dictionaries with title, authors, summary, published date, and link.
    """
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query}&start=0&max_results={max_results}"
    url = base_url + search_query

    response = requests.get(url)
    feed = feedparser.parse(response.text)

    results = []
    for entry in feed.entries:
        paper = {
            "title": entry.title.strip(),
            "authors": ", ".join(author.name for author in entry.authors),
            "summary": entry.summary.strip().replace("\n", " "),
            "published": entry.published,
            "link": entry.link
        }
        results.append(paper)

    return results
