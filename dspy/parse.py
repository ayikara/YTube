import requests
from bs4 import BeautifulSoup
import json
import re
import sys

def clean_text(text):
    """Clean whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def parse_cme_contract_page(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return {"url": url, "error": f"Failed to fetch URL: {e}"}

    soup = BeautifulSoup(response.text, "html.parser")

    result = {
        "url": url,
        "title": clean_text(soup.title.string if soup.title else ""),
        "sections": []
    }

    # Strategy: Loop through sections with headers (h2, h3, h4), then find table or paragraphs below
    for header in soup.find_all(["h2", "h3", "h4"]):
        section_title = clean_text(header.get_text())
        section = {
            "section_title": section_title,
            "content": []
        }

        # Look for sibling elements like tables or paragraphs
        sibling = header.find_next_sibling()
        while sibling and sibling.name not in ["h2", "h3", "h4"]:
            if sibling.name == "table":
                rows = sibling.find_all("tr")
                for row in rows:
                    cols = row.find_all(["td", "th"])
                    if len(cols) >= 2:
                        key = clean_text(cols[0].get_text())
                        val = clean_text(cols[1].get_text())
                        section["content"].append({key: val})
            elif sibling.name == "p":
                text = clean_text(sibling.get_text())
                if text:
                    section["content"].append(text)
            sibling = sibling.find_next_sibling()

        if section["content"]:
            result["sections"].append(section)

    return result

def parse_multiple_urls(url_list):
    results = []
    for url in url_list:
        print(f"Parsing: {url}")
        parsed = parse_cme_contract_page(url)
        results.append(parsed)
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        urls = sys.argv[1:]
    else:
        urls = [
            "https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html",
            "https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.contractSpecs.html"
        ]

    all_results = parse_multiple_urls(urls)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n✅ Parsing complete. Results saved to 'results.json'")
