import requests
from bs4 import BeautifulSoup
import json
import re
import sys

def clean_text(text):
    """Remove extra whitespace and line breaks."""
    return re.sub(r'\s+', ' ', text).strip()

def parse_cme_contract_page(url):
    """Parses a CME Group contract spec page and returns structured content."""
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

    content_root = soup.find("div", class_="cmeContent")
    if not content_root:
        return {"url": url, "error": "Could not find main content section"}

    for section in content_root.find_all(["section", "div"], recursive=False):
        header = section.find(["h2", "h3", "h4"])
        title = clean_text(header.get_text()) if header else None

        table = section.find("table")
        table_data = []

        if table:
            for row in table.find_all("tr"):
                cols = row.find_all(["td", "th"])
                if len(cols) >= 2:
                    key = clean_text(cols[0].get_text())
                    val = clean_text(cols[1].get_text())
                    table_data.append({key: val})
        else:
            paragraphs = section.find_all("p")
            paragraph_texts = [clean_text(p.get_text()) for p in paragraphs if clean_text(p.get_text())]
            if paragraph_texts:
                table_data = paragraph_texts

        if title or table_data:
            result["sections"].append({
                "section_title": title,
                "content": table_data
            })

    return result

def parse_multiple_urls(url_list):
    """Parse multiple CME contract pages."""
    results = []
    for url in url_list:
        print(f"Parsing: {url}")
        parsed = parse_cme_contract_page(url)
        results.append(parsed)
    return results


if __name__ == "__main__":
    # Accept multiple URLs as command-line arguments or use defaults
    if len(sys.argv) > 1:
        urls = sys.argv[1:]
    else:
        urls = [
            "https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html",
            "https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.contractSpecs.html"
        ]

    all_results = parse_multiple_urls(urls)

    # Write output to JSON file
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n✅ Parsing complete. Results saved to 'results.json'.")
