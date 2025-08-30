import requests
from bs4 import BeautifulSoup
import json
import re
import sys

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def find_heading_above(element):
    """Traverse upwards to find the nearest h2/h3/h4 before the element."""
    while element:
        prev = element.find_previous_sibling()
        while prev:
            if prev.name in ["h2", "h3", "h4"]:
                return clean_text(prev.get_text())
            prev = prev.find_previous_sibling()
        element = element.parent
    return None

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

    # Find all contract specs tables
    table_blocks = soup.find_all("div", class_="contractSpecs-table")

    for block in table_blocks:
        section_title = find_heading_above(block) or "Untitled Section"
        section_data = []

        table = block.find("table")
        if table:
            for row in table.find_all("tr"):
                cols = row.find_all(["td", "th"])
                if len(cols) >= 2:
                    key = clean_text(cols[0].get_text())
                    val = clean_text(cols[1].get_text())
                    section_data.append({key: val})

        if section_data:
            result["sections"].append({
                "section_title": section_title,
                "content": section_data
            })

    # Grab any FAQs or paragraphs in the overview
    overview = soup.find("div", class_="cmeContent")
    if overview:
        paras = overview.find_all("p")
        overview_text = [clean_text(p.get_text()) for p in paras if clean_text(p.get_text())]
        if overview_text:
            result["sections"].insert(0, {
                "section_title": "Overview",
                "content": overview_text
            })

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
