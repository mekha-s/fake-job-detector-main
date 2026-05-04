from duckduckgo_search import DDGS

try:
    results = DDGS().text("Google company", max_results=3)
    print("DDGS SUCCESS")
    for r in results:
        print(r['title'], r['href'])
except Exception as e:
    print("DDGS FAILED:", e)
