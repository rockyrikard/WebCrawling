import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fetch_search_results(query):
    api_key = "YOUR-API"
    cse_id= "YOUR-CSE"
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={cse_id}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    if 'items' in data:
        return [{'link': item['link'], 'title': item['title']} for item in data['items']]
    else:
        return []


def fetch_page_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([para.get_text() for para in paragraphs])
            return content
        else:
            print(f"Error fetching content from {url}: HTTP Status Code {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""


def find_most_related_pages(query, num_pages=5):
    search_results = fetch_search_results(query)
    contents = [fetch_page_content(result['link']) for result in search_results]

    # Filter out empty contents
    valid_contents = [content for content in contents if content]

    if len(valid_contents) == 0:
        print("No valid content found.")
        return []

    # Include the user's query in the list of documents to vectorize
    documents = [query] + valid_contents

    # Calculate cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    query_vector = vectors[0]  # Vector for the user's query
    content_vectors = vectors[1:]  # Vectors for the web page contents

    # Compute cosine similarity between the query vector and content vectors
    similarity_scores = cosine_similarity(query_vector, content_vectors).flatten()

    # Sort indices by similarity score in descending order
    sorted_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)

    # Select top N pages
    most_related_pages = [{'link': search_results[idx]['link'], 'title': search_results[idx]['title'],
                           'similarity_score': similarity_scores[idx]} for idx in sorted_indices[:num_pages]]

    return most_related_pages


if __name__ == "__main__":
    query = input("Please enter a topic: ")
    related_pages = find_most_related_pages(query)
    print("Most related pages:")
    for idx, page in enumerate(related_pages, start=1):
        print(f"{idx}. {page['title']} - {page['link']} (Similarity Score: {page['similarity_score']:.2f})")
