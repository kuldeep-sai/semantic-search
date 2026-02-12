import streamlit as st
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

# ---- CONFIG ----
SERP_API_KEY = "YOUR_SERPAPI_KEY"
OPENAI_API_KEY = "YOUR_OPENAI_KEY"

client = OpenAI(api_key=OPENAI_API_KEY)
import spacy
from spacy.cli import download

@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy()


# ---- FUNCTIONS ----

def get_top_urls(keyword):
    params = {
        "engine": "google",
        "q": keyword,
        "api_key": SERP_API_KEY,
        "num": 3
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return [r["link"] for r in results["organic_results"][:3]]

def scrape_text(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n")
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 150]
        return paragraphs[:20]
    except:
        return []

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def get_entities(text):
    doc = nlp(text)
    return set([ent.text.lower() for ent in doc.ents])

# ---- UI ----

st.title("AI Semantic Authority Analyzer")

keyword = st.text_input("Enter Target Keyword")
your_url = st.text_input("Enter Your URL")

if st.button("Analyze"):

    with st.spinner("Fetching competitors..."):
        competitors = get_top_urls(keyword)

    st.write("Top Competitors:")
    for c in competitors:
        st.write(c)

    with st.spinner("Scraping pages..."):
        your_paragraphs = scrape_text(your_url)

        competitor_paragraphs = []
        for url in competitors:
            competitor_paragraphs.extend(scrape_text(url))

    if not your_paragraphs or not competitor_paragraphs:
        st.error("Could not scrape content properly.")
        st.stop()

    with st.spinner("Generating embeddings..."):
        your_vectors = [get_embedding(p) for p in your_paragraphs]
        competitor_vectors = [get_embedding(p) for p in competitor_paragraphs]

    # ---- Semantic Comparison ----
    covered = 0
    similarities = []
    gaps = []

    for i, comp_vec in enumerate(competitor_vectors):
        sims = cosine_similarity([comp_vec], your_vectors)
        max_sim = max(sims[0])
        similarities.append(max_sim)

        if max_sim > 0.75:
            covered += 1
        elif max_sim < 0.65:
            gaps.append(competitor_paragraphs[i][:200])

    topic_coverage = (covered / len(competitor_vectors)) * 100
    semantic_depth = (sum(similarities) / len(similarities)) * 100

    # ---- Entity Coverage ----
    your_text_full = " ".join(your_paragraphs)
    competitor_text_full = " ".join(competitor_paragraphs)

    your_entities = get_entities(your_text_full)
    competitor_entities = get_entities(competitor_text_full)

    if competitor_entities:
        entity_coverage = (len(your_entities & competitor_entities) / len(competitor_entities)) * 100
    else:
        entity_coverage = 0

    # ---- Final Score ----
    SAS = (0.4 * topic_coverage) + (0.35 * semantic_depth) + (0.25 * entity_coverage)

    st.subheader("Results")

    st.metric("Semantic Authority Score", round(SAS, 2))
    st.metric("Topic Coverage %", round(topic_coverage, 2))
    st.metric("Semantic Depth %", round(semantic_depth, 2))
    st.metric("Entity Coverage %", round(entity_coverage, 2))

    st.subheader("Top Missing Themes")
    for gap in gaps[:5]:
        st.write("-", gap)
