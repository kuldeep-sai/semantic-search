import streamlit as st
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ---------------- #

st.set_page_config(page_title="Semantic Authority Analyzer", layout="wide")

SERP_API_KEY = st.secrets["SERP_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- HELPERS ---------------- #

def log(msg):
    st.write(f"🔹 {msg}")

@st.cache_data(ttl=86400)
def get_top_urls(keyword):
    params = {
        "engine": "google",
        "q": keyword,
        "api_key": SERP_API_KEY,
        "num": 3
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return [r["link"] for r in results.get("organic_results", [])[:3]]

@st.cache_data(ttl=86400)
def scrape_text(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        paragraphs = [
            p.strip()
            for p in text.split("\n")
            if 150 < len(p.strip()) < 1200
        ]

        return paragraphs[:15]
    except:
        return []

@st.cache_data(ttl=86400)
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def smart_chunks(paragraphs, max_len=600):
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) < max_len:
            current += " " + p
        else:
            chunks.append(current.strip())
            current = p
    if current:
        chunks.append(current.strip())
    return chunks

# ---------------- UI ---------------- #

st.title("🧠 AI Semantic Authority Analyzer")

keyword = st.text_input("🎯 Target Keyword")
your_url = st.text_input("🔗 Your Page URL")

if st.button("Analyze"):

    if not keyword or not your_url:
        st.error("Please enter both keyword and URL")
        st.stop()

    with st.spinner("Fetching competitors..."):
        log("Searching Google SERP")
        competitors = get_top_urls(keyword)

    if not competitors:
        st.error("No competitors found")
        st.stop()

    st.subheader("Top Competitors")
    for c in competitors:
        st.write(c)

    with st.spinner("Scraping content..."):
        log("Scraping your page")
        your_paragraphs = scrape_text(your_url)

        log("Scraping competitor pages")
        competitor_paragraphs = []
        for url in competitors:
            competitor_paragraphs.extend(scrape_text(url))

    if not your_paragraphs or not competitor_paragraphs:
        st.error("Content scraping failed")
        st.stop()

    # -------- SMART CHUNKING -------- #
    your_chunks = smart_chunks(your_paragraphs)[:5]
    competitor_chunks = smart_chunks(competitor_paragraphs)[:8]

    with st.spinner("Generating embeddings..."):
        log("Embedding your content")
        your_vectors = [get_embedding(c) for c in your_chunks]

        log("Embedding competitor content")
        competitor_vectors = [get_embedding(c) for c in competitor_chunks]

    # -------- SEMANTIC ANALYSIS -------- #
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
            gaps.append(competitor_chunks[i][:200])

    topic_coverage = (covered / len(competitor_vectors)) * 100
    semantic_depth = (sum(similarities) / len(similarities)) * 100

    # -------- ENTITY PROXY (FAST) -------- #
    with st.spinner("Calculating authority score..."):
        your_full = " ".join(your_chunks)
        comp_full = " ".join(competitor_chunks)

        entity_similarity = cosine_similarity(
            [get_embedding(your_full[:1500])],
            [get_embedding(comp_full[:1500])]
        )[0][0] * 100

    # -------- FINAL SCORE -------- #
    SAS = (
        0.4 * topic_coverage +
        0.35 * semantic_depth +
        0.25 * entity_similarity
    )

    # ---------------- RESULTS ---------------- #

    st.subheader("📊 Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Semantic Authority Score", round(SAS, 2))
    col2.metric("Topic Coverage %", round(topic_coverage, 2))
    col3.metric("Semantic Depth %", round(semantic_depth, 2))
    col4.metric("Entity Proxy %", round(entity_similarity, 2))

    st.subheader("🚨 Missing Content Themes")

    if gaps:
        for g in gaps[:5]:
            st.write("•", g)
    else:
        st.success("No major content gaps detected 🎉")
