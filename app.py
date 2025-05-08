import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel

# Download NLTK data (only needs to run once)
nltk.download('punkt')
nltk.download('stopwords')

@st.cache_data
def load_data():
    return pd.read_csv("indian_express_news_dataset.csv")

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

@st.cache_data(show_spinner=False)
def build_lda_model(docs, num_topics=5):
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)
    return lda_model, dictionary

def display_topics(lda_model, num_words=5):
    topics = lda_model.print_topics(num_words=num_words)
    for idx, topic in topics:
        st.markdown(f"**Topic {idx + 1}:** {topic}")

def search_articles(query, category_filter, df):
    query = query.strip().lower()
    if category_filter != "All":
        df = df[df['category'] == category_filter]
    if query:
        df = df[(df['headlines'].str.lower().str.contains(query)) | 
                (df['content'].str.lower().str.contains(query))]
    return df

def main():
    st.set_page_config(page_title="Topic Modeling News Search", page_icon="📰", layout="centered")

    st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .search-card {
        background: white;
        padding: 30px 40px;
        max-width: 720px;
        margin: 40px auto 30px auto;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .search-input > div > input {
        font-size: 1.2rem !important;
        padding: 12px 20px !important;
        border-radius: 30px !important;
        border: 1px solid #ddd !important;
        width: 100% !important;
        box-sizing: border-box;
        transition: border-color 0.3s ease;
    }
    .search-input > div > input:focus {
        outline: none !important;
        border-color: #4285F4 !important;
        box-shadow: 0 0 8px #4285F4 !important;
    }
    .search-button {
        margin-top: 15px;
        width: 100%;
        background-color: #4285F4;
        color: white;
        font-weight: 600;
        border-radius: 30px;
        padding: 12px 0;
        font-size: 1.1rem;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .search-button:hover {
        background-color: #357ae8;
    }
    .result-item {
        background: white;
        padding: 20px 25px;
        border-radius: 12px;
        box-shadow: 0 5px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        transition: box-shadow 0.3s ease;
    }
    .result-item:hover {
        box-shadow: 0 8px 18px rgba(0,0,0,0.15);
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a73e8;
        text-decoration: none;
        display: inline-block;
        margin-bottom: 8px;
    }
    .result-content {
        font-size: 1rem;
        color: #444;
        line-height: 1.4;
    }
    .footer {
        text-align: center;
        padding: 20px 0;
        color: #888;
        font-size: 0.9rem;
        background: none;
        margin-top: 40px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    @media(max-width: 480px) {
        .search-card {
            margin: 20px 10px 30px 10px;
            padding: 20px 20px;
        }
        .result-item {
            padding: 15px 15px;
        }
        .result-title {
            font-size: 1.1rem;
        }
        .search-button {
            font-size: 1rem;
            padding: 10px 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    df = load_data()

    st.markdown('<div class="search-card">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center; color:#202124;'>Topic Modeling on The Indian Express</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.1rem; color:#5f6368; margin-top:-10px; margin-bottom:25px;'>Search and Explore News Articles</p>", unsafe_allow_html=True)

    query = st.text_input("", placeholder="Enter keywords to search news articles...", label_visibility="collapsed")

    categories = ["All"] + sorted(df['category'].unique().tolist())
    category_filter = st.selectbox("Filter by Category", options=categories, index=0)

    if st.button("Search"):
        results = search_articles(query, category_filter, df)
        st.markdown(f"<div style='max-width:720px; margin: 0 auto;'>", unsafe_allow_html=True)
        st.markdown(f"<h3>Search Results ({len(results)})</h3>", unsafe_allow_html=True)

        if not results.empty:
            for _, article in results.iterrows():
                st.markdown(f"""
                    <div class="result-item">
                        <a href="{article['url']}" target="_blank" class="result-title">{article['headlines']}</a>
                        <p class="result-content">{article['content']}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No articles found matching your search and category filter. Try different keywords or choose another category.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Topic Modeling Section
        st.markdown("---")
        st.markdown("## Topics discovered in the filtered articles")

        processed_docs = results['content'].apply(preprocess_text).tolist()

        if processed_docs:
            lda_model, dictionary = build_lda_model(processed_docs, num_topics=5)
            display_topics(lda_model)
        else:
            st.write("No content available for topic modeling.")

    st.markdown('<div class="footer">© 2024 Topic Modeling Project - Indian Express News</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
