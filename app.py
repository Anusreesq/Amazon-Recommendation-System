import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import process  

st.set_page_config(
    page_title="Amazon Product Recommendation System",
    layout="wide", 
    initial_sidebar_state="expanded")

@st.cache_data
def data():
    df = pd.read_csv("C:/Users/FixLab/Desktop/project/df.csv")
    return df

def content_based(df, item, top_n=20):
    if item not in df['name'].values:
        print(f"Item '{item}' not found in the Products list")
        return pd.DataFrame()

    vec = TfidfVectorizer()
    data = vec.fit_transform(df['description'])
    nn = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine', algorithm='brute')
    nn.fit(data)

    item_index = df[df['name'] == item].index[0]
    distances, indices = nn.kneighbors(data[item_index])

    rec_indices = indices.flatten()[1:]
    details = df.iloc[rec_indices][
        ['name', 'ratings', 'no_of_ratings', 'discount_price', 'actual_price', 'image_link', 'link']
    ]

    return details

def keyword_based(keyword):
    products = df[df['description'].str.contains(keyword, case=False, na=False)]
    return products


def fuzzy_search(product_name, df, limit=5):
    matches = process.extract(product_name, df['name'], limit=limit)
    return [match[0] for match in matches] 

df = data()

st.title("Amazon Product Recommendation System")
st.header("Find products similar to your favorite items")

type = st.radio("Choose recommendation type", ("Content-Based", "Keyword-Based"))

if type == "Keyword-Based":
    keyword = st.text_input("Search for products", "")

    if keyword:
        recommendations = keyword_based(keyword)
        if not recommendations.empty:
            st.subheader(f"Recommended Products for '{keyword}':")
            for index, row in recommendations.iterrows():
                st.markdown(f"### {row['name']}")
                st.markdown(f"**Rating:** {row['ratings']}")
                st.markdown(f"**Number of ratings:** {row['no_of_ratings']}")
                st.markdown(f"**Offer Price:** {row['discount_price']}")  
                st.markdown(f"**Price:** {row['actual_price']}")
                st.image(row['image_link'], width=200)
                st.markdown(f"[Buy on Amazon]({row['link']})", unsafe_allow_html=True)
        else:
            st.warning("No products found with the keyword.")

elif type == "Content-Based":
    search_query = st.text_input("Search for a product", "")

    if search_query:
    
        matched_products = fuzzy_search(search_query, df)
        if matched_products:
            product_list = matched_products
        else:
            product_list = []
            st.warning("No products found matching your search.")
    else:
        product_list = df['name'].values

    if len(product_list) > 0:
        selected_product = st.selectbox("Select a product", product_list)
    else:
        selected_product = None
        st.warning("Please search for a product first.")

    if selected_product and st.button("Recommend"):
        with st.spinner("Finding similar products..."):
            recommendations = content_based(df, selected_product)

        if not recommendations.empty:
            st.subheader("Recommended Products:")
            for index, row in recommendations.iterrows():
                st.markdown(f"### {row['name']}")
                st.markdown(f"**Ratings:** {row['ratings']}")
                st.markdown(f"**Number of Ratings:** {row['no_of_ratings']}")
                st.image(row['image_link'], width=400) 
                st.markdown(f"[Buy on Amazon]({row['link']})", unsafe_allow_html=True)
        else:
            st.error("No recommendations found. Try another product.")
