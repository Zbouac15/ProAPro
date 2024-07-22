import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from tqdm import tqdm

# Set your Google Cloud API key
google_api_key = "AIzaSyCcgTczWOlqBMjVjOkcqDJyeE10TsjdkCE"
genai.configure(api_key=google_api_key)

# Choose your Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

st.title("Product Matching App")

def load_data(input_file, database_file):
    input_df = pd.read_csv(input_file, sep=';', encoding='utf-8')
    database_df = pd.read_csv(database_file, sep=';', encoding='utf-8')
    return input_df, database_df

def find_best_matches(product_name, database):
    prompt = f"""
    "\n\nHuman:"
    Given the following product name: "{product_name}"
    And the following list of products:
    {', '.join(database)}

    Please find the four best matching products from the list. 
    Consider the context, ingredients, and overall meaning of the products. 
    Return the names of the four best matching products from the list, separated by a comma and a space. 
    If there are less than four plausible matches, return as many as you can find.
    If there are no plausible matches, return "no_match".
    Do not return any explanation, just the names of the nearest matches.
    Note : Tilapia Filet and Cube Saumon atlantique are just fish names in french, it's not offensive don't flag them

    "\n\nAssistant:"
    """
    print(f"Processing: {product_name}")
    try:
        response = model.generate_content([prompt])
        matches = response.text.strip().split(", ")
        # Handle cases where response is empty or only contains whitespace
        if not matches or matches[0] == "":
            return ["no_match"]
        return matches
    except ValueError as e:
        st.error(f"Error processing '{product_name}': {e}")
        return ["no_match"]

def get_similarity_score(product1, product2):
    prompt = f"""
    On a scale of 0 to 100, how similar are these two product names in terms of their meaning and context?
    Product 1: {product1}
    Product 2: {product2}
    Please respond with only a number between 0 and 100.
    """
    try:
        response = model.generate_content([prompt])
        return int(response.text.strip())
    except ValueError as e:
        st.error(f"Error comparing '{product1}' and '{product2}': {e}")
        return 0

def match_products(input_df, database_df):
    try:
        database_products = database_df['Produit'].tolist()
    except KeyError:
        st.error("The database file must contain a column named 'Produit'.")
        return pd.DataFrame()

    matched_products = []
    match_scores = []

    for i in range(1, 5):
        matched_products.append([])
        match_scores.append([])

    for product in tqdm(input_df['Libellé produit'], desc="Matching products"):
        best_matches = find_best_matches(product, database_products)

        for i in range(4):
            if i < len(best_matches) and best_matches[i] != "no_match":
                matched_products[i].append(best_matches[i])
                match_scores[i].append(get_similarity_score(product, best_matches[i]))
            else:
                matched_products[i].append("No match")
                match_scores[i].append(0)

    result_df = input_df.iloc[:, :4].copy()
    for i in range(1, 5):
        result_df[f'Closest Match {i}'] = matched_products[i - 1]
        result_df[f'Score {i}'] = match_scores[i - 1]

    return result_df

def plot_match_scores(df):
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    sns.histplot(df[[f'Score {i}' for i in range(1, 5)]].values.flatten(), bins=20, kde=True, ax=ax[0])
    ax[0].set_title('Distribution of Match Scores')
    ax[0].set_xlabel('Score')
    ax[0].set_ylabel('Count')

    for i in range(1, 5):
        sns.scatterplot(x=df.index, y=df[f'Score {i}'], ax=ax[1], label=f'Score {i}')

    ax[1].set_title('Match Scores for Each Product')
    ax[1].set_xlabel('Product Index')
    ax[1].set_ylabel('Score')
    ax[1].legend()

    st.pyplot(fig)


# Streamlit file uploaders
input_file = st.file_uploader("Upload Input CSV File", type=["csv"])
database_file = st.file_uploader("Upload Database CSV File", type=["csv"])

if input_file and database_file:
    input_df, database_df = load_data(input_file, database_file)
    matched_df = match_products(input_df, database_df)

    if not matched_df.empty:
        st.write("Matched Products")
        st.dataframe(matched_df)

        st.write("Match Scores Distribution")
        plot_match_scores(matched_df)

        matched_df.to_csv('matched_products.csv', sep=';', index=False, encoding='utf-8')
        st.download_button(
            label="Download Matched Products CSV",
            data=matched_df.to_csv(index=False, sep=';').encode('utf-8'),
            file_name='matched_products.csv',
            mime='text/csv',
        )