import streamlit as st
import random
import pandas as pd
import ast
st.set_page_config(page_title="NER using BERT", page_icon="🤗")

st.title("NER using BERT by HuggingFace🤗")
st.sidebar.markdown(" ## NER using HuggingFace🤗")
df = pd.read_csv("data/entity_bert.csv")

def extract_entities(entities):
    entities = ast.literal_eval(entities)  # Convert string representation to actual list of dictionaries
    extracted_entities = [(entity['entity'], entity['word'], entity['score']) for entity in entities]
    return extracted_entities

# Select a random index
random_index = random.randint(0, len(df) - 1)

# Extract entities for the random observation
random_observation = df.iloc[random_index]
extracted_entities = extract_entities(random_observation['entities'])

# Display content and entities using Streamlit
st.subheader("Random Observation")
st.subheader("Index:")
st.write(random_index)
st.subheader("Content:")
st.write(random_observation['content'])
st.subheader("Entities:")
for entity in extracted_entities:
    st.write("- Type:", entity[0])
    st.write("  Name:", entity[1])
    st.write("  Score:", entity[2])