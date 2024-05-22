import streamlit as st
import random
import pandas as pd
import ast
st.set_page_config(page_title="NER using BERT", page_icon="🤗")

st.title("NER using BERT by HuggingFace🤗")

df = pd.read_csv("data/entity_bert.csv")
df.drop(columns=['category'], inplace=True)

st.markdown("We are using [this](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english) model which is a pre-trained BERT model fine-tuned specifically for Named Entity Recognition tasks on the CoNLL-2003 dataset.")

st.markdown(f"We are using [display_data.csv](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/display_data.csv) to train BERT. It has been derived from [master_data.csv](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/master_data.csv).")

st.markdown(f"For all the detailed steps on how to do NER using BERT from HuggingFace🤗, you can look at [NER_BERT.ipynb](https://github.com/jaideep156/ner-topicmodeling/blob/main/notebook/NER_BERT.ipynb).")

def extract_entities(entities):
    entities = ast.literal_eval(entities)  # Convert string representation to actual list of dictionaries
    extracted_entities = [(entity['entity'], entity['word'], entity['score']) for entity in entities]
    return extracted_entities

st.subheader("10 random observations from output after training:")
st.dataframe(df.sample(10)) 

random_index = random.randint(0, len(df) - 1)
random_observation = df.iloc[random_index]
extracted_entities = extract_entities(random_observation['entities'])

st.subheader(f"Displaying a random observation from the above output from row {random_index}")
st.write(random_observation['content'])
st.subheader("Below are its entities:")
for entity in extracted_entities:
    st.write("- Type:", entity[0])
    st.write("  Name:", entity[1])
    st.write("  Score:", entity[2])