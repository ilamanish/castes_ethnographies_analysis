import spacy
from spacy import displacy
import pandas as pd 
import en_core_web_sm
import matplotlib.pyplot as plt
import pickle

## Loading the pickled list of docs
with open("data/docs.pkl", "rb") as descriptions_docs:
    docs = pickle.load(descriptions_docs)
print(docs[0])

# Loading the second dataframe
nlp_df = pd.read_csv('data/nlp_castes_descriptions.csv')
print(nlp_df.iloc[0])

# spaCy exploration
# Counting the number of occurrences of each type of entity
print(nlp_df.query("ent_type != ''").ent_type.value_counts())
# Counting the number of occurrences of the top 10 words
nlp_df.query("is_stop == False & is_punct == False").lemma.value_counts().head(10).plot(kind="barh", figsize=(24, 14), alpha=.7)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20);
plt.show()

# Visualising random dependency structures
options = {"compact": True, "color": "blue", "font": "Garamond"}
displacy.serve(docs[2000], style="dep", options = options, auto_select_port = True)