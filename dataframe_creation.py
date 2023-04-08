import sys
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')
import requests as r
import bs4
from bs4 import BeautifulSoup as soup
import pandas as pd
import re
import nltk
import spacy
import en_core_web_sm
import pickle
import numpy as np


# Loading volume 1
url1 = 'https://www.gutenberg.org/cache/epub/42991/pg42991-images.html'
data1 = r.get(url1)
vol_one_data = soup(data1.text, 'html.parser')

## Loading volume 2
url2 = 'https://www.gutenberg.org/cache/epub/42992/pg42992-images.html'
data2 = r.get(url2)
vol_two_data = soup(data2.text, 'html.parser')

## Loading volume 3
url3 = 'https://www.gutenberg.org/cache/epub/42993/pg42993-images.html'
data3 = r.get(url3)
vol_three_data = soup(data3.text, 'html.parser')

## Loading volume 4
url4 = 'https://www.gutenberg.org/cache/epub/42994/pg42994-images.html'
data4 = r.get(url4)
vol_four_data = soup(data4.text, 'html.parser')

## Loading volume 5
url5 = 'https://www.gutenberg.org/cache/epub/42995/pg42995-images.html'
data5 = r.get(url5)
vol_five_data = soup(data5.text, 'html.parser')

## Loading volume 6
url6 = 'https://www.gutenberg.org/cache/epub/42996/pg42996-images.html'
data6 = r.get(url6)
vol_six_data = soup(data6.text, 'html.parser')

## Loading volume 7
url7 = 'https://www.gutenberg.org/cache/epub/42997/pg42997-images.html'
data7 = r.get(url7)
vol_seven_data = soup(data7.text, 'html.parser')

## Extracting required data
# function to add new paragraph to list of descriptions
def add_to_list(descr, list, str):
    if isinstance(descr, str):
        list.append(descr.strip())
    else:
        list.append(descr.get_text().strip()) 
# function to extract paragraphs
def get_Descriptions(descr):
    if descr.find_next_sibling('p') is not None and descr.find_next_sibling('p').find('b') is None:
        return descr.find_next_sibling('p')
    else:
        return None

# volume 1
# caste names
vol_one_headings = vol_one_data.body.find_all('b')
# print(len(vol_one_headings))
# caste descriptions
# Iterate over each heading element and extract its corresponding descriptions
vol_one_descriptions = []
# iterating over each heading element and extract its corresponding descriptions
for i in range(0,len(vol_one_headings)):
    descriptions = []
    next_paragraph = vol_one_headings[i].parent
    add_to_list(next_paragraph, descriptions, str)
    # grabbing each of the paragraphs until arriving at the next heading
    while True:
        next_paragraph = get_Descriptions(next_paragraph)
        if next_paragraph is not None:
            add_to_list(next_paragraph, descriptions, str)
        else:
            break
    # joining the collected descriptions into one string and appending it to the final list
    description_str = ' '.join(descriptions)
    description_str = '—'.join(description_str.split('—')[1:])
    description_str = re.sub('\r\n', ' ', description_str)
    description_str = re.sub('\[[^\]]*\]', '', description_str)
    vol_one_descriptions.append(description_str)
# print(len(vol_one_descriptions))

# volume 2
# caste names
vol_two_headings = vol_two_data.body.find_all('b')
# print(len(vol_two_headings))
# caste descriptions
# Iterate over each heading element and extract its corresponding descriptions
vol_two_descriptions = []
# iterating over each heading element and extract its corresponding descriptions
for i in range(0,len(vol_two_headings)):
    descriptions = []
    next_paragraph = vol_two_headings[i].parent
    add_to_list(next_paragraph, descriptions, str)
    # grabbing each of the paragraphs until arriving at the next heading
    while True:
        next_paragraph = get_Descriptions(next_paragraph)
        if next_paragraph is not None:
            add_to_list(next_paragraph, descriptions, str)
        else:
            break
    # joining the collected descriptions into one string and appending it to the final list
    description_str = ' '.join(descriptions)
    description_str = '—'.join(description_str.split('—')[1:])
    description_str = re.sub('\r\n', ' ', description_str)
    description_str = re.sub('\[[^\]]*\]', '', description_str)
    vol_two_descriptions.append(description_str)
# print(len(vol_two_descriptions))

# volume 3
# caste names
vol_three_headings = vol_three_data.body.find_all('b')
# print(len(vol_three_headings))
# caste descriptions
# Iterate over each heading element and extract its corresponding descriptions
vol_three_descriptions = []
# iterating over each heading element and extract its corresponding descriptions
for i in range(0,len(vol_three_headings)):
    descriptions = []
    next_paragraph = vol_three_headings[i].parent
    add_to_list(next_paragraph, descriptions, str)
    # grabbing each of the paragraphs until arriving at the next heading
    while True:
        next_paragraph = get_Descriptions(next_paragraph)
        if next_paragraph is not None:
            add_to_list(next_paragraph, descriptions, str)
        else:
            break
    # joining the collected descriptions into one string and appending it to the final list
    description_str = ' '.join(descriptions)
    description_str = '—'.join(description_str.split('—')[1:])
    description_str = re.sub('\r\n', ' ', description_str)
    description_str = re.sub('\[[^\]]*\]', '', description_str)
    vol_three_descriptions.append(description_str)
# print(len(vol_three_descriptions))

# volume 4
# caste names
vol_four_headings = vol_four_data.body.find_all('b')
# print(len(vol_four_headings))
# caste descriptions
# Iterate over each heading element and extract its corresponding descriptions
vol_four_descriptions = []
# iterating over each heading element and extract its corresponding descriptions
for i in range(0,len(vol_four_headings)):
    descriptions = []
    next_paragraph = vol_four_headings[i].parent
    add_to_list(next_paragraph, descriptions, str)
    # grabbing each of the paragraphs until arriving at the next heading
    while True:
        next_paragraph = get_Descriptions(next_paragraph)
        if next_paragraph is not None:
            add_to_list(next_paragraph, descriptions, str)
        else:
            break
    # joining the collected descriptions into one string and appending it to the final list
    description_str = ' '.join(descriptions)
    description_str = '—'.join(description_str.split('—')[1:])
    description_str = re.sub('\r\n', ' ', description_str)
    description_str = re.sub('\[[^\]]*\]', '', description_str)
    vol_four_descriptions.append(description_str)
# print(len(vol_four_descriptions))

# volume 5
# caste names
vol_five_headings = vol_five_data.body.find_all('b')
# print(len(vol_five_headings))
# caste descriptions
# Iterate over each heading element and extract its corresponding descriptions
vol_five_descriptions = []
# iterating over each heading element and extract its corresponding descriptions
for i in range(0,len(vol_five_headings)):
    descriptions = []
    next_paragraph = vol_five_headings[i].parent
    add_to_list(next_paragraph, descriptions, str)
    # grabbing each of the paragraphs until arriving at the next heading
    while True:
        next_paragraph = get_Descriptions(next_paragraph)
        if next_paragraph is not None:
            add_to_list(next_paragraph, descriptions, str)
        else:
            break
    # joining the collected descriptions into one string and appending it to the final list
    description_str = ' '.join(descriptions)
    description_str = '—'.join(description_str.split('—')[1:])
    description_str = re.sub('\r\n', ' ', description_str)
    description_str = re.sub('\[[^\]]*\]', '', description_str)
    vol_five_descriptions.append(description_str)
# print(len(vol_five_descriptions))

# volume 6
# caste names
vol_six_headings = vol_six_data.body.find_all('b')
# print(len(vol_six_headings))
# caste descriptions
# Iterate over each heading element and extract its corresponding descriptions
vol_six_descriptions = []
# iterating over each heading element and extract its corresponding descriptions
for i in range(0,len(vol_six_headings)):
    descriptions = []
    next_paragraph = vol_six_headings[i].parent
    add_to_list(next_paragraph, descriptions, str)
    # grabbing each of the paragraphs until arriving at the next heading
    while True:
        next_paragraph = get_Descriptions(next_paragraph)
        if next_paragraph is not None:
            add_to_list(next_paragraph, descriptions, str)
        else:
            break
    # joining the collected descriptions into one string and appending it to the final list
    description_str = ' '.join(descriptions)
    description_str = '—'.join(description_str.split('—')[1:])
    description_str = re.sub('\r\n', ' ', description_str)
    description_str = re.sub('\[[^\]]*\]', '', description_str)
    vol_six_descriptions.append(description_str)
# print(len(vol_six_descriptions))

# volume 7
# caste names
vol_seven_headings = vol_seven_data.body.find_all('b')
# print(len(vol_seven_headings))
# caste descriptions
# Iterate over each heading element and extract its corresponding descriptions
vol_seven_descriptions = []
# iterating over each heading element and extract its corresponding descriptions
for i in range(0,len(vol_seven_headings)):
    descriptions = []
    next_paragraph = vol_seven_headings[i].parent
    add_to_list(next_paragraph, descriptions, str)
    # grabbing each of the paragraphs until arriving at the next heading
    while True:
        next_paragraph = get_Descriptions(next_paragraph)
        if next_paragraph is not None:
            add_to_list(next_paragraph, descriptions, str)
        else:
            break
    # joining the collected descriptions into one string and appending it to the final list
    description_str = ' '.join(descriptions)
    description_str = '—'.join(description_str.split('—')[1:])
    description_str = re.sub('\r\n', ' ', description_str)
    description_str = re.sub('\[[^\]]*\]', '', description_str)
    vol_seven_descriptions.append(description_str)
# print(len(vol_seven_descriptions))

## Creating the final lists
# function to create the lists
def list_create(list1, list2):  
    list1.extend(list2)

# list of caste names
# function to strip tags and clean the headings
def clean_headings(list_raw, list_clean):
    for element in list_raw:
        add_to_list(element, list_clean, str)
        for i in range(0,len(list_clean)):
            list_clean[i] = re.sub('\.—', '', list_clean[i])
            list_clean[i] = re.sub('\.', '', list_clean[i])
vol_one_headings_cleaned = []
vol_two_headings_cleaned = []
vol_three_headings_cleaned = []
vol_four_headings_cleaned = []
vol_five_headings_cleaned = []
vol_six_headings_cleaned = []
vol_seven_headings_cleaned = []
clean_headings(vol_one_headings, vol_one_headings_cleaned)
clean_headings(vol_two_headings, vol_two_headings_cleaned)
clean_headings(vol_three_headings, vol_three_headings_cleaned)
clean_headings(vol_four_headings, vol_four_headings_cleaned)
clean_headings(vol_five_headings, vol_five_headings_cleaned)
clean_headings(vol_six_headings, vol_six_headings_cleaned)
clean_headings(vol_seven_headings, vol_seven_headings_cleaned)
castes_names = []
vol_one_headings_cleaned[0] = 'Abhishēka'
vol_two_headings_cleaned[0] = 'Canji'
vol_five_headings_cleaned[0] = 'Marakkāyar'
list_create(castes_names, vol_one_headings_cleaned)
list_create(castes_names, vol_two_headings_cleaned)
list_create(castes_names, vol_three_headings_cleaned)
list_create(castes_names, vol_four_headings_cleaned)
list_create(castes_names, vol_five_headings_cleaned)
list_create(castes_names, vol_six_headings_cleaned)
list_create(castes_names, vol_seven_headings_cleaned)
# print(len(castes_names))

# list of caste descriptions
castes_descriptions = []
list_create(castes_descriptions, vol_one_descriptions)
list_create(castes_descriptions, vol_two_descriptions)
list_create(castes_descriptions, vol_three_descriptions)
list_create(castes_descriptions, vol_four_descriptions)
list_create(castes_descriptions, vol_five_descriptions)
list_create(castes_descriptions, vol_six_descriptions)
list_create(castes_descriptions, vol_seven_descriptions)
# print(len(castes_descriptions))

## Creating the dataframe
column_names = ['Caste', 'Description']
df = pd.DataFrame(list(zip(castes_names, castes_descriptions)),
                  columns = column_names)
df['doc_id'] = df.index
# print(df)

## EDA
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder

# word count
word_count = []
for i in range(0, len(df)):
    description = df.loc[i]['Description']
    word_count.append(len(description.split()))
df['Word Count'] = word_count

# tokenizing the text
# function to tokenize
def tokenize_text(text: str):  
    # lowercase the text
    text = text.lower()
    # remove punctuation from text
    text = re.sub(r"[^\w\s]", "", text)
    # tokenize the text
    tokens = nltk.word_tokenize(text)
    # remove stopwords from txt_tokens and word_tokens
    from nltk.corpus import stopwords
    english_stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in english_stop_words]
    # return your tokens
    return tokens
# creating list of tokens for each description in dataframe
tokens = []
for i in range(0,len(df)):
    text = df.loc[i]['Description']
    tokens.append(tokenize_text(text = text))
# print(tokens)

# lemmatizing the tokens
# function to lemmatize
def lemmatize_tokens(tokens):
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # return your lemmatized tokens
    return lemmatized_tokens
# creating list of lemmatized tokens for each description in dataframe
lemmatized_tokens = []
for token in tokens:
    lemmatized_tokens.append(lemmatize_tokens(tokens = token))
# print(lemmatized_tokens)

# finding the most common tokens
# function to find the most common tokens
def return_top_tokens(tokens,
                      top_N = 10):
    # first, count the frequency of every unique token
    word_token_distribution = nltk.FreqDist(tokens)
    # next, filter for only the most common top_N tokens
    # also, put this in a dataframe
    top_tokens = pd.DataFrame(word_token_distribution.most_common(top_N),
                              columns=['Word', 'Frequency'])
    # return the top_tokens dataframe
    return top_tokens
# creating dataframe of top tokens for each description
top_tokens = []
for lemmatized_token in lemmatized_tokens:
    top_tokens.append(return_top_tokens(tokens = lemmatized_token,
                               top_N = 10))
# print(top_tokens)
df['Top Tokens'] = top_tokens

# finding the most common bigrams
# function to find the most common bi-grams
def return_top_bigrams(tokens,
                       top_N = 10):
    # collect bigrams
    bcf = BigramCollocationFinder.from_words(tokens)
    # put bigrams into a dataframe
    bigram_df = pd.DataFrame(data = bcf.ngram_fd.items(),
                             columns = ['Bigram', 'Frequency'])
    # sort the dataframe by frequency
    bigram_df = bigram_df.sort_values(by=['Frequency'],ascending = False).reset_index(drop=True)
    # filter for only top bigrams
    bigram_df = bigram_df[0:top_N]
    # return the bigram dataframe
    return bigram_df

# creating dataframe of most common bigrams
bigrams = []
for lemmatized_token in lemmatized_tokens:
    bigrams.append(return_top_bigrams(tokens = lemmatized_token,
                               top_N = 10))
# print(bigrams)
df['Common Bigrams'] = bigrams

# saving the dataframe as a CSV file
df.to_csv('castes_dataframe.csv')

## spaCy exploration
nlp = spacy.load("en_core_web_sm")
docs = list(nlp.pipe(df.Description))
## Pickling the list of docs
with open("docs.pkl", "wb") as descriptions_docs:
    pickle.dump(docs, descriptions_docs)
# print(docs[0])

def extract_tokens_plus_meta(doc:spacy.tokens.doc.Doc):
    """Extract tokens and metadata from individual spaCy doc."""
    tokens = []
    for i in doc:
        tokens.append([
            i.text, i.i, i.lemma_, i.ent_type_, i.tag_, 
            i.dep_, i.pos_, i.is_stop, i.is_alpha, 
            i.is_digit, i.is_punct
        ])
    return pd.DataFrame(tokens, columns=cols[1:])

cols = [
    "doc_id", "token", "token_order", "lemma", 
    "ent_type", "tag", "dep", "pos", "is_stop", 
    "is_alpha", "is_digit", "is_punct"
]

nlp_df = pd.DataFrame(columns=cols[1:])

def tidy_tokens(docs, df):
    for ix, doc in enumerate(docs):
        meta = extract_tokens_plus_meta(doc)
        df = df.append(meta.assign(doc_id=ix))
    return df.assign(doc_id=df.doc_id.astype(int)).loc[:, cols]

nlp_df = tidy_tokens(docs, nlp_df)

# converting the dataframe into an html table
html = nlp_df.to_html()
# write html to file
text_file = open("nlp_castes_descriptions.html", "w")
text_file.write(html)
text_file.close()
print(nlp_df)