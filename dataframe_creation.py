## CASTES AND TRIBES OF SOUTH INDIA (EDGAR THURSTON AND K RANGACHARI)

# importing required libraries
import sys
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')
import requests as r
from bs4 import BeautifulSoup as soup
import pandas as pd
import re
import nltk
import spacy
import pickle

# function to download data from the html file
def create_corpus(url):
    data = r.get(url)
    vol = soup(data.text, 'html.parser')
    return vol

# loading the volumes
vol_one_data = create_corpus('https://www.gutenberg.org/cache/epub/42991/pg42991-images.html')
vol_two_data = create_corpus('https://www.gutenberg.org/cache/epub/42992/pg42992-images.html')
vol_three_data = create_corpus('https://www.gutenberg.org/cache/epub/42993/pg42993-images.html')
vol_four_data = create_corpus('https://www.gutenberg.org/cache/epub/42994/pg42994-images.html')
vol_five_data = create_corpus('https://www.gutenberg.org/cache/epub/42995/pg42995-images.html')
vol_six_data = create_corpus('https://www.gutenberg.org/cache/epub/42996/pg42996-images.html')
vol_seven_data = create_corpus('https://www.gutenberg.org/cache/epub/42997/pg42997-images.html')

# functions to extract required data
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
    
# function to create table of headings
def tabulate_headings(vol):
    headings = vol.body.find_all('b')
    list_clean = []
    for element in headings:
        add_to_list(element, list_clean, str)
        for i in range(0,len(list_clean)):
            list_clean[i] = re.sub('\.—', '', list_clean[i])
            list_clean[i] = re.sub('\.', '', list_clean[i])
    return list_clean

# creating table of headings for each volume
vol_one_headings = tabulate_headings(vol_one_data)
vol_two_headings = tabulate_headings(vol_two_data)
vol_three_headings = tabulate_headings(vol_three_data)
vol_four_headings = tabulate_headings(vol_four_data)
vol_five_headings = tabulate_headings(vol_five_data)
vol_six_headings = tabulate_headings(vol_six_data)
vol_seven_headings = tabulate_headings(vol_seven_data)
vol_one_headings[0] = 'Abhishēka'
vol_two_headings[0] = 'Canji'
vol_five_headings[0] = 'Marakkāyar'

# creating final table of caste names
castes_names = []
castes_names.extend(vol_one_headings)
castes_names.extend(vol_two_headings)
castes_names.extend(vol_three_headings)
castes_names.extend(vol_four_headings)
castes_names.extend(vol_five_headings)
castes_names.extend(vol_six_headings)
castes_names.extend(vol_seven_headings)
print(len(castes_names))

# function to create table of descriptions
def tabulate_descriptions(vol, main_table):
    vol_headings = vol.body.find_all('b')
    vol_descriptions = []
    # iterating over each heading element and extracting its corresponding descriptions
    for i in range(0, len(vol_headings)):
        descriptions = []
        next_paragraph = vol_headings[i].parent
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
        vol_descriptions.append(description_str)
    main_table.extend(vol_descriptions)

# creating final table of descriptions
castes_descriptions = []
tabulate_descriptions(vol_one_data, castes_descriptions)
tabulate_descriptions(vol_two_data, castes_descriptions)
tabulate_descriptions(vol_three_data, castes_descriptions)
tabulate_descriptions(vol_four_data, castes_descriptions)
tabulate_descriptions(vol_five_data, castes_descriptions)
tabulate_descriptions(vol_six_data, castes_descriptions)
tabulate_descriptions(vol_seven_data, castes_descriptions)
print(len(castes_descriptions))

# creating the dataframe
column_names = ['Caste', 'Description']
df = pd.DataFrame(list(zip(castes_names, castes_descriptions)),
                  columns = column_names)
df['doc_id'] = df.index
# print(df)

# EDA
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

# function to tokenize the text
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

# function to lemmatize the tokens
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

# function to find the most common tokens
def return_top_tokens(tokens,
                      top_N = 10):
    # count the frequency of every unique token
    word_token_distribution = nltk.FreqDist(tokens)
    # filter for only the most common top_N tokens
    # put this in a dataframe
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


# spaCy exploration
nlp = spacy.load("en_core_web_sm")

# extracting the docs from the descriptions
docs = list(nlp.pipe(df.Description))

# pickling the list of docs
with open("docs.pkl", "wb") as descriptions_docs:
    pickle.dump(docs, descriptions_docs)
# print(docs[0])

# function to extract tokens and metadata from individual spaCy doc.
def extract_tokens_plus_meta(doc:spacy.tokens.doc.Doc):
    tokens = []
    for i in doc:
        tokens.append([
            i.text, i.i, i.lemma_, i.ent_type_, i.tag_, 
            i.dep_, i.pos_, i.is_stop, i.is_alpha, 
            i.is_digit, i.is_punct
        ])
    return pd.DataFrame(tokens, columns=cols[1:])

# function to add doc entities and tokens to a dataframe
def tidy_tokens(docs, df):
    for ix, doc in enumerate(docs):
        meta = extract_tokens_plus_meta(doc)
        df = df.append(meta.assign(doc_id=ix))
    return df.assign(doc_id=df.doc_id.astype(int)).loc[:, cols]

# creating a dataframe for the doc entities and their tokens
cols = [
    "doc_id", "token", "token_order", "lemma", 
    "ent_type", "tag", "dep", "pos", "is_stop", 
    "is_alpha", "is_digit", "is_punct"
]
nlp_df = pd.DataFrame(columns=cols[1:])

# extracting and adding the doc entities and tokens to the created dataframe
nlp_df = tidy_tokens(docs, nlp_df)

# converting the dataframe into an html table
html = nlp_df.to_html()
# write html to file
text_file = open("nlp_castes_descriptions.html", "w")
text_file.write(html)
text_file.close()
# print(nlp_df)

## SAMUEL MATEER

# importing the necessary libraries

# function to clean a text file of line breaks, page numbers, and chapter names
def clean_txt_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Remove line breaks and whitespaces from the beginning and end of the line
            cleaned_line = line.strip()
            
            # Check if the cleaned line contains only numbers (page numbers)
            if cleaned_line.isdigit():
                continue
            
            # Check if the cleaned line contains all content in capital letters (chapter names)
            if cleaned_line.isupper():
                continue
            
            # If the cleaned line doesn't meet the criteria, write it to the output file
            outfile.write(cleaned_line + ' ')

# The Gospel in South India:
input_file_path = './data/the_gospel_in_south_india.txt'
output_file_path = './data/cleaned_the_gospel_in_south_india.txt'
clean_txt_file(input_file_path, output_file_path)

# The Land of Charity
input_file_path = './data/the_land_of_charity.txt'
output_file_path = './data/cleaned_the_land_of_charity.txt'
clean_txt_file(input_file_path, output_file_path)