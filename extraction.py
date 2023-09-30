import nltk  # Importing NLTK library
import nltk.corpus  # Importing the corpus module from NLTK, used for accessing built-in corpora
from nltk.tokenize import word_tokenize  # Importing the word_tokenize method to split input text into words
from nltk.tokenize import blankline_tokenize  # Importing the blankline_tokenize method to split input text at blank lines
from nltk.corpus import wordnet as wn  # Importing the WordNet interface

prompt = ""  # Declaring a string variable to hold the input text

# Function to split the input prompt at newlines and return the first line
def splitPrompt(prompt):
    newPrompt = prompt.split('\n')  # Splitting the input string at newline characters into a list
    return newPrompt[0]  # Returning the first element of the list

# Function to determine whether a word is descriptive, based on its similarity to a list of reference words
def is_descriptive(word):
    # Defining a list of reference words that are considered descriptive
    reference_words = ['forest', 'trees', 'cottage', 'witch', 'nose', 'glowing', 'eyes', 'teeth', 'wolves', 'fight',
                       'straight', 'crooked', 'woman', 'man', 'boy', 'girl', 'horse', 'dog', 'cat', 'house', 'rock',
                       'river', 'wind', 'door', 'moutain', 'sun', 'dark', 'cave', 'shadow', 'magic', 'dragon',
                       'monster', 'elf', 'light', 'landscape', 'foreboding', 'sword', 'dagger', 'cloak', 'armor',
                       'wizard', 'beast', 'desert', 'lake', 'pond', 'meadow', 'clearing', 'brush', 'bushes', 'statue',
                       'figure', 'creature']
    reference_synsets = [wn.synsets(ref_word) for ref_word in reference_words]  # Finding synsets for each reference word
    
    word_synsets = wn.synsets(word)  # Finding synsets for the input word
    
    # Looping through each synset of the input word
    for word_synset in word_synsets:
        # Looping through each synset list of reference words
        for ref_synset_list in reference_synsets:
            # Looping through each synset in the synset list
            for ref_synset in ref_synset_list:
                similarity = word_synset.path_similarity(ref_synset)  # Calculating path similarity between synsets
                # If there is a similarity and it's greater than or equal to 0.27, return True
                if similarity and similarity >= 0.27:
                    return True
    return False  # If no suitable similarity is found, return False

# Function to extract nouns and adjectives that are descriptive from a given text
def extract_nouns_adjectives(text):
    prompt_tokens = word_tokenize(text)  # Tokenizing the input text into words
    tagged_tokens = nltk.pos_tag(prompt_tokens)  # Tagging the tokenized words with their parts of speech
    # Extracting words that are nouns or adjectives and are descriptive, based on the is_descriptive function
    prompt_nouns_adjectives = [word for word, pos in tagged_tokens if pos in ("NN", "NNS", "NNP", "JJS", "JJ", "NNPS") and is_descriptive(word)]
    return prompt_nouns_adjectives  # Returning the extracted words

# Extracting the first line from the input prompt
paragraph = splitPrompt(prompt)

# Extracting descriptive nouns and adjectives from the extracted line
nouns_adjectives = extract_nouns_adjectives(paragraph)

# Printing the extracted descriptive nouns and adjectives
print(nouns_adjectives)
