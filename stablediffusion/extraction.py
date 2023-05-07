import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.tokenize import blankline_tokenize
from nltk.corpus import wordnet as wn

prompt = ""


def splitPrompt(prompt):
    newPrompt = prompt.split('\n')
    # print(newPrompt[0])
    return newPrompt[0]


def is_descriptive(word):
    reference_words = ['forest', 'trees', 'cottage', 'witch', 'nose', 'glowing', 'eyes', 'teeth', 'wolves', 'fight', 'straight', 'crooked', 'woman', 'man', 'boy', 'girl', 'horse', 'dog', 'cat', 'house', 'rock', 'river', 'wind', 'door', 'moutain', 'sun', 'dark', 'cave', 'shadow', 'magic', 'dragon', 'monster', 'elf', 'light', 'landscape', 'foreboding', 'sword', 'dagger', 'cloak', 'armor', 'wizard', 'beast', 'desert', 'lake', 'pond', 'meadow', 'clearing', 'brush', 'bushes', 'statue', 'figure', 'creature']
    reference_synsets = [wn.synsets(ref_word) for ref_word in reference_words]

    word_synsets = wn.synsets(word)

    for word_synset in word_synsets:
        for ref_synset_list in reference_synsets:
            for ref_synset in ref_synset_list:
                similarity = word_synset.path_similarity(ref_synset)
                if similarity and similarity >= 0.27:
                    return True
    return False


def extract_nouns_adjectives(text):
    prompt_tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(prompt_tokens)
    prompt_nouns_adjectives = [word for word, pos in tagged_tokens if pos in ("NN", "NNS", "NNP", "JJS", "JJ", "NNPS") and is_descriptive(word)]
    return prompt_nouns_adjectives


paragraph = splitPrompt(prompt)
nouns_adjectives = extract_nouns_adjectives(paragraph)
print(nouns_adjectives)
