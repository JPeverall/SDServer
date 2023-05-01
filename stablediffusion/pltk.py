import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.tokenize import blankline_tokenize

prompt = "From this point forward, you will act as the narrator for a first person, choose your own adventure style story. Narrate from the third person perspective, referring to the user as 'You'. The setting should be fantasy. It should include some scary bits, but be kid friendly. Limit each reply to a maximum of 100 words. Provide a choice of two or three options in this format: 'Your reply' \n 'Do you:' \n '-first choice' \n '-second choice', etc. Let the user type the response and interpret it."


# def splitPrompt(prompt):
#     newPrompt = prompt.split('\n')
#     print(newPrompt[1])

def extract_nouns(text):
    prompt_tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(prompt_tokens)
    prompt_nouns = [word for word, pos in tagged_tokens if pos in ("NN", "NNS", "NNP", "NNPS")]
    return prompt_nouns

# splitPrompt(prompt)

nouns = extract_nouns(prompt)
print(nouns)
