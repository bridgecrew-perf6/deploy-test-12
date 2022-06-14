import nltk
import nltk
import re

from sklearn.metrics import classification_report
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import string
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer 
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

def prepare_data(str_input):
    print('Data cleaning in progress...')
    # Tokenize : dividing Sentences into words
    tokens = nltk.word_tokenize(str_input)
    print('Tokenization complete.')
    # Remove stop words
    filtered_tokens = [t for t in tokens if not t in stopwords.words("english")]
    filtered_text = " ".join(filtered_tokens)
    print('Stop words removed.')
    # remove numbers
    text_nonum = re.sub(r'\d+', '', filtered_text)
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation]) 
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    print('Numbers, punctuation and special characters removed.')
    print("Clean Input is: " + text_no_doublespace)
    return text_no_doublespace