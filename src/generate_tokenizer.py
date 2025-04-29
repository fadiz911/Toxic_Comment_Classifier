import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Constants
MAX_NUM_WORDS = 100000

# Load the training data
print("Loading training data...")
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Fill missing values
train['comment_text'] = train['comment_text'].fillna('unknown')
test['comment_text'] = test['comment_text'].fillna('unknown')

# Create and fit tokenizer
print("Creating and fitting tokenizer...")
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(list(train['comment_text']) + list(test['comment_text']))

# Save the tokenizer
print("Saving tokenizer...")
with open('../tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved successfully!")