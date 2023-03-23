import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def convert(prediction):
    if prediction[0] == 1:
        print("The comment is toxic.")
    else:
        print("The comment is not toxic.")

test1 = "You fool"
convert(model.predict([test1]))

