from cProfile import label
import pickle
import re
import nltk
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')

from PIL import Image

image = Image.open('movie.png')
st.image(image)

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def text_preprocessing(text):
    porter = PorterStemmer()
    stopword = stopwords.words('english')
    exclude = string.punctuation
    
    # remove url
    pattern = re.compile('<.*?>')
    text = pattern.sub(r'',text)
    
    # remove html tags
    pattern = re.compile(r'https?://\S+|www\.\S+')
    text = pattern.sub(r'',text)
    
    # remove Punctuation
    text = text.translate(str.maketrans('','',exclude))
    
    # Lowercasing Text
    text = text.lower()
    
    # Stemming and removing stopwords
    text = [porter.stem(word) for word in text.split() if word not in stopword]
    
    # return the Processed text
    return " ".join(text)
    #print(text)

# web app
def model():
    st.title("Sentiment Analysis of Movie Review")
    Movie_name = st.text_input("Movie Name", value="") 
    text = st.text_area("Review",value="")
    
    #if text is not None:
        #try:
            #text_bytes = text.read()
            #text_read = text_bytes.decode('utf-8')
        #except UnicodeDecodeError:
            #text_read = text_bytes.decode('latin-1')

    clean_text = text_preprocessing(text)
    clean_text = tfidf.transform([clean_text])
    prediction_id = clf.predict(clean_text)[0]
    
    category_mapping = {
            0:"Negative Review",
            1:"Positive Review"
            }
    category_name = category_mapping.get(prediction_id,"Unknown")

    if st.button('Predict Review'):
        if prediction_id == 1:
            #st.write("Movie Name : ", Movie_name)
            st.write(Movie_name, " got ",category_name,":thumbsup:")    
        else:
            #st.write("Movie Name : ", Movie_name)
            st.write( Movie_name, " got ",category_name,":thumbsdown:")
            
    else:
        st.header('No review')

# python main
if __name__ == "__main__":
    model()
