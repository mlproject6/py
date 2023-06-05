import streamlit as st
import pandas as pd
import pickle
import re
import string

st.set_page_config(
    page_title='News Detection',
    page_icon='newspaper.png',
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

model = pickle.load(open('DT.pkl','rb'))

tfidf = pickle.load(open('vectorizer.pkl','rb'))

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"


def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = tfidf.transform(new_x_test)
    pred_RFC = model.predict(new_xv_test)

    return output_label(pred_RFC[0])


def main():
    st.title("Fake News Detection")

    news = st.text_area("Enter the news", height=200)

    if st.button("Check"):
        if not news:
            st.warning("Please enter some news")
        else:
            result = manual_testing(news)
            if result == 'Fake News':
                st.error('Fake News')
            else:
                st.success('Real News')


if __name__ == '__main__':
    main()
