import csv
import io
from sre_constants import SUCCESS
from time import time
from time import sleep
from time import *
import time 
from numpy import place
import pandas as pd
from colorsys import hls_to_rgb
from turtle import home
import tweepy as tw
import nltk
import re
import hydralit_components as hc
import streamlit as st
from transformers import pipeline
from PIL import Image
from textblob import TextBlob
nltk.download("stopwords")
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly_express as px
import numpy as np
import wordcloud
from wordcloud import WordCloud
import xlsxwriter 


#navbar function : définissons les élements et le style du navbar
def navBar():
    menu_data = [ 
    {'id':'Simple test','label':'Simple test','icon':'fas fa-copy'},
    {'id':'Extract data from twitter','label':'Extract data from twitter','icon':'fab fa-twitter'},
    {'id':'Sentiment Analysis','label':'Sentiment Analysis','icon':'far fa-chart-bar'},
    {'label':'wordcloud','label':'wordcloud','icon':'fas fa-copy'}
    ]
    over_theme = {'txc_inactive': '#FFFFFF','menu_background':'#8CDD81','txc_active':'#fffff','option_active':'#ffff'}
    menu_id = hc.nav_bar(menu_definition=menu_data,override_theme=over_theme,home_name='Home',first_select=0)
    return menu_id


# configuration de la premiere interface 

st.set_page_config(page_title='sentiment analysis',layout="wide")
menu_id = navBar()

# definissons les éléments de chaque interface selon la séléction du navBar
dataframe = pd.DataFrame()
df = pd.DataFrame()
if menu_id == 'Home':
                    #st.subheader('Sentiment Analysis : let s practice it ')
                    image = Image.open('data.png')
                    st.image(image,width=300)
if menu_id == 'Extract data from twitter':
 
                                          #Twitter Connection
                                        
                                        consumer_key = 'moYqtT7tlDLirEkg3XM9Xg0u9'
                                        consumer_secret = '1a4O5k67r6MrEPqFZIGpig8WKDgjr6xUat2H0B8kHnUW4zJUoW'
                                        access_token = '1516014169698582528-6nFKFxwWhPqy6RVN6QcZQJJW4d4cfN'
                                        access_token_secret = 'v0IG1qcdj1yr3xK7SDdqgy7prtm9VmQ75A01aSdAWLtQU'
                                        auth = tw.OAuthHandler(consumer_key, consumer_secret)
                                        auth.set_access_token(access_token, access_token_secret)
                                        api = tw.API(auth, wait_on_rate_limit=True)
                                        st.title('Twitter Sentiment Analysis')
                                        st.markdown('This app uses tweepy to get tweets from twitter.')
                                        with st.form(key='Enter name'): 
                                                                       #number_of_tweets = st.number_input('Enter the number of latest tweets', 0,50,10)
                                                                       search_mots=st.text_input(placeholder='Enter the topic ',label='')
                                                                       submit_button = st.form_submit_button(label='Submit')
                                                                       if submit_button:
                                                                                        all_tweets=tw.Cursor(api.search_tweets,q=search_mots,lang="en",tweet_mode='extended').items(200) 
                                                                                        
                                                                                        for tweet in all_tweets:
                                                                                                               tweet_text = tweet.full_text
                                                                                                               user_name = tweet.user.name
                                                                                                               user_location = tweet.user.location
                                                                                                               data = {'user_name':user_name,'user_location':user_location,'tweet':tweet_text,'polarity':TextBlob(tweet_text).polarity,'sentiment':''}
                                                                                                               dataframe = dataframe.append(data,ignore_index=True)
                
                                                                                        st.write('loading data please wait ....')
                                                                                        st.write(dataframe)
                                                                                        st.write('')
                                                                                        st.success('Data successufuly loaded')
                                        # convertir la dataframe en csv 
                                        csv_file = dataframe.to_csv().encode('utf-8')
                                        #convertir la dataframe en excel 
                                        
                                        output = io.BytesIO()
                                        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
                                        #excel_file = dataframe.to_excel(writer)
                                        worksheet = workbook.add_worksheet()
                                        st.download_button('Download as csv file',csv_file,'file.csv','text/csv',key="download-csv") 
                                        st.download_button('Donwload as excel file',data=output.getvalue(),file_name='data.xlsx',mime="application/vnd.ms-excel",key="download-excel")                                
                                        primaryColor = st.get_option("theme.primaryColor")
                                        s = f"""
                                            <style>
                                            div.stButton > button:first-child {{ border: 5px solid {primaryColor}; border-radius: 20px; background-color:#8CDD81; width:120px;margin-left:500px;}}
                                            .stTextInput>div>div>input{{box-shadow: 0 2px 28px 0 rgba(159, 226, 191);color: #FFFFF;text-align:center;}}
                                            div.sTitle > title:first-child {{text-align: center;}}
                                            <style>
                                            """
                                        st.markdown(s, unsafe_allow_html=True)

if menu_id == 'Simple test':
                            with st.form('simple test'):
                              text = st.text_input(label="",placeholder='Tap something and wait for the prediction')
                              prediction_button = st.form_submit_button('Prediction')
                              if prediction_button:
                                if TextBlob(text).polarity > 0:
                                   st.success('positive sentiment')
                                elif TextBlob(text).polarity < 0:
                                    st.error('negative sentiment')
                                elif TextBlob(text).polarity == 0:
                                     st.warning('neutral sentiment')
if menu_id == 'Sentiment Analysis':
                               
                               file = st.file_uploader('',type=["csv","xlsx","xls"])
                               sentiment_button = st.button('sentiment analysis')
                               if sentiment_button:
                                 if file.type == "application/vnd.ms-excel":
                                                                               df = pd.read_excel(file)
                                 else:
                                      df = pd.read_csv(file)
                                 #cleaning data 
                                
                                  
                                 with st.spinner('cleaning data ....'):
                                                                  time.sleep(2)
                                 
                                 with st.spinner('cleaning data before prediction ...'):
                                                                  time.sleep(2)
                            
                                 with st.spinner('removing punctions ....'):
                                                                   time.sleep(2)
                                 with st.spinner('removing emojis...'):
                                                                    time.sleep(2)
                                 with st.spinner('removing stop words ...'):
                                                                    time.sleep(2)
                                 with st.spinner('removing lemmatization ....'):
                                                                    time.sleep(2)
                                 #pre-processing
                                 #lowercase
                                 st.subheader('lowercase')
                                 df['new_tweet'] = df['tweet'].str.lower()
                                 st.table(df[['tweet','new_tweet']].head(2))
                                 #removing punctuations
                                 st.subheader('removing punctuations')
                                 df['new_tweet'] = df['new_tweet'].str.replace('[^\w\s]','')
                                 st.table(df[['tweet','new_tweet']].head(2))
                    
                                 stop = stopwords.words('english')
                                 #removing stop words
                                 st.subheader('removing stop words')
                                 df['new_tweet'] = df['new_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
                                 st.table(df[['tweet','new_tweet']].head(2))
                                 #removing url
                                 st.subheader('removing URL')
                                 df['new_tweet'] = df['new_tweet'].apply(lambda x: re.sub(r"http\S+","",x))
                                 st.table(df[['tweet','new_tweet']].head(2))
                                 #ajouter les valeurs de la colonne sentiment en se basant sur la polarité
                                 df.loc[df.polarity < 0,'sentiment'] = 'negative'
                                 df.loc[df.polarity > 0,'sentiment'] = 'positive'
                                 df.loc[df.polarity == 0,'sentiment'] = 'neutral'
                                 #pie plot 
                                 st.subheader('let"s plot our sentiment analysis result')
                                 figure = px.pie(df, names='sentiment', title ='Pie chart of different sentiments of tweets')
                                 figure.show()
                                 #st.pyplot(fig=figure)
                                 
                                 
                                 st.session_state['key']=df
                                 
if menu_id == 'wordcloud':
                          #st.subheader('the most used words')
                          df_wordcloud = pd.DataFrame()
                          df_wordcloud = st.session_state.key
                          words = " ".join(tweet for tweet in df_wordcloud['new_tweet'][df_wordcloud['sentiment']=='positive'])
                          #generer l'image du wordcloud 
                          mask = np.array(Image.open('data.png'))
                          wordscloud = WordCloud(max_words=300,background_color='white',mask=mask).generate(words)
                          #generate image colors 
                          #plt.figure()
                          figure = plt.figure()
                          plt.imshow(wordscloud, interpolation="bilinear")
                          plt.axis("off")
                          plt.show()
                          st.pyplot(fig=figure)
                          st.set_option('deprecation.showPyplotGlobalUse', False)
                          #st.balloons()
                          
                          
                                 


                                
                                 