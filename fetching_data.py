#!/usr/bin/env python
# coding: utf-8

# In[7]:


from googleapiclient.discovery import build
import pandas as pd
import seaborn as sns
import emoji as em
from bs4 import BeautifulSoup

import json
import re
from spellchecker import SpellChecker


# In[8]:


api_key = "AIzaSyC3WWD7ojDKj8jfOyENcuJLftcSdqfyRXQ"
#channel_id = "UC3SdeibuuvF-ganJesKyDVQ"
# channel_ids = ["UCaizTs-t-jXjj8H0-S3ATYA",
#                "UCxladMszXan-jfgzyeIMyvw",
#                "UCJQJAI7IjbLcpsjWdSzYz0Q",
#                "UCNU_lfiiWBdtULKOw6X0Dig"
#               ]


video_ids =  ["SwSbnmqk3zY","rBPQ5fg_kiY", "jXKGsMPk1Hg"]
anime_video_ids = ["goGqVSraxk8", "zKcPIAa8OUA", "LhxbuHG4SQo"]


youtube = build("youtube","v3", developerKey = api_key)


# In[10]:


def get_channel_comments(youtube, video_ids):
    comments = []

    for video_id in video_ids:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )
       
        
        while request is not None:
            response = request.execute()
            
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append([
                    comment["authorDisplayName"],
                    comment["publishedAt"],
                    
                    comment["updatedAt"],
                    comment["likeCount"],
                    
                    comment["textDisplay"]
                ])
                
            request = youtube.commentThreads().list_next(request, response)
    
    df = pd.DataFrame(comments, columns=["author", "published_at", "updated_at", "like_count", "text"])
    return df


df = get_channel_comments(youtube,video_ids)
df.head(20)


# In[11]:


# Data cleaning section

def remove_emojis(text):
    return em.replace_emoji(text, replace='')

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()


def clean_text(comment):
    # Remove paths starting with 'c:\' and onwards, along with subsequent text
    cleaned_comment = re.sub(r'c:\\.*', '', comment)
    
    # Remove portions inside {} brackets, [], and ()
    cleaned_comment = re.sub(r'[\{\[\(].*?[\}\]\)]', '', cleaned_comment)
    
    # Remove specific symbols: $, =, }, {, <, >, #, @
    cleaned_comment = re.sub(r'[\$\=\}\{\<\>\#@]', '', cleaned_comment)
    
    # Remove patterns like ????, !!!!, !!!????, and repeated characters like ...
    cleaned_comment = re.sub(r'(\?{2,15}|!{2,15}|\.{3,15})', '', cleaned_comment)
    
    # Remove HTML tags
    cleaned_comment = re.sub(r'<.*?>', '', cleaned_comment)
    
    # Remove non-alphanumeric characters, underscores, and extra spaces
    cleaned_comment = re.sub(r'[^\w\s]', '', cleaned_comment)
    cleaned_comment = re.sub(r'\s+', ' ', cleaned_comment).strip()
    
    return cleaned_comment.strip()

    




df['text'] = df['text'].str.strip() 
df['text'].fillna("", inplace=True) 
df = df[df['text'] != '']  
df["author"] = df["author"].str[1:]
df
df["published_at"] = df["published_at"].str[0:10]
df["updated_at"] = df["updated_at"].str[0:10]
df["text"] = df["text"].str.lower()
df["text"] = df["text"].apply(remove_emojis)
df["text"] = df["text"].apply(remove_html_tags)
df.tail(200)
df["text"] = df["text"].apply(clean_text)
df.drop_duplicates(subset=['text'], keep='first', inplace=True)





# In[13]:


df.head(20)

