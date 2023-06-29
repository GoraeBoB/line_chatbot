import os
import time
import numpy as np
import pandas as pd
import openai
import json

from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi
import urllib.request

def loadDoc(path):
    data = pd.DataFrame({'title' :[], 'comment': []})

    file_list=os.listdir(path)
    for file in file_list:

        text_list = []
        
        f = open(f'{path}/{file}')
        lines = f.read().splitlines()
        for line in lines:

            if line.startswith('장소명') :
                title = line.split(':')[1].strip()

            elif line.startswith('자세한 설명') | line.startswith('태그') | line.startswith("키워드"):
                text_list.append(line.split(':')[1].strip())

            else:
                pass

            text = ' '.join(text_list)
        f.close()

        data.loc[len(data)] = [title, text]
    return data

def tokenizer(sent, stopWord):

    mecab =Mecab()
    tokens = [word for word in mecab.morphs(str(sent)) if not word in stopWord]

    return tokens
def loadStopWord():
    file = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/NLP/stop_word.txt'), 'r')
    stopWord = []
    l = ''
    while True:
        l = file.readline()
        stopWord.append(l[:-1])
        if l == '':
            break
    file.close()
    return stopWord

def re_searchDoc(corpus, query):
    flag = 0
    stopWord = loadStopWord()
    tokenized_corpus = [tokenizer(doc, stopWord) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenizer(query, stopWord)
    doc_scores = bm25.get_scores(tokenized_query)
    second_idx = np.argsort((doc_scores))[-2]
    if doc_scores[second_idx] < 4:
        flag = 1
    return second_idx, flag

def searchDoc(corpus, query):
    stopWord = loadStopWord()
    tokenized_corpus = [tokenizer(doc, stopWord) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenizer(query, stopWord)
    doc_scores = bm25.get_scores(tokenized_query)
    best_idx = np.argmax(doc_scores)
    return best_idx

def getText(data, best_idx, path):
    title = data.iloc[best_idx][0]
    f = open(f'{path}/{title}.txt')
    text = f.read()
    f.close()
    return text

def detect_lang(sentence):
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Keys/papago_api_key.json'), 'r') as f:
        papago_api = json.load(f)
        
    print(papago_api)

    client_id = papago_api['client_id']
    client_secret = papago_api['client_secret']
    
    encQuery = urllib.parse.quote(sentence)
    data = "query=" + encQuery
    url = "https://openapi.naver.com/v1/papago/detectLangs"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        decode = json.loads(response_body.decode('utf-8'))
        trans_type = decode['langCode']
        # print(trans_type)
    else:
        print("Error Code:" + rescode)
    return trans_type

def translate(sentence, det_lang, flag=0):
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Keys/papago_api_key.json'), 'r') as f:
        papago_api = json.load(f)

    client_id = papago_api['client_id']
    client_secret = papago_api['client_secret']
    
    encText = urllib.parse.quote(sentence)
    if flag == 0: #질문 -> 한국어 번역
        data = f"source={det_lang}&target=ko&text={encText}"
    else: # 한국어 -> 사용자 언어로 번역
        data = f"source=ko&target={det_lang}&text={encText}"
        
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        decode = json.loads(response_body.decode('utf-8'))
        trans_query = decode['message']['result']['translatedText']
        # print(trans_query)
    else:
        print("Error Code:" + rescode)
    return trans_query
    
    
def find_topic(query, model = "gpt-3.5-turbo", temperature = 0, verbose = False):
    messages = [
        {
            "role" : "system",
            "content" : """You are artificial intelligence that categorizes sentences into 
            'Tourist Attraction Recommendations', 'Directions to Tourist Attractions' and 'Introduction of the Chatbot' 
            The answers are limited to 'Tourist Attraction Recommendations', 'Directions to Tourist Attractions' and 'Introduction of the Chatbot' 
            You must respond strictly with 'Tourist Attraction Recommendations', 'Directions to Tourist Attractions' and 'Introduction of the Chatbot'"""
                  
        },
        {
            "role" : "assistant",
            "content": f"'''{query}'''"
        }
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=512,
        top_p = 1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    topic = response['choices'][0]['message']['content'].strip()

    return topic

def introduce(conversation, model="gpt-3.5-turbo", temperature=0, verbose=False):
    system_role = f"""You are an AI language model named "해울이", an advisory chatbot that recommends tourist attractions in 울산광역시 in Republic of Korea. 
    You are introduced to the user as a chatbot that recommends tourist attractions in Ulsan City in Republic of Korea.
    Your conversation history is as follows:
        """ + conversation + """
    You must return answer in Korean. Please answer using polite and formal.
    """
    messages = [
        {"role": "system", "content": system_role},
    ]
    
    print(messages)

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=512,
        top_p = 1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    answer = response['choices'][0]['message']['content'].strip()                              
    
    return answer

def direction(query, conversation, model="gpt-3.5-turbo", temperature=0, verbose=False):
    system_role = f"""You are an AI language model named "해울이", an advisory chatbot that recommends tourist attractions in Ulsan City in Republic of Korea. 
    You must take the given embeddings and return a very detailed explanation of the document in the language of the query. 
    Find the destination in the given conversation or query and then provide the user with directions to reach that destination.
    Your conversation history is as follows:
        """ + conversation + """
    You must return answer related to the given context.
    You must return answer in Korean. Please answer using polite and formal.
    Return a accurate answer based on the document and conversation history.
    If you already explained the attraction, you must not explain it again.
    """
    assistant_role = f"""
    The query is as follows:
        """ + query + """
    """
    messages = [
        {"role": "system", "content": system_role},
        {"role": "assistant", "content": assistant_role},
    ]
    
    print(messages)

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=512,
        top_p = 1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    answer = response['choices'][0]['message']['content'].strip()                              
    
    return answer

def recommend(query, conversation, model="gpt-3.5-turbo", temperature=0, verbose=False):
    system_role = f"""You are an AI language model named "해울이", an advisory chatbot that recommends tourist attractions in Ulsan City in Republic of Korea. 
    You must take the given embeddings and return a very detailed explanation of the document in the language of the query. 
    Provide a description of the tourist attraction based solely on the given query, similar to TripAdvisor.
    Your conversation history is as follows:
        """ + conversation + """
    Do not provide directions to the tourist attractions.
    You must return answer related to the given context.
    You must return answer in Korean. Please answer using polite and formal.
    Return a accurate answer based on the document and conversation history.
    """
    assistant_role = f"""
    The query is as follows:
        """ + query + """
    """
    
    messages = [
        {"role": "system", "content": system_role},
        {"role": "assistant", "content": assistant_role},
    ]
    
    print(messages)

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=512,
        top_p = 1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    answer = response['choices'][0]['message']['content'].strip()                              
    
    return answer
