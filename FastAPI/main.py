from fastapi import FastAPI, Request, HTTPException

import os
import numpy as np
import pandas as pd
import openai
import json
import time

from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi
from pydantic import BaseModel
from modules.util import loadDoc, getText, re_searchDoc, searchDoc, find_topic, recommend, introduce, direction, detect_lang, translate

USER_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Database")
information_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Place")
app = FastAPI()

class Message(BaseModel):
    message: str
    userID: str

@app.post("/")
async def receive_message(json_data: Message):
    init_time = time.time()
    message = json_data.message
    userID = json_data.userID
    
    print(message)
    try:
        conversation_log = pd.read_csv(os.path.join(USER_DATA_PATH, userID, "chat_log.csv"))
        
    except:
        os.makedirs(os.path.join(USER_DATA_PATH, userID))
        pd.DataFrame(columns=["sender", "message", "time"]).to_csv(os.path.join(USER_DATA_PATH, userID, "chat_log.csv"), index=False)
        conversation_log = pd.read_csv(os.path.join(USER_DATA_PATH, userID, "chat_log.csv"))
    conversation_log = pd.concat([conversation_log, pd.DataFrame([[userID, message, time.time()]], columns=["sender", "message", "time"])])
    
    #최근 3개의 대화를 불러옴.
    if len(conversation_log) > 3:
        conversation_log = conversation_log.iloc[-3:]
    else:
        conversation_log = conversation_log.iloc[:]
        
    #보낸 이 : 메시지 형태로 string을 만듦.
    message_log = ""
    for i in range(len(conversation_log)):
        message_log += conversation_log.iloc[i, 0] + " : " + conversation_log.iloc[i, 1] + "\n"
        
    det_lang = detect_lang(message)
    if det_lang != "ko":
        message = translate(message, det_lang, 0)

    
    
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Keys", "openai_api_key.json"), "r") as f:
        open_api_key = json.load(f)

    openai.api_key = open_api_key["key"]
    path = "/DaiS_internship/Chatbot/new_data"

    topic = find_topic(message)
    
    if topic == "Introduction of the Chatbot":
        message = introduce(message)
        
    elif topic == "Directions to Tourist Attractions":
        information_data = loadDoc(information_path)
        corpus = information_data.iloc[:, 1]
        best_idx = searchDoc(corpus, message_log)
        text = getText(information_data, best_idx, information_path)
        message = direction(text, message_log)
        
                            
    elif topic == "Tourist Attraction Recommendations":
        information_data = loadDoc(information_path)
        corpus = information_data.iloc[:, 1]
        best_idx = searchDoc(corpus, message)
        text = getText(information_data, best_idx, information_path)
        message = direction(text, message_log)
        
    else:
        message = introduce(message)
    
    if det_lang != "ko":
        message = translate(message, det_lang, 1)
    
    end_time = time.time()
    print("time : ", end_time - init_time)
    conversation_log = pd.concat([conversation_log, pd.DataFrame([["해울이", message, time.time()]], columns=["sender", "message", "time"])])
    conversation_log.to_csv(os.path.join(USER_DATA_PATH, userID, "chat_log.csv"), index=False)
    return message