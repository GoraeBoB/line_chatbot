## -*- coding: utf-8 -*-
import os

from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import base64
import hashlib
import hmac
import json
import urllib3


LINE_CHANNEL_SECRET = os.environ['LINE_CHANNEL_SECRET']
LINE_CHANNEL_ACCESS_TOKEN = os.environ['LINE_CHANNEL_ACCESS_TOKEN']
SERVER_URL = os.environ['SERVER_URL']

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
CONTENT_HEADER = "Content-Type"
CONTENT_HEADER_VALUE = "application/json"



def lambda_handler(event, _):
    if "body" in event:
        signature = event["headers"]['x-line-signature']
        body = event["body"]
        
        hash = hmac.new(LINE_CHANNEL_SECRET.encode('utf-8'), body.encode('utf-8'), hashlib.sha256).digest()
        signature_compare = base64.b64encode(hash).decode('utf-8')
        
        print(signature, signature_compare)
        if signature != signature_compare:
            return {"statusCode": 400, "body": "Invalid signature. Please check your channel access token/channel secret."}
        
        print("signature test passed.")
        
       
        handler.handle(body, signature)
            
        
            
        
        
    return {"statusCode": 200, "body": "OK"}




@handler.add(MessageEvent, message=(TextMessage))
def handling_message(event):
    replyToken = event.reply_token

    if isinstance(event.message, TextMessage):
        messages = event.message.text
        
        """
        messages : 사용자가 입력한 메시지. String.
        response : ChatGPT로부터 받은 답변. String.
        """
        
        print("messages : ", messages)

        json_data = {
            "message": messages
        }
        http = urllib3.PoolManager()
        response = http.request('POST', "", headers={CONTENT_HEADER: CONTENT_HEADER_VALUE}, body=json.dumps(json_data))
        response = response.data.decode('utf-8')
        print("response : ", response)
        response = response[1:-1]
        response_message = TextSendMessage(text=response)
        print("sending message...")
        line_bot_api.reply_message(reply_token=replyToken, messages=response_message)