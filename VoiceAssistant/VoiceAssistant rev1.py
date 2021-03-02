
import mysql.connector
import pandas as pd
import speech_recognition as sr
import * from text_to_speech.py
import os

db = mysql.connector.connect(host="", user="", password="", database="grocerymatedb")
cursor = db.cursor()
cursor.execute("SELECT * FROM shop_1")
result = cursor.fetchall()
df = pd.DataFrame(result, columns=['product', 'category', 'aisle'])
print(df)


cursor.execute("SELECT DISTINCT category, aisle from shop_1")
category_result = cursor.fetchall()
categories = [category[0] for category in category_result]


cursor.execute("SELECT product, aisle from shop_1")
product_result = cursor.fetchall()
products = [product[0] for product in product_result]


r = sr.Recognizer()

while True:
    with sr.Microphone() as source:
        audio = r.listen(source)
        print("audio heard")

    try:
        detected_audio = r.recognize_google(audio).lower()
        # detected_audio = "coca-cola"
        print("Google Speech Recognition thinks you said: " + '"' + detected_audio + '"')
        if detected_audio in categories:
            print(df.loc[df['category']==detected_audio])
            text = detected_audio + "could be found in aisle " + df.loc[df['category']==detected_audio]['aisle'].tolist()[0]
            print(text)
        elif detected_audio in products:
            print(df.loc[df['product']==detected_audio])
            text = detected_audio + "could be found in aisle: " + df.loc[df['product']==detected_audio]['aisle'].tolist()[0]
            print(text)
        elif detected_audio == "stop":
            text = "voice assistant shutting off"
            break
        else:
            text = "sorry, could not find " + detected_audio + " for you"
            print(text)
        speech = 'espeak -ven+f3 -k5 -s150 --punct="<characters>" "%s" 2>>/dev/null' % text
        execute_unix(speech)
    except sr.UnknownValueError:
        text = "Google Speech Recognition could not understand audio"
        print(text)
        speech = 'espeak -ven+f3 -k5 -s150 --punct="<characters>" "%s" 2>>/dev/null' % text
        execute_unix(speech)
    except sr.RequestError as e:
        text = "Could not request results from Google Speech Recognition service; {0}".format(e)
        speech = 'espeak -ven+f3 -k5 -s150 --punct="<characters>" "%s" 2>>/dev/null' % text
        execute_unix(speech)


