
import mysql.connector
import pandas as pd
import speech_recognition as sr
import subprocess
import os

def execute_unix(inputcommand):
   p = subprocess.Popen(inputcommand, stdout=subprocess.PIPE, shell=True)
   (output, err) = p.communicate()
   return output

db = mysql.connector.connect(host="grocerymate.ciihxnyiyqwe.ca-central-1.rds.amazonaws.com", user="", password="", database="grocerymatedb")
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
assistantOn = False

while True:
    with sr.Microphone() as source:
        audio = r.listen(source)
        print("audio heard")

    try:
        detected_audio = r.recognize_google(audio).lower()
        if detected_audio == "hamate" or detected_audio == "hey mate":
            assistantOn = True
            print("Google Speech Recognition thinks you said: " + "hey mate")
            print("entered into assistant mode")
            text = "hi, what are you looking for?"
        elif detected_audio in categories:
            print("Google Speech Recognition thinks you said: " + '"' + detected_audio + '"')
            print(df.loc[df['category']==detected_audio])
            text = detected_audio + " could be found in aisle " + str(df.loc[df['category']==detected_audio]['aisle'].tolist()[0]) + "."
            text += " Anything else?"
            print(text)
        elif detected_audio in products:
            print("Google Speech Recognition thinks you said: " + '"' + detected_audio + '"')
            print(df.loc[df['product']==detected_audio])
            text = detected_audio + " could be found in aisle " + str(df.loc[df['product']==detected_audio]['aisle'].tolist()[0]) + "."
            text += " Anything else?"
            print(text)
        elif detected_audio == "stop" or detected_audio == "thank you":
            print("Google Speech Recognition thinks you said: " + '"' + detected_audio + '"')
            text = "You are welcome."
            #break
        else:
            text = "sorry, could not find " + detected_audio + " for you. A staff will be with you shortly."
            print(text)
        speech = 'espeak -ven-us "%s" 2>>/dev/null' % text
        execute_unix(speech)
    except sr.UnknownValueError:
        text = "Google Speech Recognition could not understand audio"
        print(text)
        #speech = 'espeak -ven-us "%s" 2>>/dev/null' % text
        #execute_unix(speech)
    except sr.RequestError as e:
        text = "Could not request results from Google Speech Recognition service; {0}".format(e)
        #speech = 'espeak -ven-us "%s" 2>>/dev/null' % text
        #execute_unix(speech)
