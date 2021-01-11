
import mysql.connector
import pandas as pd
import speech_recognition as sr
import os

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

while True:
    with sr.Microphone() as source:
        audio = r.listen(source)
        print("audio heard")

    try:
        '''
        for testing purposes, we're just using the default API key
        to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        instead of `r.recognize_google(audio)`
        '''
        
        detected_audio = r.recognize_google(audio).lower()
        # detected_audio = "coca-cola"
        print("Google Speech Recognition thinks you said: " + '"' + detected_audio + '"')
        if detected_audio in categories:
            print(df.loc[df['category']==detected_audio])
            print(detected_audio, "could be found in aisle: ", df.loc[df['category']==detected_audio]['aisle'].tolist()[0])
            
        elif detected_audio in products:
            print(df.loc[df['product']==detected_audio])
            print(detected_audio, "could be found in aisle: ", df.loc[df['product']==detected_audio]['aisle'].tolist()[0])
        elif detected_audio == "stop":
            break
        else:
            print("sorry, could not find " + detected_audio + " for you")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


