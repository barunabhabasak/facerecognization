
from tkinter import *
from PIL import Image, ImageTk
from subprocess import *
import cv2                        
import numpy as np                
from os import listdir            
from os.path import isfile,join
import pyttsx3
import os
import shutil
import tkinter.messagebox as tmsg

root = Tk()

root.geometry("500x700")
root.resizable(0,0)

icon = ImageTk.PhotoImage(Image.open("D:\sem6proj\Face-Recognition-Project-master\\fav2.png"))
root.wm_iconphoto(False, icon)

root.title("Face Recognition")

k = pyttsx3.init()
sound = k.getProperty('voices')
k.setProperty('voice',sound[1].id)
k.setProperty('rate',130)
k.setProperty('pitch',200)


def speak(text):
    k.say(text)
    k.runAndWait()

def delete() :
    with os.scandir("D:\\sem6proj\\Face-Recognition-Project-master\\\sample\\") as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
    f = open('demo.txt', 'r+')
    f.truncate(0)
    f.close()
    a = tmsg.showinfo("Message","Data has been cleared")

def sampling() :
    
    face_classifier = cv2.CascadeClassifier("C:\Python310\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

    def face_extractor(img):

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is ():
            return None

        for(x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]

        return cropped_face



    cap = cv2.VideoCapture(1)

    count = 0
    try :
        f = open("demo.txt", "r")
        appcount = int(f.read())
        f.close()
    except :
        appcount = 0


    while True:
        ret,frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            appcount += 1
            face = cv2.resize(face_extractor(frame),(200,200))
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

            file_name_path = "D:\\sem6proj\\Face-Recognition-Project-master\\sample\\user" + str(appcount)+'.jpg'
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)

        else:
            print("Face not found !!")
            pass

        if cv2.waitKey(1) == 13 or count == 100:
            break

    f = open("demo.txt", "w")
    f.write(f"{appcount}")
    f.close()   

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples done....")
    while True :
        proc = Popen("train.py", stdout=PIPE, shell=True)
        proc = proc.communicate()
        output.insert(END, proc)
        # bool=False
        speak("congratulations, the model has been trained")
        break
    speak("click next button for face recognition")
        



def recognition() :

    try :
        data_path = "D:\\sem6proj\\Face-Recognition-Project-master\\sample\\"
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

        Training_Data,Labels = [],[]

        for i,files in enumerate(onlyfiles):
                image_path = data_path + onlyfiles[i]
                images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                Training_Data.append(np.asarray(images,dtype=np.uint8))
                Labels.append(i)

        Labels = np.asarray(Labels,dtype=np.int32)

        model = cv2.face.LBPHFaceRecognizer_create()

        model.train(np.asarray(Training_Data),np.asarray(Labels))
        print("Congratulations model is TRAINED ... *_*...")

        face_classifier = cv2.CascadeClassifier("C:\Python310\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

        def face_detector(img,size = 0.5):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)

            if faces == ():
                return img,[]

            for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
                roi = img[y:y+h,x:x+w]
                roi = cv2.resize(roi,(200,200))

            return img,roi

        cap = cv2.VideoCapture(1)
        while True:
            ret,frame = cap.read()
            image , face = face_detector(frame)


            try:
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                result = model.predict(face)

                if result[1] < 500:
                    Confidence = int(100 * (1 - (result[1])/300))


                if Confidence > 82:
                #speak2("authentication successful")
                    display_string = 'Congrats, it''s '+ str(Confidence)+' % Matched'
                    cv2.putText(image,display_string,(70,70),cv2.FONT_HERSHEY_COMPLEX,1,(213, 36, 17),2)
                    cv2.putText(image, "AUTHENTICATION SUCCESSFUL", (100, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (7, 215, 215), 2)
                    cv2.imshow("Face Cropper",image)
            

                else:
                    cv2.putText(image, "CAN'T RECOGNISE", (200, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                    cv2.imshow("Face Cropper", image)
                #speak("I can't recognize you")

            except:
                # speak("can't detect your face")
                cv2.putText(image, "PLEASE SHOW YOUR FACE", (100, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (213, 36, 17), 2)

                cv2.imshow("Face Cropper", image)
                pass
            if cv2.waitKey(1) == 13:
                break
        cap.release()
        cv2.destroyAllWindows()
    except :
        b = tmsg.showerror("Error Message", "I don't have any training data")

    


# def sample_training() :

#     speak("Starting the process")

    
    
 
        
        

bg = ImageTk.PhotoImage(Image.open("D:\\sem6proj\\Face-Recognition-Project-master\\gradient2.png"))
  
label1 = Label( root, image = bg)
label1.place(x = -12, y = 0)

f1 = Frame(root)
f1.pack(side=TOP, fill="x")

greet = Label(f1, text="Face Recognition System", bg="red", fg="white", font=("arial", 20))
greet.pack()

f2 = Frame()
f2.pack()


bt3 = Button(root, text="Exit", command=root.quit).pack(side=BOTTOM)

output = Text(root, width=40, height=4)
output.pack(side=BOTTOM)


l1 = Label (f2, text = "NOTE : Your face data will be collected", pady="2").grid(row="5", column="10")
bt = Button(root, text = "Click Here To Start",fg="blue",font=("georgia", 10), pady="10", command=sampling)
bt.place(x=190, y=500)

btn = Button(root, text="Clear Data", command=delete)
btn.place(x=215, y=65)


bt2 = Button(text="Next", command=recognition).pack(side=BOTTOM)
    
    

root.mainloop()
