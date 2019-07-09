#!/usr/bin/env python
from flask import Flask, render_template, Response, request
import cv2
import sys
import numpy
import numpy as np
from cam_access import *
import time
import subprocess

haar_file = 'haarcascade_frontalface_default.xml'


app = Flask(__name__)


class Frame():
    def __init__(self,cam):
        self.cam = cam

    @app.route('/')
    def index():
        return render_template('index.html')


    def gen():
        i=1
        while i<10:
            yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+str(i)+b'\r\n')
            i+=1

    def get_frame(self):

        camera_port=0
        ramp_frames=100
        camera=cv2.VideoCapture(camera_port) #this makes a web cam object
        while True:
            retval, im = camera.read()
            self.cam = im
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            canvas = detect(gray, im)
            imgencode=cv2.imencode('.jpg',canvas)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'
                    b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        

        del(camera)

             
    def get_data(self,name):
            im = self.cam
            ret = datasets(name,im)
            return ret

val = Frame("data")


@app.route('/calc')
def calc():
        
     return Response(val.get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/data', methods=['POST'])
def data():
    if request.method=="POST":
        name = request.form['data']
        ret = val.get_data(name)
        if ret == "ok":
            return render_template('identifier.html')

@app.route('/identify/')
def identity():
    return render_template('identifier.html')
    

    
def gen():
    i=1
    while i<10:
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+str(i)+b'\r\n')
        i+=1



def get_identify():
    import time
    (width, height) = (130, 100)
    (images, lables), names = identifier()
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, lables)
    # Part 2: Use fisherRecognizer on camera stream
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)
    while True:

        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            # Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            if prediction[1]<90:
                cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                
                x = open("attent.txt","a")
                time1 = time.ctime()
                x.write(str(time1))
                x.write("\t")
                x.write(str(names[prediction[0]]))
                x.write("\n")
                x.close()

            else:                                                                               cv2.putText(im,'Not Recognize',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
         
        key=cv2.waitKey(10)
        if key == 27:
            break
        imgencode=cv2.imencode('.jpg',im)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
    del(camera) 
        
@app.route('/identifier')
def data_identify():
     return Response(get_identify(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
