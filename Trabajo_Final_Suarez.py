import cv2 as cv
import numpy as np
import os
import sys
import math
import pdb


#-----bibliografia analizada-----
#https://docs.opencv.org/3.3.0/
#https://www.vision.uji.es/courses/Doctorado/TAVC/TAVC-flujo-optico.pdf
#https://medium.com/@jonathan_hui/self-driving-object-tracking-intuition-and-the-math-behind-kalman-filter-657d11dd0a90
#--------------------
def buscar_rgb(img):
    data = np.reshape(img, (-1,3))
    print(data.shape)
    data = np.float32(data)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv.kmeans(data,1,None,criteria,10,flags)

    print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
    return(centers[0].astype(np.int32))

#--------------------
def seleccion(frame,recorte,situacion,h1,s1,v1):    
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    min=0
    max=0
    if (situacion==1):
        h=0
        s=0
        v=0
        color_predominante=buscar_rgb(recorte)
        color_pred=np.uint8([[color_predominante]])
        color_pred_hsv=cv.cvtColor(color_pred,cv.COLOR_BGR2HSV)
        h=color_pred_hsv[0][0][0]
        s=color_pred_hsv[0][0][1]
        v=color_pred_hsv[0][0][2]
        valor=30

    else:
        h=h1
        s=s1
        v=v1
        valor=50
    #pdb.set_trace()
    lwr=np.array([h-2,s-valor,v-valor])
    upr=np.array([h+2,s+valor,v+valor])
    mask= cv.inRange(hsv,lwr,upr)
    if(situacion==1):
        while(cv.waitKey(5)&0xFF!=ord('f')):
            cv.imshow('Seleccion',mask)
    cv.destroyWindow('Seleccion')
    return(mask,h,s,v)
#----------------------------------------------
#necesario realizar esto en otro entorno porque sino no anda max
#no sacar imagenes en h s v en este entorno
def contornos(mask):
    umb=cv.adaptiveThreshold(mask,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    canny=cv.Canny(umb,100,200)
    canny=cv.GaussianBlur(canny,(5,5),5)
    _,con,_=cv.findContours(canny, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
    if con:
        vmax=max(con, key = cv.contourArea)
        momentos = cv.moments(vmax)
        try:
            cx=float(momentos['m10']/momentos['m00'])
            cy=float(momentos['m01']/momentos['m00'])            
            #buscaste el bounding box, pero generalmente no arroja resultados satifactorios
        except ZeroDivisionError as zr:
            print("Division por cero")
            punto_elegido=np.array([[[641,481]]],np.float32)
            #sys.exit()
            return(punto_elegido)
        punto_elegido=np.array([[[cx,cy]]],np.float32)
        return(punto_elegido)
            

#--------------------

if __name__=='__main__':
##    cap=cv.VideoCapture(0)
##    cap.set(cv.CAP_PROP_FPS,30)
    cap=cv.VideoCapture('pelota.mp4')
    cap.set(cv.CAP_PROP_POS_FRAMES,60)
    ok,frame=cap.read()
    if not ok:
        print("problemas con la camara")
        sys.exit()
    r=cv.selectROI(frame)
    print("Seleccionado color")
    cv.destroyWindow('ROI selector')    
    recorte=frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    h=0
    s=0
    v=0
    mask,h,s,v=seleccion(frame,recorte,1,h,s,v)
    punto_elegido=contornos(mask)
    st=0
    err=0
    old_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #pdb.set_trace()
    kfObj=cv.KalmanFilter(4,2)
    kfObj.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)
    kfObj.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]],np.float32)

    kfObj.processNoiseCov = np.array([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]],np.float32) * 0.5
    kfObj.measurementNoiseCov = np.array([[1,0],
                                     [0,1]],np.float32)*0.5

    measurement = np.array((1,2), np.float32)
    prediction = np.zeros((1,2), np.float32)
    #prediction2 = np.zeros((1,2), np.float32) #experimental calculada cuando hay oclusion
    opcion=1
    while(opcion==1):
        #_,frame=cap.read()
        while(err<1):
            _,frame=cap.read()
            new_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            puntos_predecidos=kfObj.predict()
            try:
                punto_elegido,st,err= cv.calcOpticalFlowPyrLK(new_gray,old_gray,punto_elegido,cv.OPTFLOW_USE_INITIAL_FLOW,
                                                          winSize  = (640, 480),maxLevel = 20,criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT , 20, 0.003))
                for k in punto_elegido:
                    if((punto_elegido[0][0][0]<640)&(punto_elegido[0][0][1]<480)):
                        cv.circle(frame,tuple(k[0]), 3, (0,0,255), -1)                       
                    else:
                        err=1

            except cv.error as assrt:
                print("Objeto Retirado bucle interno")
                err=1
                
            cv.imshow('testing',frame)
            cv.imshow('mascara',mask)
            if(cv.waitKey(5)& 0xFF==ord('f')):
                cv.destroyAllWindows()
                print("Terminado por usuario")
                sys.exit()

        mask,_,_,_=seleccion(frame,recorte,0,h,s,v)
        punto_elegido=contornos(mask)
        old_gray=new_gray
        new_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        err=0
        cv.imshow('testing',frame)
        cv.imshow('mascara',mask)
        if(st==0):
            print("Objeto retirado bucle externo")
            #sys.exit()
        if(cv.waitKey(5)& 0xFF==ord('f')):
            cv.destroyAllWindows()
            print("Terminado por usuario")
            sys.exit()
    #pdb.set_trace()
    
    # en el loop no reinicio el filtro con la funcion init() porque se desea almacenar los resultados anteriores
    while(opcion==0):
        _,frame=cap.read()
        mask,_,_,_=seleccion(frame,recorte,0,h,s,v)
        measurement=contornos(mask)
        if(measurement is not None):
            kfObj.correct(measurement[0][0])
            prediction=kfObj.predict()
            cv.imshow('testing',frame)
            cv.imshow('mascara',mask)            
        else:
            #cv.destroyAllWindows()            
            #continue
            #prediction2=kfObj.predict() # experimental se calcula cuando hay oclusion(taparon al objeto)
            pass
        if((prediction[0][0]>1)and(prediction[1][0]>1)and(measurement is not None)):
            cv.circle(frame,(int(prediction[0][0]),int(prediction[1][0])), 3, (255,0,255), -1)
#--------------------------experimental----------------------------------------
##        elif((prediction2[0][0]>1)and(prediction2[1][0]>1)):
##            cv.circle(frame,(int(prediction2[0][0]),int(prediction2[1][0])), 3, (0,255,0), -1)  #punto dibujado cuando hay oclusion
#--------------------------experimental----------------------------------------
        cv.imshow('testing',frame)
        cv.imshow('mascara',mask)
        #print(prediction)
        if(cv.waitKey(20)& 0xFF==ord('f')):
            cv.destroyAllWindows()
            print("Terminado por usuario")
            sys.exit()

    
    
