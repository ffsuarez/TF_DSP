#Trabajo Final de Procesamiento Digital de Señales II
#Alumno: Suarez Facundo Martin
#
#Descripción: Seguimiento de Objeto visualizado a través de camara
#   mediante Método por Flujo Optico utilizando Algoritmo de Lucas
#   Kanade, o mediante Aplicacion Filtros de Kalman
#
#Dispositivo: Raspberry Pi 3B
#
#Dependencias: Librería OpenCV 3.3.0, Python 3.4
#
#Fecha: 23/02/19
#
#
#Comentarios adicionales: Importante ejecutar la instruccion "sudo modprobe bcm2835-v4l2"
#                         en caso de utilizacion de camara Raspberry PiNoir V2.
# 
#Bibliografia Analizada:
#https://docs.opencv.org/3.3.0/
#https://www.vision.uji.es/courses/Doctorado/TAVC/TAVC-flujo-optico.pdf
#https://medium.com/@jonathan_hui/self-driving-object-tracking-intuition-and-the-math-behind-kalman-filter-657d11dd0a90
#--------------------
import cv2 as cv
import numpy as np
import os
import sys
import math
import pdb

#Función utilizada para el calculo del valor medio de color del objeto seleccionado
def buscar_rgb(img):
    data = np.reshape(img, (-1,3)) # rearma la dimension del parametro "img" para utilizacion mas simple
    print(data.shape) # imprime dimension nueva
    data = np.float32(data)# convierte tipo de datos utilizado
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)  #creacion de lista para criterio de uso de la instruccion "kmeans"
    flags = cv.KMEANS_RANDOM_CENTERS #flag utilizado para parametro de entrada en instruccion "kmeans"
    compactness,labels,centers = cv.kmeans(data,1,None,criteria,10,flags) #retorna la media de la informacion extraida a partir de seleccion inicial
    print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32))) #la variable "centers[0]" contiene la media del valor de color rgb
    return(centers[0].astype(np.int32)) #retorno del valor deseado
#--------------------

#Funcion Utilizada para Segmentación de objeto que captura la camara
def seleccion(frame,recorte,situacion,h1,s1,v1):    
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV) #pasaje de modelo de color RGB a HSV
    min=0 #inicializacion de variables minimo y maximo
    max=0
    #Al invocar la funcion "seleccion" por primera vez se realiza la determinacion del valor medio de color en modelo HSV
    if (situacion==1):
        h=0
        s=0
        v=0
        color_predominante=buscar_rgb(recorte) #invocacion funcion "buscar_rgb"
        color_pred=np.uint8([[color_predominante]]) #pasaje a formato uint8
        color_pred_hsv=cv.cvtColor(color_pred,cv.COLOR_BGR2HSV) #pasaje a modelo hsv
        h=color_pred_hsv[0][0][0] #obtencion de la media de Hue
        s=color_pred_hsv[0][0][1] #obtencion de la media de Saturacion
        v=color_pred_hsv[0][0][2] #obtencion de la media de Valor
        valor=30 #variable que maneja la extension del rango de valores para la umbralizacion

    else:
        h=h1 #utilizacion de Hue calculado inicialmente
        s=s1 #utilizacion de Saturacion calculada inicialmente
        v=v1 #utilizacion de Valor calculado inicialmente
        valor=30
    #pdb.set_trace()
    lwr=np.array([h-15,s-valor,v-valor]) #valor minimo de umbralizacion
    upr=np.array([h+15,s+valor,v+valor]) #valor maximo de umbralizacion
    mask= cv.inRange(hsv,lwr,upr) #enmascaramiento de la escena captada
    #Al inicio de la aplicacion se muestra la umbralizacion inicial para ayudar al usuario
    if(situacion==1):
        while(cv.waitKey(5)&0xFF!=ord('f')):
            cv.imshow('Seleccion',mask)
    cv.destroyWindow('Seleccion')
    return(mask,h,s,v) #retorno de escena umbralizada y valores de color en modelo HSV
#----------------------------------------------
#Funcion utilizada para extraccion de descriptores del objeto que cumple con caracteristicas definidas previamente
def contornos(mask,situacion,asp_rad):
    #En caso de utilizacion de funcion por primera vez se realiza la seleccion del objeto a seguir y se calcula
    #el descriptor correspondiente al mismo.
    if(situacion==0):
     p=cv.selectROI(mask) #determinacion de vertices del objeto recuadrado
     mask2=mask.copy() #copia umbralizacion inicial
     mask2=cv.subtract(mask2,mask2) #resta de la umbralizacion inicial con la copia para obtencion de imagen negra con igual formato que el trabajado
     mask2[int(p[1]):int(p[1]+p[3]), int(p[0]):int(p[0]+p[2])]=mask[int(p[1]):int(p[1]+p[3]), int(p[0]):int(p[0]+p[2])] #insercion de objeto recuadrado
    else:
     mask2=mask    #Si ya se ejecuto la funcion por primera vez, la imagen que se procesa es la imagen umbralizada de la escena capturada por la camara
    umb=cv.adaptiveThreshold(mask2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2) #Umbralizacion de imagen "mask2"
    canny=cv.Canny(umb,100,200) #Extraccion de bordes a partir de aplicacion de filtro de Canny
    canny=cv.GaussianBlur(canny,(5,5),5) #Aplicacion de filtro para eliminar variaciones en altas frecuencias que la camara captura e impiden el funcionamiento correcto de aplicacion
    _,con,_=cv.findContours(canny, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS) #extraccion de contornos de la imagen
    #En caso que la funcion se ejecute por primera vez se realiza la extraccion de descriptores del objeto seleccionado
    #y se extrae su aspecto de radio
    #Si la funcion ya se ejecuto por primera vez, se analiza la escena de modo que calcule el descriptor del contorno que cumpla con las caracteristicas definidas inicialmente
    if (situacion==1):
     if con:
      for i in con:
       if(cv.contourArea(i)>700):
        x,y,w,h = cv.boundingRect(i) #Se efectua el calculo de vertices del minimo rectangulo que encierra al contorno "i" el cual debe poseer un Area mayor a 700
        ASP_rad= float(w)/h #Calculo de aspecto de radio de contorno "i"
        if((asp_rad-0.2*asp_rad)<ASP_rad<(asp_rad+0.2*asp_rad)):
         momentos = cv.moments(i) #calculo del momento invariante del contorno que posee un aspecto de radio que coincide aproximadamente con el aspecto de radio calculado inicialmente
         try:
          cx=float(momentos['m10']/momentos['m00']) #obtencion de coordenada x de centro de gravedad
          cy=float(momentos['m01']/momentos['m00']) #obtencion de coordenada y de centro de gravedad
          punto_elegido=np.array([[[cx,cy]]],np.float32) #obtencion de centro de gravedad a partir de los momentos invariantes
          #print(punto_elegido)
         except ZeroDivisionError as zr:
          #print("Division por cero")
          punto_elegido=np.array([[[641,481]]],np.float32) #en caso de calculo erroneo se especifica como retorno un punto fuera de escena para no detener el ciclo de funcionamiento
          #sys.exit()
         #print(punto_elegido)
         return(punto_elegido,asp_rad) #retorno de descriptor y aspecto de radio
     punto_elegido=np.array([[[641,481]]],np.float32) #si no se captan contornos sobre la escena se especifica como retorno un punto fuera de la escena para no detener el ciclo de funcionamiento
     #print(punto_elegido)
     return(punto_elegido,asp_rad) #retorno de descriptor y aspecto de radio
    
    #Si se ejecuta la funcion por primera vez se efectua el calculo del descriptor del objeto seleccionado inicialmente
    elif con:
        vmax=max(con, key = cv.contourArea) #evaluacion de los contornos de mayor area
        #area=cv.contourArea(vmax)
        x,y,w,h = cv.boundingRect(vmax) #calculo de vertices del minimo rectangulo que encierra el contorno maximo
        asp_rad= float(w)/h #Calculo del aspecto de radio que identificara al objeto en las escenas siguientes
        momentos = cv.moments(vmax) #Calculo de momentos invariantes
        try:
            cx=float(momentos['m10']/momentos['m00']) #obtencion de coordenada x de centro de gravedad
            cy=float(momentos['m01']/momentos['m00']) #obtencion de coordenada y de centro de gravedad                      
        except ZeroDivisionError as zr:
            print("Division por cero")
            punto_elegido=np.array([[[641,481]]],np.float32) #Determinacion de punto fuera de escena en caso de no obtencion de descriptor
            #sys.exit()
            return(punto_elegido,asp_rad) #retorno de descriptor y aspecto de radio
        punto_elegido=np.array([[[cx,cy]]],np.float32) #obtencion de centro de gravedad a partir de los momentos invariantes
        #pdb.set_trace()
        return(punto_elegido,asp_rad) #retorno de descriptor y aspecto de radio
            

#--------------------




#--------------------
if __name__=='__main__':
    cap=cv.VideoCapture(0) #Inicializacion de camara
    cap.set(cv.CAP_PROP_FPS,30) #Configuracion de frames por segundo
    #-------------Experimental(trabajo con video)------------------
    #cap=cv.VideoCapture('pelota.mp4')
    #cap.set(cv.CAP_PROP_POS_FRAMES,60)
    #-----------------------------------------------------------------
    ok,frame=cap.read() #Extraccion de escena captada
    if not ok:
        print("problemas con la camara")
        sys.exit()
    r=cv.selectROI(frame) #Seleccion de objeto inicial para extraccion de color
    print("Seleccionado color")
    cv.destroyWindow('ROI selector')    
    recorte=frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] #Recorte a partir de la seleccion
    h=0 #Inicializacion variable Hue
    s=0 #Inicializacion variable Saturacion
    v=0 #Inicializacion variable Valor
    asp_rad=0 #Inicializacion variable aspecto de radio
    mask,h,s,v=seleccion(frame,recorte,1,h,s,v) #Invocacion funcion "seleccion"
    punto_elegido,asp_rad=contornos(mask,0,asp_rad) #Invocacion funcion "contornos"
    st=0 #Inicializacion variable status del Metodo por Lucas Kanade
    err=0 #Inicializacion variable error del Metodo por Lucas Kanade
    old_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY) #Extraccion de escena anterior en escala de grises
    #pdb.set_trace()
    kfObj=cv.KalmanFilter(4,2) #Inicializacion de objeto Filtro de Kalman
    kfObj.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)
    #Inicializacion de Matriz de medicion 
    kfObj.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]],np.float32)
    #Inicializacion de Matriz de transicion
    kfObj.processNoiseCov = np.array([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]],np.float32) * 0.5
    kfObj.measurementNoiseCov = np.array([[1,0],
                                     [0,1]],np.float32)*0.5
    #Inicializacion de Matriz de Covarianza de los procesos de medicion de ruido
    #de Sistema y de Medicion
    measurement = np.array((1,2), np.float32) #inicializacion vector de medicion
    prediction = np.zeros((1,2), np.float32) #inicializacion vector de posicion futura
    #-------Experimental(cuando hay oclusion)-----------
    #prediction2 = np.zeros((1,2), np.float32)
    #---------------------------------------------------
    
    #Variable para seleccion de metodo de seguimiento
    #En caso que opcion=0:  Uso de Metodo de Seguimiento por Flujo Optico a partir de Algoritmo de Lucas Kanade
    #En caso que opcion=1:  Uso de Metodo de Seguimiento por Aplicacion de Filtros de Kalman
    opcion=0
    
    # Define el codec y crea el objeto VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output3.avi',fourcc, 20.0, (640,480)) #configura la grabacion de la aplicacion
    
    
    
    #-------------Metodo de Seguimiento por Flujo Optico a partir de Algoritmo de Lucas Kanade---------
    while(opcion==1):
        #_,frame=cap.read()
        while(err<1):
            _,frame=cap.read() #Extraccion de escena captada
            new_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY) #Extraccion de escena siguiente en escala de grises            
            try:
                punto_elegido,st,err= cv.calcOpticalFlowPyrLK(new_gray,old_gray,punto_elegido,cv.OPTFLOW_USE_INITIAL_FLOW,
                                                          winSize  = (640, 480),maxLevel = 20,criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT , 20, 0.003))
                #Calculo de posicion del punto descriptor del objeto a partir de Algoritmo de Lucas Kanade
                for k in punto_elegido:
                    if((punto_elegido[0][0][0]<640)&(punto_elegido[0][0][1]<480)):
                        cv.circle(frame,tuple(k[0]), 3, (0,0,255), -1)  #Dibujo de punto color rojo del objeto seguido                     
                    else:
                        err=1 #En caso de que el punto calculado no se ubique dentro de escena se reinicia proceso de segmentacion y extraccion de descriptores

            except cv.error as assrt:
                print("Objeto Retirado bucle interno") #En caso de resolucion erronea de algoritmo se reinicia proceso de segmentacion y extraccion
                err=1
                
            cv.imshow('testing',frame) #Ilustracion de escena captada por camara
            cv.imshow('mascara',mask) #Ilustracion de enmascaramiento de escena
            #Finalizacion de Aplicacion a partir de oprimir boton "f"
            if(cv.waitKey(5)& 0xFF==ord('f')):
                cv.destroyAllWindows()
                print("Terminado por usuario")
                sys.exit()

        mask,_,_,_=seleccion(frame,recorte,0,h,s,v) #Reinicio de proceso de segmentacion
        punto_elegido,asp_rad=contornos(mask,1,asp_rad) #Reinicion de proceso de extraccion de descriptores
        old_gray=new_gray #Actualizacion de escena anterior en escala de grises
        new_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY) #Actualizacion de escena siguiente en escala de grises
        err=0 #Se permite el ingreso al loop en el que se realiza el Seguimiento por Algoritmo de Lucas Kanade
        cv.imshow('testing',frame) #Ilustracion de escena captada por camara
        cv.imshow('mascara',mask) #Ilustracion de enmascaramiento de escena
        out.write(frame) #Grabacion de escena captada 
        if(st==0):
            print("Objeto retirado bucle externo") #Aviso por consola de ejecucion erronea de Algoritmo de Lucas Kanade
            #sys.exit()
        #Finalizacion de Aplicacion a partir de oprimir boton "f"
        if(cv.waitKey(5)& 0xFF==ord('f')):
            cv.destroyAllWindows()
            print("Terminado por usuario")
            out.release()
            cap.release()
            sys.exit()
    #pdb.set_trace()
    #---------------------------------------------------------------------------------------------------------
            
            
    
    # en el loop no reinicio el filtro con la funcion init() porque se desea almacenar los resultados anteriores
    #-------------Metodo de Seguimiento por Aplicacion de Filtros de Kalman---------
    while(opcion==0):
        _,frame=cap.read() #Extraccion de escena captada
        mask,_,_,_=seleccion(frame,recorte,0,h,s,v) #Proceso de segmentacion
        measurement,asp_rad=contornos(mask,1,asp_rad) #Proceso de obtencion de extraccion de descriptores
        #measurement=contornos(mask)
        if(measurement is not None):
            kfObj.correct(measurement[0][0]) #Correccion de los estados del sistema
            prediction=kfObj.predict() #Prediccion de posicion futura de la posicion del objeto
            cv.imshow('testing',frame) #Ilustracion de escena captada por camara
            cv.imshow('mascara',mask) #Ilustracion de enmascaramiento de escena
        else:
            #prediction2=kfObj.predict() # experimental se calcula cuando hay oclusion(taparon al objeto)
            pass
        #Dibujo de la ubicacion predecida del objeto controlando que se produjo la medicion
        if((prediction[0][0]>1)and(prediction[1][0]>1)and(measurement is not None)):
            cv.circle(frame,(int(prediction[0][0]),int(prediction[1][0])), 3, (255,0,255), -1)
#--------------------------experimental----------------------------------------
##        elif((prediction2[0][0]>1)and(prediction2[1][0]>1)):
##            cv.circle(frame,(int(prediction2[0][0]),int(prediction2[1][0])), 3, (0,255,0), -1)  #punto dibujado cuando hay oclusion
#--------------------------experimental----------------------------------------
        cv.imshow('testing',frame) #Ilustracion de escena captada por camara
        cv.imshow('mascara',mask) #Ilustracion de enmascaramiento de escena
        out.write(frame) #Grabacion de escena captada 
        #print(prediction)
        #Finalizacion de Aplicacion a partir de oprimir boton "f"
        if(cv.waitKey(20)& 0xFF==ord('f')):
            cv.destroyAllWindows()
            out.release()
            cap.release()
            print("Terminado por usuario")
            sys.exit()