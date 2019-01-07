'''
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
Recordar realizar la aplicación de los comandos

*)source ~/.profile
    Para actualizar las modificaciones del sistema
    realizados en la instalación de OpenCV.
    Fuente:
        https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/
    
    Problema observado durante tutorial de instalación de la materia:
        Existen dependencias de drivers que no me permitían instalarlas.
*)workon cv
    Para cargar el entorno virtual "cv".
*)sudo modprobe bcm2835-v4l2
    Para poder realizar la carga de el driver de
    la camara de la raspberry pi





Como se pretende que sea el uso:

LK.py [--tecnica<=--lk| --shi] [<fuente_video>] [n_objetos] [--color<= --color --nocolor] [--especificacion<= -r  -b  -g  -x]

Donde: [--tecnica] decide cual tecnica tomar, Lucas Kanade o Shi-Tomasi
       [<fuente_video>] elige un archivo de video y lo lee, sino toma la camara
       [n_objetos] es el numero de objetos a seguir
       [--color] decide agregar la condicion de seguir al objeto si posee determinado color
       [--especificacion] decide si seguir color rojo,azul,verde o especifica


#https://robologs.net/2017/08/22/tutorial-de-opencv-python-tracking-de-objetos-con-el-metodo-de-lucas-kanade/

#http://pyspanishdoc.sourceforge.net/lib/module-getopt.html	   
#https://www.digitalocean.com/community/tutorials/how-to-use-the-python-debugger
#https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
#https://stackoverflow.com/questions/4117530/sys-argv1-meaning-in-script
#https://pymotw.com/2/getopt/
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
#https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html
#https://stackoverflow.com/questions/2709821/what-is-the-purpose-of-self
#https://es.stackoverflow.com/questions/202588/como-funciona-self-en-python	
https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#double%20matchShapes(InputArray%20contour1,%20InputArray%20contour2,%20int%20method,%20double%20parameter)
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
'''


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pdb


#---------------------------------------------------------------------
def nada(x):
    #pdb.set_trace()    
    #img es numpy nd array
    pass




cv.namedWindow('Color HSV')        
cv.createTrackbar('H','Color HSV',0,360,nada)
cv.createTrackbar('S','Color HSV',0,200,nada)
cv.createTrackbar('V','Color HSV',0,1,nada)




#---------------------------------------------------------------------
def puntos_objeto(frame):
    r=cv.selectROI(frame)    
    return(r)
#---------------------------------------------------------------------
def dibujo_puntos_nc(recortes,n,punto_elegido,cap,r,contours):
    _,frame=cap.read()    
    st=[None]*n
    err=[None]*n
    img=[None]*n
    img_gray=[None]*n
    for j in range(n):
        img[j]=frame[int(r[j][1]):int(r[j][1]+r[j][3]), int(r[j][0]):int(r[j][0]+r[j][2])]
        img_gray[j]=cv.cvtColor(img[j],cv.COLOR_BGR2GRAY)
        punto_elegido[j],st[j],err[j]= cv.calcOpticalFlowPyrLK(recortes[j],img_gray[j],punto_elegido[j],None, **seguidor.opciones(None,metodo)[0])
    for i in range(n):
        for k in punto_elegido[i]:
            cv.circle(img[i],tuple(k[0]), 3, (0,0,255), -1)
            recortes[i]=img_gray[i].copy()           
            frame[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]=img[i]
        #https://stackoverflow.com/questions/48829532/module-cv2-cv2-has-no-attribute-puttext
        font     = cv.FONT_HERSHEY_COMPLEX_SMALL
        bottomLeftCornerOfText = (r[i][0],r[i][1])
        fontScale    = 0.4 
        fontColor    = (0,0,0) 
        lineType    = 1
        cv.putText(frame,"{:.2f}".format(punto_elegido[i][0][0][0]), 
        bottomLeftCornerOfText, 
        font, 
        fontScale, 
        fontColor, 
        lineType)
        cv.putText(frame,"       {:.2f}".format(punto_elegido[i][0][0][1]), 
        bottomLeftCornerOfText, 
        font, 
        fontScale, 
        fontColor, 
        lineType) 

        #analizo_objeto(punto_elegido,img,n)        
    cv.imshow('testing',frame)    
	


#---------------------------------------------------------------------
class seguidor:
		
    def __init__(self,video_src):
    
            try:
                    cap=cv.VideoCapture(video_src)                            
            except:
                    cap=cv.VideoCapture(0)
            return(cap)


    def opciones(self,metod):
            
            
            if(metod=='--lk'):
                    lk_params = dict( winSize  = (500, 500),maxLevel = 20,criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
                    feature_params = dict( maxCorners = 500,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
                    return(lk_params,feature_params)
            else:
                    print('No se reconoce opcion metod:',metod)
                    print('  O existe problema con la camara')
                    sys.exit(1)
                    
    
    def run (self,puntos,cap,n,color,img):
        print('Comenzando trabajo')
        _,frame=cap.read()
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        contours=[None]*n
        maximo=[None]*n
        momentos=[None]*n
        cx=[None]*n
        cy=[None]*n
        punto_elegido=[None]*n
        r=[None]*n

        frame_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        recortes=[None]*n
        for i in range(n):
                r[i]=puntos_objeto(frame)
                puntos.append(r[i])                    
                recortes[i]=frame_gray[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]
                recortes[i]=cv.adaptiveThreshold(recortes[i],255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
                recortes[i] = cv.morphologyEx(recortes[i], cv.MORPH_OPEN, kernel)
                recortes[i] = cv.morphologyEx(recortes[i], cv.MORPH_CLOSE, kernel)
                recortes[i]=cv.bitwise_not(recortes[i])
                _,contours[i],_=cv.findContours(recortes[i], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
                #pdb.set_trace()
                maximo[i]=max(contours[i], key = cv.contourArea)
                momentos[i] = cv.moments(maximo[i])
                cx[i]=float(momentos[i]['m10']/momentos[i]['m00'])
                cy[i]=float(momentos[i]['m01']/momentos[i]['m00'])
                punto_elegido[i]=np.array([[[cx[i],cy[i]]]],np.float32)
                cv.imshow("{:d}".format(i),recortes[i])
        cv.destroyWindow('ROI selector')

        while(True):
            dibujo_puntos_nc(recortes,n,punto_elegido,cap,r,contours)					
            tecla = cv.waitKey(5) & 0xFF
            if tecla == 27:
                break                    
         

def seleccion(puntos,cap,n):
    #pdb.set_trace()
    ret,frame=cap.read()
    if(ret==False):
        print('Hubo un error')
        sys.exit(1)
    r=[None]*n
    #recortes=[None]*n
    #recortes_hsv=[None]*n
    res=[None]*n        
    #for i in range(n):
        #r[i]=puntos_objeto(frame)
        #recortes[i]=frame[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        #cv.imshow('abc',recortes[0])
    while(True):
        h=cv.getTrackbarPos('H','Color HSV')
        s=cv.getTrackbarPos('S','Color HSV')
        v=cv.getTrackbarPos('V','Color HSV')

        lwr=np.array([h,s,v])
        upr=np.array([h+5,255,255])

        mask= cv.inRange(hsv,lwr,upr)
            #res= cv.bitwise_and(recortes[i],recortes[i],mask=mask)
            #_,res=cv.threshold(res,50,255,cv.THRESH_BINARY)
        cv.imshow('Seleccion',mask)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyWindow('Seleccion')
    return(mask)
                
            




#---------------------------------------------------------------------
#https://stackoverflow.com/questions/50899692/most-dominant-color-in-rgb-image-opencv-numpy-python
def buscar_rgb(img):
    data = np.reshape(img, (-1,3))
    print(data.shape)
    data = np.float32(data)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv.kmeans(data,1,None,criteria,10,flags)

    print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
    return(centers[0].astype(np.int32))
#---------------------------------------------------------------------
import os

if __name__=='__main__':
	print(__doc__)
	import sys,getopt
	metodo=sys.argv[1]
	video_src=sys.argv[2]
	if (video_src=='0'):
            video_src=int(video_src)
	n=sys.argv[3]
	n=int(n)
	color=sys.argv[4]
	puntos=[None]*n
	tec_esc='a'
	seguidor.opciones(None,metodo)
	cap=seguidor.__init__(None,video_src)
	_,frame=cap.read()
	img=[np.zeros(frame.shape)]*n
	lala=None
	#aux=[None]*n#cv.imshow('negro',img)
	while(tec_esc != 27):            
            if(color=='--nocolor'):
                seguidor.run(None,puntos,cap,n,color,img)
            elif(color=='--color'):
                if(puntos!=None):
                    for i in range(n):
                        img[i]=seleccion(puntos,cap,n)
                    cv.imshow('imagen 1',img[0])
                    cv.imshow('imagen 2',img[1])
                    pdb.set_trace()
                    for i in range(n-1):
                        lala=cv.add(img[i],img[i-1])
                    #img[int(puntos[i][1]):int(puntos[i][1]+puntos[i][3]), int(puntos[i][0]):int(puntos[i][0]+puntos[i][2])]=aux[i]
                    while(True):
                        cv.imshow('def',lala)
                        if cv.waitKey(20) & 0xFF == 27:
                            break                        
                    pdb.set_trace()
                seguidor.run(None,puntos,cap,n,color,img[i])
            #cv.namedWindow('Test Key') #necesaria para que waitkey funcione bien
            tec_esc=cv.waitKey(0)
	cv.destroyAllWindows()
