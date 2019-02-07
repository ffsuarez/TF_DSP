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

LK.py [--tecnica<=--lk| --shi] [<fuente_video>] [n_objetos] [--color<= --color --nocolor] 

Donde: [--tecnica] decide cual tecnica tomar, Lucas Kanade o Shi-Tomasi
       [<fuente_video>] elige un archivo de video y lo lee, sino toma la camara
       [n_objetos] es el numero de objetos a seguir
       [--color] decide agregar la condicion de seguir al objeto si posee determinado color

----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
'''
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
#https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#double%20matchShapes(InputArray%20contour1,%20InputArray%20contour2,%20int%20method,%20double%20parameter)

#https://stackoverflow.com/questions/48829532/module-cv2-cv2-has-no-attribute-puttext





import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pdb


#---------------------------------------------------------------------
def nada(x):
    #pdb.set_trace()    
    #img es numpy nd array
    pass
#---------------------------------------------------------------------
def puntos_objeto(frame):
    r=cv.selectROI(frame)    
    return(r)
#---------------------------------------------------------------------
def dibujo_puntos_nc(recortes,n,punto_elegido,cap,r,contours,aux_elegido,imrecortes):
    _,frame=cap.read()    
    st=[None]*n
    err=[None]*n
    img=[None]*n
    img_gray=[None]*n
    maximo=[None]*n
    cx=[None]*n
    cy=[None]*n
    momentos=[None]*n
    for j in range(n):        
        img[j]=frame[int(r[j][1]):int(r[j][1]+r[j][3]), int(r[j][0]):int(r[j][0]+r[j][2])]
        img_gray[j]=cv.cvtColor(img[j],cv.COLOR_BGR2GRAY)
        #pdb.set_trace()
        res=cv.matchTemplate(img_gray[j],imrecortes[j],cv.TM_CCOEFF_NORMED)
        thr=0.2
        if(res>=thr):
            punto_elegido[j],st[j],err[j]= cv.calcOpticalFlowPyrLK(recortes[j],img_gray[j],punto_elegido[j],None, **seguidor.opciones(None,metodo)[0])            
            _,contours[j],_=cv.findContours(recortes[j], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
            maximo[j]=max(contours[j], key = cv.contourArea)
            momentos[j] = cv.moments(maximo[j])
            cx[j]=float(momentos[j]['m10']/momentos[j]['m00'])
            cy[j]=float(momentos[j]['m01']/momentos[j]['m00'])
            aux_elegido[j]=np.array([[[cx[j],cy[j]]]],np.float32)
            aux_elegido[j],st[j],err[j]= cv.calcOpticalFlowPyrLK(recortes[j],img_gray[j],aux_elegido[j],None, **seguidor.opciones(None,metodo)[0])
            #punto_elegido[j]=cv.goodFeaturesToTrack(recortes[j],mask=recortes[j],**seguidor.opciones(None,metodo)[1])
        else:
            break


    for i in range(n):
        if(res>=(thr-(thr*0.5))):
            for k in aux_elegido[i]:
                cv.circle(img[i],tuple(k[0]), 3, (255,0,255), -1)
                recortes[i]=img_gray[i].copy()           
                frame[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]=img[i]
        else:
            #_,contours[i],_=cv.findContours(recortes[i], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
            #maximo[i]=max(contours[i], key = cv.contourArea)
            #cx[i]=float(momentos[i]['m10']/momentos[i]['m00'])
            break



    for i in range(n):
        if(res>=thr):        
            for k in punto_elegido[i]:
                cv.circle(img[i],tuple(k[0]), 3, (0,0,255), -1)
                recortes[i]=img_gray[i].copy()           
                frame[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]=img[i]

            font     = cv.FONT_HERSHEY_COMPLEX_SMALL
            bottomLeftCornerOfText = (r[i][0],r[i][1])
            fontScale    = 0.4 
            fontColor    = (0,0,0) 
            lineType    = 1
            
            if((punto_elegido[i][0][0][0]<0)or(punto_elegido[i][0][0][1]<0)or(punto_elegido[i][0][0][1]>int(r[i][3]))or(punto_elegido[i][0][0][0]>int(r[i][2]))):
                cv.putText(frame,"FUERA DE ROI", 
                bottomLeftCornerOfText, 
                font, 
                fontScale, 
                fontColor, 
                lineType)
            else:
                cv.putText(frame,"       {:.2f}".format(punto_elegido[i][0][0][1]), 
                bottomLeftCornerOfText, 
                font, 
                fontScale, 
                fontColor, 
                lineType)
                cv.putText(frame,"{:.2f}".format(punto_elegido[i][0][0][0]), 
                bottomLeftCornerOfText, 
                font, 
                fontScale, 
                fontColor, 
                lineType)
        else:
            break

   


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
                    lk_params = dict( winSize  = (500, 500),maxLevel = 2,criteria = (cv.TERM_CRITERIA_EPS , 10, 0.003))
                    feature_params = dict( maxCorners = 4,qualityLevel = 0.6,minDistance = 7,blockSize = 7 )
                    return(lk_params,feature_params)
            else:
                    print('No se reconoce opcion metod:',metod)
                    print('  O existe problema con la camara')
                    sys.exit(1)
                    
    
    def run (self,puntos,cap,n,color):
        print('Comenzando trabajo')
        _,frame=cap.read()
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9))
        contours=[None]*n
        maximo=[None]*n
        momentos=[None]*n
        cx=[None]*n
        cy=[None]*n
        punto_elegido=[None]*n
        r=[None]*n
        aux_elegido=[None]*n
        frame_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        recortes=[None]*n
        imrecortes=[None]*n
        for i in range(n):
                r[i]=puntos_objeto(frame)
                puntos.append(r[i])                    
                imrecortes[i]=frame_gray[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]
                recortes[i]=frame_gray[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]
                recortes[i]=cv.adaptiveThreshold(recortes[i],255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
                #recortes[i] = cv.morphologyEx(recortes[i], cv.MORPH_OPEN, kernel)
                #recortes[i] = cv.morphologyEx(recortes[i], cv.MORPH_CLOSE, kernel)
                recortes[i]=cv.bitwise_not(recortes[i])
                recortes[i]=cv.Canny(recortes[i],100,200)
                _,contours[i],_=cv.findContours(recortes[i], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
                #pdb.set_trace()
                maximo[i]=max(contours[i], key = cv.contourArea)
                momentos[i] = cv.moments(maximo[i])
                cx[i]=float(momentos[i]['m10']/momentos[i]['m00'])
                cy[i]=float(momentos[i]['m01']/momentos[i]['m00'])
                aux_elegido[i]=np.array([[[cx[i],cy[i]]]],np.float32)
                punto_elegido[i]=cv.goodFeaturesToTrack(recortes[i],mask=recortes[i],**seguidor.opciones(None,metodo)[1])
                cv.imshow("{:d}".format(i),recortes[i])
        cv.destroyWindow('ROI selector')

        while(True):
            dibujo_puntos_nc(recortes,n,punto_elegido,cap,r,contours,aux_elegido,imrecortes)					
            tecla = cv.waitKey(5) & 0xFF
            if tecla == 27:
                break
            
            


    def runcolor (self,puntos,cap,n,color,img,lala):
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
        recortes=[None]*n
        for i in range(n):
                r[i]=puntos_objeto(img[i])
                puntos.append(r[i])                    
                recortes[i]=img[i][int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]
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
    cv.namedWindow('Color HSV',cv.WINDOW_NORMAL)
    cv.resizeWindow('Color HSV', 100,50)
    cv.createTrackbar('H','Color HSV',0,175,nada)
    cv.createTrackbar('S','Color HSV',0,235,nada)
    cv.createTrackbar('V','Color HSV',0,235,nada)    
    if(ret==False):
        print('Hubo un error')
        sys.exit(1)
    r=[None]*n
    res=[None]*n        
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    while(True):
        h=cv.getTrackbarPos('H','Color HSV')
        s=cv.getTrackbarPos('S','Color HSV')
        v=cv.getTrackbarPos('V','Color HSV')

        lwr=np.array([h,s,v])
        upr=np.array([h+5,s+20,v+20])
        

        mask= cv.inRange(hsv,lwr,upr)
        cv.imshow('Seleccion',mask)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyWindow('Seleccion')
    return(mask)
                
            




#---------------------------------------------------------------------
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
	while(tec_esc != 27):            
            if(color=='--nocolor'):
                seguidor.run(None,puntos,cap,n,color)
            elif(color=='--color'):
                if(puntos!=None):
                    for i in range(n):
                        img[i]=seleccion(puntos,cap,n)                   
                    for i in range(n-1):
                        img[i]=cv.add(img[i],img[i-1])
                    aux=img[i]
                seguidor.runcolor(None,puntos,cap,n,color,img,aux)
            tec_esc=cv.waitKey(0)
	cv.destroyAllWindows()
