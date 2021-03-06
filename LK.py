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


def dibujo_puntos_cc(recortes,n,punto_elegido,cap,r,contours,imrecortes):
    _,frame=cap.read()    
    st=[None]*n
    err=[None]*n
    img=[None]*n
    img_gray=[None]*n
    
    maximo=[None]*n
    momentos=[None]*n
    cx=[None]*n
    cy=[None]*n
    #pdb.set_trace()
    lwr=[None]*n
    upr=[None]*n

    hsv=[None]*n
    for j in range(n):
        img[j]=frame[int(r[j][1]):int(r[j][1]+r[j][3]), int(r[j][0]):int(r[j][0]+r[j][2])]
        img_gray[j]=cv.cvtColor(img[j],cv.COLOR_BGR2GRAY)        
        punto_elegido[j],st[j],err[j]= cv.calcOpticalFlowPyrLK(recortes[j],img_gray[j],punto_elegido[j],cv.OPTFLOW_USE_INITIAL_FLOW, **seguidor.opciones(None,metodo)[0])
    
    for i in range(n):
        lwr[i]=np.array([h[i],s[i]-20,v[i]-20])
        upr[i]=np.array([h[i]+5,s[i]+20,v[i]+20])
        if(err[i]>0.01):
            for k in punto_elegido[i]:
                #cv.circle(img[i],tuple(k[0]), 3, (0,0,255), -1)
                recortes[i]=img_gray[i].copy()           
                frame[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]=img[i]

            #https://stackoverflow.com/questions/48829532/module-cv2-cv2-has-no-attribute-puttext
                font     = cv.FONT_HERSHEY_COMPLEX_SMALL
                bottomLeftCornerOfText = (r[i][0],r[i][1])
                fontScale    = 0.4 
                fontColor    = (0,0,0) 
                lineType    = 1
    ##        
    ##        if((punto_elegido[i][0][0][0]<0)or(punto_elegido[i][0][0][1]<0)or(punto_elegido[i][0][0][1]>int(r[i][3]))or(punto_elegido[i][0][0][0]>int(r[i][2]))):
                cv.putText(frame,"{:.2f}".format(err[i][0][0]*1), 
                bottomLeftCornerOfText, 
                font, 
                fontScale, 
                fontColor, 
                lineType)                
                #recortes[i]=img_gray[i].copy()
                hsv[i]=cv.cvtColor(img[i],cv.COLOR_BGR2HSV)
                recortes[i]= cv.inRange(hsv[i],lwr[i],upr[i])
                recortes[i]=cv.Canny(recortes[i],100,200)
                _,contours[i],_=cv.findContours(recortes[i], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
                if(contours[i]):
                    maximo[i]=max(contours[i], key = cv.contourArea)
                    momentos[i] = cv.moments(maximo[i])
                #pdb.set_trace()
                if momentos[i] is not None:
                    if(momentos[i]['m00']>0.1):
                        cx[i]=float(momentos[i]['m10']/momentos[i]['m00'])
                        cy[i]=float(momentos[i]['m01']/momentos[i]['m00'])
                        cv.circle(img[i],tuple(k[0]), 3, (0,0,255), -1)
                        punto_elegido[i]=np.array([[[cx[i],cy[i]]]],np.float32)

        else:
            #recortes[i]=img_gray[i].copy()
            hsv[i]=cv.cvtColor(img[i],cv.COLOR_BGR2HSV)

            recortes[i]= cv.inRange(hsv[i],lwr[i],upr[i])

            recortes[i]=cv.Canny(recortes[i],100,200)
            _,contours[i],_=cv.findContours(recortes[i], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)

            maximo[i]=max(contours[i], key = cv.contourArea)
            momentos[i] = cv.moments(maximo[i])
            cx[i]=float(momentos[i]['m10']/momentos[i]['m00'])
            cy[i]=float(momentos[i]['m01']/momentos[i]['m00'])
            punto_elegido[i]=np.array([[[cx[i],cy[i]]]],np.float32)


##        else:
##            cv.putText(frame,"       {:.2f}".format(punto_elegido[i][0][0][1]), 
##            bottomLeftCornerOfText, 
##            font, 
##            fontScale, 
##            fontColor, 
##            lineType)
##            cv.putText(frame,"{:.2f}".format(punto_elegido[i][0][0][0]), 
##            bottomLeftCornerOfText, 
##            font, 
##            fontScale, 
##            fontColor, 
##            lineType)

            

        #analizo_objeto(punto_elegido,img,n)        
##    else:
##        for i in range(n):
##            font     = cv.FONT_HERSHEY_COMPLEX_SMALL
##            bottomLeftCornerOfText = (r[i][0],r[i][1])
##            fontScale    = 0.4 
##            fontColor    = (0,0,0) 
##            lineType    = 1
##            cv.putText(frame,"Error", 
##            bottomLeftCornerOfText, 
##            font, 
##            fontScale, 
##            fontColor, 
##            lineType)
##            
##            recortes[i]=cv.Canny(recortes[i],100,200)
##            _,contours[i],_=cv.findContours(recortes[i], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
##            
##            maximo[i]=max(contours[i], key = cv.contourArea)
##            momentos[i] = cv.moments(maximo[i])
##            cx[i]=float(momentos[i]['m10']/momentos[i]['m00'])
##            cy[i]=float(momentos[i]['m01']/momentos[i]['m00'])
##            punto_elegido[i]=np.array([[[cx[i],cy[i]]]],np.float32)
            


    cv.imshow('testing',frame)
    cv.imshow('rec',recortes[0])
    cv.imshow('imrec',imrecortes[0])






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
    res=[0]*n
    for j in range(n):        
        img[j]=frame[int(r[j][1]):int(r[j][1]+r[j][3]), int(r[j][0]):int(r[j][0]+r[j][2])]
        img_gray[j]=cv.cvtColor(img[j],cv.COLOR_BGR2GRAY)
        #pdb.set_trace()
        res[j]=cv.matchTemplate(img_gray[j],imrecortes[j],cv.TM_CCOEFF_NORMED)
        if(res[j]==None):
            res[j]=0
        thr=0.01
        if(res[j]>=thr):
            punto_elegido[j],st[j],err[j]= cv.calcOpticalFlowPyrLK(imrecortes[j],img_gray[j],punto_elegido[j],None, **seguidor.opciones(None,metodo)[0])            
            _,contours[j],_=cv.findContours(recortes[j], cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
            maximo[j]=max(contours[j], key = cv.contourArea)
            momentos[j] = cv.moments(maximo[j])
            cx[j]=float(momentos[j]['m10']/momentos[j]['m00'])
            cy[j]=float(momentos[j]['m01']/momentos[j]['m00'])
            aux_elegido[j]=np.array([[[cx[j],cy[j]]]],np.float32)
            aux_elegido[j],st[j],err[j]= cv.calcOpticalFlowPyrLK(imrecortes[j],img_gray[j],aux_elegido[j],None, **seguidor.opciones(None,metodo)[0])
            punto_elegido[j]=cv.goodFeaturesToTrack(recortes[j],mask=recortes[j],**seguidor.opciones(None,metodo)[1])
        else:
            break


    for i in range(n):
        for k in punto_elegido[i]:
            if(res[i]>=0.3):
                cv.circle(img[i],tuple(k[0]), 3, (255,0,255), -1)
                recortes[i]=img_gray[i].copy()
                frame[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]=img[i]          
                
            #_,contours[i],_=cv.findContours(recortes[i], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
            #maximo[i]=max(contours[i], key = cv.contourArea)
            #cx[i]=float(momentos[i]['m10']/momentos[i]['m00'])
            
    for i in range(n):
        if(res[i]>0.01):
            font     = cv.FONT_HERSHEY_COMPLEX_SMALL
            bottomLeftCornerOfText = (r[i][0],r[i][1]+10)
            fontScale    = 0.5 
            fontColor    = (120,60,20) 
            lineType    = 1
            cv.putText(frame,"{:.2f}".format(res[i][0][0]*100), 
            bottomLeftCornerOfText, 
            font, 
            fontScale, 
            fontColor, 
            lineType)
            
        for k in aux_elegido[i]:
            if(res[i]>=0.8):
                #cv.circle(img[i],tuple(k[0]), 3, (0,0,255), -1)
                recortes[i]=img_gray[i].copy()
                frame[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]=img[i]
                font     = cv.FONT_HERSHEY_COMPLEX_SMALL
                bottomLeftCornerOfText = (r[i][0],r[i][1])
                fontScale    = 0.4 
                fontColor    = (120,60,20) 
                lineType    = 1
                cv.putText(frame,"OBJETO DENTRO", 
                bottomLeftCornerOfText, 
                font, 
                fontScale, 
                fontColor, 
                lineType)
            
            
##            if((punto_elegido[i][0][0][0]<0)or(punto_elegido[i][0][0][1]<0)or(punto_elegido[i][0][0][1]>int(r[i][3]))or(punto_elegido[i][0][0][0]>int(r[i][2]))):
##                cv.putText(frame,"FUERA DE ROI", 
##                bottomLeftCornerOfText, 
##                font, 
##                fontScale, 
##                fontColor, 
##                lineType)
##            else:
##                cv.putText(frame,"       {:.2f}".format(punto_elegido[i][0][0][1]), 
##                bottomLeftCornerOfText, 
##                font, 
##                fontScale, 
##                fontColor, 
##                lineType)
##                cv.putText(frame,"{:.2f}".format(punto_elegido[i][0][0][0]), 
##                bottomLeftCornerOfText, 
##                font, 
##                fontScale, 
##                fontColor, 
##                lineType)
##        else:
##            break

   


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
                    lk_params = dict( winSize  = (100, 100),maxLevel = 20,criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT , 20, 5))
                    feature_params = dict( maxCorners = 4,qualityLevel = 0.1,minDistance = 3,blockSize = 10 )
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
                #puntos.append(r[i])                    
                imrecortes[i]=frame_gray[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]
                recortes[i]=frame_gray[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]
                recortes[i]=cv.GaussianBlur(recortes[i],(3,3),5)
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
            for i in range(n):
                imrecortes[i]=frame_gray[int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]
                #imrecortes[i]=cv.GaussianBlur(imrecortes[i],(3,3),5)
                #imrecortes[i]=cv.adaptiveThreshold(imrecortes[i],255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
                #recortes[i] = cv.morphologyEx(recortes[i], cv.MORPH_OPEN, kernel)
                #recortes[i] = cv.morphologyEx(recortes[i], cv.MORPH_CLOSE, kernel)
                #imrecortes[i]=cv.bitwise_not(imrecortes[i])
                #imrecortes[i]=cv.Canny(imrecortes[i],100,200)
                #_,contours[i],_=cv.findContours(imrecortes[i], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
                #pdb.set_trace()
                #maximo[i]=max(contours[i], key = cv.contourArea)
                #momentos[i] = cv.moments(maximo[i])
                #cx[i]=float(momentos[i]['m10']/momentos[i]['m00'])
                #cy[i]=float(momentos[i]['m01']/momentos[i]['m00'])
                #aux_elegido[i]=np.array([[[cx[i],cy[i]]]],np.float32)
                #punto_elegido[i]=cv.goodFeaturesToTrack(imrecortes[i],mask=imrecortes[i],**seguidor.opciones(None,metodo)[1])
                
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
        imrecortes=[None]*n
        for i in range(n):
                r[i]=puntos_objeto(img[i])
                puntos.append(r[i])                    
                recortes[i]=img[i][int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2])]
                imrecortes[i]=recortes[i]
                recortes[i]=cv.adaptiveThreshold(recortes[i],255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
##                recortes[i] = cv.morphologyEx(recortes[i], cv.MORPH_OPEN, kernel)
##                recortes[i] = cv.morphologyEx(recortes[i], cv.MORPH_CLOSE, kernel)
                recortes[i]=cv.Canny(recortes[i],100,200)
                #recortes[i]=cv.bitwise_not(recortes[i])
                _,contours[i],_=cv.findContours(recortes[i], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
                #pdb.set_trace()
                maximo[i]=max(contours[i], key = cv.contourArea)
                momentos[i] = cv.moments(maximo[i])
                cx[i]=float(momentos[i]['m10']/momentos[i]['m00'])
                cy[i]=float(momentos[i]['m01']/momentos[i]['m00'])
                punto_elegido[i]=np.array([[[cx[i],cy[i]]]],np.float32)
                cv.imshow("{:d}".format(i),recortes[i])
        cv.destroyAllWindows()

        while(True):
            dibujo_puntos_cc(recortes,n,punto_elegido,cap,r,contours,imrecortes)					
            tecla = cv.waitKey(5) & 0xFF        
            if tecla == 27:
                break                    

         

def seleccion(puntos,cap,n):
    _,frame=cap.read()
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    #esp=sys.argv[5]
    ra=[None]*n
    recortes=[None]*n
    objeto=[None]*n
    min=[None]*n
    max=[None]*n
    h=[0]*n
    s=[0]*n
    v=[0]*n
    mask=[None]*n
    for i in range(n):
        print('Encierre el objeto a seguir')
        ra[i]=puntos_objeto(frame)
        #puntos.append(ra[i])
        objeto[i]=frame[int(ra[i][1]):int(ra[i][1]+ra[i][3]), int(ra[i][0]):int(ra[i][0]+ra[i][2])]
        color_predominante=buscar_rgb(objeto[i])
        #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
        color_pred=np.uint8([[color_predominante]])
        color_pred_hsv=cv.cvtColor(color_pred,cv.COLOR_BGR2HSV)
        #min[i]=[color_pred_hsv[0][0][0],100,100]
        h[i]=color_pred_hsv[0][0][0]
        s[i]=color_pred_hsv[0][0][1]
        v[i]=color_pred_hsv[0][0][2]
        #pdb.set_trace()
        lwr=np.array([h[i]-10,s[i]-30,v[i]-30])
        upr=np.array([h[i]+10,s[i]+30,v[i]+30])
        mask[i]= cv.inRange(hsv,lwr,upr)
        cv.imshow('Seleccion',mask[i])
    #if cv.waitKey(20) & 0xFF == 27:
        #break
    #cv.destroyWindow('Seleccion')
    #pdb.set_trace()
    #cv.destroyAllWindows()
    return(mask,h,s,v)
                
            




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
	h=[0]*n
	s=[0]*n
	v=[0]*n
	while(tec_esc != 27):            
            if(color=='--nocolor'):
                seguidor.run(None,puntos,cap,n,color)
            elif(color=='--color'):
                if(puntos!=None):
                    #for i in range(n):
                    img,h,s,v=seleccion(puntos,cap,n)                   
                    #for i in range(n-1):
                        #img[i]=cv.add(img[i],img[i-1])
                    aux=img
                    seguidor.runcolor(None,puntos,cap,n,color,img,aux)
            tec_esc=cv.waitKey(0)
	cv.destroyAllWindows()
