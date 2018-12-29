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




#http://pyspanishdoc.sourceforge.net/lib/module-getopt.html	   
#https://www.digitalocean.com/community/tutorials/how-to-use-the-python-debugger
#https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
#https://stackoverflow.com/questions/4117530/sys-argv1-meaning-in-script
#https://pymotw.com/2/getopt/
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
#https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html
#https://stackoverflow.com/questions/2709821/what-is-the-purpose-of-self
#https://es.stackoverflow.com/questions/202588/como-funciona-self-en-python	
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
'''


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pdb






#---------------------------------------------------------------------
def puntos_objeto(frame):
	r=cv.selectROI(frame)
	#pdb.set_trace()
	return(r)

	
	
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
			lk_params = dict( winSize  = (15, 15),maxLevel = 2,criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
			feature_params = dict( maxCorners = 500,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )		
		elif(metod=='--shi'):
			maxCorners=25
			qualityLevel=0.01
			minDistance=10		
		else:
			print('No se reconoce opcion metod:',metod)
			print('  O existe problema con la camara')
			sys.exit(1)
			
	
	def run (self,puntos,cap,n,color):
            print('Comenzando trabajo')
            _,frame=cap.read()
            if(color=='--color'):
                    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
            elif(color=='--nocolor'):
                    frame_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            recortes=[None]*n
            for i in range(n):
                    r=puntos_objeto(frame)
                    puntos.append(r)
                    cv.destroyAllWindows()
                    if(color=='--color'):
                        recortes[i]=hsv[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                    elif(color=='--nocolor'):
                        recortes[i]=frame_gray[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                        pdb.set_trace()
                        recortes[i]=cv.adaptiveThreshold(recortes[i],255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)


		



#---------------------------------------------------------------------



import os
#import time
#import msvcrt
#from common import anorm, getsize

if __name__=='__main__':
	print(__doc__)
	import sys,getopt
	#opcs,args=getopt.getopt   
	metodo=sys.argv[1]
	video_src=sys.argv[2]
	if (video_src=='0'):
            video_src=int(video_src)
	n=sys.argv[3]
	n=int(n)
	color=sys.argv[4]

	#metodo='--lk'
	#video_src=0
	#n=2
	puntos=[None]*n
	tec_esc='a'
	seguidor.opciones(None,metodo)
	cap=seguidor.__init__(None,video_src)
	_,frame=cap.read()	
	while(tec_esc != 27):
		seguidor.run(None,puntos,cap,n,color)
		cv.namedWindow('Test Key') #necesaria para que waitkey funcione bien
		tec_esc=cv.waitKey(0)
	cv.destroyAllWindows()