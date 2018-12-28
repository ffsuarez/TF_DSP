'''
Como se pretende que sea el uso:

LK.py [--tecnica<=lk|shi-tom] [<fuente_video>] [n_objetos]

Donde: [--tecnica] decide cual tecnica tomar, Lucas Kanade o Shi-Tomasi
       [<fuente_video>] elige un archivo de video y lo lee, sino toma la camara
	   [n_objetos] es el numero de objetos a seguir
'''


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#---------------------------------------------------------------------
def selector_rois(n):
	for i in range(n):
		#https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
		r=cv.selectROI(frame)
		#seleccionar imagen
		Imcrop[i]=frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
		cv.waitKey(0)



#---------------------------------------------------------------------
class seguidor:
	def __init__(self,video_src):
		cap=cv.videoCapture(video_src)
	
	
	
	def opciones(self,metod,source):
		while(True):
			#https://stackoverflow.com/questions/4117530/sys-argv1-meaning-in-script
			#https://pymotw.com/2/getopt/
			if(metod=='lk'):
				lk_params = dict( winSize  = (15, 15),maxLevel = 2,criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
				feature_params = dict( maxCorners = 500,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
			
			elif(metod=='shi-tom'):
				maxCorners=25
				qualityLevel=0.01
				#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
				#https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html
				minDistance=10
			
			else:
				print('No se reconoce opcion metod:',metod)
				print('  O existe problema con la camara')
				sys.exit(1)
				


	def run (self):
		while(True):
			_ret,frame=cap.read()
			frame_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
			vis=frame.copy()
			selector_rois(n)
			
		


#---------------------------------------------------------------------
import os

from common import anorm, getsize

if __name__=='__main__':
	print(__doc__)
	import sys,getopt
	#opcs,args=getopt.getopt   #http://pyspanishdoc.sourceforge.net/lib/module-getopt.html
	
	metodo=sys.argv[1]
	video_src=sys.argv[2]
	n=sys.argv[3]
	#https://stackoverflow.com/questions/2709821/what-is-the-purpose-of-self
	#https://es.stackoverflow.com/questions/202588/como-funciona-self-en-python
	
	seguidor.opciones(metodo,video_src)
	seguidor.run()
	