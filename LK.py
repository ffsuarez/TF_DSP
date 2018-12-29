'''
Como se pretende que sea el uso:

LK.py [--tecnica<=lk|shi-tom] [<fuente_video>] [n_objetos]

Donde: [--tecnica] decide cual tecnica tomar, Lucas Kanade o Shi-Tomasi
       [<fuente_video>] elige un archivo de video y lo lee, sino toma la camara
	   [n_objetos] es el numero de objetos a seguir
'''
#https://www.digitalocean.com/community/tutorials/how-to-use-the-python-debugger

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pdb
#---------------------------------------------------------------------
def selector_rois(n,cap):
        _ret,frame=cap.read()
        Imcrop=[]
        for i in range(n):

            #https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
            r=cv.selectROI(frame)
            #pdb.set_trace()
            #seleccionar imagen

            Imcrop.append(frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])])
            
            cv.waitKey(0)
            cv.destroyAllWindows()
            return(Imcrop)



#---------------------------------------------------------------------
class seguidor:
	#def __init__(self):
		#self.tracks=[]
		#self.frame_idx = 0
		



	def opciones(self,metod):
		#https://stackoverflow.com/questions/4117530/sys-argv1-meaning-in-script
		#https://pymotw.com/2/getopt/
		if(metod=='--lk'):
			lk_params = dict( winSize  = (15, 15),maxLevel = 2,criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
			feature_params = dict( maxCorners = 500,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
		
		elif(metod=='--shi'):
			maxCorners=25
			qualityLevel=0.01
			#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
			#https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html
			minDistance=10
		
		else:
			print('No se reconoce opcion metod:',metod)
			print('  O existe problema con la camara')
			sys.exit(1)
			
	
	
	def seguimiento(self,Imcrop):
            #_ret,frame=cap.read()
            Imcrop_gray=[]
            umb=[]
            kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
            contours=[]
            maximo=[]
            momentos=[]
            aux=[None]*n
            I=[None]*n		
            for i in range(n):
                pdb.set_trace()
                I[i]=cv.cvtColor(Imcrop, cv.COLOR_BGR2GRAY)
                Imcrop_gray.append(I[i])
                umb.append(cv.adaptiveThreshold(Imcrop_gray[i],255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2))
                umb[i]=cv.morphologyEx(umb[i], cv.MORPH_OPEN, kernel)
                umb[i]=cv.morphologyEx(umb[i], cv.MORPH_CLOSE, kernel)
                umb[i]=cv.bitwise_not(umb)
                _,aux[i],_= cv.findContours(umb[i], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
		
		
			
	def run (self):
            try:
                cap=cv.VideoCapture(video_src)                            
            except:
                cap=cv.VideoCapture(0)                           
            #pdb.set_trace()
            Imcrop=[None]*n
            while(True):
                for i in range(n):
                    Imcrop[i]=selector_rois(n,cap)
                    Imcrop[i]=np.array(Imcrop[i])
                seguidor.seguimiento(None,Imcrop)
		


#---------------------------------------------------------------------
import os

#from common import anorm, getsize

if __name__=='__main__':
	print(__doc__)
	import sys,getopt
	#opcs,args=getopt.getopt   #http://pyspanishdoc.sourceforge.net/lib/module-getopt.html
##	metodo=sys.argv[2]
##	video_src=sys.argv[3]
##	n=sys.argv[4]
##	n=int(n)
	metodo='--lk'
	video_src=0
	n=2
	#https://stackoverflow.com/questions/2709821/what-is-the-purpose-of-self
	#https://es.stackoverflow.com/questions/202588/como-funciona-self-en-python	
	seguidor.__init__(None,video_src)    
	seguidor.opciones(None,metodo)
	seguidor.run(None)
	