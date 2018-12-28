'''
Como se pretende que sea el uso:

LK.py [--tecnica<=lk|shi-tom] [<fuente_video>]

Donde: --tecnica decide cual tecnica tomar, Lucas Kanade o Shi-Tomasi
       [<fuente_video>] elige un archivo de video y lo lee, sino toma la camara
'''


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import os

from common import anorm, getsize

if __name__=='__main__':
	print(__doc__)
	import sys,getopt
	#opcs,args=getopt.getopt   #http://pyspanishdoc.sourceforge.net/lib/module-getopt.html
	
	