import cv2 as cv
import os
import sys
import numpy as np

def seleccion_objeto(frame):
	p=cv.selectROI(frame)
	cv.destroyWindow('ROI selector')
	recorte=frame(int(r[i][1]):int(r[i][1]+r[i][3]), int(r[i][0]):int(r[i][0]+r[i][2]))
	recorte_gris=cv.cvtColor(recorte,cv.COLOR_BGR2GRAY)
	recorte_gris=cv.adaptiveThreshold(recorte_gris,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
	recorte_gris=cv.bitwise_not(recorte_gris)
	recorte_gris=cv.Canny(recorte_gris,100,200)
	_,con,_=cv.findContours(recorte_gris, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
	max=max(con, key = cv.contourArea)
	cv.drawContours(recorte, contours, max, (0,255,0), 3)
	while(True):
		cv.imshow('Recorte',recorte)
		tecla = cv.waitKey(5) & 0xFF
		if tecla == 27:
			break
	con_elegido=con[max]
	x,y,w,h = cv.boundingRect(cnt)
	aspect_radio = float(w)/h
	area=cv.contourArea(con_elegido)
	area_bounding=w*h
	extension=float(area)/area_bounding
	hull=cv.convexHull(con_elegido)
	area_hull=cv.contourArea(hull)
	solidez=float(area)/area_hull
	tecla='a'
	while(tecla=='a'):
		tecla=cv.waitKey(5) & 0xFF
	if(tecla==27):
		return(aspect_radio,extension,area_hull,solidez)
	else:
		print("El contorno seleccionado no es satisfactorio")
		sys.exit()




finalizar='a'
#loop principal
cap=cv.VideoCapture(0)
ok,frame=cap.read
if not ok:
	print("Error en apertura de camara")
	sys.exit()
aspect_radio=0
extension=0
area_hull=0
solidez=0
seleccion_objeto(frame)
while(finalizar!=27):
	gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
	umb=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)	
	umb=cv.bitwise_not(umb)
	umb=cv.Canny(umb,100,200)
	_,con,_=cv.findContours(umb, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
	max=max(con, key = cv.contourArea)
	cv.drawContours(frame, contours, max, (0,255,0), 3)
	conmax=con[max]
	x,y,w,h = cv.boundingRect(conmax)
	l_asp_rad = float(w)/h
	l_area=cv.contourArea(conmax)
	l_area_bounding=w*h
	l_extension=float(l_area)/l_area_bounding
	l_hull=cv.convexHull(conmax)
	l_area_hull=cv.contourArea(l_hull)
	l_solidez=float(l_area)/l_area_hull
	#if()
	
	finalizar=cv.waitKey(0)
