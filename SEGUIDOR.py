import cv2 as cv
import os
import sys
import numpy as np
import pdb
import math


def seleccion_objeto(frame):
    p=cv.selectROI(frame)
    cv.destroyWindow('ROI selector')
    recorte=frame[int(p[1]):int(p[1]+p[3]), int(p[0]):int(p[0]+p[2])]
    recorte_gris=cv.cvtColor(recorte,cv.COLOR_BGR2GRAY)
    #recorte_gris=cv.GaussianBlur(recorte_gris,(3,3),5)
    recorte_gris=cv.adaptiveThreshold(recorte_gris,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    recorte_gris=cv.bitwise_not(recorte_gris)
    #recorte_gris=cv.Canny(recorte_gris,100,200)
    _,con,_=cv.findContours(recorte_gris, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    vmax=max(con, key = cv.contourArea)    
    cv.drawContours(recorte, vmax, -1, (0,255,0), 1)
    while(True):
            cv.imshow('Recorte',recorte)
            tecla = cv.waitKey(5) & 0xFF
            if tecla == 27:
                    break
    con_elegido=vmax
    x,y,w,h = cv.boundingRect(con_elegido)
    aspect_radio = float(w)/h
    area=cv.contourArea(con_elegido)
    area_bounding=w*h
    extension=float(area)/area_bounding
    hull=cv.convexHull(con_elegido)
    area_hull=cv.contourArea(hull)
    solidez=float(area)/area_hull
    #tecla='a'
    #while(tecla=='a'):
            #tecla=cv.waitKey(5) & 0xFF
    #if(tecla==27):
    return(aspect_radio,extension,area_hull,solidez,con_elegido)
##    else:
##            print("El contorno seleccionado no es satisfactorio")
##            sys.exit()





if __name__=='__main__':
    cap=cv.VideoCapture(0)
    ok,frame=cap.read()
    if not ok:
        print("Error en apertura de camara")
        sys.exit()
    aspect_radio=0
    extension=0
    area_hull=0
    solidez=0
    con_elegido=None
    aspect_radio,extension,area_hull,solidez,con_elegido=seleccion_objeto(frame)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    fgbg=cv.createBackgroundSubtractorKNN()

    #pdb.set_trace()

    while(True):
        _,frame=cap.read()
        #umb = fgbg.apply(frame)
        #gray=cv.cvtColor(fgmask,cv.COLOR_BGR2GRAY)
        #umb=cv.GaussianBlur(gray,(9,9),15)
        #umb=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)	

        #canny=cv.Canny(frame,100,200)
        #umb=cv.morphologyEx(umb, cv.MORPH_GRADIENT, kernel)
        #umb=cv.bitwise_not(umb)
        umb=fgbg.apply(frame)
        _,con,_=cv.findContours(umb, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
        if (con):
            vmax=max(con, key = cv.contourArea)
            x,y,w,h = cv.boundingRect(vmax)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
 #           for c in vmax:
    #            if(cv.contourArea(c)>100):
    ##                x,y,w,h = cv.boundingRect(c)
    ##                l_asp_rad = float(w)/h
    ##                l_area=cv.contourArea(c)
    ##                l_area_bounding=w*h
    ##                l_extension=float(l_area)/l_area_bounding
    ##                l_hull=cv.convexHull(c)
    ##                l_area_hull=cv.contourArea(l_hull)
    ##                l_solidez=float(l_area)/l_area_hull
                    #pdb.set_trace()
#                res=cv.matchShapes(c,con_elegido,1,0)
                #print(res)
##                if(res<1):
##                    x,y,w,h = cv.boundingRect(c)
##                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
##                    cv.rectangle(umb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    

        #print(res)
            #cv.imshow('testing',umb)
    ##    conmax=vmax
    ##    x,y,w,h = cv.boundingRect(conmax)
    ##    l_asp_rad = float(w)/h
    ##    l_area=cv.contourArea(conmax)
    ##    l_area_bounding=w*h
    ##    l_extension=float(l_area)/l_area_bounding
    ##    l_hull=cv.convexHull(conmax)
    ##    l_area_hull=cv.contourArea(l_hull)
    ##    l_solidez=float(l_area)/l_area_hull
        #if()
        cv.imshow('frame',frame)
        cv.imshow('canny',umb)

        
        if(cv.waitKey(5) & 0xFF==ord('f')):
           break
#pdb.set_trace()
cv.destroyAllWindows()
#sys.exit()