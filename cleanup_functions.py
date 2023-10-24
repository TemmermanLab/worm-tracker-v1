# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:53:23 2021

@author: PDMcClanahan
"""

import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
import pdb, traceback, sys, code
from scipy import interpolate
from PIL import Image as Im, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, Label
from tkinter import *
import pickle
import time

# POST-PROCESSING / CLEANUP

def show_tracked_worm(vid,poses,w,f,show_img = False, dim = np.nan):
    # displays and returns a cropped frame from the raw video with the worm
    # centerline and head superimposed
    vid.set(cv2.CAP_PROP_POS_FRAMES, f-1)
    success,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.stack((img,img,img),2)
    centerline = poses[w,f]
    pts = np.int32(centerline)
    pts = pts.reshape((-1,1,2))
    #pdb.set_trace()
    
    # drawn = cv2.polylines(img, pts, True, (0,0,255), 1)
    img = cv2.circle(img, (pts[0,0,0],pts[0,0,1]), 1, (255,0,0), 1)
    R1 = int(np.min(centerline[:,1])); R1 = R1 - 10
    R2 = int(np.max(centerline[:,1])); R2 = R2 + 10
    C1 = int(np.min(centerline[:,0])); C1 = C1 - 10
    C2 = int(np.max(centerline[:,0])); C2 = C2 + 10
    if R1 < 0: R1 = 0
    if R2 > np.shape(img)[0]-1: R2 = np.shape(img)[0]-1
    if C1 < 0: C1 = 0
    if C2 > np.shape(img)[1]-1: C2 = np.shape(img)[1]-1
    img = img[R1:R2,C1:C2]
    
    #cropped = drawn[int(np.min(centerline[:,1])):int(np.max(centerline[:,1])),int(np.min(centerline[:,0])):int(np.max(centerline[:,0]))]
    if show_img:
        disp_img(img)
    
    return img
  
def remove_short_tracks(centroids, centerlines, min_f = 300):
    print('Eliminating worms tracked for fewer than '+str(min_f)+' frames...')
    for w in reversed(range(np.shape(centroids)[0])):
        if np.sum(~np.isnan(centroids[w,:,0])) < min_f:
            print('Eliminating worm track '+str(w))
            centroids = np.delete(centroids,w,0)
            centerlines.pop(w)
    print('Done removing short traces!')
    return centroids, centerlines

def remove_short_tracks_centroids_only(centroids, min_f = 300):
    print('Eliminating worms tracked for fewer than '+str(min_f)+' frames...')
    for w in reversed(range(np.shape(centroids)[0])):
        if np.sum(~np.isnan(centroids[w,:,0])) < min_f:
            print('Eliminating worm track '+str(w))
            centroids = np.delete(centroids,w,0)
            
    print('Done removing short traces!')
    return centroids

def clean_centerlines(centroids, centerlines, d_thr = 10):
    
    # 1) flips centerlines in consecutive strains so that the orientation is
    # not flipped from frame to frame
    # 2) removes worms where the position of either end shifts more than d_thr
    # in consecutive frames
    
    print('Flipping centerlines to make head tail orientation consistent' + 
          ' and removing worms where either end moves more than ' + str(d_thr) + 
          ' pixels in one frame')
    
    for w in reversed(range(len(centerlines))):
        for f in range(np.shape(centerlines[w])[0]-1):
            d11 = np.linalg.norm(np.float32(centerlines[w][f,0,:])-np.float32(centerlines[w][f+1,0,:]))
            d12 = np.linalg.norm(np.float32(centerlines[w][f,0,:])-np.float32(centerlines[w][f+1,-1,:]))
            d21 = np.linalg.norm(np.float32(centerlines[w][f,-1,:])-np.float32(centerlines[w][f+1,0,:]))
            d22 = np.linalg.norm(np.float32(centerlines[w][f,-1,:])-np.float32(centerlines[w][f+1,-1,:]))
            if np.max(np.partition((d11,d12,d21,d22), 1)[0:2]) > d_thr:
                print('Eliminating worm track '+str(w))
                centroids = np.delete(centroids,w,0)
                centerlines.pop(w)
                break
            elif d11 > d21:
                centerlines[w][f+1,:,:] = np.flipud(centerlines[w][f+1,:,:])
    print('Done cleaning centerlines!')
    return centroids, centerlines

def auto_head_tail(centerlines):
    # assumes that the end that moves the most, possibly due to foraging
    # movements, is the head
    print('Automatically detecting head / tail orientation')
    for w in range(len(centerlines)):
        sum1 = 0.0 # distance covered by first end of worm
        sum2 = 0.0 # other end of worm
        for f in range(np.shape(centerlines[w])[0]-1):
            sum1 = sum1 + np.linalg.norm(centerlines[w][f,0,:]-centerlines[w][f+1,0,:])
            #print(sum1)
            sum2 = sum2 + np.linalg.norm(centerlines[w][f,-1,:]-centerlines[w][f+1,-1,:])
        if sum2 > sum1:
            centerlines[w][:,:,:] = np.flip(centerlines[w][:,:,:],1)
            print('flipped head / tail in worm ' + str(w))
    print('Done automatically determining head-tail orientation')
    return centerlines

def create_vignettes(vid_name, vid_path, centroids, centerlines):
    
    # created cropped videos of each tracked worm, centered around the
    # centroid, thus allowing for rapid loading during manual head / tail and
    # behavioral scoring
    
    # calculate the size of window to use based on maximal extent of tracked
    # worms
    extents = np.empty(0)
    for w in range(len(centerlines)):
        for f in range(np.shape(centerlines[w])[0]):
            extent = np.linalg.norm(np.float32(centerlines[w][f,0,:])-np.float32(centerlines[w][f,-1,:]))
            extents = np.append(extents,extent)
    halfwidth = int(np.max(extents)/1.7)
    #pdb.set_trace()        
    v_out_w = halfwidth*2+1; v_out_h = v_out_w
    
    # set up
    vid = cv2.VideoCapture(vid_path+'\\'+vid_name)
    
    save_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking\\vignettes'
    if not os.path.exists(save_path):
        print('Creating directory for tracking output: '+save_path)
        os.makedirs(save_path)
    is_color = 0
    
    # create vignettes of each worm
    for w in range(len(centerlines)):
        #if w == 3: pdb.set_trace()
        save_name = 'w'+str(w)+'.avi'
        v_out = cv2.VideoWriter(save_path+ '\\' +save_name,
        cv2.VideoWriter_fourcc('M','J','P','G'), vid.get(cv2.CAP_PROP_FPS),
            (v_out_w,v_out_h), is_color)
        first = np.min(np.where(~np.isnan(centroids[w,:,0])))
        last = np.max(np.where(~np.isnan(centroids[w,:,0])))
    
        for f in range(first,last):
            msg = 'frame '+str(f-first+1)+' of '+str(last-first)+', track '+str(w)
            print(msg)
            vid.set(cv2.CAP_PROP_POS_FRAMES,f)
            frame = vid.read()[1]; frame = frame[:,:,0]
            canvas = np.uint8(np.zeros((np.shape(frame)[0]+halfwidth*2,np.shape(frame)[1]+halfwidth*2)))
            canvas[halfwidth:np.shape(frame)[0]+halfwidth,halfwidth:np.shape(frame)[1]+halfwidth] = frame
            centroid = np.uint16(np.round(centroids[w,f]))
            crop = canvas[centroid[1]:(centroid[1]+2*halfwidth),centroid[0]:(2*halfwidth+centroid[0])]
            v_out.write(crop)
        
        v_out.release()
        
    print('Done!')
    return halfwidth