# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:55:41 2021

Issues / improvements:
    -would be nice to cycle back and forth
    -steps are not labeled
    -no way to zoom image
    -possibly helpful info: blob size, 
    -no way to change frame being inspected
    -cartoon showing worm size and distance threshold, and size threshold would help
    -scale calculating function

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

import tracking_functions as track_f



def tracking_param_selector(vid_name, bkgnd_meth = 'max_merge', \
    bkgnd_nframes = 10, k_sig = 2, k_sz = (11,11), bw_thr = 10, \
    sz_bnds = [349,800], d_thr = 10, del_sz_thr = 25, um_per_pix = 3.5486, \
    min_f = 150):
    
    # vars for frame
    bkgnd_meths = ('max_merge','min_merge')
    img_type_inds = [0,1,2,3,4,5]
    img_type_ind = 5
    

    #pdb.set_trace()
    vid = cv2.VideoCapture(vid_name)
    f = 0

    #import pdb; pdb.set_trace()
    # init outcome values
    bkgnd = track_f.get_background(vid, bkgnd_nframes, bkgnd_meth)
    f_width = 600
    f_height = int(np.round(600*(np.shape(bkgnd)[0]/np.shape(bkgnd)[1])))
    img, diff, smooth, bw, bw_sz, final = track_f.show_segmentation(vid, f, bkgnd,
            k_sig, k_sz, bw_thr, sz_bnds, d_thr, db = False)
    
    # load video
    vid = cv2.VideoCapture(vid_name)
    
    # button press variable (probably will not need these)
    update_background = False
    update_images = False
    toggle_view_pressed = False
    save_and_exit_pressed = False

    def compute_background_button():
        nonlocal bkgnd, bkgnd_meth
        bkgnd_meth = enter_bkgnd_meth.get()
        bkgnd_nframes = int(enter_bkgnd_nframes.get())
        if bkgnd_meth not in bkgnd_meths:
            print('Unrecognized background method '+str(bkgnd_meth)+', please enter one the following:'+print(bkgnd_methods))
        
        bkgnd = track_f.get_background(vid, bkgnd_nframes, bkgnd_meth)
    
    def update_images_button():
        print('Updating images...')
        nonlocal bkgnd_meth, bkgnd_nframes, k_sig, k_sz, bw_thr, d_thr, sz_bnds
        nonlocal img, diff, smooth, bw, bw_sz, final, del_sz_thr, min_f, um_per_pix
        
        bkgnd_meth = enter_bkgnd_meth.get()
        bkgnd_nframes = int(enter_bkgnd_nframes.get())
        k_sig = float(enter_k_sig.get())
        k_sz = 1+2*3*round(k_sig); k_sz = (k_sz,k_sz)
        min_f = int(enter_min_f.get())
        bw_thr = int(enter_bw_thr.get())
        d_thr = int(enter_d_thr.get())
        sz_bnds[0] = int(enter_min_sz.get())
        sz_bnds[1] = int(enter_max_sz.get())
        del_sz_thr = int(enter_del_sz_thr.get())
        if enter_um_per_pix.get() == 'None':
            um_per_pix = None
        else:
            um_per_pix = float(enter_um_per_pix.get())
        
        
        img, diff, smooth, bw, bw_sz, final = track_f.show_segmentation(vid, f, bkgnd,
            k_sig, k_sz, bw_thr, sz_bnds, d_thr, db = False)
        
        update_win(img_type_ind)
            
    def cycle_image_button():
        nonlocal img_type_ind
        img_type_ind = img_type_ind+1
        if img_type_ind >= len(img_type_inds):
            img_type_ind = 0
        print(img_type_ind)
        update_win(img_type_ind)
        
    def save_exit_button():
        vid.release()
        param_insp.destroy()
        param_insp.quit()
    
    def update_win(img_type_ind):
        nonlocal frame
        # pdb.set_trace()
        print(img_type_ind)
        if  img_type_ind == 0:
            frame = img
        elif img_type_ind == 1:
            frame = diff
        elif img_type_ind == 2:
            frame = smooth
        elif img_type_ind == 3:
            frame = bw
        elif img_type_ind == 4:
            frame = bw_sz
        elif img_type_ind == 5:
            frame = final
        
        
        frame = Im.fromarray(frame)
        frame = frame.resize((f_width,f_height),Im.NEAREST)
        frame = ImageTk.PhotoImage(frame)
        img_win.configure(image = frame)
        img_win.update()
    
    # def get_scale():
    #     # get and load scale image
    #     root = tk.Tk()
    #         initialdir = os.path.dirname(vid_name)
    #         scale_img_file = tk.filedialog.askopenfile(initialdir = '/', \
    #             title = "Select the folder containing videos to be tracked \
    #             ...")
    #         root.destroy()
    #     img = cv2.imread(scale_img_file,cv2.IMREAD_GRAYSCALE)
    #     imgc = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        
    #     # display image
    #     clicks_x,clicks_y = [],[]
        
        
    #     # measure an object in the scale image
    #     return um_per_pix
        
    # set up GUI
    param_insp = tk.Toplevel()
    param_insp.title('Tracking Parameter Inspection GUI')
    param_insp.configure(background = "black")
    
    # set up video window
    frame = Im.fromarray(final)
    frame = frame.resize((f_width,f_height),Im.NEAREST)
    frame = ImageTk.PhotoImage(frame)
    img_win = Label(param_insp,image = frame, bg = "black")
    img_win.grid(row = 0, column = 0, columnspan = 4, padx = 0, pady = 0)
    
    # set up text and input windows
    Label (param_insp,text="Background method (max_merge or min_merge):", bg = "black", fg = "white") .grid(row = 1, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_bkgnd_meth = Entry(param_insp, bg = "white")
    enter_bkgnd_meth.grid(row = 1, column = 1,padx=1, pady=1, sticky = W+E)
    enter_bkgnd_meth.insert(0,bkgnd_meth)
    
    Label (param_insp,text="Number of frames in background (integer):", bg = "black", fg = "white") .grid(row = 1, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_bkgnd_nframes = Entry(param_insp, bg = "white")
    enter_bkgnd_nframes.grid(row = 1, column = 3,padx=1, pady=1, sticky = W+E)
    enter_bkgnd_nframes.insert(0,bkgnd_nframes)
    
    
    Label (param_insp,text="Smoothing sigma:", bg = "black", fg = "white") .grid(row = 2, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_k_sig = Entry(param_insp, bg = "white")
    enter_k_sig.grid(row = 2, column = 1,padx=1, pady=1, sticky = W+E)
    enter_k_sig.insert(0,str(k_sig))
    
    Label (param_insp,text="Minimum frames in a track (integer):", bg = "black", fg = "white") .grid(row = 2, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_min_f = Entry(param_insp, bg = "white")
    enter_min_f.grid(row = 2, column = 3,padx=2, pady=1, sticky = W+E)
    enter_min_f.insert(0,str(min_f))
    
    # Label (param_insp,text="Smoothing kernel width:", bg = "black", fg = "white") .grid(row = 2, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    # enter_k_sz = Entry(param_insp, bg = "white")
    # enter_k_sz.grid(row = 2, column = 3,padx=2, pady=1, sticky = W+E)
    # enter_k_sz.insert(0,str(k_sz[0]))
    
    
    Label (param_insp,text="BW threshold (1-254):", bg = "black", fg = "white") .grid(row = 3, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_bw_thr = Entry(param_insp, bg = "white")
    enter_bw_thr.grid(row = 3, column = 1,padx=1, pady=1, sticky = W+E)
    enter_bw_thr.insert(0,bw_thr)
    
    Label (param_insp,text="Distance threshold:", bg = "black", fg = "white") .grid(row = 3, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_d_thr = Entry(param_insp, bg = "white")
    enter_d_thr.grid(row = 3, column = 3,padx=1, pady=1, sticky = W+E)
    enter_d_thr.insert(0,str(d_thr))
    
    
    Label (param_insp,text="Minimum area (integer):", bg = "black", fg = "white") .grid(row = 4, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_min_sz = Entry(param_insp, bg = "white")
    enter_min_sz.grid(row = 4, column = 1,padx=1, pady=1, sticky = W+E)
    enter_min_sz.insert(0,str(sz_bnds[0]))
    
    Label (param_insp,text="Maximum area (integer)", bg = "black", fg = "white") .grid(row = 4, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_max_sz = Entry(param_insp, bg = "white")
    enter_max_sz.grid(row = 4, column = 3,padx=1, pady=1, sticky = W+E)
    enter_max_sz.insert(0,str(sz_bnds[1]))
    
    Label (param_insp,text="Size change threshold (percentage)", bg = "black", fg = "white") .grid(row = 5, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_del_sz_thr = Entry(param_insp, bg = "black")
    enter_del_sz_thr.grid(row = 5, column = 1,padx=1, pady=1, sticky = W+E)
    enter_del_sz_thr.insert(0,str(del_sz_thr))
    
    Label (param_insp,text="Scale in \u03bcm per pixel (float)", bg = "black", fg = "white") .grid(row = 5, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_um_per_pix = Entry(param_insp, bg = "white")
    enter_um_per_pix.grid(row = 5, column = 3,padx=1, pady=1, sticky = W+E)
    enter_um_per_pix.insert(0,str(um_per_pix))
    
    # set up buttons
    Button(param_insp, text = "COMPUTE BACKGROUND", command = compute_background_button) .grid(row = 6, column = 0, padx=1, pady=1, sticky = W+E+N+S)
    Button(param_insp, text = "UPDATE IMAGES", command = update_images_button) .grid(row = 6, column = 1, padx=1, pady=1, sticky = W+E+N+S)
    Button(param_insp, text = "CYCLE IMAGE", command = cycle_image_button) .grid(row = 6, column = 2, padx=1, pady=1, sticky = W+E+N+S)
    Button(param_insp, text = "SAVE AND EXIT", command = save_exit_button) .grid(row = 6, column = 3, padx=1, pady=1, sticky = W+E+N+S)
    
    param_insp.mainloop()
    
    # wrapping up
    return bkgnd_meth, bkgnd_nframes, k_sig, list(k_sz), bw_thr, list(sz_bnds), d_thr, del_sz_thr, um_per_pix, min_f

# testing
if __name__ == '__main__':
    vid_name = r'C:\Users\Temmerman Lab\Desktop\Bram data\20210923\video_AX7163_A 21-09-23 15-37-15.avi'
    bkgnd_meth, bkgnd_nframes, k_sig, k_sz, bw_thr, sz_bnds, d_thr, \
        del_sz_thr, um_per_pix, min_f = tracking_param_selector(vid_name)

