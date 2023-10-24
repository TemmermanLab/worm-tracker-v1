# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:13:11 2021

This GUI allows click-through execution of the worm tracking code. The
workflow is:
    
    1. Select videos to track (already-selected videos are displayed in the 
                               GUI window)
    2. Set tracking parameters (opens an instance of the parameter GUI)
    3. Run tracking 
    4. Exit

Improvements:
    -measure scale
    -automatically set parameters according to scale and brightness
    -load parameters from a previously-tracked video
    -estimate time remaining
    -implement R-CNN-based tracking
    -add scoring
    -option of separate parameters for each video
    -reload parameters previously chosen

@author: Temmerman Lab
"""


import tkinter as tk
import tkinter.font
from tkinter import ttk
from tkinter import *
import tkinter.filedialog as filedialog # necessary to avoid error
import numpy as np
from PIL import Image, ImageTk
import os
import cv2
import copy
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append(os.path.split(__file__)[0])

import parameter_GUI
import data_management_functions as data_f
import tracking_functions as track_f
import jan_postprocessing as jan_f
import cleanup_functions as clean_f
import numpy as np

# tracking GUI
def tracking_GUI():
    
    # internal functions
    
    # update the information in the main window
    def update_vid_inf(vid_names, has_params, is_tracked, is_scored):
        nonlocal vid_inf
        print('update_vid_inf called')
        cols = ['name','params?','trcked?','scored?']
        info = np.empty((len(vid_names),4),dtype='object')
        for v in range(len(info)):
            info[v,0] = vid_names[v]
            if has_params[v]: info[v,1] = 'Yes';
            else: info[v,1] = 'No'
            if is_tracked[v]: info[v,2] = 'Yes';
            else: info[v,2] = 'No'
            if is_scored[v]: info[v,3] = 'Yes';
            else: info[v,3] = 'No'
        
        # clear old text, if any
        vid_inf.delete("1.0","end") 
        
        # print column names
        vid_inf.insert(tk.END, cols[0]+'\t\t\t\t\t\t\t\t\t')
        vid_inf.insert(tk.END, cols[1]+'\t')
        vid_inf.insert(tk.END, cols[2]+'\t')
        vid_inf.insert(tk.END, cols[3]+'\n')
        
        # print video information
        for v in range(len(info)):
            vid_inf.insert(tk.END, info[v,0]+'\t\t\t\t\t\t\t\t\t')
            vid_inf.insert(tk.END, info[v,1]+'\t')
            vid_inf.insert(tk.END, info[v,2]+'\t')
            vid_inf.insert(tk.END, info[v,3]+'\t\n')
        

    
    # load a video or videos to be tracked
    def load_folder():
        nonlocal vid_path, vid_names, v_tot, v, has_params, is_tracked, is_scored
        if len(vid_path)==0:
            root = tk.Tk()
            vid_path = tk.filedialog.askdirectory(initialdir = '/', \
                title = "Select the folder containing videos to be tracked \
                ...")
            root.destroy()
        else:
            print('video folder already loaded')
        
        print('Fetching video info '+vid_path)
        vid_names = os.listdir(vid_path)
        for v in reversed(range(len(vid_names))):
            if len(vid_names[v])<4 or vid_names[v][-4:] != '.avi':
                del(vid_names[v])
        
        v_tot = len(vid_names)
        
        for v in range(v_tot):
            if os.path.isfile(vid_path+'/'+vid_names[v][:-4]+'_tracking/tracking_parameters.csv'):
                has_params.append(True)
            else:
                has_params.append(False)
            if os.path.isfile(vid_path+'/'+vid_names[v][:-4]+'tracking/intensity_tracking_centroids.csv'):
                is_tracked.append(True)
            else:
                is_tracked.append(False)
            if os.path.isfile(vid_path+'/'+vid_names[v][:-4]+'tracking/intensity_tracking_nict_scores.csv'):
                is_scored.append(True)
            else:
                is_scored.append(False)
        
        update_vid_inf(vid_names, has_params, is_tracked, is_scored)
    

    # button functions
    
    def load_video_folder_button():
        print('load video folder button pressed')
        load_folder()


    def set_parameters_button():
        nonlocal params
        print('set parameters button pressed')
        print(vid_path +'/'+ vid_names[0])
        # bkgnd_meth, bkgnd_nframes, k_sig, k_sz, bw_thr, sz_bnds, \
        #     d_thr, del_sz_thr = parameter_GUI.tracking_param_selector( \
        #     vid_path +'/'+ vid_names[0])
        
        params['bkgnd_meth'], params['bkgnd_nframes'], params['k_sig'], \
            params['k_sz'], params['bw_thr'], params['sz_bnds'], \
            params['d_thr'], params['del_sz_thr'], params['um_per_pix'], \
            params['min_f'] =  \
            parameter_GUI.tracking_param_selector(vid_path +'/'+ vid_names[0])
        
        vid = data_f.load_video(vid_path +'/'+ vid_names[0])[2]
        params['fps'] = vid.get(cv2.CAP_PROP_FPS)
        del vid
        for v in vid_names:
            data_f.save_params_csv(params,vid_path,v)
    
    def track_button():
        print('track button pressed')
        for v in range(len(vid_names)):
            centroids = track_f.track_worms(vid_names[v], vid_path, 
                params['bkgnd_meth'], params['bkgnd_nframes'],
                params['k_sig'], params['k_sz'], params['bw_thr'],
                params['sz_bnds'], params['d_thr'], params['del_sz_thr'], 0.5)
            data_f.write_centroids_csv(centroids,vid_path,vid_names[v],save_name = 'centroids_raw')
            centroids = clean_f.remove_short_tracks_centroids_only(centroids, params['min_f'])
            data_f.write_centroids_csv(centroids,vid_path,vid_names[v],save_name = 'centroids_zonder_short_traces')
            speed_masked = jan_f.get_avg_speeds(centroids,params['fps'],params['um_per_pix']/1000)
            data_f.write_summary_outputs_csv(speed_masked,vid_path,vid_names[v])


               
    def exit_button():
        nonlocal tracking_GUI
        print('exit button pressed')

        tracking_GUI.destroy()
        tracking_GUI.quit()

    
    # initialize variables
    vid_path = []
    vid_names = []
    v = None
    v_tot = None
    has_params = []
    params = {}
    is_tracked = []
    is_scored = []
    h = 18; w = 100 # in lines and char, based around vid inf window
    
    # set up
    
    # GUI
    tracking_GUI = tk.Tk()
    tracking_GUI.title('Tracking GUI')
    tracking_GUI.configure(background = "black")
    # get character size / line spacing in pixels
    chr_h_px = tkinter.font.Font(root = tracking_GUI, font=('Courier',12,NORMAL)).metrics('linespace')
    chr_w_px = tkinter.font.Font(root = tracking_GUI, font=('Courier',12,NORMAL)).measure('m')
    # make the main window as wide and a bit taller than the vid info window
    tracking_GUI.geometry(str(int(w*chr_w_px))+"x"+str(int(chr_h_px*(h+3))))
    
    
    # to do text
    todo_txt = tk.Label(text = 'load a folder containing videos for tracking')
    todo_txt.grid(row = 0, column = 0, columnspan = 4, padx = 0, pady = 0)


    # informational window
    vid_inf = Text(tracking_GUI, height = h, width = w)
    vid_inf.configure(font=("Courier", 12))
    vid_inf.grid(row = 1, column = 0, columnspan = 4, padx = 0, pady = 0)
    
    
    # buttons
    tk.Button(tracking_GUI, text = "LOAD VIDEO FOLDER", command = load_video_folder_button, width = 10) .grid(row = 2, column = 0, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    tk.Button(tracking_GUI, text = "SET TRACKING PARAMETERS", command = set_parameters_button, width = 10) .grid(row = 2, column = 1, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    tk.Button(tracking_GUI, text = "TRACK!", command = track_button,width = 10) .grid(row = 2, column = 2, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    tk.Button(tracking_GUI, text = "EXIT", command = exit_button,width = 10) .grid(row = 2, column = 3, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')

    
    tracking_GUI.mainloop()


tracking_GUI()

