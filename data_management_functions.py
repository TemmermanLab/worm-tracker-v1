# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:56:47 2021

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
import csv

# LOADING AND SAVING PARAMS AND DATA
# saves a dictionary of parameters used in tracking with the video in a 
# separate folder

def load_video(video_file = ''):
    if len(video_file) == 0:
        root = tk.Tk()
        root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select a video file...",filetypes=(("Video files", "*.mp4;*.avi"),
             ("All files", "*.*") ))
        video_file = root.filename
        root.destroy()
    vid_path,vid_name = os.path.split(video_file)
    vid = cv2.VideoCapture(video_file)
    return vid_name,vid_path, vid

# # testing
# vid_name = 'test.avi'
# vid_path = 'E:\20210322_suspended_arena_test_2'
# bkgnd_meth = 'max_merge'
# bkgnd_nframes = 10
# k_sig = 5
# k_sz = 11
# bw_thr = 15
# sz_bnds = (300,750)
# save_params_csv(vid_path, vid_name, bkgnd_meth, bkgnd_nframes,
#     k_sig, k_sz, bw_thr, sz_bnds)


def save_params_csv(params,vid_path,vid_name,save_name = 'tracking_parameters'):
    save_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking'
    
    if not os.path.exists(save_path):
        print('Creating directory for tracking parameters and output: '+save_path)
        os.makedirs(save_path)
    
    save_file_csv = save_path + '\\' + save_name + '.csv'
    
    with open(save_file_csv, mode='w',newline="") as csv_file: 
        keys = list(params.keys())
        
        parameters_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        row = ['Parameter','Value']
        parameters_writer.writerow(row)
        
        for r in range(len(params)):
            row = [keys[r],str(params[keys[r]])]
            parameters_writer.writerow(row)
        
    print("Tracking parameters saved as " + save_path + "/tracking_params.csv" )
    

def write_centroids_csv(centroids,vid_path,vid_name,save_name = 'centroids'):
    save_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking'
    
    if not os.path.exists(save_path):
        print('Creating directory for centroids csv and other output: '+save_path)
        os.makedirs(save_path)
    
    save_file_csv = save_path + '\\' + save_name + '.csv'
    
    with open(save_file_csv, mode='w',newline="") as csv_file: 
        
        writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        row = ['First Frame of Track','X and then Y Coordinates on Alternating Rows']
        writer.writerow(row)
        
        for t in range(len(centroids)):
            first_frame = np.where(~np.isnan(centroids[t,:,0]))[0][0]
            last_frame = np.where(~np.isnan(centroids[t,:,0]))[0][-1]
            x_row = [str(first_frame)]
            y_row = ['']
            for i in np.arange(first_frame,last_frame+1):
                x_row.append(str(round(centroids[t,i,0],1)))
                y_row.append(str(round(centroids[t,i,1],1)))
            writer.writerow(x_row)
            writer.writerow(y_row)
        
    print("Centroids saved as " + save_file_csv )


def read_centroids_csv(centroids_file):
    xs = []
    ys = []
    ffs = []
    
    with open(centroids_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                print(f'Column names are {", ".join(row)}')
                row_count += 1
            elif np.mod(row_count,2)==0:
                ys.append(np.array(row[1:],dtype='float32')); row_count += 1
            else:
                ffs.append(int(row.pop(0)))
                xs.append(np.array(row,dtype='float32')); row_count += 1
    
    # reshape into the old format
    track_ends = []
    for t in range(len(ffs)):
        track_ends.append(ffs[t] + len(xs[t]))
    dim2 = np.max(track_ends)
    dim1 = len(xs)
    dim3 = 2
    centroids = np.empty((dim1,dim2,dim3)); centroids[:] = np.nan
    
    for t in range(len(ffs)):
        centroids[t,ffs[t]:ffs[t]+len(xs[t]),0] = xs[t]
        centroids[t,ffs[t]:ffs[t]+len(xs[t]),1] = ys[t]
    
    return centroids
    

# open parameter dictionary
def load_parameter_csv_dict(vid_name,vid_path,save_name = 'tracking_parameters'):
    
    #pdb.set_trace()
    
    load_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking'
    
    load_file_csv = load_path + '\\' + save_name + '.csv'
    
    params = dict()

    with open(load_file_csv, newline="") as csv_file: 
        parameters_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for r in parameters_reader:
            if r[0] == 'sz_bnds':
                transdict = {91: None, 93: None, 40: None, 41: None}
                r[1] = r[1].translate(transdict).split(sep=',')
                r[1] =  [int(r[1][n]) for n in range(len(r[1]))]
            elif r[0] == 'k_sz':
                # pdb.set_trace()
                transdict = {91: None, 93: None, 40: None, 41: None}
                r[1] = r[1].translate(transdict).split(sep=',')
                r[1] =  [int(r[1][n]) for n in range(len(r[1]))]
            elif r[0] == 'k_sig':
                r[1] = float(r[1])
            elif not any(c.isalpha() for c in r[1]): # should be true for any windows path or vid_name due to the file ext
                r[1] = int(r[1])
            params[r[0]] = r[1]
    
    return params


def write_summary_outputs_csv(speed_masked,vid_path,vid_name,save_name = 'summary_outputs'):
    save_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking'
    
    if not os.path.exists(save_path):
        print('Creating directory for summary and other outputs: '+save_path)
        os.makedirs(save_path)
    
    save_file_csv = save_path + '\\' + save_name + '.csv'
    
    with open(save_file_csv, mode='w',newline="") as csv_file: 
        
        writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        row = ['Track Number','Total Frames','Running Frames (not Pause, not Pirouette/Burst)','Average Speed (mm/s)']
        writer.writerow(row)
        
        for t in range(len(speed_masked)):
            row = []
            row.append(str(t))
            row.append(str(len(speed_masked[t])))
            row.append(str(len(np.where(~np.isnan(speed_masked[t]))[0])))
            if len(np.where(~np.isnan(speed_masked[t]))[0]) > 0:
                row.append(str(np.nanmean(speed_masked[t])))
            else:
                row.append('n/a')
            writer.writerow(row)
        
    print("Saved " + save_file_csv )


# def save_params_csv(vid_path, vid_name, bkgnd_meth = -1, bkgnd_nframes = -1,
#     k_sig = -1, k_sz = -1, bw_thr = -1, sz_bnds = -1, d_thr = -1,
#     d_thrHT = -1, min_f = -1, halfwidth = -1):
    
#     save_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking'
#     if not os.path.exists(save_path):
#         print('Creating directory for tracking parameters and output: '+save_path)
#         os.makedirs(save_path)
#     save_file_csv = save_path + '\tracking_parameters.csv'
    
#     with open(save_file_csv, mode='w',newline="") as csv_file:
#             #pdb.set_trace()
            
#         save_dict = dict([('vid_path',vid_path),
#             ('vid_name',vid_name),
#             ('bkgnd_meth',bkgnd_meth),
#             ('bkgnd_nframes',bkgnd_nframes),
#             ('k_sig',k_sig),
#             ('k_sz',k_sz),
#             ('bw_thr',bw_thr),
#             ('sz_bnds',sz_bnds),
#             ('d_thr',d_thr),
#             ('d_thrHT',d_thrHT),
#             ('min_f',min_f),
#             ('halfwidth',halfwidth),
#             ])
            
#             parameters_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#             # complicated because csv writer only writes rows
#             row = []
#             for ww in range(len(scores)): row.append('worm '+str(ww))
#             parameters_writer.writerow(row)
            
#             num_frames = []
#             for s in scores: num_frames.append(len(s))
#             num_r = np.max(num_frames)
#             for r in range(num_r):
#                 row = []
#                 for ww in range(len(scores)):
#                     if r < len(scores[ww]):
#                         row.append(scores[ww][r])
#                     else:
#                         row.append('')
#                 parameters_writer.writerow(row)
    
    
#     print("Tracking parameters saved as " + save_path + "/tracking_params.csv" )


# def save_params(vid_path, vid_name, bkgnd_meth = -1, bkgnd_nframes = -1,
#     k_sig = -1, k_sz = -1, bw_thr = -1, sz_bnds = -1, d_thr = -1,
#     d_thrHT = -1, min_f = -1, halfwidth = -1):
    
#     save_dict = dict([('vid_path',vid_path),
#         ('vid_name',vid_name),
#         ('bkgnd_meth',bkgnd_meth),
#         ('bkgnd_nframes',bkgnd_nframes),
#         ('k_sig',k_sig),
#         ('k_sz',k_sz),
#         ('bw_thr',bw_thr),
#         ('sz_bnds',sz_bnds),
#         ('d_thr',d_thr),
#         ('d_thrHT',d_thrHT),
#         ('min_f',min_f),
#         ('halfwidth',halfwidth),
#         ])
    
#     save_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking'
#     if not os.path.exists(save_path):
#         print('Creating directory for tracking output: '+save_path)
#         os.makedirs(save_path)
    
#     pickle.dump(save_dict, open( save_path+"/tracking_params.p", "wb" ) )
    
#     print("Tracking parameters saved as " + save_path + "/tracking_params.p" )
  
# def load_params(vid_path, vid_name):
#     save_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking'
#     if os.path.exists(save_path+"//tracking_params.p"):
#         load_dict = pickle.load( open( save_path + "//tracking_params.p", "rb" ) )
#         bkgnd_meth = load_dict['bkgnd_meth']
#         bkgnd_nframes = load_dict['bkgnd_nframes']
#         k_sig = load_dict['k_sig']
#         k_sz = load_dict['k_sz']
#         bw_thr = load_dict['bw_thr']
#         sz_bnds = load_dict['sz_bnds']
#         d_thr = load_dict['d_thr']
#         d_thrHT = load_dict['d_thrHT']
#         min_f = load_dict['min_f']
#         halfwidth = load_dict['halfwidth']
        
#         print("Tracking parameters loaded from " + save_path + "/tracking_params.p" )
#         return bkgnd_meth, bkgnd_nframes, k_sig, k_sz, bw_thr, sz_bnds, d_thr, d_thrHT, min_f, halfwidth
    
#     else:
#         print('Tracking parameter file not found')
    
# def save_analysis_file(save_name = None): # call with save_analysis_file(dir())
#     # bkgnd, poses_raw, trks_raw, poses_auto_clean, trks_auto_clean,
#     # trks_man_clean, poses_man_clean, nict_auto, nict_man, angles, M
#     saveable = np.array(['vid_name','vid_path','bkgnd','trks', 'poses'])
#     to_save = {}
#     for v in saveable:
#         if v in globals():
#             to_save[v] = globals()[v]
            
#     if save_name is None:
#         save_name = get_save_name()
    
#     if os.path.isfile(save_name):
#         print('will load what is there and update as needed')
    
#     with open(save_name,'wb') as f:
#         pickle.dump(to_save,f)
    
#     cntr = 1
#     msg = 'Saved '
#     for key in to_save.keys(): 
#         if len(to_save) == 0:
#             msg = 'No variables to save'
#         elif len(to_save) == 1:
#             msg = msg+str(key);
#         elif cntr < len(to_save) and len(to_save) == 2:
#             msg = msg+str(key); cntr=cntr+1;
#         elif cntr < len(to_save):
#             msg = msg+str(key)+','; cntr=cntr+1; print(cntr)
#         else:
#             msg = msg+' and '+str(key);
#     print(msg + ' in ' + save_name)   
    
#     return save_name

# def load_analysis_file():
#     open_name = get_open_name()
#     with open(open_name, 'rb') as f:
#         data_loaded = pickle.load(f)
    
#     cntr = 1
#     msg = 'Loaded '
#     for key in data_loaded.keys(): 
#         if len(data_loaded) == 0:
#             msg = 'No variables to load'
#         elif len(data_loaded) == 1:
#             msg = msg+str(key);
#         elif cntr < len(data_loaded) and len(data_loaded) == 2:
#             msg = msg+str(key); cntr=cntr+1;
#         elif cntr < len(data_loaded):
#             msg = msg+str(key)+','; cntr=cntr+1; print(cntr)
#         else:
#             msg = msg+' and '+str(key);
#     print(msg + ' from ' + open_name)
    
#     globals().update(data_loaded)
    
#     return open_name

def test_fun(blah = None):
    if blah is None:
        print('blah was not passed')
    else:
        print('blah was passed')

def get_save_name(data_path = "/"):
    
    root = tk.Tk()
    save_name = filedialog.asksaveasfilename(initialdir = data_path,title = "Select directory and name for processed data file (.p)...")
    root.destroy()
    return save_name

def get_open_name(data_path = "/"):
    root = tk.Tk()
    open_name = filedialog.askopenfilename(initialdir = data_path,title = "Select the processed data file (.p) to open...")
    root.destroy()
    return open_name