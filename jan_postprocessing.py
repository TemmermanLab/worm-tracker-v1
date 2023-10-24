# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 17:02:30 2021

@author: Temmerman Lab
"""

import pickle
import numpy as np
import copy
import cv2


# get x and y coordinates from the worm traces
def xy_from_centroids(centroids):
    xs = []
    ys = []
    for w in range(len(centroids)):
        inds = np.where(~np.isnan(centroids[w,:,0]))
        xs.append(centroids[w,:,0][inds])
        ys.append(centroids[w,:,1][inds])
    return xs, ys


# calculate the smoothed x and y positions, assign nan to edges
def smooth_xy(xs):
    xs_smooth = []
    for w in range(len(xs)):
        xs_smooth_w = []
        for f in range(len(xs[w])):
            if f > 1 and f < len(xs[w])-1:
                xs_smooth_w.append(np.mean(xs[w][f-2:f+1]))
            else:
                xs_smooth_w.append(np.nan)
        xs_smooth.append(xs_smooth_w)
    return xs_smooth


# calculate speed (normally based on smoothed positions)
def calc_speed(xs,ys,frametime,mm_per_pix):
    speed = []
    for w in range(len(xs)):
        speed_w = []
        for f in range(len(xs[w])):
            if f>2 and f < len(xs[w])-1:
                dist = np.sqrt((xs[w][f]-xs[w][f-1])**2 + \
                    (ys[w][f]-ys[w][f-1])**2)
                speed_w.append((dist/frametime)*mm_per_pix)
            else:
                speed_w.append(np.nan)
        speed.append(speed_w)
    return speed


# find "pause" frames where smoothed speed is less than 0.01 mm / s
# pauses = 1, otherwise 0
def find_pauses_by_f(speed,thresh = 0.01):
    pause_by_f = []
    for w in range(len(speed)):
        pause_w = np.zeros(len(speed[w]))
        pause_inds = [i for i,v in enumerate(speed[w]) if v < thresh] 
        pause_w[pause_inds] = 1 # works with == but not <
        pause_by_f.append(pause_w)
    return pause_by_f


# find the 1 s bins where the smoothed speed is less that 0.01 mm / s on 
# average
def find_pauses_by_bin(speed,fps,thresh = 0.01):
    if fps == 5:
        pause_bin = [] # pause state of 1 sec bins
        pause_by_bin = [] # pause state of 1 sec bins mapped onto original frames
        bin_length = fps
        for w in range(len(speed)):
            num_bins = np.floor(len(speed[w])/5)
            rem = np.mod(len(speed[w]),5)
            pause_bin_w = []
            pause_by_bin_w = []
            for b in range(int(num_bins)):
                if rem == 0 and b == num_bins-1: # end considered not pause due to nan edge if last bin includes last frame
                    pause_bin_w.append(0)
                    for i in range(5): pause_by_bin_w.append(0)
                elif b < 1: # beginning automatically considered not pause due to nan edges
                    pause_bin_w.append(0)
                    for i in range(5): pause_by_bin_w.append(0)
                else:
                    half_1 = (speed[w][b*5]+speed[w][b*5+1]+0.5*speed[w][b*5+2])/2.5
                    half_2 = (speed[w][b*5+3]+speed[w][b*5+4]+0.5*speed[w][b*5+2])/2.5
                    if np.mean([half_1,half_2]) < thresh:
                        pause_bin_w.append(1)
                        for i in range(5): pause_by_bin_w.append(1)
                    else:
                        pause_bin_w.append(0)
                        for i in range(5): pause_by_bin_w.append(0)
            
            # pad out pause by bin with zeros if a full 1 sec bin cannot fit
            while len(speed[w]) > len(pause_by_bin_w):
                pause_by_bin_w.append(0)
                
            pause_bin.append(pause_bin_w)
            pause_by_bin.append(pause_by_bin_w)
    else:
        print('WARNING: code written specifically for 5 fps data to compare with Jan rig 2 fps data')
    return pause_by_bin


# find "pirouette" frames where the angle made by the current position and the
# positions where the path crosses a circle 300 microns is less than 80 deg.
# This is equivalent to the angle between the two path vectors coming from the
# edge of the circle in and the current position out to the edge of the circe
# being greater than 2.44 rad. In the method of calculating the angle used 
# here, opposite pointing vectors yield and angle of 2 pi, parallel vectors an
# angle of 0.

# Note: not attempt is made at interpolation of the path through the radius
# 300 micron circle

# Note: this can take awhile to run

def get_dist(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

# pirouettes = 1, not piroutte = 0, not knowable = nan
def find_pirouettes(xs,ys,mm_per_pix, rad = 0.15, ang_thr = (80/180)*np.pi):
    pirouette = []
    for w in range(len(xs)):
        print('finding pirouettes in worm track '+str(w+1)+' of '+str(len(xs)))
        pirouette_w = []
        for f in range(len(xs[w])):
            # find distances of positions in the future and past, stop when 300 microns is
            # reached, or end/beginning of track is reached
            fut_pt = []
            past_pt = []
            for ff in np.arange(f,len(xs[w]),1):
                if get_dist(xs[w][f],ys[w][f],xs[w][ff],ys[w][ff])*mm_per_pix > rad:
                    fut_pt = [xs[w][ff],ys[w][ff]]; break;
            for ff in np.arange(f,-1,-1):
                if get_dist(xs[w][f],ys[w][f],xs[w][ff],ys[w][ff])*mm_per_pix > rad:
                    past_pt = [xs[w][ff],ys[w][ff]]; break;
            
            if len(fut_pt) != 0 and len(past_pt) != 0:
                cur_pt = [xs[w][f],ys[w][f]]
                v1 = [fut_pt[0]-cur_pt[0],fut_pt[1]-cur_pt[1]]
                v2 = [cur_pt[0]-past_pt[0],cur_pt[1]-past_pt[1]]
                ang = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
                # print(ang)
                if ang < ang_thr:
                    pirouette_w.append(0)
                else:
                    pirouette_w.append(1)
            else:
                pirouette_w.append(np.nan)
        pirouette.append(pirouette_w)
            
    # # testing
    # past_pt = [1,0]
    # cur_pt = [0,0]
    # fut_pt = [1,0]
    return(pirouette)

# find pirouette_bursts, which include pirouettes and periods of less than 3.8
# s between them. Note: I need to re-examine Jan's code to see if it matters
# whether the animal is pausing or not.
# pirouettes and bursts = 1, other = 0, unknown = nan
def find_pirouette_bursts(pirouettes,fps):
    pirouette_bursts = np.array(copy.deepcopy(pirouettes),dtype=object)
    burst_thr = 3.8*fps
    for w in range(len(pirouette_bursts)):
        btwn_pirouettes = False
        for f in range(len(pirouette_bursts[w])):
            if np.isnan(pirouette_bursts[w][f]):
                btwn_pirouettes = False
            elif pirouette_bursts[w][f] == 1 and pirouette_bursts[w][f+1] == 0:
                btwn_pirouettes = True
                fb1 = f+1
                count = 1
            elif btwn_pirouettes and pirouette_bursts[w][f] == 0 and pirouette_bursts[w][f+1] == 0:
                count += 1
            elif btwn_pirouettes and pirouette_bursts[w][f] == 0 and pirouette_bursts[w][f+1] == 1:
                if count <= burst_thr:
                    pirouette_bursts[w][fb1:f] = np.ones(len(pirouette_bursts[w][fb1:f]))
                    #print('found burst, worm '+str(w)+', frame '+str(fb1))
                btwn_pirouettes = False
            else:
                pass
    return(pirouette_bursts)

def mask_speed(speed,pauses_by_bin,pirouette_bursts):
    masked_speed = copy.deepcopy(speed)
    for w in range(len(speed)):
        for f in range(len(speed[w])):
            if pauses_by_bin[w][f] == 1 or np.isnan(pauses_by_bin[w][f]):
                masked_speed[w][f] = np.nan
            elif pirouette_bursts[w][f] == 1 or np.isnan(pirouette_bursts[w][f]):
                masked_speed[w][f] = np.nan
    return masked_speed


# create heatmaps of behavioral states
import matplotlib.pyplot as plt


# smoothed speed
def plot_speed(speed):
    height = len(speed)
    lengths = []
    maxes = []
    for track in speed: lengths.append(len(track)); maxes.append(np.nanmax(track))
    width = np.max(lengths)
    max_speed = np.nanmax(maxes)
    heatmap = np.ones((height,width,3),dtype='uint8')*255
    for w,track in enumerate(speed):
        for f,frame in enumerate(track):
            if np.isnan(frame):
                heatmap[w,f,:] = [255,255,255]    
            else: 
                c = np.uint8(((frame/max_speed)*255));
                heatmap[w,f,:] = [c,c,c]
                #heatmap[w,f,:] = [255,255,255] 
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap[:,:,0],cmap = 'hot')
    plt.xlabel('frame')
    plt.ylabel('track')
    plt.title('speed (based on 3 frame smoothed position)')
    #ax.set_aspect('equal',adjustable='box')
    ax.set_aspect(1./ax.get_data_ratio())
    plt.show()


# pause by f
def plot_pauses(pauses,title = 'pause state'):
    height = len(pauses)
    lengths = []
    for track in pauses: lengths.append(len(track))
    width = np.max(lengths)
    heatmap = np.ones((height,width,3),dtype = 'uint8')*255
    for w,track in enumerate(pauses):
        for f,frame in enumerate(track):
            if np.isnan(frame): heatmap[w,f,:] = [125,125,125]
            elif frame == 0: heatmap[w][f][:] = [0,255,0];
            elif frame == 1: heatmap[w,f,:] = [255,0,0]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.xlabel('frame')
    plt.ylabel('track')
    plt.title(title)
    #ax.set_aspect('equal',adjustable='box')
    ax.set_aspect(1./ax.get_data_ratio())
    plt.show()


# pirouette
def plot_pirouettes(pirouettes):
    height = len(pirouettes)
    lengths = []
    for track in pirouettes: lengths.append(len(track))
    width = np.max(lengths)
    heatmap = np.ones((height,width,3),dtype = 'uint8')*255
    for w,track in enumerate(pirouettes):
        for f,frame in enumerate(track):
            if np.isnan(frame): heatmap[w,f,:] = [125,125,125]
            elif frame == 0: heatmap[w][f][:] = [0,0,255];
            elif frame == 1: heatmap[w,f,:] = [255,0,255]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.xlabel('frame')
    plt.ylabel('track')
    plt.title('pirouettes')
    #ax.set_aspect('equal',adjustable='box')
    ax.set_aspect(1./ax.get_data_ratio())
    plt.show()


# pirouettes and bursts
def plot_pirouette_bursts(pirouette_bursts):
    height = len(pirouette_bursts)
    lengths = []
    for track in pirouette_bursts: lengths.append(len(track))
    width = np.max(lengths)
    heatmap = np.ones((height,width,3),dtype = 'uint8')*255
    for w,track in enumerate(pirouette_bursts):
        for f,frame in enumerate(track):
            if np.isnan(frame): heatmap[w,f,:] = [125,125,125]
            elif frame == 0: heatmap[w][f][:] = [0,0,255];
            elif frame == 1:
                heatmap[w,f,:] = [150,0,0]
                if pirouette_bursts[w][f] == 1:
                    heatmap[w,f,:] = [255,0,255]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.xlabel('frame')
    plt.ylabel('track')
    plt.title('pirouettes and pirouette bursts')
    #ax.set_aspect('equal',adjustable='box')
    ax.set_aspect(1./ax.get_data_ratio())
    plt.show()

# where speed is masked
def plot_speed_masks(masked_speed):
    height = len(masked_speed)
    lengths = []
    for track in masked_speed: lengths.append(len(track))
    width = np.max(lengths)
    heatmap = np.ones((height,width,3),dtype = 'uint8')*255
    for w,track in enumerate(masked_speed):
        for f,frame in enumerate(track):
            if np.isnan(frame): heatmap[w,f,:] = [0,0,0]
            else: heatmap[w][f][:] = [125,125,125];
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.xlabel('frame')
    plt.ylabel('track')
    plt.title('places where speed is censored (in black)')
    #ax.set_aspect('equal',adjustable='box')
    ax.set_aspect(1./ax.get_data_ratio())
    plt.show()


# calculate behavioral states


# scale as measured in imagej
mm_per_pix = 5/662.24
fps = 5.0
frametime = 1.0/fps


# plot a track color-coded by behavior on an image
def plot_state_on_img(xs,ys,state = None, img = None,colors = None, thickness = 1):
    # convert the image to color
    xs = np.round(xs).astype('int32'); ys = np.round(ys).astype('int32')
    if state is None:
        state = np.zeros(len(xs))
    if colors is None:
        colors = [[255,0,0],[0,255,0],[125,125,125]]
    if img is None:
        img = np.zeros((int(np.ceil(np.max(ys))),int(np.ceil(np.max(xs)))),dtype = 'uint8')
        crop = True
    else:
        crop = False
        
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    for p in range(len(xs)-1):
        if np.isnan(state[p]):
            color = colors[2]
        else:
            color = colors[int(state[p])]
        start_pt = (xs[p],ys[p]); end_pt = (xs[p+1],ys[p+1])
        img = cv2.line(img, start_pt, end_pt, color, thickness)
        
    if crop:
        img = img[np.floor(np.min(ys)).astype('int32'):,np.floor(np.min(xs)).astype('int32'):,:]
    
    return img

def get_avg_speeds(centroids,fps,mm_per_pix,show = False):
    frametime = 1/fps
    xs,ys = xy_from_centroids(centroids)
    xs_smooth = smooth_xy(xs)
    ys_smooth = smooth_xy(ys)
    speed = calc_speed(xs_smooth,ys_smooth,frametime,mm_per_pix)
    pauses_by_f = find_pauses_by_f(speed)
    pauses_by_bin = find_pauses_by_bin(speed,fps)
    pirouettes = find_pirouettes(xs,ys,mm_per_pix)
    pirouette_bursts = find_pirouette_bursts(pirouettes,fps)
    speed_masked = mask_speed(speed,pauses_by_bin,pirouette_bursts)
    
    if show:
        plot_speed(speed)
        plot_pauses(pauses_by_f,'pause state (calculated by frame)')
        plot_pauses(pauses_by_bin,'pauses (1 s bins)')
        plot_pirouettes(pirouettes)
        plot_pirouette_bursts(pirouette_bursts)
        plot_speed_masks(speed_masked)
    
    return speed_masked

def get_avg_speeds_csv(centroids_csv_file):
    centroids = data_f.read_centroids_csv(centroids_csv_file)
    xs,ys = xy_from_centroids(centroids)
    xs_smooth = smooth_xy(xs)
    ys_smooth = smooth_xy(ys)
    speed = calc_speed(xs_smooth,ys_smooth,frametime,mm_per_pix)
    pauses_by_f = find_pauses_by_f(speed)
    pauses_by_bin = find_pauses_by_bin(speed,fps)
    pirouettes = find_pirouettes(xs,ys,mm_per_pix)
    pirouette_bursts = find_pirouette_bursts(pirouettes,fps)
    speed_masked = mask_speed(speed,pauses_by_bin,pirouette_bursts)
    
    plot_speed(speed)
    plot_pauses(pauses_by_f,'pause state (calculated by frame)')
    plot_pauses(pauses_by_bin,'pauses (1 s bins)')
    plot_pirouettes(pirouettes)
    plot_pirouette_bursts(pirouette_bursts)
    plot_speed_masks(speed_masked)
    
    return speed_masked


def get_avg_speeds_pickle(tracking_file):
    centroids = pickle.load( open(tracking_file,'rb'))
    xs,ys = xy_from_centroids(centroids)
    xs_smooth = smooth_xy(xs)
    ys_smooth = smooth_xy(ys)
    speed = calc_speed(xs_smooth,ys_smooth,frametime,mm_per_pix)
    pauses_by_f = find_pauses_by_f(speed)
    pauses_by_bin = find_pauses_by_bin(speed,fps)
    pirouettes = find_pirouettes(xs,ys,mm_per_pix)
    pirouette_bursts = find_pirouette_bursts(pirouettes,fps)
    speed_masked = mask_speed(speed,pauses_by_bin,pirouette_bursts)
    
    #import pdb; pdb.set_trace()
    
    # i = 65
    # img = plot_state_on_img(xs[i],ys[i],pirouettes[i])
    # # img = plot_state_on_img(xs[i],ys[i],pauses_by_bin[i])
    # img = plot_state_on_img(xs[i],ys[i],pirouette_bursts[i])
    # plt.imshow(img)
    # plt.show()
    
    # plot_speed(speed)
    # plot_pauses(pauses_by_f,'pause state (calculated by frame)')
    # plot_pauses(pauses_by_bin,'pauses (1 s bins)')
    #plot_pirouettes(pirouettes)
    #plot_pirouette_bursts(pirouette_bursts)
    #plot_speed_masks(speed_masked)
    
    return speed_masked

if __name__ == "__main__":
    tracking_files = [
        r'C:\Users\Temmerman Lab\Desktop\Bram data_old\20210923\video_AX7163_A 21-09-23 15-37-15_tracking\centroids_clean.p',\
        r'C:\Users\Temmerman Lab\Desktop\Bram data_old\20210923\video_AX7163_B 21-09-23 16-40-40_tracking\centroids_clean.p']
        # r'C:\Users\Temmerman Lab\Desktop\Bram data\20210923\video_N2_A 21-09-23 15-15-52_tracking\centroids_clean.p',\
        # r'C:\Users\Temmerman Lab\Desktop\Bram data\20210923\video_N2_B 21-09-23 16-19-35_tracking\centroids_clean.p',\
        # r'C:\Users\Temmerman Lab\Desktop\Bram data\20210923\video_RB1990_A 21-09-23 15-57-58_tracking\centroids_clean.p',\
        # r'C:\Users\Temmerman Lab\Desktop\Bram data\20210923\video_RB1990_B 21-09-23 17-01-53_tracking\centroids_clean.p']
        
    
    speeds = []
    numf = []
    for v in range(len(tracking_files)):
        speeds_v = []
        numf_v = []
        speed_masked = get_avg_speeds_pickle(tracking_files[v])
        for w in range(len(speed_masked)):
            speeds_v.append(np.nanmean(speed_masked[w]))
            numf_v.append(len(np.where(~np.isnan(speed_masked[w]))[0]))
        speeds.append(speeds_v)
        numf.append(numf_v)






