# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:31:26 2021

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

from scipy.spatial import distance

import data_management_functions as data_f



def get_background(vid, bkgnd_nframes = 10, method = 'max_merge'):
    supported_meths = ['max_merge','min_merge']
    #pdb.set_trace()
    if method not in supported_meths:
        raise Exception('Background method not recognized, method must be one of ' + str(supported_meths))
    else:
        print('Calculating background image...')
        num_f = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        inds = np.round(np.linspace(0,num_f-1,bkgnd_nframes)).astype(int)
        for ind in inds:
            vid.set(cv2.CAP_PROP_POS_FRAMES, ind)
            success,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if ind == inds[0]:
                img = np.reshape(img,(img.shape[0],img.shape[1],1)) 
                stack = img
            else:
                img = np.reshape(img,(img.shape[0],img.shape[1],1))
                stack = np.concatenate((stack, img), axis=2)
        if method == 'max_merge':
            bkgnd = np.amax(stack,2)
        elif method == 'min_merge':
            bkgnd = np.amin(stack,2)
        print('Background image calculated')
    return bkgnd    

def binarize(frame, background, k_sz = (101,101), k_sig = 50, bw_thresh = 15,
    db = False):
    diff = cv2.absdiff(img,bkgnd)
    diff_blr = cv2.GaussianBlur(diff,k_z,k_sig,cv2.BORDER_REPLICATE)
    thresh,bw = cv2.threshold(diff_blr,bw_thresh,255,cv2.THRESH_BINARY)
    return bw

def find_blobs(bw,amin,amax, midline_meth = 'perim_curv'):
    objs = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
    # objs: # objs, labels, stats, centroids
    #  -> stats: left, top, width, height, area
    
    labels=np.uint8(objs[1]); #disp_img(bw2)
    
    # eliminate objects that are too big or too small
    centroids_f = list()
    clines_f = list()
    qual_cntrl = False
    obj_inds = np.linspace(0,objs[0]-1,objs[0]).astype(int)
    for obj in obj_inds:
        obj_sz = objs[2][obj][4]
        if obj_sz > size_bounds[0] and obj_sz < size_bounds[1]:
            centroids_f.append(copy.deepcopy(objs[3][obj]))
            bw_worm = copy.copy(objs[1][objs[2][obj,1]:objs[2][obj,1]+objs[2][obj,3],objs[2][obj,0]:objs[2][obj,0]+objs[2][obj,2]])
            bw_worm[np.where(bw_worm == obj)]=255
            bw_worm[np.where(bw_worm!=255)]=0
            cline = np.uint16(np.round(cf.find_centerline(bw_worm)))
            cline[:,0] += objs[2][obj][0]; cline[:,1] += objs[2][obj][1] 
            clines_f.append(copy.copy(cline))
            
    # # QC
    # qual_cntrl = False
    # if qual_cntrl:
    #     bw_copy = copy.copy(bw)
    #     for w in range(np.shape(clines_f)[0]):
    #         for pt in range(np.shape(clines_f[w])[0]):
    #             bw_copy[clines_f[w][pt,1],clines_f[w][pt,0]]=127
    #     plt.imshow(bw_copy); plt.show()
    return objs, centroids_f, clines_f


def find_centerline(bw, debug = False, point_to_inspect = 5): # V1
    
    # uses perimeter curvature (improved over V1) followed by
    # an equivalent to the circle fitting method. Very slow.
    
    # find the outline of the segmented worm
    image = np.uint8(bw)
    outline = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if len(outline) > 1:
        outline = outline[0]
        if len(outline) > 1:
            outline = outline[0]
            
    outline = np.squeeze(outline) # eliminate empty first and third dimensions
    
    
    # inspect points outline
    if debug:
        plt.imshow(image,cmap='gray'); plt.plot(outline[:,0],outline[:,1],'b');
        plt.plot(outline[2,0],outline[2,1],'g.'); plt.plot(outline[-3,0],outline[-3,1],'r.')
        plt.title('Points Outline'); plt.show()
    
    # fit closed spline to outline
    x = outline[:,1]
    y = outline[:,0]
    X = (x,y)
    tck,u = interpolate.splprep(X, s=100, per = True) # s may need adjusting
    # to avoid over- or underfitting
    s = np.arange(0,1,.001)
    new_points = interpolate.splev(s, tck)
    
    # inspect spline outline
    if debug:
        plt.imshow(image,cmap='gray'); plt.plot(new_points[1],new_points[0],'b');
        plt.plot(new_points[1][2],new_points[0][2],'g.'); plt.plot(new_points[1][-3],new_points[0][-3],'r.')
        plt.title('Spline Perimeter'); plt.show()
    
    
    # calculate the curvature along avery point in the spline. All outer contours 
    # returned by cv2.findContours should be counter-clockwise due to the
    # algorithm used (Appendix A of Suzuki & Abe 1985 
    # http://pdf.xuebalib.com:1262/xuebalib.com.17233.pdf )
    # the curvature is approximated by the exterior angle of the curvature, 
    # calculated from three equidistant points which vary in spacing according to
    # the variable 'spacing', which is a proportion of the circumference
    spacing = 0.05 # in proportion of 
    spacing_discreet = int(np.round(spacing*len(new_points[0])))
    
    if spacing_discreet == 0: spacing_discreet = 1;
    
    x = np.concatenate((new_points[1][-spacing_discreet:],new_points[1],new_points[1][0:spacing_discreet+1]))
    y = np.concatenate((new_points[0][-spacing_discreet:],new_points[0],new_points[0][0:spacing_discreet+1]))
    curvature = np.empty((len(new_points[0])))
    
    for p in range(spacing_discreet,len(new_points[0])+spacing_discreet):
        v2 = np.array((y[p+spacing_discreet]-y[p],x[p+spacing_discreet]-x[p]))
        v1 = np.array((y[p]-y[p-spacing_discreet],x[p]-x[p-spacing_discreet]))
        norm_cross_product = np.cross(v1,v2)/(np.linalg.norm(v1)* \
            np.linalg.norm(v2))
        norm_dot_product = np.dot(v1,v2)/(np.linalg.norm(v1)* \
            np.linalg.norm(v2))
        angle = np.arcsin(norm_cross_product)
        
        if norm_dot_product < 0 and norm_cross_product > 0:
            angle = np.pi/2 + np.pi/2 - angle
        elif norm_dot_product < 0 and norm_cross_product < 0:
            angle = -np.pi/2 - np.pi/2 - angle
        
        curvature[p-spacing_discreet] = angle
    
    
    # inspect curvature
    # based on https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    if debug:
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap, BoundaryNorm
        x = new_points[1]; y = new_points[0]
        color_weight = curvature
        points = np.array([x,y]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        fig,axs = plt.subplots(1,1,sharex=True,sharey=True)
        norm = plt.Normalize(color_weight.min(),color_weight.max())
        lc = LineCollection(segments, cmap = 'jet',norm = norm)
        lc.set_array(color_weight)
        lc.set_linewidth(3)
        line = axs.add_collection(lc)
        fig.colorbar(line, ax=axs)
        plt.title('Curvature (Interior Angle), Spacing = '+str(spacing))
        plt.imshow(image,cmap='gray')
        # axs.set_xlim(x.min(), x.max())
        # axs.set_ylim(y.min(), y.max())
        plt.show()
        
        plt.plot(np.linspace(0,1,len(curvature)),curvature,'k')
        plt.title('Curvature Along Perimeter, Spacing = '+str(spacing))
        plt.xlabel('Perimeter Position')
        plt.ylabel('Curvature')
        plt.show()
    
       
    # find the points of maximum curvature. After finding the first one,
    # weight the curvature to bias against finding a nearby second point.
    # These are, hopefully, the two ends of the worm.
    end_1 = np.where(curvature == np.max(curvature))
    end_1 = int(end_1[0])
    
    
    ramp_up = np.linspace(0,1.5*curvature[end_1],int(0.9*(len(curvature)/2)))
    ramp_down = np.flipud(ramp_up)
    flat = np.zeros(int(np.shape(curvature)[0]-(np.shape(ramp_up)[0]+np.shape(ramp_down)[0])))
    ramp = np.concatenate((ramp_down,flat,ramp_up),axis = 0)
    correction = np.empty(len(curvature))
    if end_1 == 0:
        correction = ramp
    else:
        correction[0:end_1] = ramp[-end_1:]
        correction[end_1:] = ramp[0:len(ramp)-end_1]
    curvature_weighted = copy.copy(curvature)-correction
    end_2 = np.where(curvature_weighted == np.max(curvature_weighted))
    end_2 = int(end_2[0])

    # plot the detected endpoints
    if debug:    
        plt.plot(s,curvature,'k',label = 'curvature')
        plt.plot(s[end_1],curvature[end_1],'r.',markersize=10)
        plt.plot(s,curvature_weighted,'c',label = 'weighted curvature')
        plt.plot(s[end_2],curvature_weighted[end_2],'r.',markersize=10)
        plt.legend()
        plt.title('Perimeter-Curvature Detection of Endpoints')
        plt.xlabel('Perimeter Position')
        plt.ylabel('Perimeter Angle, Spacing = '+str(spacing))
        plt.show()
        
        # inspect the endpoint locations on the worm
        plt.imshow(image,cmap='gray'); plt.plot(new_points[1],new_points[0],'b');
        plt.plot(new_points[1][end_1],new_points[0][end_1],'r.'); plt.plot(new_points[1][end_2],new_points[0][end_2],'r.')
        plt.title('Spline Perimeter with Endpoints'); plt.show()
    
    
    
    # re-interpolate along two sidelines
    if end_1 < end_2:
        x1 = new_points[0][end_1:end_2+1]
        y1 = new_points[1][end_1:end_2+1]
        x2 = np.flipud(np.concatenate((new_points[0][end_2:], \
            new_points[0][0:end_1+1]),axis = 0))
        y2 = np.flipud(np.concatenate((new_points[1][end_2:], \
            new_points[1][0:end_1+1]),axis = 0))
    else:
        x1 = new_points[0][end_2:end_1+1]
        y1 = new_points[1][end_2:end_1+1]
        x2 = np.flipud(np.concatenate((new_points[0][end_1:], \
            new_points[0][0:end_2+1]),axis = 0))
        y2 = np.flipud(np.concatenate((new_points[1][end_1:], \
            new_points[1][0:end_2+1]),axis = 0))
    
    # divide side one into 50 points and side two into 1000 points (exact
    # numbers are somewhat arbitrary)
    X1 = (x1,y1)
    tck,u = interpolate.splprep(X1, s=100, per = False)
    ss = np.arange(0,1,.02)
    side_1_points = interpolate.splev(ss, tck)
    X2 = (x2,y2)
    tck,u = interpolate.splprep(X2, s=100, per = False)
    ss = np.arange(0,1,.001)
    side_2_points = interpolate.splev(ss, tck)
    side_2_points = np.swapaxes(side_2_points,0,1)
    
    # inspect side_1_points and side_2_points
    if debug:
        plt.imshow(image,cmap='gray');
        plt.plot(side_1_points[1][:],side_1_points[0][:],'c.');
        plt.plot(side_2_points[:,1],side_2_points[:,0],'b.');
        plt.gca().invert_yaxis()
    
    # determine the approximate length of the worm by taking the average of
    # the lengths of the two sides. This will be used as a distance threshold
    # for adding points to the set of centerline points to reduce crowding of
    # points at bends in the body. The crowding can create discontinuities in
    # the spline.
    side_1_length =  \
        np.sum(np.sqrt(np.sum(np.diff(side_1_points)**2,axis = 0)))
    side_2_length = \
        np.sum(np.sqrt(np.sum(np.diff(np.swapaxes(side_2_points,0,1))**2, \
        axis = 0)))
    distance_threshold = \
        (side_1_length+side_2_length)/(2*np.shape(side_1_points)[1])
    
    # calculate the midpoint at each side 1 point
    centerline_points = np.expand_dims(np.array((x1[0],y1[0])),axis = 0)
    for p in np.arange(1,np.shape(side_1_points)[1]-1,1):
        
        point = [(side_1_points[1][p],side_1_points[0][p])]
        
        # calculate line normal to that point
        rise = (side_1_points[0][p+1]-side_1_points[0][p-1])
        run = (side_1_points[1][p+1]-side_1_points[1][p-1])
        
        if rise == 0:
            normal_slope = 999999
        elif run == 0:
            normal_slope = 0
        else:
            normal_slope = -1/(rise/run)
        
        normal_y_intercept = side_1_points[0][p] - normal_slope* \
            side_1_points[1][p]
        normal_xs = np.linspace(0,np.shape(image)[1],np.shape(image)[1]+1)
        normal_ys = normal_slope*normal_xs + normal_y_intercept
        
        # calculate 1000 points along the normal line within the bw image
        
        # First find where the normal line enters and leaves the bw image.
        # There are four possibilies: where x or y = 0, and where x or y = the
        # height or width of the image (NB: x is the vertical dimension in the
        # image in this case). Then narrow those four to the two that are 
        # really at the edge of the image, then find 1000 points along that
        # line.
        p_y0 = np.array(((0-normal_y_intercept)/normal_slope,0))
        p_ymax = np.array(((np.shape(image)[0]- \
            normal_y_intercept)/normal_slope, np.shape(image)[0]))
        p_x0 = np.array((0,normal_y_intercept))
        p_xmax = np.array((np.shape(image)[1],normal_slope*np.shape(image)[1]+ \
            normal_y_intercept))
        four_points = np.vstack((p_y0,p_ymax,p_x0,p_xmax))
        two_points = np.empty((2,2)); fill_row = 0
        
        for pp in range(np.shape(four_points)[0]):
            if 0 <= four_points[pp,0] <= np.shape(image)[1] and 0 <=  \
                four_points[pp,1] <= np.shape(image)[0]:
                two_points[fill_row,:] = four_points[pp,:]
                fill_row = fill_row+1
                
                if fill_row == 2: break
        
        normal_xs = np.linspace(two_points[0,0],two_points[1,0],1000)
        normal_ys = normal_slope*normal_xs+normal_y_intercept
        normal_points = np.vstack((normal_xs,normal_ys))
        normal_points = np.swapaxes(normal_points,0,1)
        
        # Eliminate normal line points that are outside the worm. These can be
        # misidentified as centerline points in convoluted worms.
        image_sz = np.shape(image)
        indices_to_axe = []
        
        for pp,norm_p in enumerate(normal_points):
            if int(norm_p[1]) > image_sz[0]-1:
                indices_to_axe.append(pp)
            elif int(norm_p[0]) > image_sz[1]-1:
                indices_to_axe.append(pp)
            elif image[np.uint16(norm_p[1]),np.uint16(norm_p[0])] == 0:
                indices_to_axe.append(pp)
        
        
        normal_points_2 = np.delete(normal_points,indices_to_axe,0)
        # # check that the correct points were eliminated
        if p == point_to_inspect and debug:
            plt.imshow(image,cmap='gray')
            plt.plot(side_1_points[1][p],side_1_points[0][p],'c.',markersize = 15)
            plt.plot(normal_points[:,0],normal_points[:,1],'r--')
            plt.plot(normal_points_2[:,0],normal_points_2[:,1],'c-')
            plt.xlim(0,np.shape(image)[1])
            plt.ylim(0,np.shape(image)[0])
            plt.gca().invert_yaxis()
            plt.title('Elimination of Extranematodal Points')
            plt.show()
        
        normal_points = normal_points_2
        normal_xs = np.delete(normal_xs,indices_to_axe,0)
        normal_ys = np.delete(normal_ys,indices_to_axe,0)
        
        # calculate the distance from the original pt to the points along the
        # normal line
        distance_from_point = np.squeeze(distance.cdist(point, \
            normal_points, 'euclidean'))
        distance_from_other_side = np.squeeze(np.min(distance.cdist \
            (normal_points, np.fliplr(side_2_points), 'euclidean'),axis=1))
        absval_difference = np.abs(distance_from_point- \
            distance_from_other_side)
        index = np.where(absval_difference == min(absval_difference))
        new_point = np.swapaxes(np.array((normal_ys[index], \
            normal_xs[index])),0,1)
        last_point = centerline_points[-1,:]
        if np.linalg.norm(new_point-last_point) > distance_threshold:
            centerline_points = np.vstack((centerline_points,new_point))
        
        if p == point_to_inspect and debug:
            # plot the distance from points along the normal line to the point
            # on side 1 and from the other side, as well as the absolute value
            # of the difference between them
            plt.plot(normal_xs,distance_from_point,'c-',label = 'from point')
            plt.plot(normal_xs,distance_from_other_side,'r-',label='from opposite side')
            plt.plot(normal_xs,absval_difference,'k-',label='difference')
            plt.plot(normal_xs[index],absval_difference[index],'k.',markersize = 15)
            plt.ylabel('distance')
            plt.xlabel('normal line x coordinate')
            plt.legend()
            plt.title('Finding the Middle Along the Normal Line')
            plt.show()
            
            # plot the point along the normal line that is closest to 
            # equidistant from the original point and the other side of the worm
            plt.imshow(image,cmap='gray')
            plt.plot(side_1_points[1][p],side_1_points[0][p],'c.',markersize = 15)
            plt.plot(normal_xs,normal_ys,'c--')
            plt.plot(normal_xs[index],normal_ys[index],'.',color = 'gray',markersize = 15)
            plt.plot(normal_xs[index],normal_ys[index],'x',color = 'gray', markersize = 15)
            plt.xlim(0,np.shape(image)[1])
            plt.ylim(0,np.shape(image)[0])
            plt.plot(side_2_points[:,1],side_2_points[:,0],'r',linewidth=3)
            plt.gca().invert_yaxis()
            plt.show()
        
    
    last_point = np.expand_dims(np.array((x1[-1],y1[-1])),axis=0)
    centerline_points = np.vstack((centerline_points,last_point))
    tck,u = interpolate.splprep(np.rot90(centerline_points), s = 5, per =  \
        False)
    ss = np.arange(0,1,.01)
    centerline = interpolate.splev(ss, tck)
    
    # inspect the sides, centerline, and centerline points
    if debug:
        plt.imshow(image,cmap='gray');
        plt.plot(y1,x1,'b');
        plt.plot(y2,x2,'c');
        plt.plot(centerline_points[:,1],centerline_points[:,0],'m.',markersize=10)
        plt.plot(centerline[0][:],centerline[1][:],'r')
        plt.xlim(0,np.shape(image)[1])
        plt.ylim(0,np.shape(image)[0])
        plt.title('Interpolated Centerline with Points'); plt.show()

    return np.swapaxes(centerline,0,1)#, centerline_points

    
def find_centerline_V0(bw, db = False):
    
    # uses perimeter curvature followed by midpoints of lines
    # centerlines tend to track on the inside of bends
    
    img = np.uint8(bw)
    outln, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(outln) > 1:
        outln = outln[0]
    outln = np.squeeze(outln) # get rid of pesky first and third dimensions
    
    # fit closed spline to outline
    x = outln[:,1]
    y = outln[:,0]
    X = (x,y)
    tck,u = interpolate.splprep(X, s=100, per = True) # s may need adjusting to avoid
    # over- or underfitting
    s = np.arange(0,1,.001)
    new_pts = interpolate.splev(s, tck)
    
    # calculate the curvature along avery point in the spline
    x = np.concatenate((new_pts[1],new_pts[1][0:2]))
    y = np.concatenate((new_pts[0],new_pts[0][0:2]))
    curv = np.empty((len(new_pts[0])))
    for p in range(len(new_pts[0])):
        v1 = np.array((y[p+1]-y[p],x[p+1]-x[p]))
        v2 = np.array((y[p+2]-y[p+1],x[p+2]-x[p+1]))
        curv[p] = np.abs(np.arcsin(np.cross(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
        
    # find the points of maximum curvature that are far apart
    end_1 = np.where(curv == np.max(curv))
    end_1 = int(end_1[0])
    ramp_up = np.linspace(0,1.5*curv[end_1],int(len(curv)/2))
    ramp_down = np.flipud(ramp_up)
    ramp = np.concatenate((ramp_down,ramp_up),axis = 0)
    correction = np.empty(len(curv))
    if end_1 == 0:
        correction = ramp
    else:
        correction[0:end_1] = ramp[-end_1:]
        correction[end_1:] = ramp[0:len(ramp)-end_1]
    curv_mod = copy.copy(curv)-correction
    end_2 = np.where(curv_mod == np.max(curv_mod))
    end_2 = int(end_2[0])
    
    # re-interpolate along two sidelines
    if end_1 < end_2:
        x1 = new_pts[0][end_1:end_2+1]
        y1 = new_pts[1][end_1:end_2+1]
        x2 = np.flipud(np.concatenate((new_pts[0][end_2:],new_pts[0][0:end_1+1]),axis = 0))
        y2 = np.flipud(np.concatenate((new_pts[1][end_2:],new_pts[1][0:end_1+1]),axis = 0))
    else:
        x1 = new_pts[0][end_2:end_1+1]
        y1 = new_pts[1][end_2:end_1+1]
        x2 = np.flipud(np.concatenate((new_pts[0][end_1:],new_pts[0][0:end_2+1]),axis = 0))
        y2 = np.flipud(np.concatenate((new_pts[1][end_1:],new_pts[1][0:end_2+1]),axis = 0))
    X1 = (x1,y1)
    tck,u = interpolate.splprep(X1, s=100, per = False)
    ss = np.arange(0,1,.01)
    s1_pts = interpolate.splev(ss, tck)
    X2 = (x2,y2)
    tck,u = interpolate.splprep(X2, s=100, per = False)
    s2_pts = interpolate.splev(ss, tck)
    
    # find the centerline based on an average of the sidelines
    x_mid = np.mean(np.vstack((s1_pts[0][:],s2_pts[0][:])),0)
    y_mid = np.mean(np.vstack((s1_pts[1][:],s2_pts[1][:])),0)
    cline = np.flip(np.rot90(np.vstack((x_mid,y_mid))))
    
    return cline


def show_segmentation(vid, f = 0, bkgnd = None, k_sig = 2, k_sz = (11,11), 
    bw_thr = 20, sz_bnds = np.array((200,500)), d_thr = None, db = False):
    
    # # debugging
    # f = 0; k_sig = 2; k_sz = (11,11); bw_thr = 20; sz_bnds = np.array((200,500)); d_thr = 10
    # vid_name = r'C:\Users\Temmerman Lab\Desktop\Bram data\20210923\video_AX7163_A 21-09-23 15-37-15.avi'
    # vid = cv2.VideoCapture(vid_name)
    # bkgnd = get_background(vid, 10, 'max_merge')
    
    # read in frame 1
    vid.set(cv2.CAP_PROP_POS_FRAMES, f)
    success,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    
    # subtract background (if available) to get difference image
    if bkgnd is None:
        diff = img
    else:
        diff = cv2.absdiff(img,bkgnd)

    # apply Gaussian smoothing to difference image to get smoothed image
    smooth = cv2.GaussianBlur(diff,k_sz,k_sig,cv2.BORDER_REPLICATE)
    
    # apply a binary threshold to the smoothed image to get binary image
    thresh,bw = cv2.threshold(smooth,bw_thr,255,cv2.THRESH_BINARY)
    
    # find connected blobs in the binary image
    cc = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
    # cc: # objs, labels, stats, centroids
    #  -> stats: left, top, width, height, area
    cc_map = np.uint8(cc[1]); #disp_img(bw2)
    cc_is = np.linspace(0,cc[0]-1,cc[0]).astype(int)

    # eliminate objects that are too big or too small and make a BW image
    # without eliminated objects
    bw_ws = np.zeros(np.shape(bw),dtype = 'uint8')
    centroids_f = list()
    # clines_f = list()
    for cc_i in cc_is:
        cc_sz = cc[2][cc_i][4]
        if cc_sz > sz_bnds[0] and cc_sz < sz_bnds[1]:
            bw_ws[np.where(cc_map==cc_i)] = 255
            centroids_f.append(copy.deepcopy(cc[3][cc_i]))
            
            # # also find centerline (commented for now)
            # bw_w = copy.copy(cc[1][cc[2][cc_i,1]:cc[2][cc_i,1]+cc[2][cc_i,3],cc[2][cc_i,0]:cc[2][cc_i,0]+cc[2][cc_i,2]])
            # bw_w[np.where(bw_w == cc_i)]=255
            # bw_w[np.where(bw_w!=255)]=0
            # cline = np.uint16(np.round(find_centerline(bw_w)))
            # cline[:,0] += cc[2][cc_i][0]; cline[:,1] += cc[2][cc_i][1] 
            # clines_f.append(copy.copy(cline))
    
    # setup overlay text
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = .5
    f_thickness = 2
    f_color = (0,0,0)
    
    
    # create 'final' image showing identified worms
    final_HSV = cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2HSV)  
    
    # add red shading to all bw blobs
    final_HSV[:,:,0][np.where(bw==255)] = 120 # set hue (color)
    final_HSV[:,:,1][np.where(bw==255)] = 80 # set saturation (amount of color, 0 is grayscale)

    # change shading to green for those that are within the size bounds
    final_HSV[:,:,0][np.where(bw_ws==255)] = 65 # set hue (color)
    
    #plt.imshow(final_HSV); plt.show()
    
    # convert image to BGR (the cv2 standard)
    final = cv2.cvtColor(final_HSV,cv2.COLOR_HSV2BGR)
    
    
    
    # could also outline blobs within size bounds
    # final = cv2.cvtColor(final_HSV,cv2.COLOR_HSV2BGR)
    # contours, hierarchy = cv2.findContours(bw_ws, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(final, [contours[0]], 0, (0, 255, 0), 1) #drawing contours
    
    # # label blobs detected as worms with centerline
    # for track in range(np.shape(centroids_f)[0]):
    #     # cline
    #     pts = np.int32(clines_f[track])
    #     pts = pts.reshape((-1,1,2))
    #     final = cv2.polylines(final, pts, True, (0,0,255), 1)
    
    
    # label the size of all blobs
    for cc_i in cc_is[1:]:
        cc_sz = cc[2][cc_i][4]
        text = str(cc_sz)
        
            
        text_size = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
        text_pos = copy.copy(cc[3][cc_i]) # deepcopy avoids changing objs below
        text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
        text_pos[1] = text_pos[1] + 30
        text_pos = tuple(np.uint16(text_pos))
        if cc_sz > sz_bnds[0] and cc_sz < sz_bnds[1]:
            final = cv2.putText(final,text,text_pos,f_face,f_scale,f_color,f_thickness,cv2.LINE_AA)
        else:
            final = cv2.putText(final,text,text_pos,f_face,f_scale,(50,50,50),1,cv2.LINE_AA)

    
    # show the distance threshold
    if d_thr is not None:
        text = 'd='+str(d_thr)
        text_size = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
        pt1 = [np.shape(img)[1]-50,np.shape(img)[0]-50]
        pt2 = [pt1[0]-d_thr,pt1[1]]
        text_pos = np.array((((pt1[0]+pt2[0])/2,pt1[1])),dtype='uint16')
        text_pos[0] = text_pos[0] - text_size[0]/2 # x centering 
        text_pos[1] = text_pos[1] - 5 # y offset   
        final = cv2.polylines(final, np.array([[pt1,pt2]]), True, (0,0,255), 1)
        final = cv2.putText(final,text,text_pos,f_face,f_scale,(0,0,255),1,cv2.LINE_AA)
        del pt1, pt2
    #plt.imshow(final); plt.show()
    
    return img, diff, smooth, bw, bw_ws, final

    

    
def show_segmentation_old(vid, f = 1, bkgnd = None, k_sig = 2, k_sz = (11,11),
    bw_thr = 20, sz_bnds = np.array((600,1500)), db = False):
    print('Calculating tracking images...') 
    vid.set(cv2.CAP_PROP_POS_FRAMES, f-1)
    success,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    if bkgnd is None:
        diff = img
    else:
        diff = cv2.absdiff(img,bkgnd)
    
    smooth = cv2.GaussianBlur(diff,k_sz,k_sig,cv2.BORDER_REPLICATE)
    thresh,bw = cv2.threshold(smooth,bw_thr,255,cv2.THRESH_BINARY)
    cc = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
    # cc: # objs, labels, stats, centroids
    #  -> stats: left, top, width, height, area
    cc_map = np.uint8(cc[1]); #disp_img(bw2)
    cc_is = np.linspace(0,cc[0]-1,cc[0]).astype(int)
    
    # eliminate objects that are too big or too small and make a BW image without eliminated objects
    bw_ws = np.zeros(np.shape(bw),dtype = 'uint8')
    centroids_f = list()
    clines_f = list()
    for cc_i in cc_is:
        cc_sz = cc[2][cc_i][4]
        if cc_sz > sz_bnds[0] and cc_sz < sz_bnds[1]:
            bw_ws[np.where(cc_map==cc_i)] = 255
            centroids_f.append(copy.deepcopy(cc[3][cc_i]))
            bw_w = copy.copy(cc[1][cc[2][cc_i,1]:cc[2][cc_i,1]+cc[2][cc_i,3],cc[2][cc_i,0]:cc[2][cc_i,0]+cc[2][cc_i,2]])
            bw_w[np.where(bw_w == cc_i)]=255
            bw_w[np.where(bw_w!=255)]=0
            cline = np.uint16(np.round(find_centerline(bw_w)))
            cline[:,0] += cc[2][cc_i][0]; cline[:,1] += cc[2][cc_i][1] 
            clines_f.append(copy.copy(cline))
                    
    # text stuff
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 1
    f_thickness = 1
    f_color = (0,0,0)
    
    final = np.stack((img,img,img),2)
    for track in range(np.shape(centroids_f)[0]):
        text = str(track)
        text_size = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
        text_pos = copy.copy(centroids_f[track]) # deepcopy avoids changing objs below
        text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
        text_pos[1] = text_pos[1] + 75
        text_pos = tuple(np.uint16(text_pos))
        final = cv2.putText(final,text,text_pos,f_face,f_scale,f_color,f_thickness,cv2.LINE_AA)
        # cline
        pts = np.int32(clines_f[track])
        pts = pts.reshape((-1,1,2))
        final = cv2.polylines(final, pts, True, (0,0,255), 5)
        
        if db:
            disp_img(img)
            disp_img(bkgnd)
            disp_img(diff)
            disp_img(smooth)
            disp_img(bw)
            disp_img(final)
            
    return img, diff, smooth, bw, bw_ws, final

def setup_output_video(vid,out_name,out_scaling = 1):
    # setup output video
    
    out_w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) * out_scaling)
    out_h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * out_scaling)
    v_out = cv2.VideoWriter(out_name,
        cv2.VideoWriter_fourcc('M','J','P','G'), vid.get(cv2.CAP_PROP_FPS),
        (out_w,out_h), 1)
    # list of possible codecs: https://www.fourcc.org/codecs.php
    return v_out, out_w, out_h

def track_worms_old(vid_name, vid_path, bkgnd_meth, bkgnd_nframes, k_sig, k_sz, bw_thr,
    sz_bnds, d_thr, out_scl = 0.5):
    
    find_cline = False
    full_path_to_vid = vid_path + '//' + vid_name
    vid = data_f.load_video(full_path_to_vid)[2]
    
    full_output = True
    if full_output:
        out_scl = 1
        out_name = vid_path + '//' + os.path.splitext(vid_name)[0] + '_tracking//' + os.path.splitext(vid_name)[0] + '_tracking_bkgnd_sub.avi'
        v_out_bkgnd_sub, out_bkgnd_sub_w, out_bkgnd_sub_h = \
            setup_output_video(vid,out_name,out_scl)
        out_name = vid_path + '//' + os.path.splitext(vid_name)[0] + '_tracking//' + os.path.splitext(vid_name)[0] + '_tracking_bw.avi'
        v_out_bw, out_bw_w, out_bw_h = \
            setup_output_video(vid,out_name,out_scl)
    
    out_name = vid_path + '//' + os.path.splitext(vid_name)[0] + '_tracking//' + os.path.splitext(vid_name)[0] + '_tracking.avi'
    v_out, out_w, out_h = setup_output_video(vid,out_name,out_scl)
    
    
    bkgnd = get_background(vid, bkgnd_nframes, bkgnd_meth)
    cv2.imwrite(vid_path + '//' + os.path.splitext(vid_name)[0] + '_tracking//' + os.path.splitext(vid_name)[0] + '_background.bmp',bkgnd)
    
    num_f = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    ind_1 = 0
    inds = np.linspace(ind_1,num_f-1,int(num_f)-ind_1); ind = 0;
    
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 1
    f_thickness = 5
    f_color = (0,0,0)
    f_color2 = (255,255,255)
    
    # tracking loop
    for ind in inds:
        print('Tracking worms in frame ' + str(int(ind+1)) + ' of ' + str(int(num_f)) +'.')
        
        # load and process frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, ind)
        success,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        diff = cv2.absdiff(img,bkgnd)
        smooth = cv2.GaussianBlur(diff,tuple(k_sz),k_sig,cv2.BORDER_REPLICATE)
        thresh,bw = cv2.threshold(smooth,bw_thr,255,cv2.THRESH_BINARY)
        
        erode_dilate = False
        if erode_dilate:
            # fill holes and inlets in BW
            kernel = np.ones((5,5),np.uint8)
            bw = cv2.dilate(bw,kernel,iterations = 1)
            
            # # these lines are equivalent to MATLAB imfill
            # # source: https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
            # bw_ff = bw.copy()
            # he, wi = bw.shape[:2]
            # mask = np.zeros((he+2, wi+2), np.uint8)
            # cv2.floodFill(bw_ff, mask, (0,0), 255);
            # bw_ff = cv2.bitwise_not(bw_ff)
            # bw = bw | bw_ff
            
            bw = cv2.erode(bw,kernel,iterations = 1)
            bw = cv2.medianBlur(bw,15)
            thr,bw = cv2.threshold(bw,127,255,cv2.THRESH_BINARY)
        
        # find connected components
        cc = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
        # cc: # objs, labels, stats, centroids
        #  -> stats: left, top, width, height, area
        cc_map = np.uint8(cc[1]); #disp_img(bw2)
        cc_is = np.linspace(0,cc[0]-1,cc[0]).astype(int)

        # eliminate objects that are too big or too small, or that are touching the boundary
        centroids_f = list()
        clines_f = list()
        for cc_i in cc_is:
            cc_sz = cc[2][cc_i][4]
            if cc_sz > sz_bnds[0] and cc_sz < sz_bnds[1]:
                hits_edge = False
                obj_inds_r = np.where(cc[1]==cc_i)[0]
                obj_inds_c = np.where(cc[1]==cc_i)[1]

                if np.min(obj_inds_r) == 0 or np.min(obj_inds_c) == 0:
                    hits_edge = True
                elif np.max(obj_inds_r) == np.shape(cc[1])[0]-1 or np.max(obj_inds_c) == np.shape(cc[1])[1]-1:
                    hits_edge = True
                    
                if hits_edge is False:
                    centroids_f.append(copy.deepcopy(cc[3][cc_i]))
                    bw_w = copy.copy(cc[1][cc[2][cc_i,1]:cc[2][cc_i,1]+cc[2][cc_i,3],cc[2][cc_i,0]:cc[2][cc_i,0]+cc[2][cc_i,2]])
                    bw_w[np.where(bw_w == cc_i)]=255
                    bw_w[np.where(bw_w!=255)]=0
                    cline = np.uint16(np.round(find_centerline(bw_w)))
                    cline[:,0] += cc[2][cc_i][0]; cline[:,1] += cc[2][cc_i][1]
                    cline = cline[np.newaxis,...]
                    clines_f.append(copy.copy(cline))
        
        if ind == inds[0]:
            centroids_prev = copy.deepcopy(centroids_f)
            #clines_prev = copy.copy(clines_f)
            #pdb.set_trace()
            centroids = np.array((np.reshape(centroids_prev, \
                (np.shape(centroids_prev)[0],1,np.shape(centroids_prev)[1]))), \
                ndmin=3)
            centerlines = clines_f
            # w_poses =  np.array((np.reshape(clines_prev,
            #     (np.shape(clines_prev)[0],1,np.shape(clines_prev)[1],2))),
            #     ndmin=4)
        else:
            # calculate the distance between centroids in this frame and centroids in the
            # previous frame
            d_mat = np.empty((np.shape(centroids_prev)[0],np.shape(centroids_f)[0]))
            for row, cent_prev in enumerate(centroids_prev):            
                for col, cent in enumerate(centroids_f):
                    d_mat[row,col] = np.linalg.norm(cent_prev-cent)
            
            # (may implement a percentage size threshold in the future, but
            # not now due to the possiblity of huge apparent size changes
            # during nictation)
            # calculate the percentage change in size between the centroids in 
            # this frames and the centroids in the previous frame
            del_sz_mat = np.empty((np.shape(centroids_prev)[0],np.shape(centroids_f)[0]))
            
            # find the closest centroids, second closest, etc., crossing off matched worms
            # until either all worms from the previous frame are matched, or none of them 
            # are within the max distance of worms in the current frame or change in size
            # by a factor less than the size change threshold.
            num_to_pair =  np.min(np.shape(d_mat))
            search = True
            pair_list = list()

            if np.shape(d_mat)[0]>0 and np.shape(d_mat)[1]>0:
                while search:
                    min_dist = np.nanmin(d_mat)
                    if min_dist < d_thr:
                        result = np.where(d_mat == np.nanmin(d_mat))  
                        pair_list.append((result[0][0],result[1][0]))
                        d_mat[result[0][0],:]=np.nan
                        d_mat[:,result[1][0]]=np.nan
                    else:
                        search = False
                    
                    if len(pair_list) == num_to_pair:
                        search = False
            
            # The tracks of worms tracked in the last frame but not matched in this frame
            # are dropped (nan in the centroids matrix, or simply not appended to in the
            # centerlines list), and new worms detected in this frame are added as new row
            # in the centroids matrix, or a new item in the centerlines list
                    
            # first add tracked worms to a new column in worm_tracks and worm_poses
            new_col = np.empty((np.shape(centroids)[0],1,2))
            new_col.fill(np.nan)
            centroids = np.concatenate((centroids,new_col),axis = 1)
            
            # new_col = np.empty((np.shape(w_poses)[0],1,100,2))
            # new_col.fill(np.nan)
            # w_poses = np.concatenate((w_poses,new_col),axis = 1)

            for pair in pair_list:
                centroids[pair[0],-1,0] = copy.copy(centroids_f[pair[1]][0]) # x
                centroids[pair[0],-1,1] = copy.copy(centroids_f[pair[1]][1]) # y
                centerlines[pair[0]] = np.concatenate((centerlines[pair[0]],clines_f[pair[1]]),axis=0)
                
                
                # w_poses[pair[0],-1,:,0] = copy.copy(clines_f[pair[1]][:,0]) # x
                # w_poses[pair[0],-1,:,1] = copy.copy(clines_f[pair[1]][:,1]) # y
                
                   
            # then add new rows to worm_tracks and  new items toworm_poses for
            # new worms
            inds_new = np.linspace(0,np.shape(d_mat)[1]-1,np.shape(d_mat)[1])
            for pair in pair_list: inds_new = np.delete(inds_new,np.where(inds_new == pair[1])) 
            
            new_r = np.empty((1,np.shape(centroids)[1],2))
            new_r.fill(np.nan)
            for new_worm_ind in inds_new:
                centroids = np.concatenate((centroids,new_r),axis=0)
                centroids[-1][-1] = centroids_f[int(new_worm_ind)]
                
                centerlines.append(clines_f[int(new_worm_ind)])
                
            # new_r = np.empty((1,np.shape(w_poses)[1],100,2))
            # new_r.fill(np.nan)
            # for new_worm_ind in inds_new:
            #     w_poses = np.concatenate((w_poses,new_r),axis=0)
            #     w_poses[-1][-1] = clines_f[int(new_worm_ind)]
            
            # keep centroids from this frame for use in next iteration
            centroids_prev = list()
            for row,x_last in enumerate(centroids[:,-1,0]):
                centroids_prev.append(np.array((x_last,centroids[row,-1,1])))

            # # keep centerlines from this frame for use in the next iteration
            # clines_prev = list()
            # for row,x_last in enumerate(w_poses[:,-1,0]):
            #     clines_prev.append(np.array((x_last,w_poses[row,-1,1])))
            
            # output tracking video frame
            img_save = np.stack((img,img,img),2)
            for track in range(np.shape(centroids)[0]):
                if ~np.isnan(centroids[track,-1,0]):
                    # centroid
                    text = str(track)
                    text_size = cv2.getTextSize(text, f_face, 5, 10)[0]
                    text_pos = copy.copy(centroids[track,-1,:]) # deepcopy avoids changing objs below
                    # text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
                    text_pos[1] = text_pos[1] + 60
                    text_pos = tuple(np.uint16(text_pos))
                    img_save = cv2.putText(img_save,text,text_pos,f_face,f_scale,f_color,f_thickness,cv2.LINE_AA)
                    # cline
                    pts = np.int32(centerlines[int(track)][-1])
                    pts = pts.reshape((-1,1,2))
                    img_save = cv2.polylines(img_save, pts, True, (0,0,255), 2)
            img_save = cv2.resize(img_save, (out_w,out_h), interpolation = cv2.INTER_AREA)
            v_out.write(img_save)
            
            # full output outputs for development and debugging
            if full_output:
                # output background-subtracted version
                img_save = np.stack((diff,diff,diff),2)
                for track in range(np.shape(centroids)[0]):
                    if ~np.isnan(centroids[track,-1,0]):
                        # centroid
                        text = str(track)
                        text_size = cv2.getTextSize(text, f_face, 5, 10)[0]
                        text_pos = copy.copy(centroids[track,-1,:]) # deepcopy avoids changing objs below
                        # text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
                        text_pos[1] = text_pos[1] + 60
                        text_pos = tuple(np.uint16(text_pos))
                        img_save = cv2.putText(img_save,text,text_pos,f_face,f_scale,f_color2,f_thickness,cv2.LINE_AA)
                        # cline
                        pts = np.int32(centerlines[int(track)][-1])
                        pts = pts.reshape((-1,1,2))
                        img_save = cv2.polylines(img_save, pts, True, (0,0,255), 2)
                # (always full size)
                img_save = cv2.resize(img_save, (out_w,out_h), interpolation = cv2.INTER_AREA)
                v_out_bkgnd_sub.write(img_save)
            
                # output color-coded background-subtracted version
                img_save = np.stack((bw,bw,bw),2)
                for track in range(np.shape(centroids)[0]):
                    if ~np.isnan(centroids[track,-1,0]):
                        # centroid
                        text = str(track)
                        text_size = cv2.getTextSize(text, f_face, 5, 10)[0]
                        text_pos = copy.copy(centroids[track,-1,:]) # deepcopy avoids changing objs below
                        # text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
                        text_pos[1] = text_pos[1] + 60
                        text_pos = tuple(np.uint16(text_pos))
                        img_save = cv2.putText(img_save,text,text_pos,f_face,f_scale,f_color2,f_thickness,cv2.LINE_AA)
                        # cline
                        pts = np.int32(centerlines[int(track)][-1])
                        pts = pts.reshape((-1,1,2))
                        img_save = cv2.polylines(img_save, pts, True, (0,0,255), 2)
                # (always full size)
                img_save = cv2.resize(img_save, (out_w,out_h), interpolation = cv2.INTER_AREA)
                v_out_bw.write(img_save)
        
        # delete unecessary variables
        if ind > inds[0]:
            del success, img, diff, smooth, bw, cc, cc_map, cc_is
            # hacky way of preventing a crash if there are not worms
            if 'bw_w' in locals() and 'clines_prev' in locals():
                del bw_w, clines_f, cc_i, cc_sz, hits_edge
                del cline, clines_prev, d_mat, row, col, cent,
                del cent_prev, num_to_pair, search, pair_list, min_dist
                del new_col, inds_new, new_r, x_last
                del img_save, track, text, text_size, text_pos, pts
                del bw_ff, mask, he, wi, centroids_f
            if 'obj_inds' in locals():
                del obj_inds, obj_inds_c, result, pair
            # new_worm_ind, row, centroids_prev

    v_out.release()
    del v_out
    print('Done!')
    return centroids, centerlines


def track_worms(vid_name, vid_path, bg_meth, bg_nframes, k_sig, k_sz, bw_thr,
    sz_bnds, d_thr, del_sz_thr = None, out_scl = 0.5):
    
    vid_file = vid_path + '//' + vid_name
    vid = data_f.load_video(vid_file)[2]
    
    wrt_trk_vid = True
    find_clines = False
    
    
    
    if wrt_trk_vid:    
        out_name = vid_path + '//' + os.path.splitext(vid_name)[0] + \
        '_tracking//' + os.path.splitext(vid_name)[0] + '_tracking.avi'
        v_out, out_w, out_h = setup_output_video(vid,out_name,out_scl)
    
    bg = get_background(vid, bg_nframes, bg_meth)
    cv2.imwrite(vid_path + '//' + os.path.splitext(vid_name)[0] + '_tracking//' + os.path.splitext(vid_name)[0] + '_background.bmp',bg)
    
    num_f = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    ind_1 = 0
    inds = np.linspace(ind_1,num_f-1,int(num_f)-ind_1); ind = 0;
    
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = .75
    f_thickness = 1
    f_color = (0,0,0)
    f_color2 = (255,255,255)
    
    # tracking loop
    for ind in inds:
        print('Tracking worms in frame ' + str(int(ind+1)) + ' of ' + str(int(num_f)) +'.')
        
        # load and process frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, ind)
        ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        diff = cv2.absdiff(img,bg)
        smooth = cv2.GaussianBlur(diff,tuple(k_sz),k_sig,cv2.BORDER_REPLICATE)
        thresh,bw = cv2.threshold(smooth,bw_thr,255,cv2.THRESH_BINARY)
        
        erode_dilate = False
        if erode_dilate:
            # fill holes and inlets in BW
            kernel = np.ones((5,5),np.uint8)
            bw = cv2.dilate(bw,kernel,iterations = 1)
            
            # # these lines are equivalent to MATLAB imfill
            # # source: https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
            # bw_ff = bw.copy()
            # he, wi = bw.shape[:2]
            # mask = np.zeros((he+2, wi+2), np.uint8)
            # cv2.floodFill(bw_ff, mask, (0,0), 255);
            # bw_ff = cv2.bitwise_not(bw_ff)
            # bw = bw | bw_ff
            
            bw = cv2.erode(bw,kernel,iterations = 1)
            bw = cv2.medianBlur(bw,15)
            thr,bw = cv2.threshold(bw,127,255,cv2.THRESH_BINARY)
        
        # find connected components
        cc = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
        # cc: # objs, labels, stats, centroids
        #  -> stats: left, top, width, height, area
        cc_map = np.uint8(cc[1]); #disp_img(bw2)
        cc_is = np.linspace(0,cc[0]-1,cc[0]).astype(int)

        # eliminate objects that are too big or too small, or that are touching the boundary
        centroids_f = list()
        #clines_f = list()
        for cc_i in cc_is:
            cc_sz = cc[2][cc_i][4]
            if cc_sz > sz_bnds[0] and cc_sz < sz_bnds[1]:
                hits_edge = False
                obj_inds_r = np.where(cc[1]==cc_i)[0]
                obj_inds_c = np.where(cc[1]==cc_i)[1]

                if np.min(obj_inds_r) == 0 or np.min(obj_inds_c) == 0:
                    hits_edge = True
                elif np.max(obj_inds_r) == np.shape(cc[1])[0]-1 or np.max(obj_inds_c) == np.shape(cc[1])[1]-1:
                    hits_edge = True
                    
                if hits_edge is False:
                    centroids_f.append(copy.deepcopy(cc[3][cc_i]))
                    bw_w = copy.copy(cc[1][cc[2][cc_i,1]:cc[2][cc_i,1]+cc[2][cc_i,3],cc[2][cc_i,0]:cc[2][cc_i,0]+cc[2][cc_i,2]])
                    bw_w[np.where(bw_w == cc_i)]=255
                    bw_w[np.where(bw_w!=255)]=0
                    # cline = np.uint16(np.round(find_centerline(bw_w)))
                    # cline[:,0] += cc[2][cc_i][0]; cline[:,1] += cc[2][cc_i][1]
                    # cline = cline[np.newaxis,...]
                    # clines_f.append(copy.copy(cline))
        
        if ind == inds[0]:
            centroids_prev = copy.deepcopy(centroids_f)
            #clines_prev = copy.copy(clines_f)
            #import pdb; pdb.set_trace()
            if len(centroids_prev) > 0:
                centroids = np.array((np.reshape(centroids_prev, \
                    (np.shape(centroids_prev)[0],1,np.shape(centroids_prev)[1]))), \
                    ndmin=3)
            else:
                centroids = np.array((np.reshape(centroids_prev, \
                    (np.shape(centroids_prev)[0],1,0))), \
                    ndmin=3)
            # centerlines = clines_f

        else:
            # calculate distance between centroids in this frame and previous
            d_mat = np.empty((np.shape(centroids_prev)[0],np.shape(centroids_f)[0]))
            for row, cent_prev in enumerate(centroids_prev):            
                for col, cent in enumerate(centroids_f):
                    d_mat[row,col] = np.linalg.norm(cent_prev-cent)
            
            # may implement a percentage size threshold in the future, but
            # not now due to the possiblity of huge apparent size changes
            # during nictation. Size may also be a useful metric
            # calculate the percentage change in size between the centroids in 
            # this frames and the centroids in the previous frame
            del_sz_mat = np.empty((np.shape(centroids_prev)[0],np.shape(centroids_f)[0]))
            
            # find the closest centroids, second closest, etc., crossing off matched worms
            # until either all worms from the previous frame are matched, or none of them 
            # are within the max distance of worms in the current frame or change in size
            # by a factor less than the size change threshold.
            num_to_pair =  np.min(np.shape(d_mat))
            search = True
            pair_list = list()

            if np.shape(d_mat)[0]>0 and np.shape(d_mat)[1]>0:
                while search:
                    min_dist = np.nanmin(d_mat)
                    if min_dist < d_thr:
                        result = np.where(d_mat == np.nanmin(d_mat))  
                        pair_list.append((result[0][0],result[1][0]))
                        d_mat[result[0][0],:]=np.nan
                        d_mat[:,result[1][0]]=np.nan
                    else:
                        search = False
                    
                    if len(pair_list) == num_to_pair:
                        search = False
            
            # The tracks of worms tracked in the last frame but not matched in this frame
            # are dropped (nan in the centroids matrix, or simply not appended to in the
            # centerlines list), and new worms detected in this frame are added as new row
            # in the centroids matrix, or a new item in the centerlines list
                    
            # first add tracked worms to a new column in worm_tracks and worm_poses
            new_col = np.empty((np.shape(centroids)[0],1,2))
            new_col.fill(np.nan)
            centroids = np.concatenate((centroids,new_col),axis = 1)
            
            # new_col = np.empty((np.shape(w_poses)[0],1,100,2))
            # new_col.fill(np.nan)
            # w_poses = np.concatenate((w_poses,new_col),axis = 1)

            for pair in pair_list:
                centroids[pair[0],-1,0] = copy.copy(centroids_f[pair[1]][0]) # x
                centroids[pair[0],-1,1] = copy.copy(centroids_f[pair[1]][1]) # y
                #centerlines[pair[0]] = np.concatenate((centerlines[pair[0]],clines_f[pair[1]]),axis=0)
                
            # then add new rows to worm_tracks and  new items to worm_poses for
            # new worms
            inds_new = np.linspace(0,np.shape(d_mat)[1]-1,np.shape(d_mat)[1])
            for pair in pair_list: inds_new = np.delete(inds_new,np.where(inds_new == pair[1])) 
            
            new_r = np.empty((1,np.shape(centroids)[1],2))
            new_r.fill(np.nan)
            for new_worm_ind in inds_new:
                centroids = np.concatenate((centroids,new_r),axis=0)
                centroids[-1][-1] = centroids_f[int(new_worm_ind)]
                #centerlines.append(clines_f[int(new_worm_ind)])
                
            
            # keep centroids from this frame for use in next iteration
            centroids_prev = list()
            for row,x_last in enumerate(centroids[:,-1,0]):
                centroids_prev.append(np.array((x_last,centroids[row,-1,1])))

            # # keep centerlines from this frame for use in the next iteration
            # clines_prev = list()
            # for row,x_last in enumerate(w_poses[:,-1,0]):
            #     clines_prev.append(np.array((x_last,w_poses[row,-1,1])))
            
            # output tracking video frame
            if wrt_trk_vid:
                #import pdb; pdb.set_trace()
                img_save = np.stack((img,img,img),2)
                for track in range(np.shape(centroids)[0]):
                    if ~np.isnan(centroids[track,-1,0]):
                        # centroid
                        text = str(track)
                        text_size = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
                        text_pos = copy.copy(centroids[track,-1,:]) # deepcopy avoids changing objs below
                        text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
                        text_pos[1] = text_pos[1] + 30
                        text_pos = tuple(np.uint16(text_pos))
                        img_save = cv2.putText(img_save,text,text_pos,f_face,f_scale,f_color,f_thickness,cv2.LINE_AA)
                        # cline
                        #pts = np.int32(centerlines[int(track)][-1])
                        #pts = pts.reshape((-1,1,2))
                        #img_save = cv2.polylines(img_save, pts, True, (0,0,255), 2)
                img_save = cv2.resize(img_save, (out_w,out_h), interpolation = cv2.INTER_AREA)
                v_out.write(img_save)
            
            
        
        # delete unecessary variables
        if ind > inds[0]:
            del ret, img, diff, smooth, bw, cc, cc_map, cc_is
            # hacky way of preventing a crash if there are not worms
            if 'bw_w' in locals() and 'clines_prev' in locals():
                del bw_w,  cc_i, cc_sz, hits_edge
                del d_mat, row, col, cent,
                del cent_prev, num_to_pair, search, pair_list, min_dist
                del new_col, inds_new, new_r, x_last
                del img_save, track, text, text_size, text_pos, pts
                del bw_ff, mask, he, wi, centroids_f
                if find_clines:
                    del clines_f,cline, clines_prev
            if 'obj_inds' in locals():
                del obj_inds, obj_inds_c, result, pair
            # new_worm_ind, row, centroids_prev

    v_out.release()
    del v_out
    print('Done!')
    if find_clines:
        return centroids, centerlines
    else:
        return centroids