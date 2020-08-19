import os
import sys
import cv2
import yaml
import numpy as np
import time
import math
from collections import namedtuple
import copy
import time
from datetime import datetime

#image_path = sys.argv[1]
lidar_path = sys.argv[1]
calib_file_path = sys.argv[2]

Points = namedtuple('Points', ['xyz', 'attr'])

# make window title
title_window = 'reprojection window'

roll_angle     = 0.0
pitch_angle    = 0.0
yaw_angle      = 0.0
x_offset       = 0.0
y_offset       = 0.0
z_offset       = 0.0
sheer_off_y = 0.0
sheer_off_z = 0.0

change_flag = 0

def roll_trackbar_slide(val):
    print(val-150)
    global roll_angle
    global change_flag
    if change_flag == 0:
        roll_angle = val - 150

def pitch_trackbar_slide(val):
    print(val-150)
    global pitch_angle
    global change_flag
    if change_flag == 0:
        pitch_angle = val - 150

def yaw_trackbar_slide(val):
    print(val-150)
    global yaw_angle
    global change_flag
    if change_flag == 0:
        yaw_angle = val - 150

def X_trackbar_slide(val):
    print(val-150)
    global x_offset
    global change_flag
    if change_flag == 0:
        x_offset = val - 150

def Y_trackbar_slide(val):
    print(val-150)
    global y_offset
    global change_flag
    if change_flag == 0:
        y_offset = val - 150

def Z_trackbar_slide(val):
    print(val-150)
    global z_offset
    global change_flag
    if change_flag == 0:
        z_offset = val - 150

def track_bar_sheer_Y_slide(val):
    print(val/100 - 1)
    global change_flag
    global sheer_off_y
    if change_flag == 0:
        sheer_off_y = val/100 - 1

def track_bar_sheer_Z_slide(val):
    print(val/100 - 1)
    global change_flag
    global sheer_off_z
    if change_flag == 0:
        sheer_off_z = val/100 - 1

def make_sliders():
    cv2.namedWindow(title_window)
    track_bar_roll = 'Roll x (-%d %d)' %(150, 150)
    cv2.createTrackbar(track_bar_roll, title_window, 150, 300, roll_trackbar_slide)

    track_bar_pitch = 'Pitch x (-%d %d)' %(150, 150)
    cv2.createTrackbar(track_bar_pitch, title_window, 150, 300, pitch_trackbar_slide)

    track_bar_yaw = 'Yaw x (-%d %d)' %(150, 150)
    cv2.createTrackbar(track_bar_yaw, title_window, 150, 300, yaw_trackbar_slide)

    track_bar_X = 'X x (-%d %d)' %(150, 150)
    cv2.createTrackbar(track_bar_X, title_window, 150, 300, X_trackbar_slide)

    track_bar_Y = 'Y x (-%d %d)' %(150, 150)
    cv2.createTrackbar(track_bar_Y, title_window, 150, 300, Y_trackbar_slide)

    track_bar_Z = 'Z x (-%d %d)' %(150, 150)
    cv2.createTrackbar(track_bar_Z, title_window, 150, 300, Z_trackbar_slide)

    track_bar_sheer_Y = 'Sheer_Y x (-%d %d)' %(1, 1)
    cv2.createTrackbar(track_bar_sheer_Y, title_window, 100, 200, track_bar_sheer_Y_slide)

    track_bar_sheer_Z = 'Sheer_Z x (-%d %d)' %(1, 1)
    cv2.createTrackbar(track_bar_sheer_Z, title_window, 100, 200, track_bar_sheer_Z_slide)


def get_autoware_calib(calib_path, transform_mat = None):
    """ Get the autoware calibration matrices stored in the
    yaml file and save them in a calib dictionary. Also apply transforms
    as required."""
    
    with open(calib_path, "r") as f:
        calib = yaml.load(f)

    # get the cam to velo and velo to cam transformations
    calib['cam_to_velo'] = np.reshape(calib['CameraExtrinsicMat']['data'], (calib['CameraExtrinsicMat']['rows'], calib['CameraExtrinsicMat']['cols']))
    calib['velo_to_cam'] = np.linalg.inv(calib['cam_to_velo'])
    calib['CameraMat']   = np.reshape(calib['CameraMat']['data'], (calib['CameraMat']['rows'], calib['CameraMat']['cols']))
    #print(calib['CameraMat'])
    #print(calib['cam_to_velo'])
    calib['cam_to_image']  = np.hstack([calib['CameraMat'], [[0], [0], [0]]])
    calib['velo_to_image'] = np.matmul(calib['cam_to_image'], calib['cam_to_velo'])
    calib['DistCoeff']     = np.reshape(calib['DistCoeff']['data'], (calib['DistCoeff']['rows'], calib['DistCoeff']['cols']))
    #print(np.shape(calib['velo_to_cam']))
    #print(calib['velo_to_cam'])
    return calib

def transform_calib(calib, transform_mat = None):
    sheer = np.eye(4, 4, dtype=float)
    # sheer[1, 0] = math.tan(math.radians(30))
    sheer[0, 1] = sheer_off_y
    sheer[0, 2] = sheer_off_z
    calib['velo_to_cam']   = np.dot(calib['velo_to_cam'], transform_mat)
    calib['velo_to_cam']   = np.dot(calib['velo_to_cam'], sheer)
    return calib

def velo_points_to_cam(points, calib):
    """ Convert the points in velodyne coordinates to camera coordinates"""
    velo_xyz = np.hstack([points.xyz, np.ones([points.xyz.shape[0], 1])])
    cam_xyz  = np.matmul(velo_xyz, np.transpose(calib['velo_to_cam']))[:, :3]
    return Points(xyz=cam_xyz, attr=points.attr)

def get_cam_points(velo_points, calib=None):
    """ Load the velodyne points and convert them to the camera coordinates.
    Args: frame_idx the index of the frame to read.
    Returns Points.
    """
    cam_points  = velo_points_to_cam(velo_points, calib)
    return cam_points

def cam_points_to_image(points, calib):
    """ Convert camera points to image plane.
    Args: points: a[N, 3] float32 numpy array.
    Returns: points on image plane: a[M, 2] float32 numpy array.
    a mask indicating points: a[N, 1] boolean numpy array
    """
    cam_points_xyz = np.hstack([points.xyz, np.ones([points.xyz.shape[0],1])])
    img_points_xyz = np.matmul(cam_points_xyz, np.transpose(calib['cam_to_image']))
    img_points_xyz = img_points_xyz/img_points_xyz[:,[2]]
    img_points = Points(img_points_xyz, points.attr)
    return img_points

def get_cam_points_in_image(velo_points, calib=None, image = None):
    """ Load velo points and remove the one's not observed by the camera"""
    cam_points = get_cam_points(velo_points, calib=calib)
    height = image.shape[0]
    width  = image.shape[1]
    front_cam_points_idx = cam_points.xyz[:,2]>0.1
    front_cam_points = Points(cam_points.xyz[front_cam_points_idx, :],
    cam_points.attr[front_cam_points_idx, :])
    img_points = cam_points_to_image(front_cam_points, calib)
    img_points_in_image_idx = np.logical_and.reduce(
            [img_points.xyz[:,0]>0, img_points.xyz[:,0]<width,
            img_points.xyz[:,1]>0, img_points.xyz[:,1]<height])
    cam_points_in_img = Points(
            xyz = front_cam_points.xyz[img_points_in_image_idx,:],
            attr = front_cam_points.attr[img_points_in_image_idx,:])
    return cam_points_in_img

def get_transformation_mat():
    global roll_angle
    global pitch_angle
    global yaw_angle
    global x_offset
    global y_offset
    global z_offset
    global sheer_off_y
    global sheer_off_z

    #print(roll_angle)
    roll_mat  = np.zeros((3, 3), dtype=float)
    pitch_mat = np.zeros((3, 3), dtype=float)
    yaw_mat   = np.zeros((3, 3), dtype=float)

    # make the roll matrix
    roll_mat[0, 0] =  1
    roll_mat[1, 1] =  math.cos(math.radians(roll_angle))
    roll_mat[2, 2] =  roll_mat[1, 1]
    roll_mat[1, 2] = -math.sin(math.radians(roll_angle))
    roll_mat[2, 1] = -roll_mat[1, 2]

    # make the pitch matrix
    pitch_mat[0, 0] =  math.cos(math.radians(pitch_angle))
    pitch_mat[2, 2] =  pitch_mat[0, 0]
    pitch_mat[1, 1] =  1
    pitch_mat[0, 2] =  math.sin(math.radians(pitch_angle))
    pitch_mat[2, 0] = -math.sin(math.radians(pitch_angle))

    # make the yaw matrix
    yaw_mat[0, 0] =  math.cos(math.radians(yaw_angle))
    yaw_mat[1, 1] =  yaw_mat[0, 0]
    yaw_mat[0, 1] =  -math.sin(math.radians(yaw_angle))
    yaw_mat[1, 0] =  -yaw_mat[0, 1]
    yaw_mat[2, 2] = 1

    mat_1 = np.dot(roll_mat, pitch_mat)
    mat_2 = np.dot(mat_1,    yaw_mat)
    translation_mat = np.array([[x_offset], [y_offset], [z_offset]])
    mat_transform = np.hstack((mat_2, translation_mat))
    mat_transform = np.vstack((mat_transform, [0., 0., 0., 1.]))

    return mat_transform

def get_pcd_points(my_pcd_file):
    txt_file = my_pcd_file
    try:
        my_file  = open(txt_file, "r")
    except:
        return None
    array_points = my_file.readlines()
    points       = namedtuple('Points', ['xyz', 'attr'])
    point_cloud_array = np.zeros((len(array_points), 3), dtype=np.float32)
    reflections       = np.zeros((len(array_points), 1), dtype=np.float32)
    for i, line in enumerate(array_points):
        line_array = line.split(" ")
        x = float(line_array[0])
        y = float(line_array[1])
        z = float(line_array[2].split("\n")[0])
        point_cloud_array[i, 0] = x
        point_cloud_array[i, 1] = y
        point_cloud_array[i, 2] = z
        reflections[i, 0]       = 5    
    return points(xyz=point_cloud_array, attr=reflections)


def main():
    global change_flag
    make_sliders()
    #capture = cv2.VideoCapture(0)
    str1 = datetime.now()
    dt_string  = str1.strftime("%d_%m_%Y_%H_%M_%S_%f")[0:21]
    while 1:
        str1 = datetime.now()
        dt_string  = str1.strftime("%d_%m_%Y_%H_%M_%S_%f")[0:21]
        #_, image = capture.read()
        time.sleep(0.1)
        lidarpath = lidar_path + dt_string + ".txt"
        lidar_clouds = get_pcd_points(lidarpath)
        image_path = lidar_path + dt_string + ".tiff"
        image = cv2.imread(image_path)
        if lidar_clouds == None or image.all() == None:
            continue
        change_flag = 1
        calib1 = get_autoware_calib(calib_file_path)
        mat_transform = get_transformation_mat()
        calib1 = transform_calib(calib1, mat_transform)
        #print(calib1)
        cam_pts             = get_cam_points(lidar_clouds, calib=calib1)
        cam_points_in_image = get_cam_points_in_image(lidar_clouds, calib=calib1, image=image)
        img_points          = cam_points_to_image(cam_points_in_image, calib= calib1)
        min_distance = np.min(cam_points_in_image.xyz[:,2])
        max_distance = np.max(cam_points_in_image.xyz[:,2])
        scale = 255/(max_distance-min_distance)
        # # for rgbd
        # current_node.img = cv2.cvtColor(current_node.img, cv2.COLOR_BGR2GRAY)
        point_and_image = image.copy()
        #point_and_image = 255*np.ones((512, 640, 3), np.uint8);
        depth = cam_points_in_image.xyz[:,2]
        depth_color = cv2.applyColorMap(np.uint8(scale*(depth-min_distance)), cv2.COLORMAP_JET)
        for i, img_point in enumerate(img_points.xyz):
            color = depth_color[i, 0, :].astype(np.uint8).tolist()
            cv2.circle(point_and_image, (int(img_point[0]), int(img_point[1])), 2, color, -1, 4, 0)
        width, height = int(point_and_image.shape[1]), int(point_and_image.shape[0])
        point_and_image = cv2.resize(point_and_image, (width, height))
        change_flag = 0
        #cv2.imwrite(data_folder+save_reproject+"image"+str(j)+".jpg", point_and_image)
        cv2.imshow(title_window, point_and_image)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()