# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:08:19 2024

@author: sgs
"""
import numpy as np
import cv2
import os

def load_and_preprocess_data(data_dir):
    X = []  # To store preprocessed frames
    y = []  # To store labels
    
    # Iterate through files in the data directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.mp4'):  # Assuming videos are in .mp4 format
            video_path = os.path.join(data_dir, file_name)
            frames = extract_frames(video_path)
            for frame in frames:
                preprocessed_frame = preprocess_frame(frame)
                X.append(preprocessed_frame)
                # Extract label from file name or other source and append to y
                # Example: label = extract_label_from_filename(file_name)
                # y.append(label)
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def preprocess_frame(frame, target_size=(240, 240)):
    # Resize frame to target size
    resized_frame = cv2.resize(frame, target_size)
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values to [0, 1]
    normalized_frame = gray_frame / 255.0
    return normalized_frame

# Example usage:
data_dir = 'C:\\Users\\maila\\OneDrive\\Desktop\\FINAL YR PROJECT\\guiii\GaitDatasetA-silh\\ljg'
X, y = load_and_preprocess_data(data_dir)
