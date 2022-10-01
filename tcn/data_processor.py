# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model,load_model
import time

# Predict the next point every four points
train_size, predict_size = 4, 1

# The data fields to be used for training, only take specific fields, a total of 10
data_column = ['r-lat', 'r-lon', 'r-alt', 'vector-angle', 'vertical-deviation-angle',
               'omega_angle', 'wind_direction', 'wind_speed', 'dist-to-path', 'true-dist', 'waypoint-dist',
               'x_gyro', 'y_gyro', 'z_gyro', 'x_acc', 'y_acc', 'z_acc', 'pitch', 'yaw', 'roll']

ground_truth_column = ['r-lat', 'r-lon', 'r-alt']

path = r'relative-vector-and-angle\test'


def window_data(uav_data, stability_data, window_size):
    X_uav = []
    X_stability = []
    Y_loc = []
    Y_stability = []
    i = 0
    while (i + window_size) <= len(uav_data) - 1:
        X_uav.append(uav_data[i:i + window_size])
        X_stability.append(stability_data[i:i + window_size])
        Y_loc.append(uav_data[i + window_size, :3])
        Y_stability.append(stability_data[i + window_size])
        i += 1
    return X_uav, X_stability, Y_loc, Y_stability


def encoder_window_data(uav_encode_data, stability_data, uav_data, window_size):
    X_uav = []
    X_stability = []
    Y_loc = []
    Y_stability = []
    i = 0
    while (i + window_size) <= len(uav_encode_data) - 1:
        X_uav.append(uav_encode_data[i:i + window_size])
        X_stability.append(stability_data[i:i + window_size])
        Y_loc.append(uav_data[i + window_size, :3])
        Y_stability.append(stability_data[i + window_size])
        i += 1
    return X_uav, X_stability, Y_loc, Y_stability

# Read all csv files under the folder, save the file name in csv_file_list
def get_all_csv_file_list(path):
    csv_file_list = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1] == '.csv':
                csv_file_list.append(os.path.join(root, f))
    csv_file_list.sort(key=len)
    return csv_file_list


# Get all train data, label

def get_all_train_data_and_label_data(csv_file_list):
    X_uav = []
    X_stability = []
    y_loc = []
    y_stability = []
    X_sc = joblib.load("X_minmax_sc_20attr.save")
    X_stability_sc = joblib.load("X_minmax_sc_stability.save")

    for csv_file in csv_file_list:
        df = pd.read_csv(csv_file)
        dataset = df.loc[:, data_column].values
        stability_dataset = df.loc[:, ["stability"]].values
        training_set_scaled = X_sc.transform(dataset)
        stability_scaled = X_stability_sc.transform(stability_dataset)
        if not np.isnan(training_set_scaled).any():
            X_u, X_s, y_l, y_s = window_data(training_set_scaled, stability_scaled, train_size)
            for i in range(len(X_u)):
                X_uav.append(X_u[i])
                X_stability.append(X_s[i])
                y_loc.append(y_l[i])
                y_stability.append(y_s[i])
    return np.array(X_uav), np.array(X_stability), np.array(y_loc), np.array(y_stability)


def get_encoder_train_data_and_label_data(csv_file_list):
    X_uav = []
    X_stability = []
    y_loc = []
    y_stability = []
    X_uav_sc = joblib.load("X_minmax_sc_20attr.save")
    X_stability_sc = joblib.load("X_minmax_sc_stability.save")
    encoder = load_model("./densenet_encoder.h5")

    for csv_file in csv_file_list:
        df = pd.read_csv(csv_file)
        uav_dataset = df.loc[:, data_column].values
        stability_dataset = df.loc[:, ["stability"]].values
        uav_scaled = X_uav_sc.transform(uav_dataset)

        stability_scaled = np.array(X_stability_sc.transform(stability_dataset))

        uav_encoded = encoder.predict(np.expand_dims(np.array(uav_scaled), axis=2))


        X_u, X_s, y_l, y_s = encoder_window_data(uav_encoded, stability_scaled, np.array(uav_scaled), train_size)

        for i in range(len(X_u)):
            X_uav.append(X_u[i])
            X_stability.append(X_s[i])
            y_loc.append(y_l[i])
            y_stability.append(y_s[i])

    return np.array(X_uav), np.array(X_stability), np.array(y_loc), np.array(y_stability)

def load_train_data():
    # Get all csv files
    all_csv_file = get_all_csv_file_list(path)

    # Get the collection of train_windows data for each csv file
    X_uav, X_stability, y_loc, y_stability = get_all_train_data_and_label_data(all_csv_file)

    # Split training set, test set
    X_uav_train, X_uav_test, X_stability_train, X_stability_test, y_loc_train, y_loc_test, y_stability_train, y_stability_test = train_test_split(
        X_uav, X_stability, y_loc, y_stability, test_size=0.1, shuffle=True, random_state=2)

    return X_uav_train, X_uav_test, X_stability_train, X_stability_test, y_loc_train, y_loc_test, y_stability_train, y_stability_test


def load_encoder_train_data():
    # Get all csv files
    all_csv_file = get_all_csv_file_list(path)

    # Get the collection of train_windows data for each csv file
    X_uav, X_stability, y_loc, y_stability = get_encoder_train_data_and_label_data(all_csv_file)

    # Split training set, test set
    X_uav_train, X_uav_test, X_stability_train, X_stability_test, y_loc_train, y_loc_test, y_stability_train, y_stability_test = train_test_split(
        X_uav, X_stability, y_loc, y_stability, test_size=0.1, shuffle=True, random_state=2)

    return X_uav_train, X_uav_test, X_stability_train, X_stability_test, y_loc_train, y_loc_test, y_stability_train, y_stability_test

