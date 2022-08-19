# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:54:09 2022

@author: GuleOG
"""
import os
path = r"D:/FAKS/Meko racunarstvo/LV2/"
os.chdir(path)

def read_files(file_path):
   with open(file_path, 'r') as file:
      print(file.read())

for file in os.listdir():
   if file.endswith('.txt'):
      file_path =f"{path}/{file}"
    
