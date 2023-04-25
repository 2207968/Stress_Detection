import numpy as np 
import matplotlib.pyplot as plt  
import pickle

def addlabels(methods,values):
    for i in range(len(methods)):
        plt.text(i,values[i],values[i])


def plot1():

    # creating the dataset 
    data = {'Accuracy':97.8,'Precision':97.8} 
    methods = list(data.keys()) 
    values = list(data.values()) 
    
    fig = plt.figure(figsize = (10, 5)) 
    
    # creating the bar plot 
    plt.bar(methods, values, color ='blue', width = 0.4)

    addlabels(methods,values)
    
    plt.xlabel("Performance Metrics") 
    plt.ylabel("Accuracy & Precision") 
    plt.title("RF Model Performance") 
    plt.show() 

plot1()