# Arduino portenta detect model and send results to the TTN (lorawan)

This repository contains all the software side of things you need to run the project.  It includes :

ei_image_classification.py - The python script that runs the model and sends the data to the things network
labels.txt - This file contains the classes (the objects the model is detecting)
trained.tflite - This is the model that will make a prediction based on the data sent to it.  

Just copy the contents to the arduino portenta (with lora board attached) and play it through Openmv IDE.

Thanks 