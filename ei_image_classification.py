# Edge Impulse - OpenMV Image Classification Example

# libaries to import
import sensor, image, time, os, tf, time
from pyb import LED
from lora import *

# setup lora object for UK frequency
lora = Lora(band=BAND_EU868, poll_ms=60000, debug=False)

print("Firmware:", lora.get_fw_version())
print("Device EUI:", lora.get_device_eui())
print("Data Rate:", lora.get_datarate())
print("Join Status:", lora.get_join_status())

# Example keys for connecting to the backend
appEui = "0000000000000000"
appKey = "F39B5B54EE95B4AD661893D7CCEF2A94"

# Connect to LORA device
try:
    lora.join_OTAA(appEui, appKey)
    print("Connected.")

# You can catch individual errors like timeout, rx etc...
except LoraErrorTimeout as e:
    print("Something went wrong; are you indoor? Move near a window and retry")
    print("ErrorTimeout:", e)
except LoraErrorParam as e:
    print("ErrorParam:", e)

# set port for lora messages
lora.set_port(3)

# sets LED no's to names
red_led   = LED(1)
green_led = LED(2)
blue_led  = LED(3)

# Setup camera sensor
sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.GRAYSCALE) # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = "trained.tflite" # tensor flow model location/name
labels = [line.rstrip('\n') for line in open("labels.txt")] # opens labels for model above

clock = time.clock()

def led_control(colour, state = "off"):
#3 colours are red, blue green and state is eitehr off or on
    if colour == "red" :
        if state == "off" :
            red_led.off()
        else:
            red_led.on()
    if colour == "green" :
        if state == "off" :
            green_led.off()
        else:
            green_led.on()
    if colour == "blue" :
        if state == "off" :
            blue_led.off()
        else:
            blue_led.on()

while(True):
    clock.tick()
    led_control("blue", "off")
    led_control("red","on")
    img = sensor.snapshot()

    # default settings just do one detection... change them to search the image...
    for obj in tf.classify(net, img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        img.draw_rectangle(obj.rect())
        # This combines the labels and confidence values into a list of tuples
        predictions_list = list(zip(labels, obj.output()))

        for i in range(len(predictions_list)):
            print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))

        # Sort predictions by highest prediction first
        predictions_list.sort(key=lambda tup: tup[1],reverse=True)  # sorts in place

        # # print the top prediction out
        print("predicted :" , predictions_list[0][0], "with a value of  :" , predictions_list[0][1])

        # send sucessfully detect image to TTN network
        try:
            if lora.send_data((str(predictions_list[0][0]) + "," + str(predictions_list[0][1])), True):
                print("Message confirmed.")
            else:
                print("Message wasn't confirmed")

        except LoraErrorTimeout as e :
            print("ErrorTimeout:", e)
        except LoraErrorNoNetwork as e :
            print("ErrorTimeout, please move the device within range of a gateway", e)

    print(clock.fps(), "fps")
    led_control("red","off")
    led_control("blue", "on")

    #sleep in seconds
    time.sleep(120)
