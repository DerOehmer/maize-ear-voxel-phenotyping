from machine import Pin
from utime import sleep

step = Pin(2, Pin.OUT) #brown jumper
direction = Pin(3, Pin.OUT) #green jumper
enable = Pin(4, Pin.OUT) #blue jumper

sleep(1)


direction.on() #clockwise
#direction.off() #counterclockwise
sleep(1)

def enable_motor():
    enable.on()
    sleep(.1)

def disable_motor():
    enable.off()
    sleep(.1)

def stepping(steps):
    for x in range(steps):

        # Set one coil winding to high
        step.on()
        # Allow it to get there.
        sleep(.001) # Dictates how fast stepper motor will run
        # Set coil winding to low
        step.off()
        sleep(.001) # Dictates how fast stepper motor will run

    print("done")





