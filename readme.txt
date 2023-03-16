## NOTES For Spark Emulation v1. 

I have stripped out some components that were unessesary. Eg tools to pre-process data & train the model. 
The remaining components allow for the (pre-trained) model to peform inference on samples within the dataset.

# LIBRARIES

Python libraries that are used are noted in "requirements.txt". 

# HOW TO RUN

The script to generate predictions and associated graphics is in "version_1_demo/spark_emulation/emulator/predict.py". 
The script can take 3 optional arguments. These are
-si : starting interval [0, 21] : When the emulator starts making predictions of the fire path (0 default).
-fi : stopping interval [1,22] : When the stops making predictions of the fire path (22 default).
-idx: index of the fire sample to use [0, 170], -1 : Which fire sample to use. -1 selects a random fire (-1 default).
-ts : weather time-series. Input as an n * 4 array, string. Inputs are wind_x (m/s), wind_y (m/s), temperature (deg C), rh ([0, 1])

The file can be run as 

eg. 
python3 predict.py
python3 predict.py -si 5 
python3 predict.py -si 12 -fi 21 -idx 3
python3 predict.py -ts "[[0, 10, 30, 0.5], [0, 20, 30, 0.5], [0, 30, 30, 0.5], [0, 40, 30, 0.5], [0, 50, 30, 0.5], [0, 60, 30, 0.5]]"

# note: when using custom ts of length n the number of intervals of the emulation 1 - n



Images are output into "version_1_demo/spark_emulation/emulator/images"

The images are:
    difference.pdf   : This is equal to the simulated fire shape minus the emulated fire shape
    emuated.pdf      : The emulated fire shape
    initial.pdf      : The stating fire shape (typically a small circle if -si 0)
    land_classes.pdf : Contours of the initial fire shape, emulated final shape, and simulated final shape.
                            These are plotted on top of a coloured map of the landclasses in use.
    simulated.pdf    : The simulated fire shape
    temperature.pdf  : The temperature and relative humidity for the sample
    winds.pdf        : A plot of the wind speed and direction for the sample.

""" Note that these images are relatively time-intensive since they read in an array, generate an image, then save the image.
    There may be a more efficient way to handle this. One thought would be to read in a subset of the array values to
    generate the images eg. small_array = full_array[0::2, 0::2] which would be 1/4 the size. Smallest images are 512x512 and largest
    are around ~1500 by 2000 """

# TO DO

 * Add an argument to pass in a different weather array
 * Add an argument to suppress generating simulated images for the case above 
        (since simulation would be using a different weather array and a comparison would be pointless)
 * Add an argument to generate a subset of the images 
 * Potentially split this script into two. 
        The first component reads in the python libraries and loads the model.
        The second component reads in a sample, generates a prediction, and creates images
        (Hopefully this will reduce constantly reading in the model weights everytime a prediction is needed?)
