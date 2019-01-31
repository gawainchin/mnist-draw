
"""
CGI script that accepts image urls and feeds them into a ML classifier. Results
are returned in JSON format. 
"""

import io
import json
import sys
import os
import re
import base64
import numpy as np
from PIL import Image
from model import model
import tflearn.datasets.mnist as mnist
import cv2
# Default output
res = {"result": 0,
       "data": [], 
       "error": ''}

try:
    # Get post data
    def printing(text):
        print(text)
        if outputFile:
            outputFile.write(str(text))

    outputFile = open('outputfile.log', 'w')
    if os.environ["REQUEST_METHOD"] == "POST":
        data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))
        printing("hi09789707")


        img_str = re.search(r'base64,(.*)', data).group(1)

        image_bytes = io.BytesIO(base64.b64decode(img_str))
        im = Image.open(image_bytes)
        ar = np.array(im)
        printing(list(ar.shape))
        arr = ar[:,:,0:1]
        printing(list(ar.shape))
        printing("hi1.1")

        # Normalize and invert pixel values
        arr = (255 - arr) / 255.

        printing("hi1.2")
        printing(list(arr.shape))
        printing(list(arr.reshape([-1, 28, 28, 1]).shape))
        printing("OMG1")
        #printing("hi1.2")
        arr1 = cv2.resize(arr, dsize = (28,28)).reshape([28, 28, 1])
        printing("OMG2")
        printing(arr1.shape)
        #if arr:
        #    printing("hi1.3")

        # Load trained model

        model.load('cgi-bin/models/model.tfl')
        X, Y, testX, testY = mnist.load_data(one_hot=True)
        testX = testX.reshape([-1, 28, 28, 1])
        printing(testX[0].shape)
        #arr = testX[0]   
        # Predict class
        predictions = model.predict([arr1])[0]
        printing("OMG!@@!@@!")
        printing(np.argmax(predictions[1]))

        # Return label data
        res['result'] = 1
        res['data'] = [float(num) for num in predictions] 

except Exception as e:
    # Return error data
    res['error'] = str(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))


