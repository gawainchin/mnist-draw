

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
    if os.environ["REQUEST_METHOD"] == "POST":
        data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))
        img_str = re.search(r'base64,(.*)', data).group(1)

        image_bytes = io.BytesIO(base64.b64decode(img_str))
        im = Image.open(image_bytes)
        ar = np.array(im)
        arr = ar[:,:,0:1]

        # Normalize and invert pixel values
        arr = (255 - arr) / 255.
        arr1 = cv2.resize(arr, dsize = (28,28)).reshape([28, 28, 1])

        # Load trained model

        model.load('cgi-bin/models/model.tfl')  
        # Predict class
        predictions = model.predict([arr1])[0]

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


