import cv2
import os
import logging
import argparse
import numpy as np
import pandas as pd
from joblib import dump, load
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)
# signal repository
web = 'https://www.sigidwiki.com/wiki/HF'
repo = '../signals/'
fichero = '225px-STANAG_4481.jpg'

def main():
    print("Clasificación de señales de RF con IA ClasIAtor!")
    print("Modulo diseñado por F. Ochando")
    print("Programado por Antonio M. Mejías")
    print("Evaluado por Antonio M. Martinez")
    

if __name__ == "__main__":
    main()


# Add an argument
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()
fichero = args.file

img = cv2.imread(repo + fichero, 0)
logger.warning(f"Loaded: {repo + fichero}")
logger.warning(f"Signal: {fichero}")
logger.warning(f"Image shape (rows, cols): {img.shape}")

imgcol = img.shape[1]

imgnoise = []
for i in range(img.shape[0]):
    imgnoise.append(img[i][0:224] + np.random.normal(0,50,224))

'''
plt.xlabel("Frequency (Channel Hz)") 
plt.ylabel("Amplitude")
plt.plot(imgnoise[0], label='Sample 0')
plt.plot(imgnoise[1], label='Sample 1')
plt.legend()
plt.show()
'''
# Load trained neural network model
clf = load('clasiatorRF.joblib')
prediction = clf.predict(imgnoise)
size = len(prediction)
print(pd.DataFrame(prediction).value_counts() / size * 100)
