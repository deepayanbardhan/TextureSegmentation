import numpy as np
from PIL import Image

COLORS = [
    (255, 0, 0),   # red
    (0, 255, 0),  # green
    (0, 0, 255),   # blue
    (255, 255, 0), # yellow
    (255, 0, 255), # magenta
    (255, 128, 200), #orange
    (0, 255, 255), #cyan
    (255, 128, 128), #pink
    (128, 0, 255), #violet
    (0, 64, 0), #bottle green
]

def load_image(infilename) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data