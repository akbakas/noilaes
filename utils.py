import os
import numpy as np
import glob


def singles(path2txt: str) -> list:

    """
    Removes all txt files containing more than one bbox
    """
    allTextFiles = glob.glob(os.path.join(path2txt, '*.txt'))
    singleBbox = []
    for t in allTextFiles:
        if 37 < os.path.getsize(t) < 75:  # keep files in range (37, 75)
            singleBbox.append(
                (t.split('.')[0])  # extensionless name
            )
    return singleBbox

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xywhRead(path2txt: str) -> np.array:

    """
    Reads txt to array
    """
    coordinates = np.loadtxt(path2txt)[1:].astype(np.float32)
    return coordinates.reshape(1, -1)

def relax(xywh: "np.array([x, y, w, h])") -> np.array:

    """
    Increases bbox area
    :param c: relaxaction coefficient, must be between 1 and 2
    """
    relaxationValue = xywh[:, [2, 3]].min()
    xywh[:, 2] = xywh[:, 2] + relaxationValue
    xywh[:, 3] = xywh[:, 3] + relaxationValue
    return xywh

def cropLarge(img: "np.array", size: int) -> np.array:
    """
    Crops large image into smaller ones
    """
    assert img.ndim == 3, print("Image must have 3 dimension")
    h, w = img.shape[:-1]
    wstride = int(w / size)
    hstride = int(h/ size)
    
    for hor in range (0, size*wstride, size):
        for ver in range(0, size*hstride, size):
            print(hor, ver)
            yield img[ver:ver+size, hor:hor+size, :]
