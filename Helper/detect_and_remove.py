from imutils import paths
import cv2
import numpy as np
import argparse
import os

def dhash(image ,hashsize = 8):

    gray = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray ,(hashsize+1 ,hashsize))

    d = resized[: ,1:] > resized[: ,:-1]
    return sum([2 ** i for (i ,v) in enumerate(d.flatten()) if v])

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-r", "--remove", type=int, default=-1,
	help="whether or not duplicates should be removed (i.e., dry run)")
args = vars(ap.parse_args())

print("[INFO] computing image hashes...")

imagePaths = sorted(list(paths.list_images(args["dataset"])))
hashes = {}

for path in imagePaths:
    image = cv2.imread(path)

    if image is None:
        continue
    hash = dhash(image)
    p = hashes.get(hash ,[])
    p.append(path)
    hashes[hash] = p

print(hashes)

for (h ,hashPaths) in hashes.items():
    if len(hashPaths) > 1:

        if args["remove"] <= 0:

            montage = None

            for path in hashPaths:
                image = cv2.imread(path)
                image = cv2.resize(image ,(150 ,150))

                if montage is None:
                    montage = image
                else :
                    montage = np.hstack([montage ,image])

            print("[INFO] hash: {} ".format(h))
            cv2.imshow("montage" ,montage)
            cv2.waitKey(0)

        else:
            for path in hashPaths[1:]:
                os.remove(path)
