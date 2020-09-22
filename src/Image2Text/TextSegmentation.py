import os
import cv2
import numpy as np
import math

def createKernel(kernelSize, sigma, theta):
	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2
	
	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta
	
	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize
			
			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
			
			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel

def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):
	kernel = createKernel(kernelSize, sigma, theta)
	imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
	(_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	imgThres = 255 - imgThres

	if cv2.__version__.startswith('3.'):
		(_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		(components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	res = []
	for c in components:
		if cv2.contourArea(c) < minArea:
			continue
		currBox = cv2.boundingRect(c) # returns (x, y, w, h)
		(x, y, w, h) = currBox
		currImg = img[y:y+h, x:x+w]
		res.append((currBox, currImg))

	return sorted(res, key=lambda entry:entry[0][0])

def prepareImg(img, height):
    assert(len(img.shape) in (2, 3))
    print(len(img.shape))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)

def main():
    files = os.listdir('../data/sample')
    for (i, f) in enumerate(files):
        img = prepareImg(cv2.imread('../data/sample/%s'%f), 2479)
        res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
        if not os.path.exists('../data/sample/out/%s'%f):
	        os.mkdir('../data/sample/out/%s'%f)
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite('../data/sample/out/%s/%d.png'%(f, j), wordImg) # save word
            cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image

        cv2.imwrite('../data/sample/out/%s/summary.png'%f, img)

if __name__ == "__main__":
    main()