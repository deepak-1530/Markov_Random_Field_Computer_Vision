# Implement Markov Random Field based image segmentation given initial estimates from a CNN mask
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import math
import random
import matplotlib.pyplot as plt

# Use 16 neighbourhood problems
# Energy Function to be minimized is -> summation(unary potentials + pairwise potentials)
# Unary potential is -log(prob(x))
# Pairwise potential is using potts model summation(abs(xi - xj)) where xi is the current pixel and xj is the pixel 
# in the neighbourhood
# Define Optimization parameters

N = 1 #no. of cycles
n = 1 #steps in cycles
nR = 0

def addNoise(im):
    mu = 0.25
    sig = 0.1
    gaussian = np.random.normal(mu, sig, im.shape)
    return im + 0.1*gaussian

def isValid(shape_, s):
    if s[0]< shape_[0] and s[1] < shape_[1]:
        return True

def simAnnealing(im, initE, labels):
# Optimization using simulated annealing
    na = 0.0
    p1 = 0.75
    pN = 0.001
    t1 = -1.0/math.log(p1) # Initial temperature
    tN = -1.0/math.log(pN) # Final temperature
    frac = (tN/t1)**(1.0/(n-0.99999))
    im_curr = im
    im_fin = im
    accept = False
    t = t1 
    prev_label = 0
    #initial temperature
    # We have the initial image and energy
    # For each index, assign the label and check which configuration minimizes the energy
    DeltaE_avg = 0.000001
    count = 0
    minE = initE  #set the minimum energy to be the initial energy at first
    for i in range(N):
        for j in range(n):
            for row in range(nR, im.shape[0]):
                for col in range(im.shape[1]):
                    count = count + 1
                    print("Iteration index is : " + str(i) + " and " + str(j))
 
                    #row = random.randint(nR, im.shape[0]-1)
                    #col = random.randint(0, im.shape[1]-1)
                    print([row, col])
                    curr_label = np.random.choice(labels)
                    print("Current Label is: " + str(curr_label))

                    if im_curr[row, col] == curr_label:
                        continue

                    prev_label = im_curr[row, col]
                    im_curr[row, col] = curr_label
                    currE = pottsEnergy(im_curr, row, col)

                    print("Current energy is: " + str(currE))
                    DeltaE = abs(currE-minE[row, col])
                    E_dif = currE - minE[row, col]
                    print("Current energy difference is: " + str(E_dif))
            
                    if (E_dif>0):
                        if (i==0 and j==0): DeltaE_avg = DeltaE
                        p = math.exp(-DeltaE/((DeltaE_avg+0.01) * t))
                        print("p is :" + str(p))
                    if (p>p1):
                        accept = True
                        print("P in accepted case is: " + str(p))
                        p1 = p1 + 0.01
                    else:
                        accept = False
                        im_curr[row, col] = prev_label

                    if E_dif < 0:
                        accept = True

                    if (accept==True):
                        im_fin[row, col] = curr_label
                        na = na + 1.0
                        DeltaE_avg = ((DeltaE_avg + 0.01) * (na-1.0) +  DeltaE) / na
                        minE[row,col] = currE
                
                
                t = frac * t
                print("t at current iteration : " + str(i) + " is :" + str(t) )
    return im_fin



def pottsEnergy(im, i, j):   #unary potential energy is also added in this.
    pE = 0.0
    variance = 0.0
    mean = 0
    # Calculate potts model energy using neighbouring pixels

    '''
[i-2,j-2]    [i-2, j-1]   [i-2,j]   [i-2,j+1]    [i-2,j+2]
[i-1,j-2]    [i-1,j-1]    [i-1,j]   [i-1, j+1]   [i-1,j+2]
[i,j-2]      [i,j-1]      [i,j]     [i,j+1]      [i,j+2]
[i+1,j-2]    [i+1,j-1]   [i+1,j]    [i+1,j+1]    [i+1,j+2]
[i+2,j-2]    [i+2,j-1]   [i+2,j]    [i+2,j+1]    [i+2,j+2]

    '''
    nps = [[i-2,j-2], [i-1,j-2], [i,j-2], [i+1,j-2], [i+2,j-2], [i-2,j+2], [i-1, j+2], [i,j+2], [i+1,j+2], [i+2,j+2],[i,j+1],[i,j-1], [i+1,j], [i-1,j], [i-1,j+1], [i+1,j-1], [i+1, j+1], [i-1,j-1], [i-2,j-1], [i-2,j], [i-2,j+1], [i+2,j-1], [i+2,j], [i+2,j+1]]
    for k in range(len(nps)):
        if isValid(im.shape, nps[k]):
            pE += 5*abs(im[i,j] - im[nps[k][0],nps[k][1]])**2
            mean += im[nps[k][0],nps[k][1]]
    mean = mean/len(nps)

    for [l,m] in nps:
        if isValid(im.shape,[l,m]):
            variance += (im[l,m] - mean)**2

    return 0.50*pE + 30.0*variance - 0.0000001*math.log(im[i,j] + 0.0001)   # log(prob) is used for unary potential calculation.


def calcEnergy(im): # Calculate total unary and pairwise energy using MRF
    Energy = 0.0
    for i in range(nR,im.shape[0]):
        for j in range(0,im.shape[1]):
            Energy  +=  pottsEnergy(im,i,j) #- 0.0001*(math.log(im[i,j] + 0.001))

    return Energy


def MRF(im, initE, labels):
    # 8 point neighbourhood for each pixel in the mask
    im_final = simAnnealing(im, initE, labels)
    return im_final


if __name__=="__main__":
    init_mask = cv2.imread("um_000010.png",0)
    init_mask = init_mask/255.0
    print(np.unique(init_mask))

    labels = [0.0,0.25, 0.50, 0.75, 1.00] 

    E_init = np.zeros(init_mask.shape,dtype=np.float32)
    for i in range(init_mask.shape[0]):
        for j in range(init_mask.shape[1]):
            E_init[i,j] = pottsEnergy(init_mask, i,j)

    
    plt.imshow(E_init)
    plt.show()

    fin_mask = MRF(init_mask, E_init, labels)
    cv2.imwrite("Markov_mask.png", fin_mask*255)
    plt.imshow(fin_mask)
    plt.show()
    plt.figure()
    f, axarr = plt.subplots(3,1) 
    axarr[0].imshow(init_mask)
    axarr[1].imshow(E_init)
    axarr[2].imshow(fin_mask)
    plt.show()