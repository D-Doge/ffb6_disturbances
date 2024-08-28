import numpy as np
import math

def eul2rot(theta) :
    # theta must be in rads

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

def decompose_rotation_matrix(rotation_matrix):
    #Radians https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    #Does beta alpha gamma YXZ, meaning Z has no dependencys
    if((rotation_matrix[2,0] != 1) or (rotation_matrix[2,0] != -1)):
        beta1 = -math.asin(rotation_matrix[2,0])
        beta2 = math.pi - beta1

        alpha1 = math.atan2(rotation_matrix[2,1]/math.cos(beta1), rotation_matrix[2,2]/math.cos(beta1))
        alpha2 = math.atan2(rotation_matrix[2,1]/math.cos(beta2), rotation_matrix[2,2]/math.cos(beta2))

        gamma1 = math.atan2(rotation_matrix[1,0]/math.cos(beta1), rotation_matrix[0,0]/math.cos(beta1))
        gamma2 = math.atan2(rotation_matrix[1,0]/math.cos(beta2), rotation_matrix[0,0]/math.cos(beta2))

        return np.array(((alpha1, beta1, gamma1), (alpha2, beta2, gamma2)))
    else:
        gamma = 0
        if(rotation_matrix[2,0] == -1):
            beta = math.pi/2
            alpha = gamma + math.atan2(rotation_matrix[0,1], rotation_matrix[0,2])
        else:
            beta = -math.pi/2
            alpha = -gamma + math.atan2(-rotation_matrix[0,1], -rotation_matrix[0,2])


    return np.array((alpha, beta, gamma))

# TODO function that does with X independent and Y indipendent (mazbe use scipy)