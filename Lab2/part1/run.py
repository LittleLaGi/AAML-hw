#!/usr/bin/env python3

from os import openpty
from PIL import Image
import numpy as np
import torch
from multiprocessing import Pool
import multiprocessing
import sys
import torchvision.transforms as transforms
from torch.autograd import Variable
img_to_tensor = transforms.ToTensor()

num_cpus = multiprocessing.cpu_count()

def loadImage():
    # image = Image.open('weights/input.jpg')
    # # convert image to numpy array: <class 'numpy.ndarray'>
    # # (224, 224, 3) h w c
    # # kernel: n c h w
    # data = np.asarray(image) 
    # norm = np.linalg.norm(data)
    # return data / norm
    img = Image.open('weights/input.jpg')
    img = img.resize((224, 224))
    tensor = img_to_tensor(img)
    tensor = tensor.resize_(3, 224, 224)
    return Variable(tensor)

def padArray(array, size):
    if size == 0:
        return array
    array_pad = np.zeros((array.shape[0], array.shape[1] + 2 * size, array.shape[2] + 2 * size), dtype=float)
    array_pad[:, size:-size, size:-size] = array
    return array_pad

def job_conv3D(arg):
    offset, num_work = arg[0], arg[1]
    l = arg[2]
    kernel, new_array, array_pad, bias = arg[3], arg[4], arg[5], arg[6]
    for co in range(num_work):      
        for i in range(l):
            for j in range(l):
                tmp = 0
                for ci in range(kernel.shape[1]):               
                    for k1 in range(3):
                        for k2 in range(3):
                            tmp += array_pad[ci][i + k1][j + k2] * kernel[co + offset][ci][k1][k2]
                tmp += bias[co + offset]
                if tmp > 0.0:
                    new_array[co + offset][i][j] = tmp
    
    return new_array
    
def conv3D(array, kernel, bias, stride=1, pad=1):
    if pad > 0:
        array_pad = padArray(array, pad)
    else:
        array_pad = array
    
    l = array.shape[1]
    new_array = np.zeros(shape=(kernel.shape[0], l, l), dtype=float)

    pool = Pool(num_cpus)
    inputs = []
    offset = 0
    for i in range(num_cpus):
        if i == num_cpus - 1:
            num_work = int(kernel.shape[0] / num_cpus + kernel.shape[0] % num_cpus)
        else:
            num_work = int(kernel.shape[0] / num_cpus)
        inputs.append((offset, num_work, l, kernel, new_array, array_pad, bias))
        offset += int(kernel.shape[0] / num_cpus)
    new_arrays = pool.map(job_conv3D, inputs)

    return np.add.reduce(new_arrays)

def job_maxPooling(arg):
    offset, num_work, l, array, new_array = arg[0], arg[1], arg[2], arg[3], arg[4]
    for c in range(num_work):
        for i in range(int(l/2)):
            for j in range(int(l/2)):
                max = -sys.float_info.max
                for k1 in range(2):
                    for k2 in range(2):
                        tmp = array[c + offset][2 * i + k1][2 * j + k2]
                        if tmp > max:
                            max = tmp
                new_array[c + offset][i][j] = max

    return new_array

def maxPooling(array):
    l = array.shape[1]
    c = array.shape[0]

    new_array = np.zeros(shape=(c, int(l/2), int(l/2)), dtype=float)
    
    pool = Pool(num_cpus)
    inputs = []
    offset = 0
    for i in range(num_cpus):
        if i == num_cpus - 1:
            num_work = int(c / num_cpus + c % num_cpus)
        else:
            num_work = int(c / num_cpus)
        inputs.append((offset, num_work, l, array, new_array))
        offset += int(c / num_cpus)
    new_arrays = pool.map(job_maxPooling, inputs)

    return np.add.reduce(new_arrays) 


def relu(x):
	return max(0.0, x)


def job_Fc(arg):
    offset, num_work, weights, bias, new_array, array, flag = arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6]
    for i in range(num_work):
        for j in range(weights.shape[1]):
            new_array[i + offset] += weights[i + offset][j] * array[j]
        new_array[i + offset] += bias[i + offset]
        if (flag):
            new_array[i + offset] = relu(new_array[i + offset])

    return new_array

def Fc(array, weights, bias, flag):
    if array.ndim != 1:
        array = array.flatten()
    
    new_array = np.zeros(shape=(weights.shape[0]), dtype=float)

    pool = Pool(num_cpus)
    inputs = []
    offset = 0
    for i in range(num_cpus):
        if i == num_cpus - 1:
            num_work = int(weights.shape[0] / num_cpus + weights.shape[0] % num_cpus)
        else:
            num_work = int(weights.shape[0] / num_cpus)
        inputs.append((offset, num_work, weights, bias, new_array, array, flag))
        offset += int(weights.shape[0] / num_cpus)
    new_arrays = pool.map(job_Fc, inputs)

    return np.add.reduce(new_arrays) 
        

def main():
    input = loadImage()

    print("Conv1")
    weights = np.load('weights/Conv1_weights.npy')
    bias = np.load('weights/Conv1_bias.npy')
    input = conv3D(input, weights, bias)
    print("Conv2")
    weights = np.load('weights/Conv2_weights.npy')
    bias = np.load('weights/Conv2_bias.npy')
    input = conv3D(input, weights, bias)
    print("maxPooling")
    input = maxPooling(input)
    print("Conv3")
    weights = np.load('weights/Conv3_weights.npy')
    bias = np.load('weights/Conv3_bias.npy')
    input = conv3D(input, weights, bias)
    print("Conv4")
    weights = np.load('weights/Conv4_weights.npy')
    bias = np.load('weights/Conv4_bias.npy')
    input = conv3D(input, weights, bias)
    print("maxPooling")
    Conv5_in = maxPooling(input)
    print("Conv5")
    weights = np.load('weights/Conv5_weights.npy')
    bias = np.load('weights/Conv5_bias.npy')
    input = conv3D(Conv5_in, weights, bias)
    print("Conv6")
    weights = np.load('weights/Conv6_weights.npy')
    bias = np.load('weights/Conv6_bias.npy')
    input = conv3D(input, weights, bias)
    print("Conv7")
    weights = np.load('weights/Conv7_weights.npy')
    bias = np.load('weights/Conv7_bias.npy')
    input = conv3D(input, weights, bias)
    print("maxPooling")
    input = maxPooling(input)
    print("Conv8")
    weights = np.load('weights/Conv8_weights.npy')
    bias = np.load('weights/Conv8_bias.npy')
    input = conv3D(input, weights, bias)
    print("Conv9")
    weights = np.load('weights/Conv9_weights.npy')
    bias = np.load('weights/Conv9_bias.npy')
    input = conv3D(input, weights, bias)
    print("Conv10")
    weights = np.load('weights/Conv10_weights.npy')
    bias = np.load('weights/Conv10_bias.npy')
    input = conv3D(input, weights, bias)
    print("maxPooling")
    input = maxPooling(input)
    print("Conv11")
    weights = np.load('weights/Conv11_weights.npy')
    bias = np.load('weights/Conv11_bias.npy')
    input = conv3D(input, weights, bias)
    print("Conv12")
    weights = np.load('weights/Conv12_weights.npy')
    bias = np.load('weights/Conv12_bias.npy')
    input = conv3D(input, weights, bias)
    print("Conv13")
    weights = np.load('weights/Conv13_weights.npy')
    bias = np.load('weights/Conv13_bias.npy')
    input = conv3D(input, weights, bias)
    print("maxPooling")
    input = maxPooling(input)

    # print("avgPooling")
    # avgPooling = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))
    # input = avgPooling(torch.from_numpy(input))
    # input = input.numpy()
    print("Fc14")
    weights = np.load('weights/Fc14_weights.npy')
    bias = np.load('weights/Fc14_bias.npy')
    input = Fc(input, weights, bias, 1)
    print("Fc15")
    weights = np.load('weights/Fc15_weights.npy')
    bias = np.load('weights/Fc15_bias.npy')
    input = Fc(input, weights, bias, 1)
    print("Fc16")
    weights = np.load('weights/Fc16_weights.npy')
    bias = np.load('weights/Fc16_bias.npy')
    output = Fc(input, weights, bias, 0)

    # ans[435]
    ans = np.load('weights/output.npy')
    print()
    if np.argmax(ans) == np.argmax(output):
        print("Pass!!")
        print(f"index of the max: {np.argmax(ans)}")
    else:
        print("Incorrect!!")
    
    max_ans = max(ans)
    max_mine = max(output)
    print()
    print(f"max ans: {max_ans}")
    print(f"max mine: {max_mine}")

if __name__ == "__main__":
    main()

