import numpy as np
import cv2
import glob
import os
import pandas as pd
from skimage import data
from skimage.feature import match_template
import template_matching3
import shutil

#def template_matching(image, corners):
#    dif = 
#    coin = image[83:83+19, 15:15+18]
#
#    result = match_template(image, coin)
#    ij = np.unravel_index(np.argmax(result), result.shape)
#    x, y = ij[::-1]
#
#    fig = plt.figure(figsize=(8, 3))
#    ax1 = plt.subplot(1, 3, 1)
#    ax2 = plt.subplot(1, 3, 2)
#    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
#
#    ax1.imshow(coin, cmap=plt.cm.gray)
#    ax1.set_axis_off()
#    ax1.set_title('template')
#
#    ax2.imshow(image, cmap=plt.cm.gray)
#    ax2.set_axis_off()
#    ax2.set_title('image')
#    # highlight matched region
#    hcoin, wcoin = coin.shape
#    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
#    ax2.add_patch(rect)
#
#    ax3.imshow(result)
#    ax3.set_axis_off()
#    ax3.set_title('`match_template`\nresult')
#    # highlight matched region
#    ax3.autoscale(False)
#    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
#
#    plt.show()



def detection(line, images, columns):
    
    x = columns
    y = line
    
    dots = images

    dbg_image_circles = []

    found, corners = cv2.findCirclesGrid(dots, (x,y), cv2.CALIB_CB_SYMMETRIC_GRID)
    if found == True:
        dbg_image_circles = dots.copy()
        cv2.drawChessboardCorners(dbg_image_circles, (x, y), corners, found)
        img = dbg_image_circles
    else:
        img = dots
    
    return found, img, corners
    
def run_1(images, grid_form):
    block = [0, 1, 0, 1]
    img_tmp = cv2.imread(images)
    cv2.imwrite("./pic.jpg",img_tmp)
    for i in range(grid_form[0],3,-1):
        #images = './pic/pic3.JPG'
        found, img, corners = detection(i, img_tmp, grid_form[1])
        if found == True:
            break
    return img, found, corners, block

def run_2(images, grid_form):
    hmx = 1
    h = 0
    found_mx = False
    block_mx = [0, 1, 0, 1]
    img_tmp = cv2.imread(images)
    img_mx = img_tmp
    corners_mx = []
    height, width = img_tmp.shape[:2]
    df = pd.read_csv('divide.csv')
    case = int(len(df.columns) / 4)
    for n in range(case):
        df_tmp = df.iloc[:,n*4:(n+1)*4].dropna(how='any')
        div = len(df_tmp)
        for m in range(div):
            block = [df_tmp.iloc[m,0], df_tmp.iloc[m,1], df_tmp.iloc[m,2], df_tmp.iloc[m,3]]
            img_tmp = img_tmp[int(height*block[0]):int(height*block[1]),int(width*block[2]):int(width*block[3])]
            img_tmp = cv2.resize(img_tmp, dsize=(width, height))
            for i in range(grid_form[0],3,-1):
                found, img, corners = detection(i, img_tmp, grid_form[1])
                if found == True:
                    h = i*10
                    break
            if h > hmx:
                hmx = h
                img_mx = img
                found_mx = True
                corners_mx = corners
                block_mx = block
        if found_mx:
            break
    if found_mx:
        add = np.array([[[block_mx[0]*width, block_mx[2]*height]]] * len(corners_mx))
        corners_mx = corners_mx + add
    return img_mx, found_mx, corners_mx, block_mx

def predict_wl(corners, grid_form, interval, buffer):
    PN = len(corners) / grid_form[1]
    WL = 9.995 - buffer - PN*interval/100.0  #水位（設置箇所上部の標高基準）
    print(WL)
    
    return WL

#####MAIN#######
#case1
in_path = "./pic_mask_1/"
out_path = "./pic_out_1/"
grid_form = [10, 4]
interval = 20.0  #(cm)
buffer = 0.03 + 0.14/2  #(m)

#case2
#in_path = "./pic_mask_2/"
#out_path = "./pic_out_2/"
#grid_form = [20, 4]
#interval = 10.0  #(cm)
#buffer = 0.065 + 0.07/2  #(m)

#case3
in_path = "./01_kumorigawahimon_cut_3/"
out_path = "./pic_out_3/"
grid_form = [10, 4]
interval = 20.0  #(cm)
buffer = 0.03 + 0.14/2 +0.07  #(m)
max_point = 40


H_list = []
flist = glob.glob(in_path + "*.jpg")
for file in flist:
    #量水標検知
    #img, found, corners, block = run_1(file, grid_form)
    #if found == False:
    #    img, found, corners, block = run_2(file, grid_form)
    #    if found == False:
    #        print("NOT FOUND")
    found, point = template_matching3.cal(file)
    point = min(point, max_point)
    if found:
        #corners = template_matching(cv2.imread(file,0), corners)
        height, width = cv2.imread(file).shape[:2]
        #水位算定
        #H = predict_wl(corners, grid_form, interval, buffer)
        H = 9.995 - buffer - int(point/4-1)*interval/100.0
        H_list.append([os.path.basename(file), H])
        #cv2.imwrite(out_path + os.path.basename(file), img)
        shutil.copy('tmp_1.png', out_path + os.path.splitext(os.path.basename(file))[0]+"_1.png")
        shutil.copy('tmp_2.png', out_path + os.path.splitext(os.path.basename(file))[0]+"_2.png")
    else:
        H_list.append([os.path.basename(file), "miss"])
df = pd.DataFrame(H_list, columns=['time', 'H'])
df.to_csv('./H_series3.csv')
