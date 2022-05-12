import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import match_template

import cv2

def cal(file):

    image = cv2.imread(file, 0)
    coin = image[80:97, 27:42]

    result = match_template(image, coin)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(coin, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('template')

    ax2.imshow(image, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('image')
    # highlight matched region
    hcoin, wcoin = coin.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    ax3.imshow(result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    #plt.show()
    plt.savefig(f'./tmp_1.png', dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    pts = np.array(np.where(result > 0.7)).T
    pts = pts[:, [1, 0]]

    pts2 = []
    confidences = []
    for i in range(len(pts)):
        is_neighbors = np.sqrt(np.sum((pts - pts[i]) ** 2., axis=1)) < 10
        pt_num = (is_neighbors).sum()
        confidences.append(pt_num)
        pts2.append(pts[is_neighbors].mean(axis=0))

    pts2 = np.array(pts2)
    pts2 = np.unique(pts2, axis=0)
    
    if len(pts2)==0:
        return False, 0
    else:
        plt.scatter(pts2[:, 0], pts2[:, 1])
        #plt.show()
        plt.savefig(f'./tmp_2.png', dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        return True, len(pts2)


if __name__ == '__main__':
    cal()
    