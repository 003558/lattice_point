from pathlib import Path
import numpy as np
import cv2

for fn in Path('01_kumorigawahimon').glob('*.jpg'):
    img = cv2.imread(str(fn))
    img_cut = img[482:679, 939:1009]
    cv2.imwrite('./01_kumorigawahimon_cut/{}.jpg'.format(fn.stem), img_cut)
