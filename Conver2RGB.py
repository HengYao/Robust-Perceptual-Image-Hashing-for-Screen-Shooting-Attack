import os
import cv2
from PIL import Image
import imghdr
import re

read_path = r" "
save_path = r" "

file_list = os.listdir(read_path)

for i in file_list:
    img_path = os.path.join(read_path, i)
    img = Image.open(img_path)
    f = img.getbans()

    if str(f) != "('R', 'G', 'B')":
        src = cv2.imread(img_path)
        img1 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, img1)
        print(i + "已转换成RGB图片")