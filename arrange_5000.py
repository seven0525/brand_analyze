import os
import re
import random

path = "./datasets/yoji_expand"
img_list = os.listdir(path)
count = 0
for img in img_list:
    index = re.search(".jpg", img)
    if index:
        count += 1

while count > 5000:
    chosen_img = random.choice(img_list)
    if chosen_img != ".DS_Store":
        os.remove(os.path.join(path, chosen_img))