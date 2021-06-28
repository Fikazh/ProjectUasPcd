# importing PIL Module
from PIL import Image
import os

dir = os.getcwd()
categories = ['Beracun', 'BisaDimakan']

for category in categories:
        path = os.path.join(dir, category)
        i=24
        j=48
        k=72

        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            image = Image.open(imgpath)

            vertical_img = image.transpose(method=Image.FLIP_TOP_BOTTOM)
            vertical_img.save(f'{path}/{i}.png')

            horizontal_img = image.transpose(method=Image.FLIP_LEFT_RIGHT)
            horizontal_img.save(f'{path}/{j}.png')

            vertical_horizontal_img = horizontal_img.transpose(method=Image.FLIP_TOP_BOTTOM)
            vertical_horizontal_img.save(f'{path}/{k}.png')

            i+=1
            j+=1
            k+=1