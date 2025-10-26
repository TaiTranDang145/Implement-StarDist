import cv2

path = './data/stage1_train/0b2e702f90aee4fff2bc6e4326308d50cf04701082e718d4f831c8959fbcda93/masks/3235d8c9a0f43a97eeaea47331f62672253eacb740f2c81ac2ab3997256afb4d.png'
img = cv2.imread(path)

print(img.shape)