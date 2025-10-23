import os
data_folder = os.path.join('data', 'stage1_train')
print(data_folder)
image_ids = next(os.walk((data_folder)))[1][0]
print(image_ids)
mask = os.path.join(data_folder, image_ids, 'masks')
print(mask)
x = next(os.walk(mask))
print(x)