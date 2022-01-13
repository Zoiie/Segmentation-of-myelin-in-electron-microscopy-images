import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

listdir="./test_unseen_images"

list_numbers=len(os.listdir(listdir))
print(list_numbers,int(list_numbers /int(np.sqrt(list_numbers))))
lists=os.listdir(listdir)
print(lists)

# fig, ax = plt.subplots(figsize=(int(list_numbers/int(np.sqrt(list_numbers))), int(np.sqrt(list_numbers))))
row=int(int(np.sqrt(list_numbers)))
rank=int(list_numbers/int(np.sqrt(list_numbers)))
plt.figure()
i=1
for list in lists:
    plt.subplot(row,rank,i)
    image_dir="test_unseen_images/"+list
    img = mpimg.imread(image_dir)
    plt.title(list)
    plt.imshow(img)
    i=i+1
plt.show()