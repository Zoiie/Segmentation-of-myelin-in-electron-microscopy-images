import os
from PIL import Image
import numpy as np

test_seg_address="data/SPLIT"
test_seg_lists=os.listdir(test_seg_address)
for i in test_seg_lists:
    print(i)
    if "test" in i:
        im=Image.open(test_seg_address+"/"+i)
        test_seg_im_arr=np.array(im)
        print("im_arr:",test_seg_im_arr.shape,test_seg_im_arr)


