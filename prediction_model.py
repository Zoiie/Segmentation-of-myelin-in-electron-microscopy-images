#!/usr/bin/env python
# coding: utf-8

# System imports
import os

# from google.colab import drive
from glob import glob

# Package imports
from PIL import Image
from typing import Sequence, Union
import matplotlib.pyplot as plt
import numpy as np
import monai
from monai.config import print_config
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    AddChannel,
    Compose,
    LoadImage,
    Rotate90,
    ScaleIntensity,
    ToTensor,
    RandFlip
)
from monai.utils import set_determinism
from monai.data import ArrayDataset
from monai.transforms.compose import Transform
import torch

print_config()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ## Drive mounting (if using Google Colab)

# ## User Inputs

# **Needed inputs**
# * data_dir: str (data directory)
# * batch_size: int (the size of the training batches)
# * nb_channels: int (the number of channels in the input image)
# * lr: float (the learning rate used during training)
# * output_dim: list of integers (the dimension of the output image)
# * val_interval: integer (compute validation loss value every val_interval iterations)
# * labels: list of integers (the labels to select)
# * outdir: str (the output directory)

# Input parameters
args = {
    # 'data_dir' : '/content/drive/My Drive/projet_myeline/data',
    'data_dir': './data',

    'batch_size': 10,
    'nb_channels': 1,
    'nb_epochs': 1,
    'lr': 0.1,
    'output_dim': [448, 672],
    'val_interval': 1,
    # 'labels' : [1, 2, 3],
    'labels': [2],
    'outdir': './data/SPLIT'

}

# ## Classes and functions
def my_round(x):
    """ Round a float value

      Parameters
      ----------
      x: float (mandatory)
          the input value

      Returns
      -------
      x_out: float
          the output rounded value

     Command:
     -------
     x_out = my_round(x)
     """

    # Round
    x_out = np.round(x * 100) / 100
    return x_out

def create_data_dict(imsdir):

    """ Load the images and their associated manual segmentations,
    and put them into a dictionary

      Parameters
      ----------
      imsdir: str (mandatory)
          the data full path
      masksdir: str (mandatory)
          the manual segmentation full path

      Returns
      -------
      data_dict: dictionary
          the dictionary associated images with their manual segmentation

     Command:
     -------
     data_dict = create_data_dict(imsdir, masksdir)
     """
    data_dict = {}
    impaths = []
    gtpaths = []
    print("创建图像列表")
    for impath in sorted(glob(os.path.join(imsdir, '*.tif'))):
        imname = os.path.basename(impath).split('.')[0]
        gtfile = imname + '_segmented.png'
        print("impath",impath)

        # Convert the tif into png if necessary
        im_png_path = os.path.join(imsdir, imname + '.png')
        im_arr = np.array(Image.open(impath))
        im = Image.fromarray(im_arr)
        # print(im.dtype)
        # fig,ax=plt.subplots()
        # ax.imshow(im, cmap='gray')
        # plt.show()
        im.save(im_png_path)

        # Keep the image only if it is associated with a segmentation
        impaths.append(im_png_path)
    data_dict = {'image': impaths, 'label': gtpaths}
    print("完成")
    return data_dict


class my_resample(Transform):
    """
    Resample an image of size s1 x s2 into an image of size new_s1 x new_s2,
    possibly with new_s1 != new_s2

    Parameters
    ----------
    pixdim: list of integers [s1_new, s2_new] (mandatory)
        the dimension of the output image

    mode: str ('nn' or 'bilinear')
      the method used for interpolation (nearest-neighbor or bilinear)

    Returns
    -------
    output_arr: array
        the output array of dimension s1_new x s2_new x n

   Command:
   -------
   data_dict = create_data_dict(imsdir, masksdir)

    """

    def __init__(
            self,
            pixdim: Union[Sequence[int], int],
            mode: str,
    ) -> None:
        self.pixdim = pixdim
        self.mode = mode

    def __call__(self, img: np.ndarray) -> np.ndarray:

        # Choose the resample method
        if self.mode == 'nn':
            order = Image.NEAREST
        elif self.mode == 'bilinear':
            order = Image.BILINEAR

        # Resample using PIL
        pil_img = Image.fromarray(img)
        output_arr = np.array(pil_img.resize((self.pixdim[1],
                                              self.pixdim[0]),
                                             order))

        return output_arr

# Make the code reproducible (to get the same result at each run)
def main(path):
    set_determinism(seed=0)

    # Get the EM images and associated segmentations directories
    # im_test_dir = os.path.join(args['data_dir'], 'SPLIT')
    im_test_dir = os.path.join(path)


    # label_test_dir = os.path.join(args['data_dir'], 'SEGMENT_TEST')

    # Display database statistics
    # print('{0} training images'.format(len(os.listdir(im_train_dir))))
    # print('{0} training manual segmentations'.format(len(os.listdir(label_train_dir))))
    # print('{0} validation images'.format(len(os.listdir(im_valid_dir))))
    # print('{0} validation manual segmentations'.format(len(os.listdir(label_valid_dir))))
    # print('{0} test images'.format(len(os.listdir(im_test_dir))))
    # print('{0} test manual segmentations'.format(len(os.listdir(label_test_dir))))

    # Get parameters for the Unet
    nb_classes = len(args['labels'])

    # print('{0} classes, {1}D image'.format(nb_classes, dimension))

    # # Training

    # In[ ]:

    # Select the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the loss function = mean Dice of the labels, background excluded
    # loss_function = DiceLoss(softmax=True)

    # loss_function = topoloss_pytourch


    # # Training

    # In[ ]:

    # start a typical PyTorch training


    # Go through each epoch

    # In[ ]:

    # logsdir = "./comments"



    # ## Test

    # In[ ]:
    # test_data_dict = create_data_dict(im_test_dir, label_test_dir)
    test_data_dict = create_data_dict(im_test_dir)

    nb_classes = len(args['labels'])
    if nb_classes == 1:
        nb_classes += 1
    # dimension = len(im_arr.shape)
    # class_ids = np.unique(gt_arr)
    # Select a model
    trained_model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=args['nb_channels'],
        out_channels=nb_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    trained_model.load_state_dict(torch.load(os.path.join(args['outdir'],'final_model.pth')));

    # print(trained_model)
    # In[ ]:

    # Test image transforms
    test_imtrans = Compose([LoadImage(image_only=True),
                            my_resample(pixdim=args['output_dim'], mode='bilinear'),
                            ScaleIntensity(),
                            AddChannel(),
                            ToTensor()])

    # Test segmentation transforms

    # In[ ]:

    # create the test data loader
    # test_ds = ArrayDataset(test_data_dict['image'], test_imtrans,
    #                        test_data_dict['label'], test_segtrans)
    test_ds = ArrayDataset(test_data_dict['image'], test_imtrans)
    print("?????",test_ds.dataset)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True,
                             num_workers=2,
                             pin_memory=torch.cuda.is_available())

    # In[ ]:

    # Test the model on unseen images
    # tot_dices = []
    # se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    # testdir = "./data/SPLIT"
    testdir = path
    # print(testdir)

    # trained_model.eval()
    # print("??????",test_loader)
    with torch.no_grad():
        for cnt_test, test_data in enumerate(test_loader):

            # Load each couple of image/manual segmentation
            im_torch = test_data[0].to(device)
            # gt_torch = test_data[1].to(device)

            # Apply the model to the unknown image
            # shape=im_torch.shape
            # print(shape)

            im_torch=torch.reshape(im_torch,[1,1,448,672])
            # print(im_torch.shape)
            pred_torch = trained_model(im_torch)

            # Hard automatic segmentation
            seg_arr = pred_torch[0, :, :, :].detach().cpu()
            seg_arr = np.argmax(seg_arr.numpy(), axis=0)
            seg_arr = seg_arr+1
            print(cnt_test)

            # Predicted labels
            pred_labs = np.unique(seg_arr)
            pred_labs = pred_labs[pred_labs > 0]

            # Display and save the results
            im_arr = im_torch[0, 0, :, :].detach().cpu()
            print(im_arr.numpy().shape)
            im_save=Image.fromarray(im_arr.numpy())
            # im_save.show()
            # im_save.convert("P").show()
            # print(im_save.max)
            # im_save.save("{0}_{1}dddd.png".format(testdir,cnt_test),"png")

            fig, ax = plt.subplots(ncols=2,nrows=1)
            ax[0].imshow(seg_arr, cmap='gray')
            ax[1].imshow(im_arr, cmap='gray')
            for label, color in zip(pred_labs, ['r', 'b', 'k']):
                ax[1].contour(seg_arr == label, [0.5], colors=color, linewidths=[1, 1])

            # ax[0].save(cnt_test+".png")

            fig.savefig(os.path.join(testdir, 'compare_{0}_pred.png'.format(cnt_test)))

            fig, ax = plt.subplots()
            ax.imshow(seg_arr, cmap='gray')
            ax.axis('off')
            # ax.set_title('Predicted segmentation')
            # plt.axis('off')
            plt.margins(0, 0)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            fig.savefig(os.path.join(testdir, 'test{0}_pred.png'.format(cnt_test)),bbox_inches = 'tight', pad_inches = 0)

            fig, ax = plt.subplots()
            ax.imshow(im_arr, cmap='gray')
            for label, color in zip(pred_labs, ['r', 'b', 'k']):
                ax.contour(seg_arr == label, [0.5], colors=color, linewidths=[1, 1])
            ax.axis('off')
            ax.set_title('EM image and contours of the predicted labels')
            fig.savefig(os.path.join(testdir, 'test{0}_im+pred.png'.format(cnt_test)))

            # im = Image.fromarray(np.uint8(im_arr))
            # seg=Image.fromarray(np.uint8(seg_arr))
            # im.save('{0}_pred.png'.format(cnt_test))
            # seg.save('{0}_predmask.png'.format(cnt_test))

            # fig[0].savefig(os.path.join(testdir, '{0}_pred.png'.format(cnt_test)))
            # fig[1].savefig(os.path.join(testdir, '{0}_predmask.png'.format(cnt_test)))

            plt.show()

if __name__ == '__main__':
    path="./data/SPLIT"
    main(path)