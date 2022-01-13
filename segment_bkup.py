#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------
# # Segmentation of EM images
# ------------------------------------------------------------------------------

# **Description of the notebook**
#
# Segment EM images into myelin, axons and background.
#
# This code should be modified to perform the binary segmentation of the myelin using topological losses.
#
# The data are included in a 'data' directory you have to create (on your Drive if you are using Google Colab). This directory also contains 6 directories:
# * TRAIN (EM images used for training)
# * VALIDATION (EM images used for validation)
# * TEST (EM images used for test)
# * SEGMENT_TRAIN (3-class manual segmentations used for training)
# * SEGMENT_VALIDATION (3-class manual segmentations used for validation)
# * SEGMENT_TEST (3-class manual segmentations used for test)
#
# The database distribution used in [1] is provided in the file distribution.csv
#
# **References**
# * [1] Le Couedic, T., Caillon, R., Rossant, F., Joutel, A., Urien, H., & Rajani, R. M. (2020, November). Deep-learning based segmentation of challenging myelin sheaths. In 2020 Tenth International Conference on Image Processing Theory, Tools and Applications (IPTA) (pp. 1-6). IEEE.

# ## Installations

# In[1]:


# Install Monai
# get_ipython().system('python -c "import monai" || pip install -q "monai[nibabel, tqdm]"')
# get_ipython().system('python -c "import matplotlib" || pip install -q matplotlib')
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Imports

# In[ ]:


# System imports
import json
import os
import time
# from google.colab import drive
from glob import glob

# Package imports
from scipy import ndimage as ndi
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
# from skimage.transform import resize
import torch
from torch import nn
# from sklearn.decomposition import PCA
from torch.autograd import Variable
# print_config()
import topoloss_pytourch

# print("hhh")
# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ## Drive mounting (if using Google Colab)

# In[ ]:


# Mount the drive
# drive.mount('/content/drive')


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

# In[ ]:


# Input parameters
args = {
    # 'data_dir' : '/content/drive/My Drive/projet_myeline/data',
    'data_dir': './data',

    'batch_size': 20,
    'nb_channels': 1,
    'nb_epochs': 40,
    'lr': 0.1,
    'output_dim': [448, 672],
    'val_interval': 1,
    # 'labels' : [1, 2, 3],
    'labels': [2],
    # 'outdir' : '/content/drive/My Drive/projet_myeline/results/test'
    'outdir': './results/test'

}
print(len(args['labels']))


# ## Classes and functions

# In[ ]:


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


# In[ ]:


def create_data_dict(imsdir, masksdir):
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
    for impath in sorted(glob(os.path.join(imsdir, '*.tif'))):
        imname = os.path.basename(impath).split('.')[0]
        gtfile = imname + '_segmented.png'

        # Convert the tif into png if necessary
        im_png_path = os.path.join(imsdir, imname + '.png')
        im_arr = np.array(Image.open(impath))
        im = Image.fromarray(im_arr)
        im.save(im_png_path)

        # Keep the image only if it is associated with a segmentation
        if not os.path.exists(os.path.join(masksdir, gtfile)):
            print('Warning : no segmentation for {0}'.format(imname))
        else:
            impaths.append(im_png_path)
            gtpaths.append(os.path.join(masksdir, gtfile))
    data_dict = {'image': impaths, 'label': gtpaths}

    return data_dict


# In[ ]:


class seg2Dtoseg3D(Transform):
    """
    Convert a 2D segmentation (size s1 x s2) with n labels to an image of
    dimension s1 x s2 x n image,
    each channel being the binary image of the associated label

    Parameters
    ----------
    labels: list of integers (mandatory)
        the list of the n needed label values

    Returns
    -------
    output_arr: array
        the output array of dimension s1 x s2 x n

   Command:
   -------
   data_dict = create_data_dict(imsdir, masksdir)

    """

    def __init__(
            self,
            labels: Union[Sequence[int], int],
    ) -> None:
        self.labels = labels

    def __call__(self,
                 img: np.ndarray) -> np.ndarray:
        result = []
        for class_id in self.labels:
            if len(self.labels) == 1:
                result.append(img != class_id)

            result.append(img == class_id)

            # Output array
            out_arr = np.stack(result, axis=0).astype(np.float32)
        return out_arr


# In[ ]:


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


# ## Reproducibility

# In[ ]:

def topo_module(lh, gt):
    topoloss_sum = []
    # print(lh.shape)
    lh_bu = lh
    gt_bu = gt
    # print("lh:",lh,gt)
    for i in range(0, len(lh)):
        print(i)
        lh_re = lh_bu[i, :, :, :].detach().cpu()
        lh_np = np.argmax(lh_re.numpy(), axis=0)
        lh = lh_np
        lh = torch.from_numpy(lh)

        gt_re = gt_bu[i, :, :, :].detach().cpu()
        gt_np = np.argmax(gt_re.numpy(), axis=0)
        gt = gt_np
        gt = torch.from_numpy(gt)

        loss_topo = topoloss_pytourch.getTopoLoss(lh, gt)
        # print("loss_topo:",loss_topo)
        topoloss_sum.append(loss_topo)

    loss_topo_out = 0
    # print("topoloss_sum:",topoloss_sum)
    for i in topoloss_sum:
        loss_topo_out = loss_topo_out + i
    print("topoloss average:", loss_topo_out / len(topoloss_sum))
    return loss_topo_out / len(topoloss_sum)


# Make the code reproducible (to get the same result at each run)
if __name__ == "__main__":
    set_determinism(seed=0)

    # ## Data preparation

    # In[ ]:

    # #Create output directory if needed
    # if not os.path.isdir(args['outdir']):
    #     os.makedirs(args['outdir'])
    #
    # #Create logs output directory if needed
    # logsdir = os.path.join(args['outdir'], 'logs')
    # if not os.path.isdir(logsdir):
    #     os.makedirs(logsdir)
    #
    # #Create training output directory if needed
    # traindir = os.path.join(args['outdir'], 'train')
    # if not os.path.isdir(traindir):
    #     os.makedirs(traindir)
    #
    # #Create test output directory if needed
    # testdir = os.path.join(args['outdir'], 'test')
    # if not os.path.isdir(testdir):
    #     os.makedirs(testdir)

    # In[ ]:

    # Get the EM images and associated segmentations directories

    im_train_dir = os.path.join(args['data_dir'], 'TRAIN')
    label_train_dir = os.path.join(args['data_dir'], 'SEGMENT_TRAIN')

    im_valid_dir = os.path.join(args['data_dir'], 'VALID')
    label_valid_dir = os.path.join(args['data_dir'], 'SEGMENT_VALID')

    im_test_dir = os.path.join(args['data_dir'], 'TEST')
    label_test_dir = os.path.join(args['data_dir'], 'SEGMENT_TEST')

    # Display database statistics
    print('{0} training images'.format(len(os.listdir(im_train_dir))))
    print('{0} training manual segmentations'.format(len(os.listdir(label_train_dir))))
    print('{0} validation images'.format(len(os.listdir(im_valid_dir))))
    print('{0} validation manual segmentations'.format(len(os.listdir(label_valid_dir))))
    print('{0} test images'.format(len(os.listdir(im_test_dir))))
    print('{0} test manual segmentations'.format(len(os.listdir(label_test_dir))))

    # In[ ]:

    # Create the training/validation and test data dictionaries
    train_data_dict = create_data_dict(im_train_dir, label_train_dir)
    valid_data_dict = create_data_dict(im_valid_dir, label_valid_dir)
    # print("train_data_dict:",train_data_dict)
    test_data_dict = create_data_dict(im_test_dir, label_test_dir)

    # In[ ]:

    # Display an image and its associated manual segmentation
    impaths = train_data_dict['image']
    # print(impaths)
    im_arr = np.array(Image.open(train_data_dict['image'][0]))
    # print("im_arr:",im_arr)
    gt_arr = np.array(Image.open(train_data_dict['label'][0]))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(im_arr / np.max(im_arr), cmap='gray')
    fig.suptitle('EM image of shape {0}'.format(im_arr.shape))
    ax.axis('off')
    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(gt_arr / np.max(gt_arr), cmap='gray')
    fig.suptitle('Manual segmentation of shape {0}'
                 ' with {1} labels'.format(gt_arr.shape,
                                           len(np.unique(gt_arr))))
    ax.axis('off')

    # In[ ]:

    # Get parameters for the Unet
    nb_classes = len(args['labels'])
    if nb_classes == 1:
        nb_classes += 1
    dimension = len(im_arr.shape)
    class_ids = np.unique(gt_arr)
    print('{0} classes, {1}D image'.format(nb_classes, dimension))

    # # Training

    # In[ ]:

    # define transforms for image and segmentation
    train_imtrans = Compose(
        [
            LoadImage(image_only=True),
            my_resample(pixdim=args['output_dim'], mode='bilinear'),
            ScaleIntensity(),
            AddChannel(),
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            ToTensor(),
        ]
    )

    train_segtrans = Compose(
        [
            LoadImage(image_only=True, dtype=np.uint8),
            my_resample(pixdim=args['output_dim'], mode='nn'),
            seg2Dtoseg3D(labels=args['labels']),
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            ToTensor(),
        ])

    val_imtrans = Compose([LoadImage(image_only=True),
                           my_resample(pixdim=args['output_dim'], mode='bilinear'),
                           ScaleIntensity(),
                           AddChannel(),
                           ToTensor()])

    val_segtrans = Compose([LoadImage(image_only=True),
                            my_resample(pixdim=args['output_dim'], mode='nn'),
                            seg2Dtoseg3D(labels=args['labels']),
                            ToTensor()])

    # In[ ]:

    # create a training data loader
    train_ds = ArrayDataset(train_data_dict['image'], train_imtrans,
                            train_data_dict['label'], train_segtrans)

    # In[ ]:

    train_loader = DataLoader(train_ds,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              num_workers=2,
                              pin_memory=torch.cuda.is_available())

    # In[ ]:

    # create a validation data loader
    val_ds = ArrayDataset(valid_data_dict['image'], val_imtrans,
                          valid_data_dict['label'], val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True,
                            num_workers=2,
                            pin_memory=torch.cuda.is_available())

    # In[ ]:

    # Select the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the network architecture
    model = monai.networks.nets.UNet(
        dimensions=dimension,
        in_channels=args['nb_channels'],
        out_channels=nb_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Define the loss function = mean Dice of the labels, background excluded
    loss_function = DiceLoss(softmax=True)

    # loss_function = topoloss_pytourch

    # Choose an optimizer
    optimizer = torch.optim.SGD(model.parameters(), args['lr'])

    # # Training

    # In[ ]:

    # start a typical PyTorch training
    train_mean_imlosses = []
    valid_mean_imlosses = []
    train_resume = []

    # Go through each epoch
    for epoch in range(args['nb_epochs']):

        # Save comments
        comment = '-' * args['nb_epochs']
        # print(comment)
        train_resume.append(comment)

        comment = 'epoch {0}/{1} '.format(epoch + 1, args['nb_epochs'])
        print(comment)
        train_resume.append(comment)

        # Initialize training paramaters
        model.train()
        sum_epoch_imloss = 0
        nb_step_ims = 0

        # print("train_loader:",train_loader)

        # Go through each batch
        for cnt_batch, batch_data in enumerate(train_loader):
            # Get the batch images and manual segmentations
            # print("train_loader:", batch_data)
            ims_torch = batch_data[0].to(device)
            gts_torch = batch_data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Learn (forward)
            preds_torch = model(ims_torch)
            # print("ims_torch:",ims_torch.shape)
            # print(preds_torch.shape, gts_torch.shape)

            # Compute the loss
            if epoch > 10:
                loss = loss_function(preds_torch, gts_torch)

            else:
                loss = topo_module(preds_torch, gts_torch)

            flag = False
            if flag == True:
                preds_torch_shape = preds_torch.detach().numpy().shape
                # print("preds_torch_shape",preds_torch_shape)
                size = 1
                for i in preds_torch_shape:
                    size = size * i
                size_sqrt = int(np.sqrt(size / 4)) + 1
                # print("preds_torch",preds_torch.shape)

                new_preds = np.resize(preds_torch.detach().numpy(), (size_sqrt, size_sqrt, 4))
                new_gts = np.resize(gts_torch.detach().numpy(), (size_sqrt, size_sqrt, 4))
                # print("new_gts",new_gts.shape)
                cal_topoloss = topoloss_pytourch.getTopoLoss(torch.from_numpy(new_preds), torch.from_numpy(new_gts))
                print("loss:", loss)
                print("loss:", loss.detach().numpy())
                print("cal_topoloss:", cal_topoloss)
                new_loss = loss.detach().numpy() + cal_topoloss.detach().numpy() * 0.00000002
                loss = torch.tensor([new_loss], requires_grad=True)
                print("new loss", loss)

                # print("preds_torch:",preds_torch.shape,"gts_torch",gts_torch.shape)
                # plt.imshow(preds_torch.detach().numpy())
                # plt.show()

                # loss=topoloss_pytourch.getTopoLoss(preds_torch,gts_torch)

                # print("loss:",loss)
                # train(pretrain_epoch, topo_epoch):
                #    for i in range(0, topo_epoch):
                #       if (i<= pretrain_epoch):
                #         loss = nn.CrossEntropyLoss();
                #       else:
                #         loss = nn.CrossEntropyLoss() + lambda * getTopoLoss(lh, gt);
                # Backward + optimize
                # loss.mean()
                # print("loss.mean():",loss)
            loss = Variable(loss, requires_grad=True)
            loss.backward()
            print("loss.backward():", loss)
            optimizer.step()

            # Save batch parameters
            nb_batch_ims = ims_torch.shape[0]
            sum_epoch_imloss += loss.item() * nb_batch_ims
            nb_step_ims += nb_batch_ims
            epoch_len = len(train_ds) // train_loader.batch_size

            # Save comments
            comment = '{0}/{1}, train loss = {2}'.format(cnt_batch + 1, epoch_len,
                                                         my_round(loss.item()))
            # print("comment",comment)
            train_resume.append(comment)

        # Compute and save the mean epoch loss
        mean_imloss = sum_epoch_imloss / nb_step_ims
        train_mean_imlosses.append(mean_imloss)

        # Save comments
        comment = 'epoch {0} average batch loss {1}'.format(epoch + 1,
                                                            my_round(mean_imloss))
        # print(comment)
        train_resume.append(comment)

        if (epoch + 1) % args['val_interval'] == 0:
            model.eval()
            with torch.no_grad():
                sum_epoch_imloss = 0
                nb_step_ims = 0
                for cnt_val, val_data in enumerate(val_loader):
                    # Get the validation images and manual segmentations
                    val_images = val_data[0].to(device)
                    val_labels = val_data[1].to(device)

                    # Compute the validation loss
                    nb_batch_ims = val_images.shape[0]
                    nb_step_ims += nb_batch_ims
                    outputs = model(val_images)
                    loss = loss_function(outputs, val_labels)
                    sum_epoch_imloss += loss.item() * nb_batch_ims

                # Compute and save the mean epoch loss
                mean_imloss = sum_epoch_imloss / nb_step_ims
                valid_mean_imlosses.append(mean_imloss)

    # Save the last model

    torch.save(model.state_dict(), os.path.join(args['outdir'], 'final_model.pth'))

    # In[ ]:

    logsdir = "./comments"

    ##Save comments
    with open(os.path.join(logsdir, 'train_resume.txt'), 'w') as text_file:
        for comment in train_resume:
            text_file.write(comment)
            text_file.write('\n')

    # In[ ]:

    # Save input arguments
    with open(os.path.join(logsdir, 'inputs.json'), 'wt') as open_file:
        json.dump(args, open_file, indent=2)

    # In[ ]:

    # Get the training epochs
    train_epochs = list(range(1, args['nb_epochs'] + 1))

    # Get the validation epochs
    val_epochs = train_epochs[args['val_interval'] - 1::args['val_interval']]

    traindir = "./training_loss"

    # Plot the batch training loss
    fig, ax = plt.subplots(figsize=(10, 4));
    ax.plot(train_epochs, train_mean_imlosses, label='training')
    ax.plot(val_epochs, valid_mean_imlosses, label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    ax.legend(loc='upper left')
    fig.suptitle('Mean image training loss');
    ax.set_xlim(1, args['nb_epochs'])
    ax.set_ylim(0, 1)
    fig.savefig(os.path.join(traindir, 'im_train_loss.png'))

    # ## Test

    # In[ ]:

    # Select a model
    trained_model = monai.networks.nets.UNet(
        dimensions=dimension,
        in_channels=args['nb_channels'],
        out_channels=nb_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    trained_model.load_state_dict(torch.load(os.path.join(args['outdir'],
                                                          'final_model.pth')));

    # In[ ]:

    # Test image transforms
    test_imtrans = Compose([LoadImage(image_only=True),
                            my_resample(pixdim=args['output_dim'], mode='bilinear'),
                            ScaleIntensity(),
                            AddChannel(),
                            ToTensor()])

    # Test segmentation transforms
    test_segtrans = Compose([LoadImage(image_only=True),
                             my_resample(pixdim=args['output_dim'], mode='nn'),
                             seg2Dtoseg3D(labels=args['labels']),
                             ToTensor()])

    # In[ ]:

    # create the test data loader
    test_ds = ArrayDataset(test_data_dict['image'], test_imtrans,
                           test_data_dict['label'], test_segtrans)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True,
                             num_workers=2,
                             pin_memory=torch.cuda.is_available())

    # In[ ]:

    # Test the model on unseen images
    tot_dices = []
    se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    testdir = "./test_unseen_images"

    trained_model.eval()
    with torch.no_grad():
        for cnt_test, test_data in enumerate(test_loader):

            # Load each couple of image/manual segmentation
            im_torch = test_data[0].to(device)
            gt_torch = test_data[1].to(device)

            # Apply the model to the unknown image
            pred_torch = trained_model(im_torch)

            # Hard automatic segmentation
            seg_arr = pred_torch[0, :, :, :].detach().cpu()
            seg_arr = np.argmax(seg_arr.numpy(), axis=0)
            seg_arr += 1

            # Hard manual segmentation
            gt_arr = gt_torch[0, :, :, :].detach().cpu()
            gt_arr = np.argmax(gt_arr.numpy(), axis=0)
            gt_arr += 1

            # Predicted labels
            pred_labs = np.unique(seg_arr)
            pred_labs = pred_labs[pred_labs > 0]

            # Display the results
            im_arr = im_torch[0, 0, :, :].detach().cpu()

            fig, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(seg_arr, cmap='gray')
            ax.axis('off')
            ax.set_title('Predicted segmentation', fontsize=30)
            fig.savefig(os.path.join(testdir, 'test{0}_pred.png'.format(cnt_test)))

            fig, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(im_arr, cmap='gray')
            for label, color in zip(pred_labs, ['r', 'b', 'k']):
                ax.contour(seg_arr == label, [0.5], colors=color, linewidths=[2, 2])
            ax.axis('off')
            ax.set_title('EM image and contours of the predicted labels',
                         fontsize=30)
            fig.savefig(os.path.join(testdir, 'test{0}_im+pred.png'.format(cnt_test)))

            fig, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(gt_arr, cmap='gray')
            ax.axis('off')
            ax.set_title('Manual segmentation', fontsize=30)
            fig.savefig(os.path.join(testdir, 'test{0}_gt.png'.format(cnt_test)))

