#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Xiaoling Hu
# Created Date: Tue June 22 9:00:00 PDT 2021
# =============================================================================

import time
import numpy
import gudhi as gd
from pylab import *
import torch
import os

t0 = time.time();

def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0.03, pers_thresh_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighouboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thresh_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process

    """
    ##Add
    # print("gt_dgm:",gt_dgm.shape)
    # print("lh_dgm:",lh_dgm.shape)
    if (lh_dgm.shape[0]==0):
        # shape_gt=gt_dgm.shape
        # lh_dgm=np.zeros(shape_gt,dtype=float)
        # lh_pers = np.array([[]])
        # print(lh_dgm)
        lh_dgm=np.array([[0.5, 0.5]])
        # lh_dgm =torch.Tensor([[0.5, 0.5]])###
        lh_pers=abs(lh_dgm[:, 1] - lh_dgm[:, 0])
        # print("lh_dgm0:",lh_dgm.shape)
        # print("abs:",lh_pers)
    else:
        lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
        # print("abs:", lh_pers)

    ###
    # lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])

    if (gt_dgm.shape[0] == 0):
        gt_pers = None;
        gt_n_holes = 0;
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size  # number of holes in gt

    if (gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = list();
        idx_holes_to_remove = list(set(range(lh_pers.size)))
        idx_holes_perfect = list();
    else:
        # check to ensure that all gt dots have persistence 1
        # tmp = gt_pers > pers_thresh_perfect

        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1; it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values

        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
        # print("tmp",tmp)
        # lh_pers=np.asarray(lh_pers)###
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        # lh_pers=torch.Tensor(lh_pers_sorted_indices)###
        # print(lh_pers)
        # lh_pers_sorted_indices = torch.Tensor.argsort(lh_pers)[::-1]###
        # print(tmp)

        if np.sum(tmp) >= 1:
            lh_n_holes_perfect = tmp.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect];
        else:
            idx_holes_perfect = list();

        # find top gt_n_holes indices
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes];

        # the difference is holes to be fixed to perfect
        map_idx_holes_to_fix_or_perfect=map(int,np.array(idx_holes_to_fix_or_perfect))
        map_idx_holes_perfect=map(int,np.array(idx_holes_perfect))
        # print("idx_holes_to_fix_or_perfect:",len(idx_holes_to_fix_or_perfect),set(list(map_idx_holes_to_fix_or_perfect)),"idx_holes_perfect:",len(idx_holes_perfect),set(list(map_idx_holes_perfect)))

        if lh_dgm.shape[0]==0:
            idx_holes_to_fix=[0]
        else:

            idx_holes_to_fix = list(
                # set(list(map(int,np.array(idx_holes_to_fix_or_perfect)))) - set(list(map(int,np.array(idx_holes_perfect)))))
                set(map(int, idx_holes_to_fix_or_perfect)) - set(map(int, idx_holes_perfect)))
            # print(len(idx_holes_to_fix))
            # print(idx_holes_to_fix)

        # idx_holes_to_fix = list(
        #     set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        # remaining holes are all to be removed
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:];

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    pers_thd = pers_thresh
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)


    ##Add
    shape_lh=lh_dgm.shape
    if (lh_dgm.shape[0]==0):
        # shape_gt=gt_dgm.shape
        lh_dgm=list([[0,0]])
        # print(idx_holes_to_fix)
        force_list=list([0,0])
    else:
        force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
        force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]
        # print(lh_dgm,idx_holes_to_fix,lh_dgm[idx_holes_to_fix, 0],lh_dgm[idx_holes_to_fix, 1])
        # print(force_list)
    #####

    # push each hole-to-fix to (0,1)
    # force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    # force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    if shape_lh[0]==0:
        force_list=list([0,0])
    else:
        force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                             math.sqrt(2.0)
        force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                             math.sqrt(2.0)
        # print(force_list)

    if (do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove


def getCriticalPoints(likelihood):
    """
    Compute the critical points of the image (Value range from 0 -> 1)

    Args:
        likelihood: Likelihood image from the output of the neural networks

    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.

    """
    lh = 1 - likelihood
    lh_vector = np.asarray(lh).flatten()
    # lh_vector=torch.Tensor(lh_vector)###
    lh_shape=lh.shape
    # print(lh_shape,lh_vector.shape)

    if len(lh_shape)==3:
        nx = lh_shape[0]*lh_shape[2]/2
        ny = lh_shape[1]*lh_shape[2]/2
    else:
        nx=lh_shape[0]
        ny=lh_shape[1]

    lh_cubic = gd.CubicalComplex(
        dimensions=[nx, ny],
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()
    # print("pairs_lh shape:",len(pairs_lh),pairs_lh)
    # if pairs_lh[0]==[]:
        # print("hhhhhh")
        # new_pairs_lh=[[]]
        # print(pairs_lh)
        # for i in range(len(pairs_lh[0][1])):
        #     new_pairs_lh[i][0]=pairs_lh[0][1][i][0]
        #     new_pairs_lh[i][1]=pairs_lh[0][1][i][1]
        # print("new_pairs_lh:",new_pairs_lh)
        # pairs_lh=new_pairs_lh
        # return 0, 0, 0, False

    # If the paris is 0, return False to skip
    # print("//////////////////////")
    # print("len pairs:",pairs_lh[0],"len:",len(pairs_lh))
    if (len(pairs_lh[0]) == 0): return 0, 0, 0, False
    # print("pairs_lh:",pairs_lh,"lh.shape[1]:",lh.shape[1])
    # return persistence diagram, birth/death critical points
    pd_lh = numpy.array([[lh_vector[pairs_lh[0][0][i][0]], lh_vector[pairs_lh[0][0][i][1]]] for i in range(len(pairs_lh[0][0]))])
    # pd_lh = torch.Tensor([[lh_vector[pairs_lh[0][0][i][0]], lh_vector[pairs_lh[0][0][i][1]]] for i in range(len(pairs_lh[0][0]))])###

    # print("pd_lh::",pd_lh)
    # for i in range(len(pairs_lh[0][0])):
    #     print("lh_vector[pairs_lh[0][0][i][0]]=",lh_vector[pairs_lh[0][0][i][0]])
    #     print("lh_vector[pairs_lh[0][0][i][1]]=",lh_vector[pairs_lh[0][0][i][1]])
    #     print("pairs_lh[0][0][i][0] % lh.shape[1]=",pairs_lh[0][0][i][0] % lh.shape[1])
    #     print("pairs_lh[0][0][i][0] // lh.shape[1]=",pairs_lh[0][0][i][0] // lh.shape[1])

    # print(numpy.array(
    #     [[pairs_lh[0][0][i][0] // lh.shape[1], pairs_lh[0][0][i][0] % lh.shape[1]] for i in range(len(pairs_lh[0][0]))]))

    bcp_lh = numpy.array([[pairs_lh[0][0][i][0] // lh.shape[1], pairs_lh[0][0][i][0] % lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
    # bcp_lh = torch.Tensor([[pairs_lh[0][0][i][0] // lh.shape[1], pairs_lh[0][0][i][0] % lh.shape[1]] for i in range(len(pairs_lh[0][0]))])###

    # bcp_lh=[[]]
    # for i in range(len(pairs_lh[0][0])):
    #     print("pairs_lh[0][0][i][0] // lh.shape[1]:",pairs_lh[0][0][i][0] // lh.shape[1])
    #     print("pairs_lh[0][0][i][0] % lh.shape[1]:",pairs_lh[0][0][i][0] % lh.shape[1])
    #     bcp_lh[i][0] = numpy.array([pairs_lh[0][0][i][0] // lh.shape[1]])
    #     bcp_lh[i][1] = numpy.array([pairs_lh[0][0][i][0] % lh.shape[1]])


    dcp_lh = numpy.array([[pairs_lh[0][0][i][1] // lh.shape[1], pairs_lh[0][0][i][1] % lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
    # dcp_lh = torch.Tensor([[pairs_lh[0][0][i][1] // lh.shape[1], pairs_lh[0][0][i][1] % lh.shape[1]] for i in range(len(pairs_lh[0][0]))])###

    # print(pd_lh, bcp_lh, dcp_lh)
    # print(bcp_lh)
    # print("//////////")
    # if (pd_lh== []): return 0, 0, 0, False
    # print("pd_lh, bcp_lh, dcp_lh",pd_lh, bcp_lh, dcp_lh)
    return pd_lh, bcp_lh, dcp_lh, True


def getTopoLoss(likelihood_tensor, gt_tensor, topo_size=100):
    """
    Calculate the topology loss of the predicted image and ground truth image
    Warning: To make sure the topology loss is able to back-propagation, likelihood
    tensor requires to clone before detach from GPUs. In the end, you can hook the
    likelihood tensor to GPUs device.

    Args:
        likelihood_tensor:   The likelihood pytorch tensor.
        gt_tensor        :   The groundtruth of pytorch tensor.
        topo_size        :   The size of the patch is used. Default: 100

    Returns:
        loss_topo        :   The topology loss value (tensor)

    """

    likelihood = torch.sigmoid(likelihood_tensor).clone()
    gt = gt_tensor.clone()

    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()

    topo_cp_weight_map = np.zeros(likelihood.shape)
    # topo_cp_weight_map=torch.Tensor(topo_cp_weight_map)###
    topo_cp_ref_map = np.zeros(likelihood.shape)
    # topo_cp_ref_map=torch.Tensor(topo_cp_ref_map)###

    for y in range(0, likelihood.shape[0], topo_size):
        for x in range(0, likelihood.shape[1], topo_size):

            lh_patch = likelihood[y:min(y + topo_size, likelihood.shape[0]),
                       x:min(x + topo_size, likelihood.shape[1])]
            gt_patch = gt[y:min(y + topo_size, gt.shape[0]),
                       x:min(x + topo_size, gt.shape[1])]

            if (np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if (np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue

            # Get the critical points of predictions and ground truth

            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints(lh_patch)
            # print("\\\\\\\\\\\\\\\\\\\\\\\\")
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints(gt_patch)

            # If the pairs not exist, continue for the next loop

            # print("pairs_lh_pa:",pd_lh, bcp_lh, dcp_lh,pairs_lh_pa)
            if not (pairs_lh_pa): continue
            if not (pairs_lh_gt): continue

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thresh=0.03)
            # force_list=torch.Tensor(force_list)
            # idx_holes_to_fix=torch.Tensor(idx_holes_to_fix)
            # idx_holes_to_remove=torch.Tensor(idx_holes_to_remove)

            if bcp_lh!=[]:
                if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):

                    for hole_indx in idx_holes_to_fix:

                        # print("hole index:",hole_indx,"???","idx_holes_to_fix:", len(idx_holes_to_fix), "bcp_lh:", bcp_lh.shape)
                        # if bcp_lh==[]:
                        #     bcp_lh
                        # print("////////////////////////////////////")
                        # print("bcp_lh",bcp_lh)
                        # print(bcp_lh[hole_indx])
                        # print(bcp_lh[0])
                        # print(hole_indx)
                        # print("likelihood.shape[0]",likelihood.shape[0],int(bcp_lh[hole_indx][0]) >= 0)

                        if (int(bcp_lh[hole_indx][0]) >= 0 and
                                int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                                bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                            topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                                bcp_lh[hole_indx][1])] = 1  # push birth to 0 i.e. min birth prob or likelihood
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 0
                        # print("/////////")
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                                dcp_lh[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                    for hole_indx in idx_holes_to_remove:
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                                bcp_lh[hole_indx][1])] = 1  # push birth to death  # push to diagonal
                            if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                                0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                    likelihood.shape[1]):
                                topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                    likelihood[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                            else:
                                topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                                dcp_lh[hole_indx][1])] = 1  # push death to birth # push to diagonal
                            if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                                0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                    likelihood.shape[1]):
                                topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                                    likelihood[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                            else:
                                topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0
                # print("topo_cp_ref_map:",topo_cp_ref_map)
    # topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    # topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float, requires_grad=True).cpu()
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float, requires_grad=True).cpu()
    # topo_cp_weight_map = torch.Tensor(topo_cp_weight_map)
    # topo_cp_ref_map = torch.Tensor(topo_cp_ref_map)

    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((likelihood_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()
    # loss_topo=torch.Tensor(loss_topo)
    print("Topo torch::",loss_topo)
    return loss_topo/10000

if __name__ == "__main__":

    train_lh_list=os.listdir("./data/TRAIN")
    train_gt_list=os.listdir("./data/SEGMENT_TRAIN")
    # print("train_gt_list:",train_gt_list)
    # print("train_lh_list:",train_lh_list)
    train_lh_list_new=[]
    for i in train_lh_list:
        if ".tif" in i:
            continue
        else:
            train_lh_list_new.append(i)

    topoloss_result=[]
    # print(train_lh_list_new)

    for i in range(0,len(train_gt_list)):
        gt = 1 - imread("./data/SEGMENT_TRAIN/"+train_gt_list[i])
        lh = 1 - imread("./data/TRAIN/"+train_lh_list_new[i])

        # print(train_gt_list[i])
        # print(train_lh_list_new[i])
        gt=torch.from_numpy(gt)
        lh=torch.from_numpy(lh)

        print("shape:",gt.shape,lh.shape)
        print(gt)

        loss_topo = getTopoLoss(lh, gt)
        loss_topo=loss_topo.detach().numpy()
        topoloss_result.append(loss_topo)
        print("loss_topo{0}:".format(i),loss_topo)

    print("topoloss_result:",topoloss_result)
    total=0
    for i in topoloss_result:
        total=total+i
    print("average of topoloss:",total/len(topoloss_result))


    # gt = 1 - imread('test0_gt.png')
    # lh = 1 - imread('test0_pred.png')
    #
    # gt=torch.from_numpy(gt)
    # lh=torch.from_numpy(lh)
    #
    # loss_topo = getTopoLoss(lh, gt)
    # print(loss_topo)
    # print('time %.3f' % (time.time() - t0))