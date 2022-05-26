from logging import debug
import os
import time
import argparse
import json
import random
import math

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data, prepare_train_dataset, prepare_train_dataloader

import torch    
import torch.nn.functional as F
import numpy as np

import etent
import tent


import models.Res as Resnet



def validate(val_loader, model, criterion, args, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    # if mode == 'eval':
    #     model.eval()
    # elif mode == 'train':
    #     model.train()
    # else:
    #     assert False, "not support"

    # print(model.model.training)
    # print(model.training)

    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            output = model(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0: # args.print_freq
                progress.display(i)
                # print(model.num_samples_update_2)
                # print(model.num_samples_update_1)
    return top1.avg, top5.avg


def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # path
    parser.add_argument('--data', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_corruption', default='/dockerdata/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--output', default='/apdcephfs/private_huberyniu/etta_exps/rebuttal', help='the output directory of this experiment')

    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')

    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    parser.add_argument('--rotation', default=False, type=bool, help='if use the rotation ssl task for training (this is TTTs dataloader).')

    parser.add_argument('--fisher_clip_by_norm', type=float, default=10.0, help='Clip fisher before it is too large')

    # corruption dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')

    # Test time training hyper-parameters
    parser.add_argument('--arch', default='resnet50', type=str, help='the default architecture')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    subnet = Resnet.__dict__[args.arch](pretrained=False)
    if args.arch.endswith("50"):
        init = torch.load("/apdcephfs/private_huberyniu/cli_pretrained_models/resnet50-19c8e357.pth")
    elif args.arch.endswith("101"):
        init = torch.load("/apdcephfs/private_huberyniu/cli_pretrained_models/resnet101-5d3b4d8f.pth")
    elif args.arch.endswith("152"):
        init = torch.load("/apdcephfs/private_huberyniu/cli_pretrained_models/resnet152-b121ed2d.pth")
    else:
        assert False, NotImplementedError

    subnet.load_state_dict(init)
    subnet = subnet.cuda()

    if not os.path.exists(args.output): # and args.local_rank == 0
        os.makedirs(args.output, exist_ok=True)

    debug = False
    if debug:
        logger = get_logger(name="project", output_directory="/apdcephfs/private_huberyniu/etta_exps/rebuttal/debug", log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-log.txt", debug=False)
    else:
        logger = get_logger(name="project", output_directory=args.output, log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-small_testset.txt", debug=False)
    
    # common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    common_corruptions = ['gaussian_noise']
    logger.info(args)

    tent_acc1s = []
    eata_acc1s = []
    eata_num1s = []
    eata_num2s = []
    eta_acc1s = []
    eta_num1s = []
    eta_num2s = []
    for data_size in [256, 512, 1024, 2048, 4096, 10000]:
        logger.info(f"dataset size is {data_size}")
        for corrupt in common_corruptions:
            args.corruption = corrupt
            logger.info(args.corruption)

            val_dataset, val_loader = prepare_test_data(args)
            val_dataset.switch_mode(True, False)
            val_dataset.set_dataset_size(data_size)

            validate_ori = False
            if validate_ori:
                # validate the original accuracy
                top1, top5 = validate(val_loader, subnet, None, args, mode='eval')
                logger.info(f"With Max Architecture {args.corruption} Original Accuracy: top1: {top1:.5f} and top5: {top5:.5f}")

                assert False

            # tent evaluation
            subnet = Resnet.__dict__[args.arch](pretrained=False)
            subnet.load_state_dict(init)
            subnet = subnet.cuda()

            subnet = tent.configure_model(subnet)
            params, param_names = tent.collect_params(subnet)
            optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)
            tented_model = tent.Tent(subnet, optimizer)

            for i in range(1):
                top1, top5 = validate(val_loader, tented_model, None, args, mode='eval')
                logger.info(f"With Max Architecture {args.corruption} After Tent: Original Accuracy: top1: {top1:.5f} and top5: {top5:.5f}")
            tent_acc1s.append(top1.item())

            # ETA evaluation
            subnet = Resnet.__dict__[args.arch](pretrained=False)
            subnet.load_state_dict(init)
            subnet = subnet.cuda()
            
            import etent as mytent

            subnet = mytent.configure_model(subnet)
            params, param_names = mytent.collect_params(subnet)

            optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)
            tented_model = mytent.Tent(subnet, optimizer, e_margin=math.log(1000)*0.40)
            logger.info("tened model is: " + "mytent.Tent(subnet, optimizer, e_margin=math.log(1000)*0.40)")

            top1, top5 = validate(val_loader, tented_model, None, args, mode='eval')
            logger.info(f"With Max Architecture {args.corruption} After ETA: Original Accuracy: top1: {top1:.5f} and top5: {top5:.5f}")
            logger.info(f"num filters are {tented_model.num_samples_update_1}, {tented_model.num_samples_update_2}")

            eta_acc1s.append(top1.item())
            eta_num1s.append(tented_model.num_samples_update_1)
            eta_num2s.append(tented_model.num_samples_update_2)


            # EATA evaluation
            subnet = Resnet.__dict__[args.arch](pretrained=False)
            subnet.load_state_dict(init)
            subnet = subnet.cuda()

            fisher_type = 'avg' # avg or sum
            args.fisher_alpha = 2000.  # 2500.0
            args.fisher_size = 2000

            # compute fisher informatrix
            args.corruption = 'original'
            fisher_dataset, fisher_loader = prepare_test_data(args)
            fisher_dataset.set_dataset_size(args.fisher_size)
            fisher_dataset.switch_mode(True, False)

            import etent as mytent

            # print logs
            logger.info("fisher alpha is "+str(args.fisher_alpha))
            logger.info("fisher type is "+fisher_type)
            logger.info("fisher data are: "+args.corruption)
            logger.info("fisher size is: "+str(args.fisher_size))
            logger.info("corrpution type is "+args.corruption)

            subnet = mytent.configure_model(subnet)
            params, param_names = mytent.collect_params(subnet)
            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = nn.CrossEntropyLoss().cuda()
            for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    targets = targets.cuda(args.gpu, non_blocking=True)
                outputs = subnet(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in subnet.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if fisher_type == 'sum':
                            if fisher.norm() > args.fisher_clip_by_norm:
                                fisher = fisher / fisher.norm() * args.fisher_clip_by_norm
                        elif fisher_type == 'avg':
                            if iter_ == len(fisher_loader):
                                fisher = fisher / iter_
                        else:
                            assert False, NotImplementedError
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            logger.info("compute fisher matrices finished")
            del ewc_optimizer

            optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)
            tented_model = mytent.Tent(subnet, optimizer, fishers, args.fisher_alpha, e_margin=math.log(1000)*0.40)
            logger.info("e_margin is: " + "tented_model = mytent.Tent(subnet, optimizer, fishers, args.fisher_alpha, e_margin=math.log(1000)*0.40)")

            top1, top5 = validate(val_loader, tented_model, None, args, mode='eval')
            logger.info(f"With Max Architecture {args.corruption} After EATA: Original Accuracy: top1: {top1:.5f} and top5: {top5:.5f}")
            logger.info(f"num filters are {tented_model.num_samples_update_1}, {tented_model.num_samples_update_2}")

            eata_acc1s.append(top1.item())
            eata_num1s.append(tented_model.num_samples_update_1)
            eata_num2s.append(tented_model.num_samples_update_2)
        logger.info(f"tent_acc1s are {tent_acc1s}")
        logger.info(f"eata_acc1s are {eata_acc1s}")
        logger.info(f"eta_acc1s are {eta_acc1s}")
        logger.info(f"eata_num1s are {eata_num1s}")
        logger.info(f"eata_num2s are {eata_num2s}")
        logger.info(f"eta_num1s are {eta_num1s}")
        logger.info(f"eta_num2s are {eta_num2s}")