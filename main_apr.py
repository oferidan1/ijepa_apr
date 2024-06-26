"""
Entry point training and testing TransPoseNet
"""
import sys, os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import argparse
import torch
import numpy as np
import json
import logging
from pose.util import utils
import time
from pose.datasets.CameraPoseDataset import CameraPoseDataset
from pose.datasets.MSCameraPoseDataset import MSCameraPoseDataset
from pose.pose_losses import CameraPoseLoss
from pose_regressor import PoseRegressor
from os.path import join


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()    
    arg_parser.add_argument("--mode", help="train or eval", default='train')
    arg_parser.add_argument("--backbone_path", help="path to backbone .pth", default="checkpoint/IN1K-vit.h.14-300e.pth.tar")
    arg_parser.add_argument("--dataset_path", help="path to the physical location of the dataset", default="/dsi/scratch/home/dsi/rinaVeler/datasets")
    arg_parser.add_argument("--labels_file", help="path to a file mapping images to their poses", default="pose/datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_train.csv")
    arg_parser.add_argument("--test_labels_file", help="path to a file mapping images to their poses", default="pose/datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_test.csv")
    arg_parser.add_argument("--checkpoint_path", help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")
    arg_parser.add_argument("--gpu", help="gpu id", default="0")
    arg_parser.add_argument("--patch_size", help="ijepa_patchsize", type= int, default= 14)

    args = arg_parser.parse_args()
    
    utils.init_logger()

    # Record execution details
    logging.info("Start PoseRegressor with {}".format(args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open('pose/config.json', "r") as read_file:
        config = json.load(read_file)    
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = 'cuda:' + args.gpu 
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model
    model = PoseRegressor(args.backbone_path, args.patch_size,config).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Freeze parts of the model if indicated
        freeze = config.get("freeze")
        freeze_exclude_phrase = config.get("freeze_exclude_phrase")
        if isinstance(freeze_exclude_phrase, str):
            freeze_exclude_phrase = [freeze_exclude_phrase]
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                    parameter.requires_grad_(False)

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)
        pose_loss_valid = CameraPoseLoss(config).to(device)

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        no_augment = config.get("no_augment")
        if no_augment:
            transform = utils.test_transforms.get('baseline')
        else:
            transform = utils.train_transforms.get('baseline')

        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)

        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        #validation loader
        transform_valid = utils.test_transforms.get('baseline')
        dataset_valid = CameraPoseDataset(args.dataset_path, args.test_labels_file, transform_valid)
        loader_params = {'batch_size': config.get('batch_size'),
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, **loader_params)


        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []        
        loss_vals_valid = []
        sample_count_valid = []        
        
        best_valid_loss = 1000000000
        
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device).to(dtype=torch.float32)
                gt_pose = minibatch.get('pose')
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                if freeze: 
                    model.eval()
                    with torch.no_grad():
                        backbone_res = model.forward_backbone(minibatch)
                    model.train()

                # Zero the gradients
                optim.zero_grad()

                # Forward pass to estimate the pose
                if freeze:
                    res = model.forward_heads(backbone_res)
                else:
                    res = model(minibatch)

                est_pose = res.get('pose')
                criterion = pose_loss(est_pose, gt_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0
            valid_loss = 0

            #validation
            with torch.no_grad():
                for batch_idx, minibatch in enumerate(dataloader_valid, 0):
                    for k, v in minibatch.items():
                        minibatch[k] = v.to(device).to(dtype=torch.float32)
                    gt_pose = minibatch.get('pose')
                    # Forward pass to predict the pose
                    est_pose = model(minibatch).get('pose')
                    # Evaluate error
                    criterion = pose_loss_valid(est_pose, gt_pose)
                    
                    valid_loss += criterion.item()                    
                    running_loss += criterion.item()                                        
                    n_samples += batch_size
                    loss_vals_valid.append(criterion.item())
                    sample_count_valid.append(n_total_samples)
                    
                     # Record loss and performance on train set
                    if batch_idx % n_freq_print == 0:
                        posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
                        logging.info("[Validatin Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                    "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                            batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                            posit_err.mean().item(),
                                                                            orient_err.mean().item()))
                    
                    if batch_idx > 100:
                        break
                    
            if valid_loss < best_valid_loss:
                logging.info("Saving Best checkpoint - Epoch-{}".format(epoch+1))
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint_best.pth')
                best_valid_loss = valid_loss
                best_epoch = epoch+1
                                

            # Scheduler update
            scheduler.step()

        logging.info('Training completed, saving final model. Best checkpoint at epoch: {}'.format(best_epoch))
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

        # Plot the loss function
        loss_fig_path = checkpoint_prefix + "_loss_fig.png"
        utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)
        loss_fig_path = checkpoint_prefix + "_valid_loss_fig.png"
        utils.plot_loss_func(sample_count_valid, loss_vals_valid, loss_fig_path)

    else: # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_path, args.test_labels_file, transform)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device).to(dtype=torch.float32)        

                gt_pose = minibatch.get('pose')

                # Forward pass to predict the pose
                tic = time.time()
                est_pose = model(minibatch).get('pose')
                toc = time.time()

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic)*1000

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.test_labels_file))
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
        logging.info(
            "Var pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanstd(stats[:, 0])**2, np.nanstd(stats[:, 1])**2))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))