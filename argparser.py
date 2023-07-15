import argparse
import json

import tasks


def modify_command_options(opts):
    if opts.dataset == "monusac":
        opts.num_classes = 5
    if opts.dataset == "consep":
        opts.num_classes = 4

    if opts.dataset in ["consep", "monusac"]:
        # opts.debug = True
        opts.crop_val = False
        opts.optim = "Adam"
        opts.batch_size = 8
        opts.crop_size = 320
        opts.lr = 1e-4
        opts.weight_decay = 1e-8
        opts.epochs = 100
        opts.val_interval = 1

    if opts.step == 0:
        opts.unce = True
    else:
        opts.puce = True
        opts.loss_pcon = 0.1
        opts.loss_prd = 0.1
        opts.hard = True

    opts.feat_dim = 2208

    if not opts.visualize:
        opts.sample_num = 0

    opts.no_overlap = not opts.overlap
    opts.no_cross_val = not opts.cross_val

    return opts


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--random_seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="number of workers (default: 1)"
    )

    # Dataset Options
    parser.add_argument(
        "--data_root", type=str, default="./data", help="path to Dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["monusac", "consep"],
        help="Name of dataset",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="num classes (default: None), set by method modify_command_options()",
    )

    # Train Options
    parser.add_argument(
        "--epochs", type=int, default=30, help="epoch number (default: 30)"
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size (default: 8)"
    )
    parser.add_argument(
        "--crop_size", type=int, default=512, help="crop size (default: 513)"
    )

    parser.add_argument(
        "--lr", type=float, default=0.007, help="learning rate (default: 0.007)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum for SGD (default: 0.9)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay (default: 1e-4)"
    )

    parser.add_argument(
        "--lr_policy",
        type=str,
        default="poly",
        choices=["poly", "step"],
        help="lr schedule policy (default: poly)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=5000,
        help="decay step for stepLR (default: 5000)",
    )
    parser.add_argument(
        "--lr_decay_factor",
        type=float,
        default=0.1,
        help="decay factor for stepLR (default: 0.1)",
    )
    parser.add_argument(
        "--lr_power", type=float, default=0.9, help="power for polyLR (default: 0.9)"
    )

    parser.add_argument(
        "--feat_dim",
        type=int,
        default=2048,
        help="Dimensionality of the features space (default: 2048 as in Resnet-101)",
    )

    # Validation Options
    parser.add_argument(
        "--val_on_trainset",
        action="store_true",
        default=False,
        help="enable validation on train set (default: False)",
    )
    parser.add_argument(
        "--cross_val",
        action="store_true",
        default=False,
        help="If validate on training or on validation (default: Train)",
    )
    parser.add_argument(
        "--crop_val",
        action="store_false",
        default=True,
        help="do crop for validation (default: True)",
    )

    # Logging Options
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="path to Log directory (default: ./logs)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="name of the experiment - to append to log directory (default: Experiment)",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=0,
        help="number of samples for visualization (default: 0)",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="verbose option"
    )
    parser.add_argument(
        "--visualize",
        action="store_false",
        default=True,
        help="visualization on tensorboard (def: Yes)",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=10,
        help="print interval of loss (default: 10)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="epoch interval for eval (default: 15)",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=1,
        help="epoch interval for saving model (default: 1)",
    )
    parser.add_argument("--visdir", type=str, default=None)

    # Test and Checkpoint options
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Whether to train or test only (def: train and test)",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        type=str,
        help="path to trained model. Leave it None if you want to retrain your model",
    )

    parser.add_argument(
        "--loss_de_prototypes",
        type=float,
        default=0.0,  # Distillation on Encoder
        help="Set this hyperparameter to a value greater than "
        "0 to enable loss_de with prototypes (idea 1b)",
    )

    # Incremental parameters
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=tasks.get_task_list(),
        help="Task to be executed (default: 19-1)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="The incremental step in execution (default: 0)",
    )
    # Consider the dataset as done in
    # http://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Michieli_Incremental_Learning_Techniques_for_Semantic_Segmentation_ICCVW_2019_paper.pdf
    # and https://arxiv.org/pdf/1911.03462.pdf : same as disjoint scenario (default) but with label of old classes in
    # new images, if present.
    parser.add_argument(
        "--no_mask",
        action="store_true",
        default=False,
        help="Use this to not mask the old classes in new training set, i.e. use labels of old classes"
        " in new training set (if present)",
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        default=False,
        help="Use this to not use the new classes in the old training set",
    )
    parser.add_argument(
        "--step_ckpt",
        default=None,
        type=str,
        help="path to trained model at previous step. Leave it None if you want to use def path",
    )
    parser.add_argument(
        "--opt_level", type=str, choices=["O0", "O1", "O2", "O3"], default="O0"
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--optim", type=str, default="Adam")

    # CoNuSeg
    parser.add_argument(
        "--unce",
        default=False,
        action="store_true",
        help="Enable Unbiased Cross Entropy instead of CrossEntropy",
    )
    parser.add_argument(
        "--loss_prd",
        type=float,
        default=0.0,
        help="weight for Prototype-wise Relation Distillation Loss.",
    )
    parser.add_argument(
        "--loss_pcon",
        type=float,
        default=0.0,
        help="weight for Prototype-wise Contrastive Loss.",
    )
    parser.add_argument(
        "--puce",
        default=False,
        action="store_true",
        help="Use confidence and cosine similarity to filter pseudo label, then perform unbiased cross entropy loss.",
    )
    parser.add_argument(
        "--hard",
        default=False,
        action="store_true",
        help="Hard sampling for prototype construction.",
    )
    parser.add_argument("--base_threshold", default=0.75, type=float)

    return parser
