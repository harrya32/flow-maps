import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train MFM Lightning")

    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to config file"
    )

    ####### ITERATES IN THE CODE #######
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds to iterate over",
    )
    parser.add_argument(
        "--t_exclude",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Time points to exclude (iterating over)",
    )
    ####################################

    parser.add_argument(
        "--working_dir",
        type=str,
        default="./",
        help="Working directory",
    )
    parser.add_argument(
        "--resume_flow_model_ckpt",
        type=str,
        default=None,
        help="Path to the flow model to resume training",
    )
    parser.add_argument(
        "--load_geopath_model_ckpt",
        type=str,
        default=None,
        help="Path to the geopath model to resume training",
    )

    ######### DATASETS #################
    parser = datasets_parser(parser)
    ####################################

    ######### IMAGE DATASETS ###########
    parser = image_datasets_parser(parser)
    ####################################

    ######### METRICS ##################
    parser = metric_parser(parser)
    ####################################

    ######### General Training #########
    parser = general_training_parser(parser)
    ####################################

    ######### Training GeoPath Network ####
    parser = geopath_network_parser(parser)
    ####################################

    ######### Training Flow Network ####
    parser = flow_network_parser(parser)
    ####################################

    return parser.parse_args()


def datasets_parser(parser):
    parser.add_argument("--dim", type=int, default=5, help="Dimension of data")

    parser.add_argument(
        "--data_type",
        type=str,
        default="scrna",
        help="Type of data, now wither scrna or one of toys",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="cite",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--whiten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whiten the data",
    )
    parser.add_argument("--maizels_dataset_path", type=str, default="")
    parser.add_argument("--maizels_classifier_path", type=str, default="")
    parser.add_argument(
        "--maizels_pair_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "ot_plain",
            "endpoint_interpolant",
            "ot_endpoint_interpolant",
        ],
    )
    parser.add_argument("--maizels_n_pairs", type=int, default=500_000)
    parser.add_argument("--maizels_validation_pairs", type=int, default=16_384)
    parser.add_argument("--maizels_holdout_fraction", type=float, default=0.2)
    parser.add_argument("--maizels_holdout_seed", type=int, default=701)
    parser.add_argument(
        "--maizels_lineage_transition_mode",
        type=str,
        default="descendant",
        choices=["descendant", "direct"],
    )
    parser.add_argument("--maizels_interpolant_check_times", type=int, default=50)
    parser.add_argument("--maizels_classifier_prob_threshold", type=float, default=0.0)
    parser.add_argument(
        "--maizels_classifier_margin_threshold", type=float, default=0.0
    )
    parser.add_argument("--maizels_classifier_batch_size", type=int, default=8192)
    parser.add_argument("--maizels_geopath_filter_chunk_size", type=int, default=2048)
    parser.add_argument("--maizels_rejection_chunk_size", type=int, default=50_000)
    parser.add_argument(
        "--maizels_rejection_max_candidates", type=int, default=5_000_000
    )
    parser.add_argument("--maizels_ot_candidate_chunk_size", type=int, default=50_000)
    parser.add_argument(
        "--maizels_ot_infeasible_fallback",
        type=str,
        default="partial",
        choices=["error", "partial"],
    )
    parser.add_argument("--maizels_ot_cache_dir", type=str, default="")
    parser.add_argument(
        "--maizels_ot_progress_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--maizels_ot_verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--maizels_eval_points_per_time", type=int, default=1024)
    parser.add_argument("--maizels_eval_euler_steps", type=int, default=50)
    parser.add_argument("--maizels_eval_wasserstein_projections", type=int, default=256)
    parser.add_argument("--maizels_eval_every_n_steps", type=int, default=500)
    return parser


def image_datasets_parser(parser):
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Size of the image",
    )
    parser.add_argument(
        "--x0_label",
        type=str,
        default="dog",
        help="Label for x0",
    )
    parser.add_argument(
        "--x1_label",
        type=str,
        default="cat",
        help="Label for x1",
    )
    return parser


def metric_parser(parser):
    parser.add_argument(
        "--mfm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If geopathing (True for MFM, False for CFM)",
    )
    parser.add_argument(
        "--n_centers",
        type=int,
        default=100,
        help="Number of centers for RBF network",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help="Kappa parameter for RBF network",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.001,
        help="Rho parameter in Riemanian Velocity Calculation",
    )
    parser.add_argument(
        "--velocity_metric",
        type=str,
        default="rbf",
        help="Metric for velocity calculation",
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.2, 0.2],
        help="Gamma parameter in Riemanian Velocity Calculation",
    )
    parser.add_argument(
        "--metric_epochs",
        type=int,
        default=50,
        help="Number of epochs for metric learning",
    )
    parser.add_argument(
        "--metric_train_batches",
        type=int,
        default=0,
        help="Cap metric-fitting train batches per epoch; 0 uses the full loader.",
    )
    parser.add_argument(
        "--metric_val_batches",
        type=int,
        default=0,
        help="Cap metric-fitting validation batches; 0 uses the full loader.",
    )
    parser.add_argument(
        "--metric_patience",
        type=int,
        default=5,
        help="Patience for metric learning",
    )
    parser.add_argument(
        "--metric_lr",
        type=float,
        default=1e-2,
        help="Learning rate for metric learning",
    )
    parser.add_argument(
        "--alpha_metric",
        type=float,
        default=1.0,
        help="Alpha parameter for metric learning",
    )
    parser.add_argument(
        "--image_size_metric",
        type=int,
        default=64,
        help="Size of the image for metric learning",
    )

    return parser


def general_training_parser(parser):
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for CFM training"
    )
    parser.add_argument(
        "--optimal_transport_method",
        type=str,
        default="exact",
        help="Use optimal transport in CFM training",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=None,
        help="Decay for EMA",
    )
    parser.add_argument(
        "--split_ratios",
        nargs=2,
        type=float,
        default=[0.9, 0.1],
        help="Split ratios for training/validation data in CFM training",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum optimizer steps per geopath/flow phase; -1 uses epochs.",
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=500,
        help="Validation interval in optimizer steps when max_steps is used.",
    )
    parser.add_argument(
        "--reference_loss_batches",
        type=int,
        default=0,
        help="Cap geopath reference-loss batches; 0 uses the full loader.",
    )
    parser.add_argument(
        "--accelerator", type=str, default="cpu", help="Training accelerator"
    )
    parser.add_argument(
        "--geopath_accelerator",
        type=str,
        default="auto",
        help="Geopath/metric accelerator. Auto uses CPU when Apple MPS is detected.",
    )
    parser.add_argument(
        "--flow_accelerator",
        type=str,
        default="auto",
        help="Flow-phase accelerator. Auto inherits --accelerator.",
    )
    parser.add_argument(
        "--sim_num_steps",
        type=int,
        default=1000,
        help="Number of steps in simulation",
    )
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="")
    return parser


def geopath_network_parser(parser):
    parser.add_argument(
        "--patience_geopath",
        type=int,
        default=5,
        help="Patience for training geopath model",
    )
    parser.add_argument(
        "--hidden_dims_geopath",
        nargs="+",
        type=int,
        default=[64, 64, 64],
        help="Dimensions of hidden layers for GeoPath model training",
    )
    parser.add_argument(
        "--time_geopath",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use time in GeoPath model",
    )
    parser.add_argument(
        "--activation_geopath",
        type=str,
        default="selu",
        help="Activation function for GeoPath",
    )
    parser.add_argument(
        "--geopath_optimizer",
        type=str,
        default="adam",
        help="Optimizer for GeoPath training",
    )
    parser.add_argument(
        "--geopath_lr",
        type=float,
        default=1e-4,
        help="Learning rate for GeoPath training",
    )
    parser.add_argument(
        "--geopath_weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for GeoPath training",
    )
    parser.add_argument(
        "--unet_num_channels_geopath",
        type=int,
        default=64,
        help="Number of channels for UNet",
    )
    parser.add_argument(
        "--unet_num_res_blocks_geopath",
        type=int,
        default=2,
        help="Number of res blocks for UNet",
    )
    parser.add_argument(
        "--unet_channel_mult_geopath",
        nargs="+",
        type=int,
        default=[1, 2, 2],
        help="Channel multiplier for UNet",
    )
    parser.add_argument(
        "--unet_dropout_geopath",
        type=float,
        default=0.0,
        help="Dropout for UNet",
    )
    return parser


def flow_network_parser(parser):
    parser.add_argument(
        "--sigma", type=float, default=0.1, help="Sigma parameter for CFM (variance)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopping in CFM training",
    )
    parser.add_argument(
        "--hidden_dims_flow",
        nargs="+",
        type=int,
        default=[64, 64, 64],
        help="Dimensions of hidden layers for CFM training",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=10,
        help="Check validation every N epochs during CFM training",
    )
    parser.add_argument(
        "--activation_flow",
        type=str,
        default="selu",
        help="Activation function for CFM",
    )
    parser.add_argument(
        "--flow_optimizer",
        type=str,
        default="adamw",
        help="Optimizer for GeoPath training",
    )
    parser.add_argument(
        "--flow_lr",
        type=float,
        default=1e-3,
        help="Learning rate for GeoPath training",
    )
    parser.add_argument(
        "--flow_weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for GeoPath training",
    )
    parser.add_argument(
        "--unet_num_channels",
        type=int,
        default=128,
        help="Number of channels for UNet",
    )
    parser.add_argument(
        "--unet_num_res_blocks",
        type=int,
        default=4,
        help="Number of res blocks for UNet",
    )
    parser.add_argument(
        "--unet_channel_mult",
        nargs="+",
        type=int,
        default=[2, 2, 2],
        help="Channel multiplier for UNet",
    )
    parser.add_argument(
        "--unet_dropout",
        type=float,
        default=0.1,
        help="Dropout for UNet",
    )
    parser.add_argument(
        "--unet_resblock_updown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use resblock updown in UNet",
    )
    parser.add_argument(
        "--unet_use_new_attention_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use new attention order in UNet",
    )
    parser.add_argument(
        "--unet_attention_resolutions",
        type=str,
        default="16",
        help="Resolutions for attention in UNet",
    )
    parser.add_argument(
        "--unet_num_heads",
        type=int,
        default=1,
        help="Number of heads for attention in UNet",
    )
    return parser
