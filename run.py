import subprocess
import argparse
import os

def get_args_parser():
    parser = argparse.ArgumentParser(description='quantization PTQ+QAT', add_help=False)

    #PTQ and QAT same arguments

    parser.add_argument('--device', default='cuda', help='cpu vs cuda')

    #data load
    #/mnt/d/data/image/ILSVRC/Data/CLS-LOC
    parser.add_argument('--data-set', default='CIFAR', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'])
    parser.add_argument('--data_path', default='/mnt/d/data/image', type=str, help='path to ImageNet data')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    #model
    parser.add_argument('--model', default='resnet18', type=str, help='model name',
                        choices=['resnet18.a1_in1k','resnet18'])
    parser.add_argument('--pretrained', default='/mnt/c/Users/tang3/OneDrive/바탕 화면/code/my_code/efficient_model/deit/output_dir_resnet18/best_checkpoint.pth', help='get pretrained weights from checkpoint')
    
    # quantization parameters
    parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')

    #save
    parser.add_argument('--output_dir', default='./output_dir', type=str, help='path where to save scale, empty for no saving')

    #onnx inference
    parser.add_argument('--onnx_inference', default=True, help='perform onnx inference')
    
    # PTQ specific arguments

    #evaluate
    parser.add_argument('--no_evaluate', action='store_true', help='whether evaluate quantized model')

    parser.add_argument('--calibrator', default='max', choices=['entropy', "max"], help='calibrator for activation quantization')
    parser.add_argument('--num_samples', default=100, type=int, help='size of the calibration dataset')
    parser.add_argument('--calib_batch_size', default=10, type=int, help='number of iterations for calibration')

    #QAT specific arguments

    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')


    parser.add_argument('--batch_size', default=32, type=int, help='batch size for training')

    parser.add_argument("--quantized_model", default="/mnt/c/Users/tang3/OneDrive/바탕 화면/code/my_code/quantization/output_dir/resnet18.a1_in1k/resnet18.a1_in1k_quant.pth")

    #optimizer
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    #run
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    #save and log
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument("--save_freq", default=1, type=int)

    #cuda 
    parser.add_argument("--num_cuda", default=1, type=int)

    return parser


def main(args):
    print("Run PTQ and QAT\n")
    print("\n" + "*" * 30)
    print("***** PTQ Start... *****")
    print("*" * 30 + "\n")
    ptq_command = [
        "python", "PTQ_main.py",
        "--device", args.device,
        "--data-set", args.data_set,
        "--data_path", args.data_path,
        "--eval-crop-ratio", str(args.eval_crop_ratio),
        "--input-size", str(args.input_size),
        "--model", args.model,
        "--pretrained", args.pretrained,
        "--n_bits_w", str(args.n_bits_w),
        "--n_bits_a", str(args.n_bits_a),
        "--calibrator", args.calibrator,
        "--num_samples", str(args.num_samples),
        "--calib_batch_size", str(args.calib_batch_size),
        "--output_dir", args.output_dir,
    ]

    if args.no_evaluate:
        ptq_command.append("--no_evaluate")

    ptq_result = subprocess.run(ptq_command, check=True)
    print("\nPTQ_main.py:", ptq_result.returncode)

    print("\n" + "*" * 30)
    print("***** QAT Start... *****")
    print("*" * 30 + "\n")

    quantized_model_path = os.path.join(args.output_dir, args.model, args.model + "_quant.pth")

    qat_command = [
    "torchrun", "--nnodes", str(1),  "--nproc_per_node", str(args.num_cuda), "QAT_main.py",
    "--dist_url", args.dist_url,
    "--local_rank", str(args.local_rank),
    "--dist-bn", args.dist_bn,
    "--device", args.device,
    "--data-set", args.data_set,
    "--data_path", args.data_path,
    "--eval-crop-ratio", str(args.eval_crop_ratio),
    "--input-size", str(args.input_size),
    "--batch_size", str(args.batch_size),
    "--model", args.model,
    "--pretrained", args.pretrained,
    "--quantized_model", quantized_model_path,  
    "--n_bits_w", str(args.n_bits_w),
    "--n_bits_a", str(args.n_bits_a),
    "--lr", str(args.lr),
    "--momentum", str(args.momentum),
    "--weight_decay", str(args.weight_decay),
    "--epochs", str(args.epochs),
    "--start_epoch", str(args.start_epoch),
    "--resume", args.resume,
    "--print_freq", str(args.print_freq),
    "--save_freq", str(args.save_freq),
    "--output_dir", args.output_dir,
    ]

    if args.distributed:
        qat_command.append("--distributed")

    qat_result = subprocess.run(qat_command, check=True)
    print("\nQAT_main.py:", qat_result.returncode)

if __name__=="__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
