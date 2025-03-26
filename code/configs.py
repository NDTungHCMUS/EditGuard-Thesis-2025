# ----- VN Start -----
import argparse
import options.options as option

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--ckpt', type=str, default='/userhome/NewIBSN/EditGuard_open/checkpoints/clean.pth',
                        help='Path to pre-trained model.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    opt = option.parse(args.opt, is_train=True)
    
    if args.launcher == 'none':
        opt['dist'] = False
    else:
        opt['dist'] = True
    
    return opt, args

config, args = get_config()

# ----- VN End -----