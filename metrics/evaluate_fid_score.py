import argparse
from cleanfid import fid
import os

def main(args):
    if str.lower(args.mode) == 'fid':
        score = fid.compute_fid(args.dir1, args.dir2)
        print(f'FID score: {score}')
    elif str.lower(args.mode) == 'kid':
        score = fid.compute_kid(args.dir1, args.dir2)
        print(f'KID score: {score}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FID/KID score between two directories.')
    parser.add_argument('--dir1', type=str, required=True, help='path/to/generated_images')
    parser.add_argument('--dir2', type=str, required=True, help='path/to/generated_images')
    parser.add_argument('--mode', type=str, default='KID', choices=['FID', 'KID', 'fid', 'kid'], help='Select mode between FID / KID')
    args = parser.parse_args()
    main(args)