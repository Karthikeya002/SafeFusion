#!/usr/bin/env python3
"""Demo script for SafeFusion."""

import argparse
from src.safefusion import SafeFusion

def main():
    parser = argparse.ArgumentParser(description='SafeFusion Demo')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--output', type=str, default='output.mp4')
    args = parser.parse_args()
    
    model = SafeFusion(args.config)
    model.process_video(args.video, args.output)
    print(f'Output saved to {args.output}')

if __name__ == '__main__':
    main()
