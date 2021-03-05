from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')
    parser.add_argument('--testset', type=int, required=True)
    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images',
                        default='/media/zyd/Elements/EndoVis/0original_all')
    parser.add_argument('--model_name', type=str,
                        help='path to a trained model to use', required=True)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    return parser.parse_args()


def paths_list(dataset, idx):
    paths = []
    dataset = os.path.join(dataset, "d{}".format(idx))
    for i in range(1, 5):
        d_k_ = os.path.join(dataset, "k{}/Left".format(i))
        paths.extend(glob.glob(os.path.join(d_k_, '*0.{}'.format(args.ext))))
    return paths


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    l_disp = l_disp.cpu().numpy()
    r_disp = r_disp.cpu().numpy()
    _, _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def test_simple(args):
    # Function to predict for a single image or folder of images
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = os.path.join("/media/zyd/Elements/OMEN Ubuntu backup/monodepth2_models", args.model_name, "models/weights_19")
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(50, False)  # original：18
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    paths = paths_list(args.image_path, args.testset)
    output_directory = os.path.join("/media/zyd/Elements/OMEN Ubuntu backup/respository/monodepth2_results", args.model_name)
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            # print("original_width: ", original_width)
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            # print("feed_width", feed_width)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            input_image_flip = torch.flip(input_image, [3])


            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
            disp = outputs[("disp", 0)]          

            input_image_flip = input_image_flip.to(device)
            features_flip = encoder(input_image_flip)
            outputs_flip = depth_decoder(features_flip)
            disp_flip = outputs_flip[("disp", 0)]
            disp_flip = torch.flip(disp_flip, [3])

            # disp_resized = torch.nn.functional.interpolate(
            #                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            scaled_disp_flip, _ = disp_to_depth(disp_flip, 0.1, 100)
            scaled_disp = batch_post_process_disparity(scaled_disp, scaled_disp_flip)            
            scaled_disp = scaled_disp[0, 0, :, :]  # shape: (512, 640)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            # scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())
            np.save(name_dest_npy, scaled_disp)

            # Saving colormapped depth image
            # disp_resized_np = disp_resized.squeeze().cpu().numpy()  
            vmax = np.percentile(scaled_disp, 95)  # disp_resized_np
            normalizer = mpl.colors.Normalize(vmin=scaled_disp.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  # gray for 灰白深度图
            colormapped_im = (mapper.to_rgba(scaled_disp)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
