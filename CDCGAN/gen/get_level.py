import os
import pickle
import time

import argparse

import numpy as np
import torch
from image_gen.asset_map import get_asset_map
from image_gen.fixer import PipeFixer
from image_gen.image_gen import GameImageGenerator
from torch.distributions import MultivariateNormal

from data_loader import BinaryDataset
from models.custom import Generator

# torch.manual_seed(75)


class GetLevel:
    def __init__(self, netG, gen, fixer, prev_frame, curr_frame, conditional_channels):
        self.netG = netG
        self.gen = gen
        self.fixer = fixer
        self.init_prev_frame = prev_frame
        self.init_curr_frame = curr_frame
        self.prev_frame = prev_frame
        self.curr_frame = curr_frame
        self.full_level = None
        self.conditional_channels = conditional_channels

    def reset(self, prev_frame=None, curr_frame=None):
        self.full_level = None
        self.prev_frame = prev_frame if prev_frame is not None else self.init_prev_frame
        self.curr_frame = curr_frame if curr_frame is not None else self.init_curr_frame

    def generate_frames(self, noises, var=0.07, frame_count=14):
        noise_dis = MultivariateNormal(
            torch.from_numpy(noises).type("torch.FloatTensor"),
            torch.eye(len(noises)) * var,
        )
        for i in range(frame_count):
            noise = noise_dis.sample().reshape((1, 1, 16, 16))
            gen_input = torch.cat(
                (noise, self.prev_frame[:, self.conditional_channels, 8:-8, :]), dim=1
            )

            fake = self.netG(gen_input).data
            stitched = torch.cat(
                (self.prev_frame, fake[:, :, :, :]), dim=3 #todo (self.prev_frame, fake[:, :, :, 16:]), dim=3
            )  # stitch the context frame
            #tod level_frame = fake[:, :, 8:-8, 16:-2].data.cpu().numpy()  # without padding
            level_frame = fake[:, :, 8:-8, :].data.cpu().numpy()  # without padding
            self.prev_frame = torch.cat(
                (self.curr_frame[:, :, :, :], fake[:, :, :, :]), dim=3 #todo(self.curr_frame[:, :, :, -2:], fake[:, :, :, 16:-2]), dim=3
            )  # with padding added to front

            if self.full_level is None:
                stitched = np.argmax(
                    stitched[:, :, 8:-8, :].data.cpu().numpy(), axis=1
                )
                self.full_level = stitched[0]
            else:
                level_frame = np.argmax(level_frame, axis=1)
                self.full_level = np.concatenate(
                    (self.full_level, level_frame[0]), axis=1
                )

            self.full_level = self.fixer.fix(self.full_level)

        self.gen.render(
            image_array=self.full_level, sprite_dims=(16, 16),
        )

        return self.full_level

    def save_full_level(self, file_name="full_level"):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name + ".pkl", "wb") as fp:
            pickle.dump(self.full_level, fp)

NUM_CHANNELS = 2  # 13

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_data", default="lvls.json", help="location of dataset"
    )
    opt = parser.parse_args()
    return opt



def get_binary_asset_map():
    from PIL import Image

    base_path = "./binary_assets/{}.png"
    asset_map = {}
    for i in range(11):
        try:
            asset_map[i] = Image.open(base_path.format(i))
        except FileNotFoundError:
            continue
    return asset_map

if __name__ == "__main__":
    print("Generating Level")
    opt = parse_arguments()

    conditional_channels = [
        0,
        1,
    ]  # channels on which generator is conditioned on
    dataset = BinaryDataset(opt.src_data)

    netG = Generator(
        latent_size=(len(conditional_channels) + 1, 16, 16), out_size=(NUM_CHANNELS, 32, 32)
    )

    # netG.load_state_dict(torch.load("./trained_models/netG_epoch_300000_0_32.pth", map_location=torch.device('cpu')))
    #netG.load_state_dict(torch.load("./cluster_test/netG_epoch_8.pth", map_location=torch.device('cpu')))
    #netG.load_state_dict(torch.load("/datasets/mgumpu/sample_cluster/netG_epoch_30000_0_32.pth", map_location=torch.device('cpu')))
    #netG.load_state_dict(torch.load("/datasets/mgumpu/cgan_data/netG_epoch_10000_0_32.pth", map_location=torch.device('cpu')))
    # netG.load_state_dict(torch.load("/datasets/mgumpu/cgan_data/netG_epoch_99.pth", map_location=torch.device('cpu')))
    netG.load_state_dict(torch.load("netG_epoch_10000_0_32.pth", map_location=torch.device('cpu')))

    # 300000
    binary_map = get_binary_asset_map()
    gen = GameImageGenerator(asset_map=binary_map)
    #prev_frame, curr_frame = dataset[[120]]  # 51
    prev_frame, curr_frame = dataset[[torch.randint(0, len(dataset), (1,))]]

    full_level = None

    PART = -4
    MAX_ITER = 1
    for i in range(10):

        noise = torch.rand((1, 1, 16, 16)).normal_(0, 1)

        if i == 0:
            gen_input = torch.cat(
                (noise, prev_frame[:, conditional_channels, 8:-8, :]), dim=1
            )
        else:
            gen_input = torch.cat(
                (noise, prev_frame[:, conditional_channels, 8:-8, -16:]), dim=1
                # (noise, prev_frame[:, conditional_channels, 8:-8, :-16]), dim=1
            )

        fake = netG(gen_input).data

        # level_frame = fake[:, :, 8:-8, 16:PART].data.cpu().numpy()  # without padding
        level_frame = fake[:, :, 8:-8, -16:].data.cpu().numpy()  # without padding

        prev_frame = torch.cat(
        (curr_frame[:, :, :, PART:], fake[:, :, :, 16:PART]), dim=3
        # (curr_frame[:, :, :, :], fake[:, :, :, :-16]), dim=3
        )  # with padding added to front
        curr_frame = fake[:, :, :, -16:]

        if full_level is None:
            stitched = torch.cat(
                (prev_frame, fake[:, :, :, :]), dim=3
                # (prev_frame, curr_frame), dim=3
            )  # stitch the context frame

            stitched = np.argmax(stitched[:, :, 8:-8, :PART].data.cpu().numpy(), axis=1)
            # stitched = np.argmax(stitched[:, :, 8:-8, :].data.cpu().numpy(), axis=1)

            full_level = stitched[0]
        else:
            level_frame = np.argmax(level_frame, axis=1)
            full_level = np.concatenate((full_level, level_frame[0]), axis=1)

        primary_sample = full_level[:, -24:-8]
        secondary_sample = full_level[:, -16:]
        gen.render(image_array=primary_sample, sprite_dims=(16, 16))
        gen.save_gen_level(img_name=f"b_primary_sample{i}")
        # generated_level = level_frame
        gen.render(image_array=secondary_sample, sprite_dims=(16, 16))
        gen.save_gen_level(img_name=f"b_secondary_sample{i}")
    generated_level = full_level

    # generated_level = level_frame
    gen.render(image_array=generated_level, sprite_dims=(16, 16))
    gen.save_gen_level(img_name=f"binary_cluster")
    # generated_level = level_frame

    # time.sleep(2)
    with open("full_level.pkl", "wb") as fp:
        pickle.dump(generated_level, fp)

    print("Level Generated")
    # print(level_frame)
