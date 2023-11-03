# NUM_CHANNELS= 2 # 13
#
# def get_binary_asset_map():
#     from PIL import Image
#
#     base_path = "./binary_assets/{}.png"
#     asset_map = {}
#     for i in range(11):
#         try:
#             asset_map[i] = Image.open(base_path.format(i))
#         except FileNotFoundError:
#             continue
#     return asset_map
#
# if __name__ == "__main__":
#     conditional_channels = [
#         0,
#         1,
#     ]  # channels on which generator is conditioned on
#     dataset = BinaryDataset()
#     # dataset = MarioDataset()
#
#     netG = Generator(
#         latent_size=(len(conditional_channels) + 1, 14, 14), out_size=(NUM_CHANNELS, 32, 32)
#     )
#
#     # netG.load_state_dict(torch.load("./trained_models/netG_epoch_300000_0_32.pth", map_location=torch.device('cpu')))
#     netG.load_state_dict(torch.load("./binary_test/netG_epoch_9.pth", map_location=torch.device('cpu')))
#
#     # 300000
#     binary_map = get_binary_asset_map()
#     gen = GameImageGenerator(asset_map=binary_map)
#     # prev_frame, curr_frame = dataset[[120]]  # 51
#     prev_frame, curr_frame = dataset[[torch.randint(0, len(dataset), (1,))]]
#
#     full_level = None