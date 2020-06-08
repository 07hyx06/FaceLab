import sys
import os

o_path = os.getcwd()
sys.path.append(o_path)

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from StyleGAN.net.StyleGenerator import style_generator
# from .net.StyleGenerator import style_generator
import os
import imageio


class API:
    def __init__(self):
        # StyleGAN权重路径
        self.STYLE_GAN_WEIGHT_PATH = r'E:\GitHub项目\pretrained model\karras2019stylegan-ffhq-1024x1024.for_g_all.pt'

        # VGG权重路径
        self.VGG_WEIGHT_PATH = r'E:\GitHub项目\pretrained model\imagenet_vgg16.pth'

        # 使用显卡或CPU
        self.DEVICE = 'cuda'

        # 直接从噪声生成图片的配置：
        # 输入：随机种子
        # 输出：StyleGAN生成器产生一张随机图片
        # 结果保存地址
        self.NOISE_IMG_SAVE_PATH = r'E:\CODE\StyleGAN\from_noise.jpg'
        # 随机种子，如果为0则完全随机
        self.RANDOM_SEED = 259
        # 重定向保存地址
        self.NOISE_IMG_LATENT_SAVE_PATH = r'fromNoise.npy'

        # 由预训练的向量生成图片的配置：
        # 输入：一张训练好的图片的latent
        # 输出：latent对应的图片
        # 预训练的向量地址
        self.PRETRAIN_LATENT_PATH = r'E:\CODE\StyleGAN\rec_lab_latent.npy'
        # 结果保存地址
        self.LATENT_IMG_SAVE_PATH = r'E:\CODE\StyleGAN\from_latent.jpg'
        # 重定向保存地址
        self.LATENT_IMG_LATENT_SAVE_PATH = r'fromLatent.npy'

        # 向量融合的配置：
        # 输入：两张训练好的图片的latent与图像混合的层次
        # 输出：一张混合图片
        # 两个预训练向量的地址
        self.PRETRAIN_LATENT1_PATH = r'E:\GitHub项目\pretrained_latent\Leijun.npy'
        self.PRETRAIN_LATENT2_PATH = r'E:\GitHub项目\pretrained_latent\Luhan.npy'
        # 两张图片的混合层级
        self.MIX_LEVEL = [5, 6, 7, 8, 9]
        # 结果保存地址
        self.MIX_IMG_SAVE_PATH = r'E:\CODE\StyleGAN\mix_latent.jpg'

        # 生成向量之间的过渡gif
        # 输入：两张训练好的图片的latent
        # 输出：过渡图片与过度gif
        # 两个预训练向量的地址
        self.GIF_LATENT1_PATH = r'E:\CODE\StyleGAN\DEMO\hxl.npy'
        self.GIF_LATENT2_PATH = r'E:\CODE\StyleGAN\DEMO\hxa.npy'
        # 保存过渡图片的文件夹
        self.MIDTERM_DIR = r'E:\CODE\StyleGAN\DEMO\gif'
        # GIF大小
        self.GIF_IMG_SIZE = 300
        # 过度图片数量
        self.MIDTERM_NUM = 20
        # GIF帧数
        self.FPS = 5
        # GIF保存地址
        self.GIF_SAVE_PATH = r'E:\CODE\StyleGAN\DEMO\transfer.gif'

        # 图像编辑的配置：
        # 输入：原始图像latent，方向latent，变形幅度
        # 输出：编辑图像
        # 原始latent
        self.ORIGIN_MANI_LATENT_PATH = r'E:\GitHub项目\pretrained_latent\Luxun.npy'
        # 方向latent
        self.MANI_LATENT_PATH = r'E:\GitHub项目\pretrained_latent\hyperplane\stylegan_ffhq_smile_w_boundary.npy'
        # 变形幅度
        self.MANI_FACTOR = 1
        # 变形latent保存地址
        self.AFTER_MANI_LATENT_SAVE_PATH = r'abc.npy'
        # 编辑图像保存地址
        self.MANI_SAVE_PATH = r'mani.jpg'

    def get_latent(self, latent_path):
        latent = np.load(latent_path)  # [18,512]
        latent = torch.tensor(latent, dtype=torch.float32).unsqueeze(0)  # [1,18,512]
        # print(latent.dtype)
        return latent

    def latent2image(self, latent, g_s):
        latent = latent.to(self.DEVICE)
        with torch.no_grad():
            res = g_s(latent)
            res = (res[0].clamp(-1, 1) + 1) / 2.0
        res = res.to('cpu')
        res = transforms.ToPILImage()(res)
        return res

    def present_from_latent(self, latent, res_save_path):
        np.save(self.LATENT_IMG_LATENT_SAVE_PATH, latent[0])
        _, g_s = style_generator()
        g_s.to(self.DEVICE)
        g_s.eval()
        res = self.latent2image(latent, g_s)
        res.save(res_save_path)

    def present_from_noise(self, res_save_path):
        g_m, g_s = style_generator()
        g_s.to(self.DEVICE)
        g_s.eval()
        g_m.to(self.DEVICE)
        g_m.eval()
        if self.RANDOM_SEED != 0:
            torch.manual_seed(self.RANDOM_SEED)
        latent = torch.randn(1, 512).to(self.DEVICE)
        latent = g_m(latent)
        np.save(self.NOISE_IMG_LATENT_SAVE_PATH, latent[0].cpu().detach().numpy())
        res = self.latent2image(latent, g_s)
        res.save(res_save_path)

    def mix_latent(self, latent1, latent2, level):
        latent = latent1.clone()
        for i in level:
            latent[0][i] = latent2[0][i]
        return latent

    def freeze_grad(self, net):
        for param in net.parameters():
            param.requires_grad = False

    def create_mid_img_from_latents(self):
        latent1 = self.get_latent(self.GIF_LATENT1_PATH).to(self.DEVICE)
        latent2 = self.get_latent(self.GIF_LATENT2_PATH).to(self.DEVICE)
        _, g_s = style_generator()
        g_s.to(self.DEVICE)
        g_s.eval()
        for i in range(self.MIDTERM_NUM + 1):
            latent = ((self.MIDTERM_NUM - i) * latent1 + i * latent2) / self.MIDTERM_NUM
            res = self.latent2image(latent, g_s)
            res = transforms.Resize((self.GIF_IMG_SIZE, self.GIF_IMG_SIZE))(res)
            name = str(i) + '.jpg'
            res.save(os.path.join(self.MIDTERM_DIR, name))

    def create_gif_from_image(self):
        image_list = []
        for i in range(self.MIDTERM_NUM + 1):
            name = str(i) + '.jpg'
            image_list.append(imageio.imread(os.path.join(self.MIDTERM_DIR, name)))
        imageio.mimsave(self.GIF_SAVE_PATH, image_list, fps=self.FPS)

    # 直接从噪声生成一张图片
    def noise_style(self):
        self.present_from_noise(self.NOISE_IMG_SAVE_PATH)

    # 从训练好的latent随机生成图片
    def latent_style(self):
        self.present_from_latent(self.get_latent(self.PRETRAIN_LATENT_PATH), self.LATENT_IMG_SAVE_PATH)

    # 混合两张图片
    def mix_style(self):
        latent = self.mix_latent(self.get_latent(self.PRETRAIN_LATENT1_PATH),
                                 self.get_latent(self.PRETRAIN_LATENT2_PATH), self.MIX_LEVEL)
        self.present_from_latent(latent, self.MIX_IMG_SAVE_PATH)

    # 生成两张图片转化的gif图片
    def gif_style(self):
        self.create_mid_img_from_latents()
        self.create_gif_from_image()

    # 编辑图片
    def mani_style(self):
        latent = self.get_latent(self.ORIGIN_MANI_LATENT_PATH) + self.MANI_FACTOR * self.get_latent(
            self.MANI_LATENT_PATH)
        self.present_from_latent(latent, self.MANI_SAVE_PATH)


if __name__ == '__main__':
    api = API()
    api.ORIGIN_MANI_LATENT_PATH = r'cache/mani_image.npy'
    api.MANI_LATENT_PATH = r'E:\GitHub项目\pretrained_latent\hyperplane\stylegan_ffhq_age_c_gender_boundary.npy'
    api.MANI_SAVE_PATH = r'cache/mani_image.jpg'
    api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/mani_image.npy'
    api.mani_style()
