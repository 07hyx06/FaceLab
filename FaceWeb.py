import streamlit as st
import torch

import os
from PIL import Image

from StyleGAN.FaceLabAPI import API


PRETRAIN_LATENT_DIR = r'E:\GitHub项目\pretrained_latent'
PRETRAIN_DIRECTION_LATENT_DIR = r'E:\GitHub项目\pretrained_latent\hyperplane'


def InitApi():
    api = API()
    api.NOISE_IMG_SAVE_PATH = api.LATENT_IMG_SAVE_PATH = r'cache/faceweb_image.jpg'
    api.NOISE_IMG_LATENT_SAVE_PATH = api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/faceweb_image.npy'
    return api


if __name__ == "__main__":
    api = InitApi()
    latent_dict = {"随机": 'random',
                   "雷军": 'Leijun',
                   "鲁迅": 'Luxun',
                   "刘德华": 'Liudehua',
                   "杨颖": 'Angelababy',
                   "安吉丽娜朱莉": 'Angelina_Jolie',
                   "C罗": 'Cristiano_Ronaldo',
                   "特朗普": 'Donald_Trump',
                   "范冰冰": 'Fanbingbing',
                   "高圆圆": 'Gaoyuanyuan',
                   "黄晓明": 'Huangxiaoming',
                   "莱昂纳多": 'Leonardo_DiCaprio',
                   "鹿晗": 'Luhan',
                   "玛丽莲梦露": 'Marilyn_Monroe',
                   "马云": 'Mayun',
                   "普京": 'Putin',
                   "乔布斯": 'Steve_Jobs',
                   "吴亦凡": 'Wuyifan',
                   "杨幂": 'Yangmi',
                   "基努里维斯": 'Keanu_Reeves',
                   }
    st.sidebar.markdown(
        '''
        # Welcome to FaceWeb v0.1
        一个基于**StyleGAN**与**StreamLit**的可视化人脸编辑平台。
        ## 人脸编辑
        你可以在这里对**随机人像**，或者**名人模型**进行实时人脸编辑。如果选择随机人像，还需要设置随机种子。
        '''
    )

    option = st.sidebar.selectbox('', list(latent_dict.keys()))
    seed = st.sidebar.number_input(
        "", min_value=0, max_value=99999999, value=0, key='seed')
    api.RANDOM_SEED = seed

    st.sidebar.markdown(
        '''
        下面是人像编辑的部分，**滑钮为0时代表原图，可调节范围从-1到1**。
        '''
    )
    smile = st.sidebar.slider('笑脸', min_value=-1.0,
                              max_value=1.0, value=0.0, key='smile', step=0.01)
    age = st.sidebar.slider('年龄', min_value=-1.0,
                            max_value=1.0, value=0.0, key='age', step=0.01)
    sex = st.sidebar.slider('性别', min_value=-1.0,
                            max_value=1.0, value=0.0, key='sex', step=0.01)
    pos = st.sidebar.slider('姿势', min_value=-1.0,
                            max_value=1.0, value=0.0, key='pos', step=0.01)

    if st.sidebar.button('一键生成', key='face2'):
        smile_latent_path = os.path.join(
            PRETRAIN_DIRECTION_LATENT_DIR, 'stylegan_ffhq_smile_w_boundary.npy')
        age_latent_path = os.path.join(
            PRETRAIN_DIRECTION_LATENT_DIR, 'stylegan_ffhq_age_w_boundary.npy')
        sex_latent_path = os.path.join(
            PRETRAIN_DIRECTION_LATENT_DIR, 'stylegan_ffhq_gender_w_boundary.npy')
        pos_latent_path = os.path.join(
            PRETRAIN_DIRECTION_LATENT_DIR, 'stylegan_ffhq_pose_w_boundary.npy')

        if option == "随机":
            api.noise_style()
            latent_path = r'cache/faceweb_image.npy'
        else:
            latent_name = latent_dict[option] + '.npy'
            latent_path = os.path.join(PRETRAIN_LATENT_DIR, latent_name)
        latent = api.get_latent(latent_path) + smile * api.get_latent(smile_latent_path) + age * api.get_latent(
            age_latent_path) + sex * api.get_latent(sex_latent_path) + pos * api.get_latent(pos_latent_path)
        api.present_from_latent(latent, r'cache/faceweb_image.jpg')
        st.markdown(
            "# StyleGAN 生成图像:"
        )
        st.image(Image.open(r'cache/faceweb_image.jpg'), clamp=True, width=500)
