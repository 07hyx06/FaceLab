import sys
import os

o_path = os.getcwd()
sys.path.append(o_path)

from PyQt5.QtWidgets import *
from PIL import Image
from PyQt5.QtGui import *
from app.FaceLabUI import Ui_Form
from StyleGAN.FaceLabAPI import API
from torchvision import transforms
import os

INIT_IMAGE_PATH = r'app/图像资源/空白人像.jpg'
PRETRAIN_LATENT_DIR = r'E:\GitHub项目\pretrained_latent'
PRETRAIN_DIRECTION_LATENT_DIR = r'E:\GitHub项目\pretrained_latent\hyperplane'
MAX_RANDOM_SEED = 1e9


class FaceLab(QMainWindow):
    def __init__(self):
        super(FaceLab, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle('FaceLab v1.0')
        self.setFixedSize(1024, 768)
        self.latent_dict = {"雷军": 'Leijun',
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
                            "基努里维斯": 'Keanu_Reeves'}

        # 防止混合图像逻辑错误
        self.page2_gate1 = 0
        self.page2_gate2 = 0
        self.page3_gate1 = 0
        self.page3_gate2 = 0
        self.page4_gate = 0

        self.last_smile = 0
        self.last_pose = 0
        self.last_gender = 0
        self.last_age = 0

        self.initUi()
        self.api = API()

    def initUi(self):
        # 设置背景
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("app/图像资源/bkg.jpg")))
        self.setPalette(palette)
        # 添加FaceLab标志
        self.ui.label.resize(224, 100)
        self.ui.label.move(0, 0)
        self.put_image_on_label(self.ui.label, 'app/图像资源/FaceLabLogo.png', 100, 224)
        # 设置侧边导航栏
        self.ui.listWidget.setFrameShape(QListWidget.NoFrame)  # 去掉边框
        # 随机人像页
        self.put_image_on_label(self.ui.label_6, INIT_IMAGE_PATH)  # 放置空白人像
        self.ui.spinBox.setMaximum(MAX_RANDOM_SEED)
        self.ui.pushButton.clicked.connect(self.page1_f1)  # 关联随机种子与一键生成按钮
        # 人像混合页
        self.put_image_on_label(self.ui.label_8, 'app/图像资源/空白1.jpg', 300, 300)  # 放置空白人像
        self.put_image_on_label(self.ui.label_9, 'app/图像资源/空白2.jpg', 300, 300)  # 放置空白人像
        self.put_image_on_label(self.ui.label_10, 'app/图像资源/空白3.jpg', 300, 300)  # 放置空白人像
        self.ui.pushButton_2.clicked.connect(self.page2_f1)
        self.ui.pushButton_3.clicked.connect(self.page2_f2)
        self.ui.pushButton_4.clicked.connect(self.page2_f3)
        # 人像过渡页
        self.put_image_on_label(self.ui.label_14, 'app/图像资源/空白1.jpg', 300, 300)  # 放置空白人像
        self.put_image_on_label(self.ui.label_17, 'app/图像资源/空白2.jpg', 300, 300)  # 放置空白人像
        self.put_image_on_label(self.ui.label_16, 'app/图像资源/空白3.jpg', 300, 300)  # 放置空白人像
        self.ui.pushButton_7.clicked.connect(self.page3_f1)
        self.ui.pushButton_6.clicked.connect(self.page3_f2)
        self.ui.pushButton_5.clicked.connect(self.page3_f3)
        # 人像编辑页
        self.put_image_on_label(self.ui.label_18, INIT_IMAGE_PATH)  # 放置空白人像
        self.ui.pushButton_8.clicked.connect(self.page4_f1)
        self.ui.horizontalSlider_smile.sliderReleased.connect(self.page4_f2)
        self.ui.horizontalSlider_pose.sliderReleased.connect(self.page4_f3)
        self.ui.horizontalSlider_gender.sliderReleased.connect(self.page4_f4)
        self.ui.horizontalSlider_age.sliderReleased.connect(self.page4_f5)

    # 随机人像页，关联随机种子与一键生成按钮
    def page1_f1(self):
        seed = self.ui.spinBox.value()
        self.api.RANDOM_SEED = seed
        self.api.NOISE_IMG_SAVE_PATH = r'cache/noise_image.jpg'
        self.api.NOISE_IMG_LATENT_SAVE_PATH = r'cache/noise_image.npy'
        self.api.noise_style()
        self.put_image_on_label(self.ui.label_6, self.api.NOISE_IMG_SAVE_PATH)

    # 人像混合页，关联图像1的复选框与一键生成按钮
    def page2_f1(self):
        self.page2_gate1 = 1
        style = self.ui.comboBox.currentText()
        if style == '随机':
            seed = self.ui.spinBox_2.value()
            self.api.RANDOM_SEED = seed
            self.api.NOISE_IMG_SAVE_PATH = r'cache/mix_image1.jpg'
            self.api.NOISE_IMG_LATENT_SAVE_PATH = r'cache/mix_image1.npy'  # 保存latent，为了混合图像
            self.api.noise_style()
            self.put_image_on_label(self.ui.label_8, self.api.NOISE_IMG_SAVE_PATH, 300, 300)
        else:
            latent_name = self.latent_dict[style] + '.npy'
            self.api.PRETRAIN_LATENT_PATH = os.path.join(PRETRAIN_LATENT_DIR, latent_name)
            self.api.LATENT_IMG_SAVE_PATH = r'cache/mix_image1.jpg'
            self.api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/mix_image1.npy'
            self.api.latent_style()
            self.put_image_on_label(self.ui.label_8, self.api.LATENT_IMG_SAVE_PATH, 300, 300)

    # 人像混合页，关联图像2的复选框与一键生成按钮
    def page2_f2(self):
        self.page2_gate2 = 1
        style = self.ui.comboBox_2.currentText()
        if style == '随机':
            seed = self.ui.spinBox_3.value()
            self.api.RANDOM_SEED = seed
            self.api.NOISE_IMG_SAVE_PATH = r'cache/mix_image2.jpg'
            self.api.NOISE_IMG_LATENT_SAVE_PATH = r'cache/mix_image2.npy'  # 保存latent，为了混合图像
            self.api.noise_style()
            self.put_image_on_label(self.ui.label_9, self.api.NOISE_IMG_SAVE_PATH, 300, 300)
        else:
            latent_name = self.latent_dict[style] + '.npy'
            self.api.PRETRAIN_LATENT_PATH = os.path.join(PRETRAIN_LATENT_DIR, latent_name)
            self.api.LATENT_IMG_SAVE_PATH = r'cache/mix_image2.jpg'
            self.api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/mix_image2.npy'
            self.api.latent_style()
            self.put_image_on_label(self.ui.label_9, self.api.LATENT_IMG_SAVE_PATH, 300, 300)

    # 人像混合页，响应图像混合按钮
    def page2_f3(self):
        if self.page2_gate1 * self.page2_gate2 == 0:
            return
        self.api.PRETRAIN_LATENT1_PATH = r'cache/mix_image1.npy'
        self.api.PRETRAIN_LATENT2_PATH = r'cache/mix_image2.npy'
        self.api.MIX_IMG_SAVE_PATH = r'cache/mix_image.jpg'
        self.api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/mix_image.npy'
        self.api.mix_style()
        self.put_image_on_label(self.ui.label_10, self.api.MIX_IMG_SAVE_PATH, 300, 300)

    # 人像过渡页，关联图像1的复选框与一键生成按钮
    def page3_f1(self):
        self.page3_gate1 = 1
        style = self.ui.comboBox_3.currentText()
        if style == '随机':
            seed = self.ui.spinBox_4.value()
            self.api.RANDOM_SEED = seed
            self.api.NOISE_IMG_SAVE_PATH = r'cache/mix_image1.jpg'
            self.api.NOISE_IMG_LATENT_SAVE_PATH = r'cache/mix_image1.npy'  # 保存latent，为了混合图像
            self.api.noise_style()
            self.put_image_on_label(self.ui.label_14, self.api.NOISE_IMG_SAVE_PATH, 300, 300)
        else:
            latent_name = self.latent_dict[style] + '.npy'
            self.api.PRETRAIN_LATENT_PATH = os.path.join(PRETRAIN_LATENT_DIR, latent_name)
            self.api.LATENT_IMG_SAVE_PATH = r'cache/mix_image1.jpg'
            self.api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/mix_image1.npy'
            self.api.latent_style()
            self.put_image_on_label(self.ui.label_14, self.api.LATENT_IMG_SAVE_PATH, 300, 300)

    # 人像过渡页，关联图像2的复选框与一键生成按钮
    def page3_f2(self):
        self.page3_gate2 = 1
        style = self.ui.comboBox_4.currentText()
        if style == '随机':
            seed = self.ui.spinBox_5.value()
            self.api.RANDOM_SEED = seed
            self.api.NOISE_IMG_SAVE_PATH = r'cache/mix_image2.jpg'
            self.api.NOISE_IMG_LATENT_SAVE_PATH = r'cache/mix_image2.npy'  # 保存latent，为了混合图像
            self.api.noise_style()
            self.put_image_on_label(self.ui.label_17, self.api.NOISE_IMG_SAVE_PATH, 300, 300)
        else:
            latent_name = self.latent_dict[style] + '.npy'
            self.api.PRETRAIN_LATENT_PATH = os.path.join(PRETRAIN_LATENT_DIR, latent_name)
            self.api.LATENT_IMG_SAVE_PATH = r'cache/mix_image2.jpg'
            self.api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/mix_image2.npy'
            self.api.latent_style()
            self.put_image_on_label(self.ui.label_17, self.api.LATENT_IMG_SAVE_PATH, 300, 300)

    # 人像过渡页，响应生成GIF按钮
    def page3_f3(self):
        if self.page3_gate1 * self.page3_gate2 == 0:
            return
        self.api.GIF_LATENT1_PATH = r'cache/mix_image1.npy'
        self.api.GIF_LATENT2_PATH = r'cache/mix_image2.npy'
        self.api.MIDTERM_DIR = r'cache/gif'
        self.api.GIF_SAVE_PATH = r'cache/mix.gif'
        self.api.GIF_IMG_SIZE = 300
        self.api.gif_style()
        load_gif = QMovie(self.api.GIF_SAVE_PATH)
        self.ui.label_16.setMovie(load_gif)
        load_gif.start()

    # 人像编辑页，响应一键生成按钮
    def page4_f1(self):
        self.page4_gate = 1
        style = self.ui.comboBox_5.currentText()
        if style == '随机':
            seed = self.ui.spinBox_6.value()
            self.api.RANDOM_SEED = seed
            self.api.NOISE_IMG_SAVE_PATH = r'cache/mani_image.jpg'
            self.api.NOISE_IMG_LATENT_SAVE_PATH = r'cache/mani_image.npy'  # 保存latent，为了混合图像
            self.api.noise_style()
            self.put_image_on_label(self.ui.label_18, self.api.NOISE_IMG_SAVE_PATH, 450, 450)
        else:
            latent_name = self.latent_dict[style] + '.npy'
            self.api.PRETRAIN_LATENT_PATH = os.path.join(PRETRAIN_LATENT_DIR, latent_name)
            self.api.LATENT_IMG_SAVE_PATH = r'cache/mani_image.jpg'
            self.api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/mani_image.npy'
            self.api.latent_style()
            self.put_image_on_label(self.ui.label_18, self.api.LATENT_IMG_SAVE_PATH, 450, 450)

        self.ui.horizontalSlider_age.setValue(50)
        self.ui.horizontalSlider_gender.setValue(50)
        self.ui.horizontalSlider_pose.setValue(50)
        self.ui.horizontalSlider_smile.setValue(50)
        self.last_smile = 50
        self.last_pose = 50
        self.last_gender = 50
        self.last_age = 50

    # 人像编辑页，响应微笑滑钮
    def page4_f2(self):
        if self.page4_gate == 0:
            return
        self.api.ORIGIN_MANI_LATENT_PATH = r'cache/mani_image.npy'
        self.api.MANI_LATENT_PATH = os.path.join(PRETRAIN_DIRECTION_LATENT_DIR, 'stylegan_ffhq_smile_w_boundary.npy')
        self.api.MANI_FACTOR = self.get_scale_slide(self.ui.horizontalSlider_smile.value(), 1) \
                               - self.get_scale_slide(self.last_smile, 1)
        self.api.MANI_SAVE_PATH = r'cache/mani_image.jpg'
        self.api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/mani_image.npy'
        self.api.mani_style()
        self.put_image_on_label(self.ui.label_18, self.api.MANI_SAVE_PATH, 450, 450)
        self.last_smile = self.ui.horizontalSlider_smile.value()

    # 人像编辑页，响应姿势滑钮
    def page4_f3(self):
        if self.page4_gate == 0:
            return
        self.api.ORIGIN_MANI_LATENT_PATH = r'cache/mani_image.npy'
        self.api.MANI_LATENT_PATH = os.path.join(PRETRAIN_DIRECTION_LATENT_DIR, 'stylegan_ffhq_pose_w_boundary.npy')
        self.api.MANI_FACTOR = self.get_scale_slide(self.ui.horizontalSlider_pose.value(), 1) \
                               - self.get_scale_slide(self.last_pose, 1)
        self.api.MANI_SAVE_PATH = r'cache/mani_image.jpg'
        self.api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/mani_image.npy'
        self.api.mani_style()
        self.put_image_on_label(self.ui.label_18, self.api.MANI_SAVE_PATH, 450, 450)
        self.last_pose = self.ui.horizontalSlider_pose.value()

    # 人像编辑页，响应性别滑钮
    def page4_f4(self):
        if self.page4_gate == 0:
            return
        self.api.ORIGIN_MANI_LATENT_PATH = r'cache/mani_image.npy'
        self.api.MANI_LATENT_PATH = os.path.join(PRETRAIN_DIRECTION_LATENT_DIR, 'stylegan_ffhq_gender_w_boundary.npy')
        self.api.MANI_FACTOR = self.get_scale_slide(self.ui.horizontalSlider_gender.value(), 1) \
                               - self.get_scale_slide(self.last_gender, 1)
        self.api.MANI_SAVE_PATH = r'cache/mani_image.jpg'
        self.api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/mani_image.npy'
        self.api.mani_style()
        self.put_image_on_label(self.ui.label_18, self.api.MANI_SAVE_PATH, 450, 450)
        self.last_gender = self.ui.horizontalSlider_gender.value()

    # 人像编辑页，响应年龄滑钮
    def page4_f5(self):
        if self.page4_gate == 0:
            return
        self.api.ORIGIN_MANI_LATENT_PATH = r'cache/mani_image.npy'
        self.api.MANI_LATENT_PATH = os.path.join(PRETRAIN_DIRECTION_LATENT_DIR, 'stylegan_ffhq_age_w_boundary.npy')
        self.api.MANI_FACTOR = self.get_scale_slide(self.ui.horizontalSlider_age.value(), 1) \
                               - self.get_scale_slide(self.last_age, 1)
        self.api.MANI_SAVE_PATH = r'cache/mani_image.jpg'
        self.api.LATENT_IMG_LATENT_SAVE_PATH = r'cache/mani_image.npy'
        self.api.mani_style()
        self.put_image_on_label(self.ui.label_18, self.api.MANI_SAVE_PATH, 450, 450)
        self.last_age = self.ui.horizontalSlider_age.value()

    # [0,100] -> [-k,k]
    def get_scale_slide(self, val, k):
        shift = 50 / k
        return (val - 50) / shift

    # 把本地图片放到label上面
    def put_image_on_label(self, label, img_path, target_width=450, target_height=450):
        ori_img = Image.open(img_path)
        resized_img = transforms.Resize((target_width, target_height))(ori_img)
        resized_img.save(img_path)
        bkg_image = QPixmap(img_path)
        label.setPixmap(bkg_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_wnd = FaceLab()
    main_wnd.show()
    app.exec()
