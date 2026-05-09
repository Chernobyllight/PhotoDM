from PIL import Image
import numpy as np
import cv2
from cv2.ximgproc import guidedFilter
from pathlib import Path

import os
import re


class GIFSmoothing():
#     def forward(self, *input):
#         pass
        
    def __init__(self, r, eps):
        super(GIFSmoothing, self).__init__()
        self.r = r
        self.eps = eps

    def process(self, initImg, contentImg):
        return self.process_opencv(initImg, contentImg)

    def process_opencv(self, initImg, contentImg):
        '''
        :param initImg: intermediate output. Either image path or PIL Image
        :param contentImg: content image output. Either path or PIL Image
        :return: stylized output image. PIL Image
        '''
        if type(initImg) == str:
            init_img = cv2.imread(initImg)
        else:
            init_img = np.array(initImg[:, :, ::-1]*255, dtype=np.uint8)#.copy()

        if type(contentImg) == str:
            cont_img = cv2.imread(contentImg)
        else:
            cont_img = np.array(contentImg[:, :, ::-1]*255, dtype=np.uint8)#.copy()
            
        if init_img.shape != cont_img.shape:
            cont_img = cv2.resize(cont_img, (init_img.shape[1], init_img.shape[0]))

        output_img = guidedFilter(guide=cont_img, src=init_img, radius=self.r, eps=self.eps)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        output_img = Image.fromarray(output_img)
        return output_img




def sort_files_by_number(folder_path):
    """
    读取文件夹中的文件，并按文件名中的数字顺序排序
    
    参数:
        folder_path (str): 文件夹路径
        
    返回:
        List[str]: 按数字排序后的文件名列表
    """
    # 获取文件夹中所有文件和目录
    all_items = os.listdir(folder_path)
    
    # 只保留文件（排除目录）
    files = [f for f in all_items if os.path.isfile(os.path.join(folder_path, f))]
    
    # 定义一个函数从文件名中提取数字
    def extract_number(filename):
        # 使用正则表达式匹配文件名中的数字
        numbers = re.findall(r'\d+', filename)
        # 如果找到数字，返回第一个数字（转换为整数）
        return int(numbers[0]) if numbers else 0
    
    # 按文件名中的数字排序
    sorted_files = sorted(files, key=extract_number)
    
    return sorted_files

if __name__ == "__main__":
    smooth = GIFSmoothing(r=50, eps=(0.02 * 255) ** 2) # default: (r=50, eps=(0.02 * 255) ** 2)

    stylised_folder = "./figs_full_efdm"
    content_folder = "./content_efdm"
    output_smooth_folder = "./output_smooth_efdm"

    workfolder_stylised = os.path.abspath(stylised_folder)
    workfolder_content = os.path.abspath(content_folder)
    workfolder_output = os.path.abspath(output_smooth_folder)
    
    output_smooth_folder2 = Path(output_smooth_folder)
    if not output_smooth_folder2.exists():
        output_smooth_folder2.mkdir(parents=True, exist_ok=True)
    
    stylised_files = sort_files_by_number(stylised_folder)
    print(stylised_files)

    content_files = sort_files_by_number(content_folder)
    print(content_files)

    cnt = -1
    for stylised_file in stylised_files:
        # index1 = int(cnt/5)
        cnt+=1
        content_file = content_files[cnt]

        output_filename = str(cnt+1) + "_smooth.jpg"
        output_smooth_file = os.path.join(workfolder_output, output_filename)

        stylised_file = os.path.join(workfolder_stylised, stylised_file)
        content_file = os.path.join(workfolder_content, content_file)
        print('--------------------------------')
        print("now process:")
        print("stylised image:", stylised_file)
        print("content image:", content_file)
        smooth.process(stylised_file,content_file).save(output_smooth_file)
        print("output smooth image:", output_smooth_file)
        print('--------------------------------')




    # smooth.process('/group/40063/chernoliu/PhotoDM/TEST/STYLE_TRANSFER/figs_full_efdm/3.jpg',
    # '/group/40063/chernoliu/PhotoDM/TEST/STYLE_TRANSFER/content_efdm/3.jpg').save('o_smooth_3efdm1.jpg')
    # smooth.process('/group/40063/chernoliu/style_4layer_kv/TEST_AE/Transfer/figs_full_efdm_wokvi/2.jpg','/group/40063/chernoliu/style_4layer_kv/TEST_AE/Transfer/content_output/2.jpg').save('o_smooth_2efdmwokvi.jpg')
