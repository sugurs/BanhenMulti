import os
from PIL import Image
 
def convert_png_to_jpg(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
 
    # 遍历文件夹中的每个文件
    for file in files:
        # 检查文件是否为PNG图片文件
        # --------------------------去除944--------------------------
        # print(type(file))
        file1 = file.replace(r"944 (", r"")
        file2 = file1.replace(r").jpg", r".jpg")
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, file2))

        # if file.endswith('.jpg') and not("-" in file):
        #     # 构建文件的完整路径
        #     file_path = os.path.join(folder_path, file)
            
        #     # 打开PNG图片
        #     image = Image.open(file_path)

        #     # 将PNG图片转换为JPEG格式
        #     file_dir = os.path.dirname(file_path)
        #     new_file_name = prefix + os.path.basename(file_path)
        #     new_file_path = os.path.join(file_dir, new_file_name)
        #     image.convert('RGB').save(new_file_path, 'JPEG')
 
        #     print(f"转换文件：{file} -> {os.path.basename(new_file_path)}")
            
        #     os.remove(file_path)
        
        # if file.endswith('.png') or file.endswith('.JPG') or file.endswith('.jpeg'):
        #     # 构建文件的完整路径
        #     file_path = os.path.join(folder_path, file)
            
        #     # 打开PNG图片
        #     image = Image.open(file_path)

        #     # 将PNG图片转换为JPEG格式
        #     new_file_path = os.path.splitext(file_path)[0] + '.jpg'
        #     image.convert('RGB').save(new_file_path, 'JPEG')
 
        #     print(f"转换文件：{file} -> {os.path.basename(new_file_path)}")
            
        #     os.remove(file_path)
        
        # if not(file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png')):
        #     print(file)
 
 
# 指定图片的文件夹路径
# prefix = "235-"
# folder_path = r'/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/增生性瘢痕208/'
# folder_path = r'/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/瘢痕疙瘩943/'
folder_path = r'/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/瘢痕疙瘩（第二批）650/'
# folder_path = r'/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/瘢痕癌114/'
# folder_path = r'/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/萎缩性瘢痕235/'

# 调用函数进行PNG到JPG的转换
convert_png_to_jpg(folder_path)
