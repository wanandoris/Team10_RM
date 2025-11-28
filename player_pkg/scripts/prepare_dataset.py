#!/usr/bin/env python3
# prepare_dataset.py

import os
import shutil
import random

def train_test_split(items, train_ratio=0.8):
    """手动实现训练集验证集分割"""
    random.shuffle(items)
    split_index = int(len(items) * train_ratio)
    return items[:split_index], items[split_index:]

def prepare_yolo_dataset(data_dir, output_dir, train_ratio=0.8):
    """
    准备YOLO格式的数据集
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # 分割训练集和验证集
    train_files, val_files = train_test_split(image_files, train_ratio=train_ratio)
    
    # 创建输出目录结构
    output_images_train = os.path.join(output_dir, "images", "train")
    output_images_val = os.path.join(output_dir, "images", "val")
    output_labels_train = os.path.join(output_dir, "labels", "train")
    output_labels_val = os.path.join(output_dir, "labels", "val")
    
    for dir_path in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 复制文件
    for file_list, images_dest, labels_dest in [
        (train_files, output_images_train, output_labels_train),
        (val_files, output_images_val, output_labels_val)
    ]:
        for img_file in file_list:
            # 复制图像
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(images_dest, img_file)
            shutil.copy2(src_img, dst_img)
            
            # 复制标签
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(labels_dest, label_file)
            
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
    
    # 创建数据集配置文件
    create_dataset_config(output_dir, len(train_files), len(val_files))

def create_dataset_config(output_dir, train_count, val_count):
    """
    创建YOLO数据集配置文件
    """
    config_content = f"""# Armor Dataset Configuration
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Number of classes
nc: 5

# Class names
names: ['armor_red_1', 'armor_red_2', 'armor_red_3', 'armor_red_4', 'armor_red_5']

# Dataset statistics
train_count: {train_count}
val_count: {val_count}
total_count: {train_count + val_count}
"""
    
    config_path = os.path.join(output_dir, "armor_dataset.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"数据集配置已保存到: {config_path}")
    print(f"训练集: {train_count} 张图像")
    print(f"验证集: {val_count} 张图像")

if __name__ == "__main__":
    # 准备数据集
    prepare_yolo_dataset("armor_dataset_realtime", "armor_yolo_dataset")