#!/usr/bin/env python3
# train_simple.py

import os
import sys

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import numpy as np
        print(f"NumPy版本: {np.__version__}")
    except ImportError:
        print("错误: 未安装NumPy包")
        print("请运行: pip install numpy")
        return False
    
    try:
        from ultralytics import YOLO
        print("Ultralytics YOLO 已安装")
        return True
    except ImportError:
        print("错误: 未安装ultralytics包")
        print("请运行: pip install ultralytics")
        return False

def fix_numpy_issue():
    """尝试修复NumPy问题"""
    try:
        # 清除可能存在的NumPy缓存
        import shutil
        numpy_cache = os.path.join(os.path.expanduser('~'), '.cache', 'numpy')
        if os.path.exists(numpy_cache):
            shutil.rmtree(numpy_cache)
            print("已清除NumPy缓存")
    except Exception as e:
        print(f"清除缓存时出错: {e}")

def check_dataset():
    """检查数据集配置"""
    dataset_path = "armor_yolo_dataset/armor_dataset.yaml"
    
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集配置文件不存在: {dataset_path}")
        
        # 检查可能的路径
        possible_paths = [
            "armor_yolo_data/armor_dataset.yaml",
            "armor_dataset.yaml",
            "../armor_dataset.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到可能的配置文件: {path}")
                return path
        
        print("请检查以下目录结构:")
        print("armor_yolo_dataset/")
        print("├── images/")
        print("│   ├── train/")
        print("│   └── val/")
        print("├── labels/")
        print("│   ├── train/")
        print("│   └── val/")
        print("└── armor_dataset.yaml")
        return None
    
    # 检查YAML文件内容
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print("数据集配置文件内容:")
            print(content)
    except Exception as e:
        print(f"读取数据集配置文件出错: {e}")
    
    return dataset_path

def create_dataset_yaml():
    """创建数据集配置文件"""
    yaml_content = """# Armor Detection Dataset
path: ./armor_yolo_dataset  # 数据集根目录
train: images/train  # 训练图像路径
val: images/val      # 验证图像路径

# 类别数量
nc: 1

# 类别名称
names: ['armor']

# 下载脚本/URL (可选)
# download: https://example.com/dataset.zip
"""
    
    os.makedirs("armor_yolo_dataset", exist_ok=True)
    with open("armor_yolo_dataset/armor_dataset.yaml", 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print("已创建数据集配置文件: armor_yolo_dataset/armor_dataset.yaml")

def train_armor_detector():
    """
    简化的训练脚本
    """
    print("=== 装甲板检测模型训练 ===")
    
    # 首先尝试修复NumPy问题
    fix_numpy_issue()
    
    if not check_dependencies():
        return
    
    from ultralytics import YOLO
    
    # 检查数据集
    dataset_path = check_dataset()
    if dataset_path is None:
        print("是否要创建默认的数据集配置文件? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            create_dataset_yaml()
            dataset_path = "armor_yolo_dataset/armor_dataset.yaml"
        else:
            print("请先准备数据集")
            return
    
    print(f"使用数据集配置: {dataset_path}")
    
    # 检查预训练模型
    if not os.path.exists('yolov8n.pt'):
        print("下载YOLOv8预训练模型...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            print("预训练模型下载完成")
        except Exception as e:
            print(f"下载预训练模型失败: {e}")
            return
    
    print("开始训练装甲板检测模型...")
    
    try:
        # 加载预训练模型
        model = YOLO('yolov8n.pt')
        
        # 训练参数
        train_args = {
            'data': dataset_path,
            'epochs': 100,
            'imgsz': 640,
            'batch': 8,
            'workers': 2,  # 减少workers数量以避免多进程问题
            'patience': 10,
            'save': True,
            'exist_ok': True,
            'name': 'armor_detection_simple'
        }
        
        # 检查是否有GPU
        try:
            import torch
            if torch.cuda.is_available():
                print(f"检测到GPU: {torch.cuda.get_device_name()}")
                train_args['device'] = 0  # 使用第一个GPU
            else:
                print("使用CPU进行训练")
                train_args['device'] = 'cpu'
        except Exception as e:
            print(f"设备检测失败: {e}，使用CPU")
            train_args['device'] = 'cpu'
        
        print("训练参数:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        
        # 训练模型
        results = model.train(**train_args)
        
        print("训练完成!")
        
        # 导出为ONNX格式
        best_model = 'runs/detect/armor_detection_simple/weights/best.pt'
        if os.path.exists(best_model):
            print("导出模型为ONNX格式...")
            model = YOLO(best_model)
            model.export(format='onnx')
            print("模型已导出为ONNX格式")
            
            # 验证导出的模型
            onnx_model = best_model.replace('.pt', '.onnx')
            if os.path.exists(onnx_model):
                print(f"ONNX模型保存位置: {onnx_model}")
        else:
            print("未找到训练好的模型")
            
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 提供具体的解决方案
        if "numpy" in str(e).lower():
            print("\nNumPy相关问题解决方案:")
            print("1. 尝试重新安装NumPy: pip uninstall numpy && pip install numpy")
            print("2. 检查Python环境是否有冲突")
            print("3. 尝试使用conda环境")

if __name__ == "__main__":
    train_armor_detector()