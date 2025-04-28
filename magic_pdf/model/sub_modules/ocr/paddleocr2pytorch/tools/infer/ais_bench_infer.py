import os
import time
import numpy as np

from ais_bench.infer.interface import InferSession,MultiDeviceSession
from ais_bench.infer.common.utils import logger_print

model_path_rec = "/home/aicc/mineru/model/d_n_recfix.om"
model_path_det = "/home/aicc/mineru/model/d_n_decfix_linux_aarch64.om"



class AisBenchInfer:
    _instance = None  # 单例模式的类变量

    def __new__(cls, device_id=1):
        # 单例模式实现：如果实例不存在则创建，否则返回已有实例
        if cls._instance is None:
            cls._instance = super(AisBenchInfer, cls).__new__(cls)
            cls._instance._initialized = False  # 标记是否已经初始化
        return cls._instance

    def __init__(self, device_id=1):
        """
        初始化推理模型
        
        Args:
            device_id: 设备ID
            model_path: 模型路径
        """
        # 只在第一次初始化时执行
        if not self._initialized:
            self.device_id = device_id
            self.model_path_rec = model_path_rec
            self.session_rec = InferSession(device_id, self.model_path_rec)
            self.model_path_det = model_path_det
            # self.session_det = InferSession(device_id, self.model_path_det)
            
            self.multi_session_det = MultiDeviceSession(self.model_path_det)
            
            # self.session_det.set_staticbatch()
            print("初始化完成:")
            self._initialized = True  # 标记为已初始化
    
    def muti_infer_det(self, norm_img_batch: np.ndarray):
        """
        执行推理
        
        Args:
            norm_img_batch: 输入的图像批次数据
            
        Returns:
            推理输出结果
        """
        
        
        outputs = self.multi_session_det.infer({self.device_id: [[norm_img_batch]]}, mode='dymshape', custom_sizes=1000000)
        print("推理成功")
        # print(outputs)
        return outputs
    
    def infer_rec(self, norm_img_batch: np.ndarray):
        """
        执行推理
        
        Args:
            norm_img_batch: 输入的图像批次数据
            
        Returns:
            推理输出结果
        """
        outputs = self.session_rec.infer([norm_img_batch], mode='dymbatch')
        print("推理成功")
        return outputs
    def infer_det(self, norm_img_batch: np.ndarray):
        """
        执行推理
        
        Args:
            norm_img_batch: 输入的图像批次数据
            
        Returns:
            推理输出结果
        """
        # model_path_det = "/home/aicc/mineru/model/d_n_decfix_linux_aarch64.om"
        # session_det = InferSession(self.device_id, model_path_det)
        outputs = self.session_det.infer([norm_img_batch], mode='dymshape')
        print("type(outputs):", type(outputs))          # 应输出 <class 'list'>
        print("type(outputs[0]):", type(outputs[0]))       # 应输出 <class 'numpy.ndarray'>
        print("outputs[0].dtype:", outputs[0].dtype)       # 应输出 float32
        print("outputs[0].shape:", outputs[0].shape)       # 例如 (6, 25, 6625)
        print("outputs:", outputs)       # 例如 (6, 25, 6625)
        print(len(outputs))       # 例如 (6, 25, 6625)
        
        print("推理成功")
        # outputs = self.session_det.infer([norm_img_batch], mode='dymshape')
        # print("推理成功")
        # session_det.free_resource()
        return outputs
    
    def free_resource(self):
        """释放模型资源"""
        if hasattr(self, 'session'):
            self.session.free_resource()
    
    @staticmethod
    def infer_with_file(bin_file_path, device_id=0, model_path='/home/aicc/mineru/model/d_model_rec_linux_aarch64.om'):
        """
        使用文件执行动态批量推理
        
        Args:
            bin_file_path: 二进制输入文件路径
            device_id: 设备ID
            model_path: 模型路径
            
        Returns:
            推理输出结果
        """
        session = InferSession(device_id, model_path)
        
        # 读取数据
        ndata = np.fromfile(bin_file_path, dtype=np.float32)
        print("ndata shape:", ndata.shape)
        print("ndata元素数量:", ndata.size)
        print("ndata数据类型:", ndata.dtype)
        
        # 重塑数据
        ndata = ndata.reshape(6, 3, 48, 320)
        print("重塑后的ndata shape:", ndata.shape)
        
        # 执行推理
        outputs = session.infer([ndata], mode='dymshape')
        
        # 打印输出信息
        print(type(outputs))          # 应输出 <class 'list'>
        print(type(outputs[0]))       # 应输出 <class 'numpy.ndarray'>
        print(outputs[0].dtype)       # 应输出 float32
        print(outputs[0].shape)       # 例如 (6, 25, 6625)
        
        # 释放资源
        session.free_resource()
        
        return outputs
    @staticmethod
    def infer_with_file_det(bin_file_path, device_id=0, model_path='/home/aicc/mineru/model/d_n_decfix_linux_aarch64.om'):
        """
        使用文件执行动态批量推理
        
        Args:
            bin_file_path: 二进制输入文件路径
            device_id: 设备ID
            model_path: 模型路径
            
        Returns:
            推理输出结果
        """
        session = InferSession(device_id, model_path)
        
        # 读取数据
        ndata = np.fromfile(bin_file_path, dtype=np.float32)
        print("ndata shape:", ndata.shape) 
        print("ndata元素数量:", ndata.size)
        print("ndata数据类型:", ndata.dtype)
        
        # 重塑数据
        ndata = ndata.reshape(1, 3, 800, 704)
        print("重塑后的ndata shape:", ndata.shape)
        
        # 执行推理
        outputs = session.infer([ndata], mode='dymshape')
        
        # 打印输出信息
        print(type(outputs))          # 应输出 <class 'list'>
        print(type(outputs[0]))       # 应输出 <class 'numpy.ndarray'>
        print(outputs[0].dtype)       # 应输出 float32
        print(outputs[0].shape)       # 例如 (6, 25, 6625)
        
        # 释放资源
        session.free_resource()
        
        return outputs

    @staticmethod
    def infer_folder_det(folder_path, device_id=0, model_path='/home/aicc/mineru/model/d_n_decfix_linux_aarch64.om'):
        """
        处理文件夹中的所有bin文件进行检测推理
        
        Args:
            folder_path: 包含bin文件和shape.txt文件的文件夹路径
            device_id: 设备ID
            model_path: 模型路径
            
        Returns:
            所有bin文件的推理结果字典，键为bin文件名，值为推理输出
        """
        session = MultiDeviceSession( model_path)
        # session.set_staticbatch()
        results = {}
        
        # 获取文件夹中所有bin文件
        bin_files = [f for f in os.listdir(folder_path) if f.endswith('.bin') and not f.endswith('.shape.txt')]
        
        for bin_file in bin_files:
            bin_file_path = os.path.join(folder_path, bin_file)
            shape_file_path = bin_file_path + '.shape.txt'
            
            # 检查shape文件是否存在
            if not os.path.exists(shape_file_path):
                print(f"跳过 {bin_file}: 找不到shape文件")
                continue
            
            # 读取shape数据
            with open(shape_file_path, 'r') as f:
                shape_str = f.read().strip()
            
            # 解析shape数据
            shape = tuple(map(int, shape_str.split(',')))
            
            # 读取bin数据
            ndata = np.fromfile(bin_file_path, dtype=np.float32)
            print(f"处理 {bin_file}")
            print(f"原始数据shape: {ndata.shape}")
            print(f"从shape文件读取的形状: {shape}")
       
            
            # 重塑数据
            try:
                ndata = ndata.reshape(shape)
                print(f"重塑后的数据shape: {ndata.shape}")
                
                # 执行推理
                outputs = session.infer({device_id: [[ndata]]}, mode='dymshape', custom_sizes=10000000)
                print(f"{bin_file} 推理成功")
                
                # 记录结果
                results[bin_file] = outputs
                
            except Exception as e:
                print(f"处理 {bin_file} 时出错: {e}")
        
        # 释放资源
        # session.free_resource()
        
        return results
    
    
    @staticmethod
    def infer_folder_rec(folder_path, device_id=0, model_path='/home/aicc/mineru/model/d1001_n_recfix_linux_aarch64.om'):
        """
        处理文件夹中的所有bin文件进行识别推理
        
        Args:
            folder_path: 包含bin文件和shape.txt文件的文件夹路径
            device_id: 设备ID
            model_path: 模型路径
            
        Returns:
            所有bin文件的推理结果字典，键为bin文件名，值为推理输出
        """
        session = InferSession(device_id, model_path)
        results = {}
        
        # 获取文件夹中所有bin文件
        bin_files = [f for f in os.listdir(folder_path) if f.endswith('.bin') and not f.endswith('.shape.txt')]
        
        for bin_file in bin_files:
            bin_file_path = os.path.join(folder_path, bin_file)
            shape_file_path = bin_file_path + '.shape.txt'
            
            # 检查shape文件是否存在
            if not os.path.exists(shape_file_path):
                print(f"跳过 {bin_file}: 找不到shape文件")
                continue
            
            # 读取shape数据
            with open(shape_file_path, 'r') as f:
                shape_str = f.read().strip()
            
            # 解析shape数据
            shape = tuple(map(int, shape_str.split(',')))
            
            # 读取bin数据
            ndata = np.fromfile(bin_file_path, dtype=np.float32)
            print(f"处理 {bin_file}")
            print(f"原始数据shape: {ndata.shape}")
            print(f"从shape文件读取的形状: {shape}")
            
            # 重塑数据
            try:
                ndata = ndata.reshape(shape)
                print(f"重塑后的数据shape: {ndata.shape}")
                
                # 执行推理
                outputs = session.infer([ndata], mode='dymbatch')
                print(f"{bin_file} 推理成功")
                
                # 记录结果
                results[bin_file] = outputs
                
            except Exception as e:
                print(f"处理 {bin_file} 时出错: {e}")
        
        # 释放资源
        session.free_resource()
        
        return results

# 使用示例:

# import acl

# infer_model = AisBenchInfer()
# result = infer_model.infer_det(np.zeros((1, 3, 608, 704), dtype=np.float32))
# result = infer_model.infer_det(np.zeros((1, 3, 608, 704), dtype=np.float32))

# 使用 muti 推理多个 ，muti每次都会创建InferSession。    使用推理接口时才会在指定的几个devices的每个进程中新建一个InferSession。
# result = infer_model.muti_infer_det(np.zeros((1, 3, 800, 704), dtype=np.float32))
# result = infer_model.muti_infer_det(np.zeros((1, 3, 608, 704), dtype=np.float32))


# infer_model.free_resource()

# 或者直接使用静态方法:
# result = AisBenchInfer.infer_with_file('/home/aicc/mineru/MinerU_1.3.0/demo/preprocessed_data/rec/rec_input_batch_0_20250421_091529_142.bin')
# result = AisBenchInfer.infer_with_file_det('/home/aicc/mineru/MinerU_1.3.0/demo/preprocessed_data/det/det_input_20250421_034746_105.bin')

# results = AisBenchInfer.infer_folder_det('/home/aicc/mineru/MinerU_1.3.0/demo/preprocessed_data/det')
# results = AisBenchInfer.infer_folder_rec('/home/aicc/mineru/MinerU_1.3.0/demo/preprocessed_data/rec')
# print("检测推理结果:", results)






"""
det(-1,3,-1,-1) d_n_decfix_linux_aarch64.om
rec(-1,3,48,320) d_n_recfix.om


"""