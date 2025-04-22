# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import aclruntime
import numpy as np

from ais_bench.infer.common.utils import logger_print


class AclRuntimeInfer:
    def __init__(self, device_id=0, model_path='/home/aicc/mineru/model/d_n_decfix_linux_aarch64.om'):
        """
        初始化推理模型
        
        Args:
            device_id: 设备ID
            model_path: 模型路径
        """
        self.device_id = device_id
        self.model_path = model_path
        self.options = aclruntime.session_options()
        # 初始化时不创建会话，改为在推理时创建
        logger_print("初始化完成")
    
    def _create_session(self):
        """创建新会话"""
        return aclruntime.InferenceSession(self.model_path, self.device_id, self.options)
    
    def infer_single(self, bin_file_path):
        """
        单文件推理
        
        Args:
            bin_file_path: 二进制输入文件路径
            
        Returns:
            推理输出结果
        """
        # 创建新会话
        session = self._create_session()
        
        # 读取数据
        ndata = np.fromfile(bin_file_path, dtype=np.float32)
        # 重塑数据
        ndata = ndata.reshape(1, 3, 256, 256)
        
        shapes = []
        feeds = []
        # 设置输入shape
        shape0 = [1, 3, 256, 256]
        shapes.append(shape0)
        
        # 将数据移至设备
        tensor0 = aclruntime.Tensor(ndata)
        tensor0.to_device(self.device_id)
        feeds.append(tensor0)
        
        # 设置动态shape
        dym_list = []
        indesc = session.get_inputs()
        for i, shape in enumerate(shapes):
            str_shape = [str(val) for val in shape]
            dyshape = "{}:{}".format(indesc[i].name, ",".join(str_shape))
            dym_list.append(dyshape)
        dyshapes = ';'.join(dym_list)
        session.set_dynamic_shape(dyshapes)
        
        # 设置自定义输出大小
        outdesc = session.get_outputs()
        custom_sizes = 100000
        custom_sizes = [custom_sizes] * len(outdesc)
        session.set_custom_outsize(custom_sizes)
        
        # 推理
        outnames = [meta.name for meta in session.get_outputs()]
        outputs = session.run(outnames, feeds)
        
        # 处理输出
        outarray = []
        for out in outputs:
            # 将输出数据移至主机内存
            out.to_host()
            # 转换为numpy数组
            outarray.append(np.array(out))
        
        logger_print("infer avg:%s ms" % np.mean(session.sumary().exec_time_list))
        
        # 释放会话资源
        del session
        return outarray
    
    def infer_folder_det(self, folder_path):
        """
        处理文件夹中的所有bin文件进行检测推理
        
        Args:
            folder_path: 包含bin文件和shape.txt文件的文件夹路径
            
        Returns:
            所有bin文件的推理结果字典，键为bin文件名，值为推理输出
        """
        results = {}
        
        # 获取文件夹中所有bin文件
        bin_files = [f for f in os.listdir(folder_path) if f.endswith('.bin') and not f.endswith('.shape.txt')]
        
        for bin_file in bin_files:
            bin_file_path = os.path.join(folder_path, bin_file)
            shape_file_path = bin_file_path + '.shape.txt'
            
            # 检查shape文件是否存在
            if not os.path.exists(shape_file_path):
                logger_print(f"跳过 {bin_file}: 找不到shape文件")
                continue
            
            # 读取shape数据
            with open(shape_file_path, 'r') as f:
                shape_str = f.read().strip()
            
            # 解析shape数据
            shape = tuple(map(int, shape_str.split(',')))
            
            # 读取bin数据
            ndata = np.fromfile(bin_file_path, dtype=np.float32)
            logger_print(f"处理 {bin_file}")
            logger_print(f"原始数据shape: {ndata.shape}")
            logger_print(f"从shape文件读取的形状: {shape}")
            
            # 重塑数据并推理
            try:
                ndata = ndata.reshape(shape)
                logger_print(f"重塑后的数据shape: {ndata.shape}")
                
                # 为每个文件创建新会话
                session = self._create_session()
                
                shapes = []
                feeds = []
                shapes.append(list(shape))
                
                # 移动数据到设备
                tensor0 = aclruntime.Tensor(ndata)
                tensor0.to_device(self.device_id)
                feeds.append(tensor0)
                
                # 设置动态shape
                dym_list = []
                indesc = session.get_inputs()
                for i, shape in enumerate(shapes):
                    str_shape = [str(val) for val in shape]
                    dyshape = "{}:{}".format(indesc[i].name, ",".join(str_shape))
                    dym_list.append(dyshape)
                dyshapes = ';'.join(dym_list)
                session.set_dynamic_shape(dyshapes)
                
                # 设置自定义输出大小
                outdesc = session.get_outputs()
                custom_sizes = 100000
                custom_sizes = [custom_sizes] * len(outdesc)
                session.set_custom_outsize(custom_sizes)
                
                # 推理
                outnames = [meta.name for meta in session.get_outputs()]
                outputs = session.run(outnames, feeds)
                
                # 处理输出
                outarray = []
                for out in outputs:
                    out.to_host()
                    outarray.append(np.array(out))
                
                logger_print(f"{bin_file} 推理成功")
                results[bin_file] = outarray
                
                # 释放会话资源
                del session
                
            except Exception as e:
                logger_print(f"处理 {bin_file} 时出错: {e}")
        
        return results
    
    def free_resource(self):
        """释放资源"""
        pass  # 会话资源已在每次推理后释放


# 示例使用
if __name__ == "__main__":
    infer = AclRuntimeInfer()
    # 单文件推理
    # result = infer.infer_single('/home/aicc/mineru/MinerU_1.3.0/demo/preprocessed_data/det/det_input_20250421_034744_121.bin')
    
    # 文件夹批量推理
    results = infer.infer_folder_det('/home/aicc/mineru/MinerU_1.3.0/demo/preprocessed_data/det')
    infer.free_resource()
