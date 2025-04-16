"""
这个是最原始的串行处理的代码： @batch_analyze copy.py 
请将修改我们优化的代码，取消模块内部的并行，只需要在模块之间做并行。也就是说，取消OCR、table内部的并行策略
"""

import time
import cv2
from loguru import logger
from tqdm import tqdm
import concurrent.futures
import torch
import torch_npu

from magic_pdf.config.constants import MODEL_NAME
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram, crop_img, get_res_list_from_layout_res)
from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.ocr_utils import (
    get_adjusted_mfdetrec_res, get_ocr_result_list)

YOLO_LAYOUT_BASE_BATCH_SIZE = 1
MFD_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16


class BatchAnalyze:
    def __init__(self, model_manager, batch_ratio: int, show_log, layout_model, formula_enable, table_enable):
        self.model_manager = model_manager
        self.batch_ratio = batch_ratio
        self.show_log = show_log
        self.layout_model = layout_model
        self.formula_enable = formula_enable
        self.table_enable = table_enable

    def __call__(self, images_with_extra_info: list) -> list:
        if len(images_with_extra_info) == 0:
            return []
        
        # 使用并行优化版本的处理方法
        return self.process_page_parallel(images_with_extra_info)

    def process_page_parallel(self, images_with_extra_info: list) -> list:
        """
        并行处理页面的优化版本
        """
        if len(images_with_extra_info) == 0:
            return []
            
        total_start_time = time.time()
        
        # 1. 初始化模型（必须的步骤）
        _, fst_ocr, fst_lang = images_with_extra_info[0]
        self.model = self.model_manager.get_model(
            fst_ocr, self.show_log, fst_lang, self.layout_model, 
            self.formula_enable, self.table_enable
        )
        
        # 2. 布局检测（所有后续步骤的依赖，必须先执行）
        layout_start_time = time.time()
        images_layout_res = self._run_layout_detection(images_with_extra_info)
        layout_time = round(time.time() - layout_start_time, 2)
        logger.info(f'layout detection time: {layout_time}, image num: {len(images_with_extra_info)}')
        
        # 3. 并行执行三个独立分支的处理，减少最大并行度为2
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # 公式检测与识别分支
            formula_future = None
            if self.model.apply_formula:
                formula_future = executor.submit(
                    self._process_formulas, images_with_extra_info, images_layout_res
                )
            
            # 文本块检测分支
            ocr_det_future = executor.submit(
                self._process_text_blocks, images_with_extra_info, images_layout_res
            )
            
            # 获取处理结果和时间
            formula_times = {"MFD": 0, "MFR": 0}
            if formula_future:
                formula_times = formula_future.result()
            
            det_time = ocr_det_future.result()
            
        # 表格识别分支单独执行，避免并发冲突
        table_time = 0
        if self.model.apply_table:
            table_time = self._process_tables(images_with_extra_info, images_layout_res)
        
        # 4. OCR文本识别（依赖文本块检测结果）
        rec_start_time = time.time()
        self._process_ocr_recognition(images_layout_res)
        rec_time = round(time.time() - rec_start_time, 2)
        
        # 计算总时间和输出性能报告
        total_time = round(time.time() - total_start_time, 2)
        
        # 收集所有处理阶段的时间数据
        time_data = {
            "Layout": layout_time,
            "MFD": formula_times.get("MFD", 0),
            "MFR": formula_times.get("MFR", 0),
            "det": det_time,
            "rec": rec_time,
            "Table": table_time,
            "Total": total_time
        }
        
        # 计算每页平均时间
        page_count = len(images_with_extra_info)
        time_data_per_page = {k: round(v / page_count, 2) for k, v in time_data.items()}
        
        # 输出表格标题
        headers = ["Layout", "MFD", "MFR", "det", "rec", "Table", "Total"]
        header_str = "\t".join(headers)
        
        # 输出总耗时
        total_values = [str(round(time_data.get(h, 0), 2)) for h in headers]
        total_str = "\t".join(total_values)
        
        # 输出单页平均耗时
        per_page_values = [str(round(time_data_per_page.get(h, 0), 2)) for h in headers]
        per_page_str = "\t".join(per_page_values)
        
        # 打印时间统计表格
        logger.info("=" * 80)
        logger.info("PDF处理时间统计 (单位: 秒)")
        logger.info("=" * 80)
        logger.info(header_str)
        logger.info("-" * 80)
        logger.info(f"总耗时:\t{total_str}")
        logger.info(f"平均每页:\t{per_page_str}")
        logger.info("=" * 80)

        # 简化的性能统计
        logger.info(f"总处理时间: {total_time}秒, 共处理{page_count}页, 平均每页: {round(total_time/page_count, 2)}秒")

        return images_layout_res
        
    def _run_layout_detection(self, images_with_extra_info: list) -> list:
        """
        布局检测模块
        """
        images = [image for image, _, _ in images_with_extra_info]
        images_layout_res = []
        
        if self.model.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            # layoutlmv3
            for image in images:
                layout_res = self.model.layout_model(image, ignore_catids=[])
                images_layout_res.append(layout_res)
        elif self.model.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # doclayout_yolo
            layout_images = []
            for image_index, image in enumerate(images):
                layout_images.append(image)

            images_layout_res += self.model.layout_model.batch_predict(
                # layout_images, self.batch_ratio * YOLO_LAYOUT_BASE_BATCH_SIZE
                layout_images, YOLO_LAYOUT_BASE_BATCH_SIZE
            )
            
        return images_layout_res

    def _process_formulas(self, images_with_extra_info: list, images_layout_res: list) -> dict:
        """
        处理公式检测与识别，增加错误处理和内存清理
        """
        times = {"MFD": 0, "MFR": 0}
        
        try:
            # 公式检测
            mfd_start_time = time.time()
            images = [image for image, _, _ in images_with_extra_info]
            images_mfd_res = self.model.mfd_model.batch_predict(
                images, MFD_BASE_BATCH_SIZE
            )
            times["MFD"] = round(time.time() - mfd_start_time, 2)
            logger.info(f'mfd time: {times["MFD"]}, image num: {len(images)}')
            
            # 清理中间状态
            if torch_npu.npu.is_available():
                torch_npu.npu.empty_cache()
            # clean_vram()

            # 公式识别
            mfr_start_time = time.time()
            images_formula_list = self.model.mfr_model.batch_predict(
                images_mfd_res,
                images,
                batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE,
            )
            
            mfr_count = 0
            for image_index in range(len(images)):
                images_layout_res[image_index] += images_formula_list[image_index]
                mfr_count += len(images_formula_list[image_index])
                
            times["MFR"] = round(time.time() - mfr_start_time, 2)
            logger.info(f'mfr time: {times["MFR"]}, formula num: {mfr_count}')
        
        except Exception as e:
            logger.error(f"Error in formula processing: {str(e)}")
        finally:
            # 强制清理内存
            if torch_npu.npu.is_available():
                torch_npu.npu.empty_cache()
            # clean_vram()
            
        return times

    def _process_text_blocks(self, images_with_extra_info: list, images_layout_res: list) -> float:
        """
        处理文本框检测
        返回: 文本框检测时间
        """
        det_start = time.time()
        det_count = 0
        ocr_res_list_all_page = []
        
        # 预处理: 获取需要OCR处理的区域
        for index in range(len(images_with_extra_info)):
            _, ocr_enable, _lang = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_array_img = images_with_extra_info[index][0]

            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )

            ocr_res_list_all_page.append({
                'ocr_res_list': ocr_res_list,
                'lang': _lang,
                'ocr_enable': ocr_enable,
                'np_array_img': np_array_img,
                'single_page_mfdetrec_res': single_page_mfdetrec_res,
                'layout_res': layout_res,
            })
        
        # 顺序处理不同页面的OCR检测
        for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
            # 处理每个需要OCR处理的区域
            _lang = ocr_res_list_dict['lang']
            
            # 获取当前语言的OCR模型
            atom_model_manager = AtomModelSingleton()
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name='ocr',
                ocr_show_log=False,
                det_db_box_thresh=0.3,
                lang=_lang
            )
            
            for res in ocr_res_list_dict['ocr_res_list']:
                new_image, useful_list = crop_img(
                    res, ocr_res_list_dict['np_array_img'], crop_paste_x=50, crop_paste_y=50
                )
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                )

                # OCR检测(不包含识别)
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                ocr_res = ocr_model.ocr(
                    new_image, mfd_res=adjusted_mfdetrec_res, rec=False
                )[0]

                # 整合结果
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(
                        ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], 
                        new_image, _lang
                    )
                    ocr_res_list_dict['layout_res'].extend(ocr_result_list)
                    
            det_count += len(ocr_res_list_dict['ocr_res_list'])
            
        det_time = round(time.time() - det_start, 2)
        logger.info(f'ocr detection time: {det_time}, text block num: {det_count}')
        
        return det_time

    def _process_tables(self, images_with_extra_info: list, images_layout_res: list) -> float:
        """
        处理表格识别，移除内部并行
        """
        table_start = time.time()
        table_res_list_all_page = []
        
        # 预处理: 获取需要处理的表格
        for index in range(len(images_with_extra_info)):
            _, _, _lang = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_array_img = images_with_extra_info[index][0]

            _, table_res_list, _ = get_res_list_from_layout_res(layout_res)

            for table_res in table_res_list:
                table_img, _ = crop_img(table_res, np_array_img)
                table_res_list_all_page.append({
                    'table_res': table_res,
                    'lang': _lang,
                    'table_img': table_img,
                })
        
        # 顺序处理表格，移除内部并行
        table_count = len(table_res_list_all_page)
        for table_res_dict in tqdm(table_res_list_all_page, desc="Table Predict"):
            try:
                self._process_single_table(table_res_dict)
            except Exception as e:
                logger.error(f"Error processing table: {str(e)}")
                
        table_time = round(time.time() - table_start, 2)
        logger.info(f'Table recognition total time: {table_time}, table num: {table_count}')
        
        return table_time
        
    def _process_single_table(self, table_res_dict):
        """
        处理单个表格
        """
        _lang = table_res_dict['lang']
        atom_model_manager = AtomModelSingleton()
        ocr_engine = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            ocr_show_log=False,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            lang=_lang
        )
        table_model = atom_model_manager.get_atom_model(
            atom_model_name='table',
            table_model_name='rapid_table',
            table_model_path='',
            table_max_time=400,
            device='cpu',
            ocr_engine=ocr_engine,
            table_sub_model_name='slanet_plus'
        )
        
        single_table_start = time.time()
        html_code, table_cell_bboxes, logic_points, elapse = table_model.predict(table_res_dict['table_img'])
        single_table_time = round(time.time() - single_table_start, 2)
        
        # 判断是否返回正常
        if html_code:
            expected_ending = html_code.strip().endswith(
                '</html>'
            ) or html_code.strip().endswith('</table>')
            if expected_ending:
                table_res_dict['table_res']['html'] = html_code
            else:
                logger.warning(
                    f'Table recognition processing fails, not found expected HTML table end, time: {single_table_time}s'
                )
        else:
            logger.warning(
                f'Table recognition processing fails, not get html return, time: {single_table_time}s'
            )
        
        return True

    def _process_ocr_recognition(self, images_layout_res: list) -> float:
        """
        处理OCR文本识别，移除内部并行
        """
        # 分析需要OCR识别的项目
        need_ocr_lists_by_lang = {}  # 按语言分组的列表字典
        img_crop_lists_by_lang = {}  # 按语言分组的图像列表字典

        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                if layout_res_item['category_id'] in [15]:
                    if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                        lang = layout_res_item['lang']

                        # 初始化该语言的列表
                        if lang not in need_ocr_lists_by_lang:
                            need_ocr_lists_by_lang[lang] = []
                            img_crop_lists_by_lang[lang] = []

                        # 添加到相应语言的列表
                        need_ocr_lists_by_lang[lang].append(layout_res_item)
                        img_crop_lists_by_lang[lang].append(layout_res_item['np_img'])

                        # 移除已添加到列表的字段
                        layout_res_item.pop('np_img')
                        layout_res_item.pop('lang')
        
        # 处理OCR识别
        rec_start = time.time()
        total_processed = 0
        
        # 顺序处理各语言组，移除内部并行
        for lang, img_crop_list in img_crop_lists_by_lang.items():
            if len(img_crop_list) > 0:
                try:
                    self._process_lang_ocr(lang, img_crop_list, need_ocr_lists_by_lang[lang])
                    total_processed += len(img_crop_list)
                except Exception as e:
                    logger.error(f"Error in OCR recognition for language {lang}: {str(e)}")

        rec_time = round(time.time() - rec_start, 2)
        logger.info(f'OCR recognition total time: {rec_time}, total images processed: {total_processed}')
        
        # 清理内存
        # clean_vram()
        
        return rec_time
        
    def _process_lang_ocr(self, lang, img_crop_list, need_ocr_list):
        """
        处理单一语言的OCR识别
        """
        lang_start_time = time.time()
        
        # 获取此语言的OCR模型
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            ocr_show_log=False,
            det_db_box_thresh=0.3,
            lang=lang
        )
        
        # 批量OCR识别
        ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]

        # 验证数量匹配
        assert len(ocr_res_list) == len(need_ocr_list), \
            f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)} for lang: {lang}'

        # 处理OCR结果
        for index, layout_res_item in enumerate(need_ocr_list):
            ocr_text, ocr_score = ocr_res_list[index]
            layout_res_item['text'] = ocr_text
            layout_res_item['score'] = float(round(ocr_score, 2))

        lang_time = round(time.time() - lang_start_time, 2)
        logger.info(f'OCR recognition for language {lang} time: {lang_time}, items: {len(img_crop_list)}')
        
        return True
