import time
import cv2
from loguru import logger
from tqdm import tqdm

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
    
        images_layout_res = []
        total_start_time = time.time()
        layout_start_time = time.time()
        _, fst_ocr, fst_lang = images_with_extra_info[0]
        self.model = self.model_manager.get_model(fst_ocr, self.show_log, fst_lang, self.layout_model, self.formula_enable, self.table_enable)

        images = [image for image, _, _ in images_with_extra_info]

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

        layout_time = round(time.time() - layout_start_time, 2)
        logger.info(
            f'layout detection time: {layout_time}, image num: {len(images)}'
        )

        if self.model.apply_formula:
            # 公式检测
            mfd_start_time = time.time()
            images_mfd_res = self.model.mfd_model.batch_predict(
                # images, self.batch_ratio * MFD_BASE_BATCH_SIZE
                images, MFD_BASE_BATCH_SIZE
            )
            mfd_time = round(time.time() - mfd_start_time, 2)
            logger.info(
                f'mfd time: {mfd_time}, image num: {len(images)}'
            )

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
            mfr_time = round(time.time() - mfr_start_time, 2)
            logger.info(
                f'mfr time: {mfr_time}, formula num: {mfr_count}'
            )

        # 清理显存
        # clean_vram(self.model.device, vram_threshold=8)

        ocr_res_list_all_page = []
        table_res_list_all_page = []
        # preprocessing_start_time = time.time()
        
        for index in range(len(images)):
            _, ocr_enable, _lang = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_array_img = images[index]

            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )

            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list,
                                          'lang':_lang,
                                          'ocr_enable':ocr_enable,
                                          'np_array_img':np_array_img,
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res,
                                          'layout_res':layout_res,
                                          })

            for table_res in table_res_list:
                table_img, _ = crop_img(table_res, np_array_img)
                table_res_list_all_page.append({'table_res':table_res,
                                                'lang':_lang,
                                                'table_img':table_img,
                                              })
        
        # preprocessing_time = round(time.time() - preprocessing_start_time, 2)
        # logger.info(f'Result preprocessing time: {preprocessing_time}')

        # 文本框检测
        det_start = time.time()
        det_count = 0
        # for ocr_res_list_dict in ocr_res_list_all_page:
        for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
            # Process each area that requires OCR processing
            _lang = ocr_res_list_dict['lang']
            # Get OCR results for this language's images
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

                # OCR-det
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                ocr_res = ocr_model.ocr(
                    new_image, mfd_res=adjusted_mfdetrec_res, rec=False
                )[0]

                # Integration results
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], new_image, _lang)
                    ocr_res_list_dict['layout_res'].extend(ocr_result_list)
            det_count += len(ocr_res_list_dict['ocr_res_list'])
        det_time = round(time.time()-det_start, 2)
        logger.info(f'ocr time: {det_time}, text block num: {det_count}')


        # 表格识别 table recognition
        if self.model.apply_table:
            table_start = time.time()
            table_count = len(table_res_list_all_page)
            # for table_res_list_dict in table_res_list_all_page:
            for table_res_dict in tqdm(table_res_list_all_page, desc="Table Predict"):
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
            table_time = round(time.time() - table_start, 2)
            logger.info(f'Table recognition total time: {table_time}, table num: {table_count}')

        # Create dictionaries to store items by language
        need_ocr_lists_by_lang = {}  # Dict of lists for each language
        img_crop_lists_by_lang = {}  # Dict of lists for each language

        # ocr_prep_start_time = time.time()
        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                if layout_res_item['category_id'] in [15]:
                    if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                        lang = layout_res_item['lang']

                        # Initialize lists for this language if not exist
                        if lang not in need_ocr_lists_by_lang:
                            need_ocr_lists_by_lang[lang] = []
                            img_crop_lists_by_lang[lang] = []

                        # Add to the appropriate language-specific lists
                        need_ocr_lists_by_lang[lang].append(layout_res_item)
                        img_crop_lists_by_lang[lang].append(layout_res_item['np_img'])

                        # Remove the fields after adding to lists
                        layout_res_item.pop('np_img')
                        layout_res_item.pop('lang')
        
        # ocr_prep_time = round(time.time() - ocr_prep_start_time, 2)
        # logger.info(f'OCR prep time: {ocr_prep_time}')

        if len(img_crop_lists_by_lang) > 0:
            # Process OCR by language
            rec_start = time.time()
            total_processed = 0

            # Process each language separately
            for lang, img_crop_list in img_crop_lists_by_lang.items():
                if len(img_crop_list) > 0:
                    lang_start_time = time.time()
                    # Get OCR results for this language's images
                    atom_model_manager = AtomModelSingleton()
                    ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name='ocr',
                        ocr_show_log=False,
                        det_db_box_thresh=0.3,
                        lang=lang
                    )
                    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]

                    # Verify we have matching counts
                    assert len(ocr_res_list) == len(
                        need_ocr_lists_by_lang[lang]), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}'

                    # Process OCR results for this language
                    for index, layout_res_item in enumerate(need_ocr_lists_by_lang[lang]):
                        ocr_text, ocr_score = ocr_res_list[index]
                        layout_res_item['text'] = ocr_text
                        layout_res_item['score'] = float(round(ocr_score, 2))

                    total_processed += len(img_crop_list)
                    lang_time = round(time.time() - lang_start_time, 2)
                    logger.info(f'OCR recognition for language {lang} time: {lang_time}, items: {len(img_crop_list)}')

            rec_time = round(time.time() - rec_start, 2)
            logger.info(f'OCR recognition total time: {rec_time}, total images processed: {total_processed}')

        total_time = round(time.time() - total_start_time, 2)
        
        # 收集所有处理阶段的时间数据
        time_data = {
            "Layout": layout_time,
            "MFD": 0,
            "MFR": 0,
            "det": det_time,
            "rec": 0 if 'rec_time' not in locals() else rec_time,
            "Table": 0 if not self.model.apply_table or 'table_time' not in locals() else table_time,
        }
        
        if self.model.apply_formula:
            time_data["MFD"] = mfd_time
            time_data["MFR"] = mfr_time
        
        # 计算总时间
        time_data["Total"] = total_time
        
        # 计算每页平均时间
        page_count = len(images)
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
