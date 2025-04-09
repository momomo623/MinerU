# coding=UTF-8
# Copyright (c) Opendatalab. All rights reserved.
import os
import glob

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

def process_pdf_file(pdf_file_path, output_base_dir=None):
    name_without_extension = os.path.basename(pdf_file_path).split('.')[0]
    
    if output_base_dir is None:
        __dir__ = os.path.dirname(os.path.abspath(__file__))
        output_base_dir = os.path.join(__dir__, "output")
    
    local_image_dir = os.path.join(output_base_dir, name_without_extension, "images")
    local_md_dir = os.path.join(output_base_dir, name_without_extension)
    image_dir = str(os.path.basename(local_image_dir))
    os.makedirs(local_image_dir, exist_ok=True)
    
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
    
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_path) 
    
    ds = PymuDocDataset(pdf_bytes)
    
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)
    
    model_inference_result = infer_result.get_infer_res()
    pipe_result.draw_layout(os.path.join(local_md_dir, "{}_layout.pdf".format(name_without_extension)))
    
    pipe_result.draw_span(os.path.join(local_md_dir, "{}_spans.pdf".format(name_without_extension)))
    
    md_content = pipe_result.get_markdown(image_dir)
    
    pipe_result.dump_md(md_writer, "{}.md".format(name_without_extension), image_dir)
    
    content_list_content = pipe_result.get_content_list(image_dir)
    
    pipe_result.dump_content_list(md_writer, "{}_content_list.json".format(name_without_extension), image_dir)
    middle_json_content = pipe_result.get_middle_json()
    pipe_result.dump_middle_json(md_writer, '{}_middle.json'.format(name_without_extension))
    
    print("处理完成: {}".format(pdf_file_path))
    return local_md_dir

def process_multiple_pdfs(pdf_dir=None, pdf_files=None, output_dir=None):
    """处理多个PDF文件
    
    Args:
        pdf_dir: 包含PDF文件的目录
        pdf_files: PDF文件路径列表
        output_dir: 输出目录
    """
    if pdf_dir is None and pdf_files is None:
        __dir__ = os.path.dirname(os.path.abspath(__file__))
        pdf_dir = os.path.join(__dir__, "pdfs")
    
    all_pdf_files = []
    if pdf_dir:
        all_pdf_files.extend(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if pdf_files:
        all_pdf_files.extend(pdf_files)
    
    if not all_pdf_files:
        print("未找到PDF文件")
        return
    
    results = []
    for pdf_file in all_pdf_files:
        print("处理文件: {}".format(pdf_file))
        result_dir = process_pdf_file(pdf_file, output_dir)
        results.append((pdf_file, result_dir))
    
    return results

if __name__ == "__main__":
    # 解析命令行参数来处理多个文件
    import sys
    
    if len(sys.argv) > 1:
        # 从命令行获取文件路径
        pdf_files = []
        output_dir = None
        
        for arg in sys.argv[1:]:
            if arg.startswith("--output="):
                output_dir = arg.split("=")[1]
            elif arg.endswith(".pdf"):
                pdf_files.append(arg)
        
        if pdf_files:
            process_multiple_pdfs(pdf_files=pdf_files, output_dir=output_dir)
        else:
            print("用法: python demo_muti.py 文件1.pdf 文件2.pdf ... [--output=输出目录]")
    else:
        # 默认处理
        process_multiple_pdfs(pdf_files=["pdfs/demo2.pdf", "成人肥胖食养指南（2024 年版）.pdf"], output_dir='output')

