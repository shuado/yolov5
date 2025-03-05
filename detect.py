# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlpackage          # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse # 解析命令行参数    
import csv # 读写CSV文件
import os # 操作系统相关操作
import platform # 获取操作系统信息
import sys # 系统路径管理
from pathlib import Path # 路径管理




# from utils.event_triggers import trigger_events
import torch # 深度学习框架

FILE = Path(__file__).resolve() # 获取当前文件的绝对路径
ROOT = FILE.parents[0]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将ROOT添加到系统路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend # 检测多后端
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams # 数据加载器
from utils.general import (
    LOGGER, # 日志记录
    Profile, # 性能分析
    check_file, # 检查文件
    check_img_size, # 检查图像尺寸
    check_imshow, # 检查图像显示
    check_requirements, # 检查要求
    colorstr, # 颜色字符串
    cv2, # 图像处理
    increment_path, # 路径增量
    non_max_suppression, # 非最大抑制
    print_args, # 打印参数
    scale_boxes, # 缩放框
    strip_optimizer, # 优化器
    xyxy2xywh, # 坐标转换
)
from utils.torch_utils import select_device, smart_inference_mode # 设备选择和智能推理模式


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # 模型权重路径
    source=ROOT / "data/images",  # 输入源路径
    data=ROOT / "data/coco128.yaml",  # 数据集配置文件
    imgsz=(640, 640),  # 推理图像尺寸
    conf_thres=0.25,  # 置信度阈值
    iou_thres=0.45,  # NMS IOU阈值
    max_det=1000,  # 每张图像最大检测数量
    device="",  # 设备选择（CPU/GPU）
    view_img=False,  # 是否显示结果
    save_txt=False,  # 是否保存文本结果
    save_format=0,  # 保存框坐标的格式（0表示YOLO格式，1表示Pascal-VOC格式）
    save_csv=False,  # 是否保存CSV格式结果
    save_conf=False,  # 是否在保存的标签中包含置信度
    save_crop=False,  # 是否保存裁剪的预测框
    nosave=False,  # 是否不保存图像/视频
    classes=None,  # 按类别过滤
    agnostic_nms=False,  # 是否使用类别无关的NMS
    augment=False,  # 是否使用增强推理
    visualize=False,  # 是否可视化特征
    update=False,  # 是否更新所有模型
    project=ROOT / "runs/detect",  # 结果保存项目路径
    name="exp",  # 结果保存名称
    exist_ok=False,  # 是否允许现有项目/名称存在
    line_thickness=3,  # 边界框线条粗细
    hide_labels=False,  # 是否隐藏标签
    hide_conf=False,  # 是否隐藏置信度
    half=False,  # 是否使用FP16半精度推理
    dnn=False,  # 是否使用OpenCV DNN进行ONNX推理
    vid_stride=1,  # 视频帧率步长
):
    """
    Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL. Default is 'yolov5s.pt'.
        source (str | Path): Input source, which can be a file, directory, URL, glob pattern, screen capture, or webcam
            index. Default is 'data/images'.
        data (str | Path): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        imgsz (tuple[int, int]): Inference image size as a tuple (height, width). Default is (640, 640).
        conf_thres (float): Confidence threshold for detections. Default is 0.25.
        iou_thres (float): Intersection Over Union (IOU) threshold for non-max suppression. Default is 0.45.
        max_det (int): Maximum number of detections per image. Default is 1000.
        device (str): CUDA device identifier (e.g., '0' or '0,1,2,3') or 'cpu'. Default is an empty string, which uses the
            best available device.
        view_img (bool): If True, display inference results using OpenCV. Default is False.
        save_txt (bool): If True, save results in a text file. Default is False.
        save_csv (bool): If True, save results in a CSV file. Default is False.
        save_conf (bool): If True, include confidence scores in the saved results. Default is False.
        save_crop (bool): If True, save cropped prediction boxes. Default is False.
        nosave (bool): If True, do not save inference images or videos. Default is False.
        classes (list[int]): List of class indices to filter detections by. Default is None.
        agnostic_nms (bool): If True, perform class-agnostic non-max suppression. Default is False.
        augment (bool): If True, use augmented inference. Default is False.
        visualize (bool): If True, visualize feature maps. Default is False.
        update (bool): If True, update all models' weights. Default is False.
        project (str | Path): Directory to save results. Default is 'runs/detect'.
        name (str): Name of the current experiment; used to create a subdirectory within 'project'. Default is 'exp'.
        exist_ok (bool): If True, existing directories with the same name are reused instead of being incremented. Default is
            False.
        line_thickness (int): Thickness of bounding box lines in pixels. Default is 3.
        hide_labels (bool): If True, do not display labels on bounding boxes. Default is False.
        hide_conf (bool): If True, do not display confidence scores on bounding boxes. Default is False.
        half (bool): If True, use FP16 half-precision inference. Default is False.
        dnn (bool): If True, use OpenCV DNN backend for ONNX inference. Default is False.
        vid_stride (int): Stride for processing video frames, to skip frames between processing. Default is 1.

    Returns:
        None

    Examples:
        ```python
        from ultralytics import run

        # Run inference on an image
        run(source='data/images/example.jpg', weights='yolov5s.pt', device='0')

        # Run inference on a video with specific confidence threshold
        run(source='data/videos/example.mp4', weights='yolov5s.pt', conf_thres=0.4, device='0')
        ```
    """
    
      # ==================== 添加 Metal 设备检查 ==================== 
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        LOGGER.info(f"🚀 Metal Acceleration (MPS) Enabled")
    else:
        device = torch.device("cpu")
        LOGGER.warning("Metal acceleration not available, using CPU")
    # ==================== 修改结束 ====================
    
    
    # 将source转换为字符串
    source = str(source)
    # 确定是否保存推理图像
    save_img = not nosave and not source.endswith(".txt")
    # 检查是否为文件
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 检查是否为URL
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    # 检查是否为网络摄像头
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    # 检查是否为屏幕截图
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories 目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    model = model.to(device)  # 🚨 关键修改：强制模型加载到 Metal 设备
        


    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size 检查图像尺寸

    # Dataloader 数据加载器
    bs = 1  # batch_size 批量大小
    if webcam:
        view_img = check_imshow(warn=True) # 检查图像显示
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) # 加载流数据集
        bs = len(dataset) # 获取批量大小
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) # 加载图像数据集
    vid_path, vid_writer = [None] * bs, [None] * bs # 视频路径和视频写入器

    # Run inference 运行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup 预热
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device)) # 已处理图像数、窗口列表、时间测量器
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)  # 🚨 数据传送到 Metal
            # im = torch.from_numpy(im).to(model.device) # 将numpy数组转换为torch张量并移动到模型设备
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 将uint8转换为fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0 归一化到0.0-1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim 扩展为批量维度
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0) # 将图像分割为多个张量

        # Inference 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False # 可视化路径
            if model.xml and im.shape[0] > 1:
                pred = None # 预测结果  
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0) # 将预测结果扩展为批量维度
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0) # 将预测结果连接起来
                pred = [pred, None] # 预测结果列表
            else:
                pred = model(im, augment=augment, visualize=visualize) # 推理
        # NMS 非最大抑制
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) # 非最大抑制

        # Second-stage classifier (optional) 第二阶段分类器
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file 定义CSV文件路径
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file 创建或追加到CSV文件
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions 处理预测结果
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count # 路径、图像副本、帧数
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0) # 路径、图像副本、帧数

            p = Path(p)  # to Path 转换为路径
            save_path = str(save_dir / p.name)  # im.jpg 保存路径
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt 标签路径
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string 打印字符串
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh 归一化增益
            imc = im0.copy() if save_crop else im0  # for save_crop 保存裁剪图像    
            annotator = Annotator(im0, line_width=line_thickness, example=str(names)) # 注释器
            if len(det):
                # Rescale boxes from img_size to im0 size 从img_size缩放到im0大小
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round() 

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class 每类检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string 添加到字符串  

                # Write results 写入结果        
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class 整数类
                    label = names[c] if hide_conf else f"{names[c]}" # 标签
                    confidence = float(conf) # 置信度
                    confidence_str = f"{confidence:.2f}" # 置信度字符串

                    if save_csv:  # 保存CSV文件
                        write_to_csv(p.name, label, confidence_str) 

                    if save_txt:  # Write to file 写入文件  
                        if save_format == 0:  # YOLO格式
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )  # normalized xywh
                        else:  # Pascal-VOC格式
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy 归一化坐标
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)  # label格式
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")  # 写入文件

                    if save_img or save_crop or view_img:  # Add bbox to image 添加边界框到图像
                        c = int(cls)  # integer class 整数类
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}") # 标签
                        annotator.box_label(xyxy, label, color=colors(c, True)) # 添加边界框到图像
                    if save_crop:  # 保存裁剪图像
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True) # 保存裁剪图像

            # Stream results 流结果
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections) 保存结果（带检测的图像）
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)  # 保存图像 
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # 新视频
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # 释放之前的视频写入器
                        if vid_cap:  # 视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # 强制结果视频为*.mp4后缀
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)  # 写入视频

        # Print time (inference-only) 打印时间（仅推理）
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # Print results 打印结果
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image 每张图像的速度
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t) 
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")  # 结果保存到{colorstr('bold', save_dir)}{s}
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning) 更新模型（修复SourceChangeWarning）


def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser() # 解析命令行参数
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL") # 模型路径或Triton URL
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)") # 文件/目录/URL/glob/屏幕/0(网络摄像头)
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path") # (可选)数据集配置文件路径
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w") # 推理尺寸h,w
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold") # 置信度阈值
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold") # NMS IOU阈值
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image") # 每张图像最大检测数量
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu") # 设备选择（CPU/GPU）
    parser.add_argument("--view-img", action="store_true", help="show results") # 显示结果
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt") # 保存结果到*.txt
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    ) # 是否保存框坐标（YOLO格式或Pascal-VOC格式）  
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format") # 保存结果为CSV格式
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels") # 保存置信度在--save-txt标签中    
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes") # 保存裁剪预测框  
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos") # 不保存图像/视频    
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3") # 按类别过滤 
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS") # 类别无关的NMS   
    parser.add_argument("--augment", action="store_true", help="augmented inference") # 增强推理    
    parser.add_argument("--visualize", action="store_true", help="visualize features") # 可视化特征    
    parser.add_argument("--update", action="store_true", help="update all models") # 更新所有模型       
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name") # 保存结果到项目/名称
    parser.add_argument("--name", default="exp", help="save results to project/name") # 保存结果到项目/名称 
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment") # 项目/名称存在，不增量   
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)") # 边界框厚度（像素）   
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels") # 隐藏标签 
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences") # 隐藏置信度    
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference") # 使用FP16半精度推理   
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference") # 使用OpenCV DNN进行ONNX推理    
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride") # 视频帧率步长
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand 扩展 
    print_args(vars(opt)) # 打印参数    
    return opt


def main(opt):
    """
    Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Note:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
