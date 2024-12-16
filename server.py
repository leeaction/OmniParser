import base64
import json
import os
import uuid
import socket
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler
from PIL import Image
import io

import numpy as np
import torch


import base64, os
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image

import logging

TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

class MyApp(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_ip = self._get_local_ip()
        self.initialize_app()

    def initialize_app(self):
        print("Performing application initialization...")
        print(f"Local IP Address: {self.local_ip}")
        icon_detect_model =  "weights/icon_detect_v1_5/model_v1_5.pt"
        icon_caption_model = "florence2"
        self.yolo_model = get_yolo_model(model_path=icon_detect_model)
        if icon_caption_model == 'florence2':
            self.caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
        elif icon_caption_model == 'blip2':
            self.caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2")

    def _get_local_ip(self):
        # 创建UDP socket并获取真实IP地址（无需实际发送数据）
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # 连接到远程地址，不会实际发送数据
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"  # 备用地址
            

    def process(self, image_input_base64: str, box_threshold: float, iou_threshold: float, use_paddleocr: bool, imgsz: tuple, icon_process_batch_size: int) -> tuple:
        try:
            image_data = base64.b64decode(image_input_base64)
            image = Image.open(io.BytesIO(image_data))
            image_save_path = "imgs/temp_image.jpg"
            image.save(image_save_path)
            print(f"Image saved to {image_save_path}")
        except Exception as e:
            raise ValueError(f"Failed to save image: {e}")
        
        
        box_overlay_ratio = image.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        # import pdb; pdb.set_trace()
    
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
        text, ocr_bbox = ocr_bbox_rslt
        # print('prompt:', prompt)
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_save_path, self.yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz, batch_size=icon_process_batch_size)  
        out_image_data = base64.b64decode(dino_labled_img)
        out_image = Image.open(io.BytesIO(out_image_data))

        # Save the image to a unique file
        unique_filename = f"{uuid.uuid4().hex}.jpg"
        out_image_save_path = os.path.join(TEMP_IMAGE_DIR, unique_filename)
        out_image.save(out_image_save_path)
        print(f"Image saved to {out_image_save_path}")

        # Generate the URL with the local IP address
        image_url = f"http://{self.local_ip}:8899/{TEMP_IMAGE_DIR}/{unique_filename}"    

        print('finish processing')
        # parsed_content_list = '\n'.join(parsed_content_list)
        # parsed_content_list = '\n'.join([f'type: {x['type']}, content: {x["content"]}, interactivity: {x["interactivity"]}' for x in parsed_content_list])
        return image_url, parsed_content_list

class ProcessHandler(RequestHandler):
    def initialize(self, app):
        self.app = app

    def post(self):
        try:
            # Parse JSON request
            request_data = json.loads(self.request.body)

            # Extract parameters
            image_input_base64 = request_data.get("image_input_base64")
            box_threshold = float(request_data.get("box_threshold", 0.05))
            iou_threshold = float(request_data.get("iou_threshold", 0.1))
            use_paddleocr = bool(request_data.get("use_paddleocr", True))
            imgsz = tuple(request_data.get("imgsz", (640, 640)))
            icon_process_batch_size = int(request_data.get("icon_process_batch_size", 64))

            # Validate input
            if not image_input_base64 or not isinstance(image_input_base64, str):
                self.set_status(400)
                self.write({"error": "Invalid image_input_base64"})
                return

            # Call process function
            image_url, description = self.app.process(
                image_input_base64,
                box_threshold,
                iou_threshold,
                use_paddleocr,
                imgsz,
                icon_process_batch_size,
            )

            # Create response with image URL
            response_data = {
                "image_url": image_url,
                "description": description,
            }

            self.set_status(200)
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps(response_data))
        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})

class ImageHandler(RequestHandler):
    def get(self, filename):
        try:
            image_path = os.path.join(TEMP_IMAGE_DIR, filename)
            if not os.path.exists(image_path):
                self.set_status(404)
                self.write({"error": "Image not found"})
                return

            with open(image_path, "rb") as f:
                self.set_header("Content-Type", "image/jpeg")
                self.write(f.read())
        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})

if __name__ == "__main__":
    app = MyApp([
        (r"/process", ProcessHandler, dict(app=None)),
        (r"/temp_images/(.*)", ImageHandler),
    ])

    # Update handlers to pass app instance correctly
    app.add_handlers(".*$", [
        (r"/process", ProcessHandler, dict(app=app)),
    ])

    app.listen(8899)
    print("Server running on http://localhost:8899")
    IOLoop.current().start()
