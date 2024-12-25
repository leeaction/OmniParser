import base64
import json
import os
import tempfile
import uuid
import socket
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler
from PIL import Image
import io

import base64, os
from omniparser import Omniparser
from PIL import Image
import logging

PORT = 8899
DINO_LABLED_IMAGE_DIR = "dino_labled_images"

class MyApp(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_ip = self._get_local_ip()
        self._port = PORT
        self.initialize_app()

    def initialize_app(self):
        config = {
            'som_model_path': 'weights/icon_detect_v1_5/model_v1_5.pt',
            'device': 'cpu',
            'caption_model_path': 'Salesforce/blip2-opt-2.7b',
            'draw_bbox_config': {
                'text_scale': 0.8,
                'text_thickness': 2,
                'text_padding': 3,
                'thickness': 3,
            },
            'BOX_TRESHOLD': 0.05
        }

        os.makedirs(DINO_LABLED_IMAGE_DIR, exist_ok=True)
        self._parser = Omniparser(config)
        

    def _get_local_ip(self):
        # 创建UDP socket并获取真实IP地址（无需实际发送数据）
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # 连接到远程地址，不会实际发送数据
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"  # 备用地址
        
    def caption(self, image_input_base64: str, prompt: str):
        image_data = base64.b64decode(image_input_base64)
        image = Image.open(io.BytesIO(image_data))

        temp_dir = tempfile.gettempdir()
        random_filename = str(uuid.uuid4()) + ".png"
        temp_image_file = os.path.join(temp_dir, random_filename)
        image.save(temp_image_file)

        dino_labled_img, parsed_content_list = self._parser.parse(temp_image_file, prompt)

        # Save the image to a unique file
        unique_filename = f"{uuid.uuid4().hex}.png"
        out_image_save_path = os.path.join(DINO_LABLED_IMAGE_DIR, unique_filename)
        dino_labled_img.save(out_image_save_path,  format='PNG')
        print(f"Image saved to {out_image_save_path}")

        # Generate the URL with the local IP address
        image_url = f"http://{self._local_ip}:{self._port}/{DINO_LABLED_IMAGE_DIR}/{unique_filename}"    

        print('finish processing')

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
            prompt = request_data.get("prompt")

            # Validate input
            if not image_input_base64 or not isinstance(image_input_base64, str):
                self.set_status(400)
                self.write({"error": "Invalid image_input_base64"})
                return
            
            if not prompt or not isinstance(prompt, str):
                self.set_status(400)
                self.write({"error": "Invalid prompt"})
                return

            # Call caption function
            image_url, parsed_content_list = self.app.caption(
                image_input_base64,
                prompt,
            )

            # Create response with image URL
            response_data = {
                "image_url": image_url,
                "description": str(parsed_content_list),
            }

            self.set_status(200)
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps(response_data))
        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})
            logging.exception(e)

class ImageHandler(RequestHandler):
    def get(self, filename):
        try:
            image_path = os.path.join(DINO_LABLED_IMAGE_DIR, filename)

            if not os.path.exists(image_path):
                self.set_status(404)
                self.write({"error": "Image not found"})
                return

            with open(image_path, "rb") as f:
                self.set_header("Content-Type", "image/png")
                self.write(f.read())
        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})

if __name__ == "__main__":
    app = MyApp([
        (r"/caption", ProcessHandler, dict(app=None)),
        (r"/dino_labled_images/(.*)", ImageHandler),
    ])

    # Update handlers to pass app instance correctly
    app.add_handlers(".*$", [
        (r"/caption", ProcessHandler, dict(app=app)),
    ])

    app.listen(PORT)
    print("Server running on http://localhost:8899")
    IOLoop.current().start()
