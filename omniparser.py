from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
from typing import Dict, Tuple, List
import io
import base64


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


class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        
        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        # self.caption_model_processor = get_caption_model_processor(config['caption_model_path'], device=cofig['device'])
        # self.caption_model_processor['model'].to(torch.float32)

    def parse(self, image_path: str, prompt: str = None):
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=True)
        text, ocr_bbox = ocr_bbox_rslt

        draw_bbox_config = self.config['draw_bbox_config']
        BOX_TRESHOLD = self.config['BOX_TRESHOLD']
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_path, self.som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=False, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=None, ocr_text=text,use_local_semantics=False, prompt=prompt, batch_size=1)
        
        image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        # formating output
        return_list = [{'shape': {'x':coord[0], 'y':coord[1], 'width':coord[2], 'height':coord[3]},
                        'text': parsed_content_list[i]["content"], 'type':'text'} for i, (k, coord) in enumerate(label_coordinates.items()) if i < len(parsed_content_list)]
        return_list.extend(
            [{'shape': {'x':coord[0], 'y':coord[1], 'width':coord[2], 'height':coord[3]},
                        'text': 'None', 'type':'icon'} for i, (k, coord) in enumerate(label_coordinates.items()) if i >= len(parsed_content_list)]
              )

        return [image, return_list]
    

if __name__ == '__main__':
    parser = Omniparser(config)
    image_path = 'examples/saved_image_demo.png'
    prompt = "这张图片展示了微博应用页面里的UI元素，请用一句话说明这个UI元素展示的内容？"

    #  time the parser
    import time
    s = time.time()
    image, parsed_content_list = parser.parse(image_path, prompt)
    device = config['device']
    print(f'Time taken for Omniparser on {device}:', time.time() - s)
    

