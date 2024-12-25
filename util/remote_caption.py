import concurrent
import requests
import base64
import io

config = {
    'base_url': 'http://10.235.11.63:11434/api/chat',
    'model': 'minicpm-v:8b-2.6-fp16',
    'temperature': 0.3
}


def generate_api(images, query) -> list:
    captions = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image,image, query): i for i, image in enumerate(images)}
        
        for future in concurrent.futures.as_completed(futures):
            caption = future.result()
            captions.append(caption)
    
    return captions

def process_image(image, query) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # 指定保存的格式，例如 "JPEG", "PNG" 等
    encoded_string = base64.b64encode(buffer.getvalue()).decode('ascii')
    
    try:
        caption = chat(query, base64_img = encoded_string)
        if not caption:
            caption = "icon"
        return caption
    except Exception as e:
        return "icon"

def chat(prompt: str, full_base64_img: str = None, base64_img: str = None) -> str:
    url = config['base_url']

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer "
    }

    images = []
    if full_base64_img:
        images.append(full_base64_img)

    if base64_img:
        images.append(base64_img)

    payload = {
        "model": config['model'],
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images
            }
        ],
        "options": {
            "temperature": config['temperature'],
        },
        "stream": False
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    message = response.json()["message"]
    return message["content"]