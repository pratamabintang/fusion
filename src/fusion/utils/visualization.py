import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_detections(image: Image.Image | np.ndarray | torch.Tensor, detections: torch.Tensor | np.ndarray, labels: list = None, color=(0, 255, 0)) -> Image.Image:
    if isinstance(image, torch.Tensor):
        # Assuming CxHxW, float 0-1 or byte 0-255
        if image.is_floating_point():
            image = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        else:
            image = image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image.copy()
        
    draw = ImageDraw.Draw(img)
    
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()
        
    for det in detections:
        if len(det) >= 4:
            x1, y1, x2, y2 = det[:4]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            text = ""
            if len(det) >= 6:
                conf = det[4]
                cls = int(det[5])
                label_str = labels[cls] if labels and cls < len(labels) else str(cls)
                text = f"{label_str} {conf:.2f}"
            elif len(det) == 5:
                conf = det[4]
                text = f"{conf:.2f}"
                
            if text:
                draw.text((x1, max(y1 - 15, 0)), text, fill=color)
                
    return img

def create_side_by_side_vis(vis_img: Image.Image, therm_img: Image.Image, detections: torch.Tensor | np.ndarray, title: str = "Multispectral Detection") -> Image.Image:
    vis_res = draw_detections(vis_img, detections)
    therm_res = draw_detections(therm_img, detections)
    
    w1, h1 = vis_res.size
    w2, h2 = therm_res.size
    
    header_height = 40
    total_width = w1 + w2
    max_height = max(h1, h2) + header_height
    
    combined = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    
    draw = ImageDraw.Draw(combined)
    draw.text((w1 // 2 - 40, 10), "Visible (RGB)", fill=(0, 0, 0))
    draw.text((w1 + w2 // 2 - 40, 10), "Thermal (IR)", fill=(0, 0, 0))
    
    combined.paste(vis_res, (0, header_height))
    combined.paste(therm_res, (w1, header_height))
    
    return combined
