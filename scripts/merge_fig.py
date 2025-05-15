import fire
from PIL import Image, ImageDraw, ImageFont
import os

def merge_images(category='bear'):
    input_dir = f'/opt/dlami/nvme/zhe/text2image_eval/scripts/figs/{category}'
    output_path = os.path.join(input_dir, f'{category}.pdf')
    image_order = ['emu3', 'januspro', 'sana', 'lumina', 'cogview', 'flux']
    images = []
    
    total_width = 0
    total_height = 0
    
    for name in image_order:
        img_path = os.path.join(input_dir, f'{name}.png')
        img = Image.open(img_path)
        total_width += img.width
        total_height += img.height
        images.append(img)
    
    avg_width = total_width // len(images)
    avg_height = total_height // len(images)
    
    target_ratio = avg_width / avg_height
    resized_images = []
    
    for img in images:
        current_ratio = img.width / img.height
        
        if current_ratio > target_ratio:
            new_width = int(avg_height * current_ratio)
            new_height = avg_height
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            left = (new_width - avg_width) // 2
            img = img.crop((left, 0, left + avg_width, new_height))
        else:
            new_width = avg_width
            new_height = int(avg_width / current_ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            top = (new_height - avg_height) // 2
            img = img.crop((0, top, new_width, top + avg_height))
            
        resized_images.append(img)
    
    result = Image.new('RGB', (avg_width * 3, avg_height * 2))
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    for idx, (img, name) in enumerate(zip(resized_images, image_order)):
        row = idx // 3
        col = idx % 3
        x = col * avg_width
        y = row * avg_height
        result.paste(img, (x, y))
        draw.text((x + 10, y + 10), name, fill='white', font=font, stroke_width=2, stroke_fill='black')
    
    result.save(output_path)
    return output_path

def process_all_categories():
    categories = ['bear', 'woman', 'hand', 'bottle', 'anime']
    for category in categories:
        merge_images(category)

if __name__ == '__main__':
    fire.Fire({
        'merge': merge_images,
        'all': process_all_categories
    })