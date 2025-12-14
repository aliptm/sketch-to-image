"""
Generate synthetic training data (sketches and corresponding images)
This creates sample data for testing the training pipeline
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import argparse
import random


def create_simple_sketch(width=256, height=256):
    """Create a simple synthetic sketch"""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Random seed for reproducibility
    np.random.seed(random.randint(0, 10000))
    
    # Draw random shapes to create a sketch
    num_shapes = random.randint(3, 8)
    
    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle', 'line', 'polygon'])
        
        if shape_type == 'circle':
            x = random.randint(20, width - 20)
            y = random.randint(20, height - 20)
            radius = random.randint(15, 50)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                        outline='black', width=2)
        
        elif shape_type == 'rectangle':
            x1 = random.randint(10, width - 60)
            y1 = random.randint(10, height - 60)
            x2 = x1 + random.randint(30, 80)
            y2 = y1 + random.randint(30, 80)
            draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
        
        elif shape_type == 'line':
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            draw.line([x1, y1, x2, y2], fill='black', width=2)
        
        elif shape_type == 'polygon':
            num_points = random.randint(3, 6)
            points = []
            center_x = random.randint(50, width - 50)
            center_y = random.randint(50, height - 50)
            radius = random.randint(20, 60)
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                points.append((x, y))
            draw.polygon(points, outline='black', width=2)
    
    # Convert to grayscale for sketch-like appearance
    img = img.convert('L')
    return img


def create_beautiful_image(sketch, width=256, height=256):
    """Create a 'beautiful' version of the sketch with colors and effects"""
    # Start with the sketch
    base = sketch.convert('RGB')
    
    # Create a colored version
    colored = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(colored)
    
    # Get sketch edges (simple edge detection simulation)
    sketch_array = np.array(sketch)
    
    # Create gradient background
    for y in range(height):
        for x in range(width):
            # Create a gradient based on position
            r = int(200 + 55 * (x / width))
            g = int(200 + 55 * (y / height))
            b = int(220)
            colored.putpixel((x, y), (r, g, b))
    
    # Add colored regions based on sketch
    # Find regions in sketch and fill with colors
    sketch_array = np.array(sketch)
    threshold = 200  # Threshold for sketch lines
    
    # Create color palette
    colors = [
        (255, 200, 200),  # Light red
        (200, 255, 200),  # Light green
        (200, 200, 255),  # Light blue
        (255, 255, 200),  # Light yellow
        (255, 200, 255),  # Light magenta
        (200, 255, 255),  # Light cyan
    ]
    
    # Fill regions with colors
    for y in range(0, height, 5):
        for x in range(0, width, 5):
            if sketch_array[y, x] < threshold:
                # Near a sketch line, add color
                color = random.choice(colors)
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if random.random() > 0.7:  # Sparse coloring
                                colored.putpixel((nx, ny), color)
    
    # Blend with original sketch
    result = Image.blend(colored, base, 0.3)
    
    # Add some smoothing
    result = result.filter(ImageFilter.SMOOTH_MORE)
    
    # Enhance colors
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Color(result)
    result = enhancer.enhance(1.3)
    
    return result


def create_advanced_sketch(width=256, height=256):
    """Create a more advanced sketch with patterns"""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Create a scene-like sketch
    # Draw horizon
    horizon_y = height // 2 + random.randint(-30, 30)
    draw.line([0, horizon_y, width, horizon_y], fill='black', width=2)
    
    # Draw some objects
    # Sun/moon
    sun_x = random.randint(width - 80, width - 20)
    sun_y = random.randint(20, horizon_y - 20)
    draw.ellipse([sun_x - 15, sun_y - 15, sun_x + 15, sun_y + 15], 
                outline='black', width=2)
    
    # Trees or buildings
    for i in range(random.randint(2, 4)):
        x = random.randint(20, width - 20)
        # Building/tree shape
        height_obj = random.randint(40, 100)
        draw.rectangle([x - 15, horizon_y - height_obj, x + 15, horizon_y], 
                      outline='black', width=2)
        # Top detail
        if random.random() > 0.5:
            # Triangle top (tree)
            draw.polygon([(x - 15, horizon_y - height_obj), 
                         (x, horizon_y - height_obj - 20),
                         (x + 15, horizon_y - height_obj)], 
                        outline='black', width=2)
    
    # Some clouds or decorative elements
    for _ in range(random.randint(2, 5)):
        cloud_x = random.randint(0, width)
        cloud_y = random.randint(20, horizon_y - 40)
        for offset in [(0, 0), (10, 5), (-10, 5), (5, -5)]:
            draw.ellipse([cloud_x + offset[0] - 8, cloud_y + offset[1] - 8,
                         cloud_x + offset[0] + 8, cloud_y + offset[1] + 8],
                        outline='black', width=1)
    
    return img.convert('L')


def create_advanced_image(sketch, width=256, height=256):
    """Create a beautiful version of advanced sketch"""
    base = sketch.convert('RGB')
    colored = Image.new('RGB', (width, height))
    
    sketch_array = np.array(sketch)
    
    # Create sky gradient
    for y in range(height):
        for x in range(width):
            if y < height // 2:
                # Sky - blue gradient
                intensity = 1.0 - (y / (height / 2)) * 0.3
                r = int(135 * intensity)
                g = int(206 * intensity)
                b = int(250 * intensity)
            else:
                # Ground - green/brown gradient
                ground_y = y - height // 2
                intensity = 0.7 + (ground_y / (height / 2)) * 0.3
                r = int(139 * intensity)
                g = int(180 * intensity)
                b = int(100 * intensity)
            colored.putpixel((x, y), (r, g, b))
    
    # Add colors to objects
    colors = {
        'sun': (255, 255, 100),
        'tree': (34, 139, 34),
        'building': (139, 69, 19),
        'cloud': (255, 255, 255),
    }
    
    # Simple region coloring based on sketch
    for y in range(0, height, 3):
        for x in range(0, width, 3):
            if sketch_array[y, x] < 200:
                # Near sketch line
                if y < height // 2:
                    # Sky region - keep blue or add cloud white
                    if random.random() > 0.9:
                        colored.putpixel((x, y), colors['cloud'])
                else:
                    # Ground region - add green/brown
                    color_choice = random.choice([colors['tree'], colors['building']])
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if random.random() > 0.8:
                                    colored.putpixel((nx, ny), color_choice)
    
    # Blend
    result = Image.blend(colored, base, 0.2)
    result = result.filter(ImageFilter.SMOOTH)
    
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Color(result)
    result = enhancer.enhance(1.2)
    
    return result


def generate_dataset(output_dir, num_samples=100, image_size=256, advanced=False):
    """Generate a complete dataset"""
    sketch_dir = os.path.join(output_dir, 'train', 'sketches')
    image_dir = os.path.join(output_dir, 'train', 'images')
    
    os.makedirs(sketch_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    print(f"Generating {num_samples} training samples...")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Mode: {'Advanced' if advanced else 'Simple'}")
    print()
    
    for i in range(num_samples):
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_samples} samples...")
        
        # Generate sketch
        if advanced and i % 2 == 0:
            sketch = create_advanced_sketch(image_size, image_size)
            image = create_advanced_image(sketch, image_size, image_size)
        else:
            sketch = create_simple_sketch(image_size, image_size)
            image = create_beautiful_image(sketch, image_size, image_size)
        
        # Save files
        sketch_filename = f"sketch_{i+1:04d}.png"
        image_filename = f"image_{i+1:04d}.png"
        
        sketch_path = os.path.join(sketch_dir, sketch_filename)
        image_path = os.path.join(image_dir, image_filename)
        
        sketch.save(sketch_path)
        image.save(image_path)
    
    print(f"\n[SUCCESS] Generated {num_samples} samples successfully!")
    print(f"  Sketches: {sketch_dir}")
    print(f"  Images: {image_dir}")
    print(f"\nYou can now train your model with:")
    print(f"  python train.py --data_dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                       help='Output directory for generated dataset')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size (width and height)')
    parser.add_argument('--advanced', action='store_true',
                       help='Generate more complex sketches (scenes, objects)')
    
    args = parser.parse_args()
    
    generate_dataset(
        args.output_dir,
        args.num_samples,
        args.image_size,
        args.advanced
    )


if __name__ == "__main__":
    main()

