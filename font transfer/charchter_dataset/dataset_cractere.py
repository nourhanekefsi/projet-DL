import os
from PIL import Image, ImageDraw, ImageFont
import random

# Directory setup
output_dir = "dataset/"  # Path to save generated characters
characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"  # Characters to generate
image_size = 128  # Size of each character image

# Color options for text and backgrounds
text_colors = ['black']
background_colors = ['white']

# Rotation angles
rotations = [0,180, 270]

# List of known system fonts (system fonts available in Windows)
common_fonts = [
    "arial.ttf", "times.ttf", "courier.ttf", "verdana.ttf", 
    "calibri.ttf", "consolas.ttf", "comic.ttf", "tahoma.ttf"
]

# Windows system font directory
fonts_dir = r"C:\Windows\Fonts"  # Directory of system fonts on Windows

def generate_character_images():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over the list of known font styles (common_fonts)
    for font_name in common_fonts:
        font_path = os.path.join(fonts_dir, font_name)
        
        # Vérifier si la police existe dans le répertoire système
        if not os.path.isfile(font_path):
            print(f"Font file {font_name} not found in {fonts_dir}. Skipping this font.")
            continue

        font_output_dir = os.path.join(output_dir, font_name)
        os.makedirs(font_output_dir, exist_ok=True)

        # Charger la police
        try:
            font = ImageFont.truetype(font_path, image_size // 2)  # Ajuster la taille de la police pour qu'elle tienne dans l'image
        except Exception as e:
            print(f"Could not load font {font_name}: {e}")
            continue

        # Générer les images pour chaque caractère
        for char in characters:
            for text_color in text_colors:
                for bg_color in background_colors:
                    if text_color != bg_color:
                        # Créer une image avec la couleur de fond
                        img = Image.new("RGB", (image_size, image_size), bg_color)
                        draw = ImageDraw.Draw(img)

                        # Obtenir la taille du texte et calculer la position pour centrer le texte
                        text_bbox = draw.textbbox((0, 0), char, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        position = ((image_size - text_width) / 2, (image_size - text_height) / 2)

                        # Dessiner le caractère
                        draw.text(position, char, fill=text_color, font=font)

                        # Appliquer les rotations
                        for angle in rotations:
                            rotated_img = img.rotate(angle, expand=True)
                            rotated_img = rotated_img.resize((image_size, image_size))

                            # Sauvegarder les images avec les rotations
                            char_filename = f"{char}_{text_color}_{bg_color}_{angle}.png"
                            rotated_img.save(os.path.join(font_output_dir, char_filename))

    print("Dataset generation complete!")

generate_character_images()
