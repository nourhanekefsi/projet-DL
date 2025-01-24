import nltk
import random
from nltk.corpus import brown
import os
from PIL import Image, ImageDraw, ImageFont

def generate_random_texts(num_texts=1500, min_length=5, max_length=25):
    words = brown.words()
    texts = []
    for _ in range(num_texts):
        length = random.randint(min_length, max_length)
        text = " ".join(random.choices(words, k=length))
        texts.append(text)
    return texts

def filter_unsupported_characters(text, font_path):
    """Filtre les caractères non pris en charge par une police."""
    font = ImageFont.truetype(font_path, 40)
    supported_text = "".join([char for char in text if font.getbbox(char)[2] > 0])
    return supported_text

def wrap_text(text, font, max_width):
    """Divise le texte en plusieurs lignes pour qu'il tienne dans l'image."""
    lines = []
    words = text.split()
    current_line = words[0]

    for word in words[1:]:
        test_line = f"{current_line} {word}"
        if font.getbbox(test_line)[2] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return "\n".join(lines)

def generate_text_image(paragraph, font_path, IMAGE_SIZE):
    """Génère une image avec un paragraphe rendu dans une police spécifique."""
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)
    max_width = IMAGE_SIZE[0] - 40

    # Divise le paragraphe en lignes
    wrapped_text = wrap_text(paragraph, font, max_width)

    # Crée l'image
    img = Image.new("RGB", IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(img)

    # Réduit la taille de la police si nécessaire
    while draw.textbbox((0, 0), wrapped_text, font=font)[3] > IMAGE_SIZE[1] - 40 and font_size > 10:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)

    # Centre le texte
    w, h = draw.textbbox((0, 0), wrapped_text, font=font)[2:]
    position = ((IMAGE_SIZE[0] - w) // 2, (IMAGE_SIZE[1] - h) // 2)
    draw.text(position, wrapped_text, font=font, fill="black")
    return img

def generate_dataset(FONT_DIR, output_dir, IMAGE_SIZE):
    """Génère un dataset avec des images pour les paires de polices."""
    fontA, fontB = os.path.join(FONT_DIR, 'arial.ttf'), os.path.join(FONT_DIR, 'calibri.ttf')
    os.makedirs(output_dir, exist_ok=True)
    paired_data_dir = os.path.join(output_dir, "pairs")
    os.makedirs(paired_data_dir, exist_ok=True)

    fontA_name = os.path.basename(fontA).split(".")[0]
    fontB_name = os.path.basename(fontB).split(".")[0]

    TEXT_PARAGRAPHS = generate_random_texts()
    # Génère des images pour la paire
    for j in range(1500):
        paragraph = TEXT_PARAGRAPHS[j]  # Paragraphe aléatoire

        # Filtre les caractères non pris en charge
        paragraphA = filter_unsupported_characters(paragraph, fontA)
        paragraphB = filter_unsupported_characters(paragraph, fontB)

        if not paragraphA or not paragraphB:
            continue  # Ignore si le texte devient vide

        input_img = generate_text_image(paragraphA, fontA, IMAGE_SIZE)
        target_img = generate_text_image(paragraphB, fontB, IMAGE_SIZE)

        # Sauvegarde les images
        input_img.save(os.path.join(paired_data_dir, f"{fontA_name}_{fontB_name}_input_{j}.png"))
        target_img.save(os.path.join(paired_data_dir, f"{fontA_name}_{fontB_name}_target_{j}.png"))

    print(f"Dataset généré dans : {output_dir}")

# Paramètres
FONT_DIR = r"C:\Windows\Fonts"  # Chemin vers les polices du système
OUTPUT_DIR = "font_transfer_dataset"  # Répertoire de sortie pour le dataset

IMAGE_SIZE = (256, 256)  # Dimensions des images
if __name__ == "__main__":
    generate_dataset(FONT_DIR, OUTPUT_DIR, IMAGE_SIZE)
