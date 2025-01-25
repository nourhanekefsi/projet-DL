import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Charger le modèle
model = tf.keras.models.load_model("model.h5")

# Liste des polices disponibles
input_fonts = ["Times New Roman", "Lobster"]
output_fonts = ["Times New Roman", "Lobster"]

# Fonction pour le traitement de l'image
def font_style_transfer(image, input_font, output_font):
    # Redimensionner l'image au format attendu par le modèle
    image = image.resize((256, 256)).convert("L")  # Conversion en niveaux de gris
    image_array = np.array(image) / 255.0  # Normalisation
    image_array = np.expand_dims(image_array, axis=(0, -1))  # Préparer pour le modèle
    
    # Utiliser le modèle pour générer une nouvelle image
    generated_image = model.predict(image_array)[0]
    generated_image = (generated_image * 255).astype(np.uint8)  # Re-normaliser pour l'affichage
    
    # Convertir en image PIL pour l'affichage
    output_image = Image.fromarray(generated_image)
    return output_image

# Configuration de l'interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Font Style Transfer")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload your image")
        output_image = gr.Image(label="Transformed Image")

    with gr.Row():
        input_font_dropdown = gr.Dropdown(input_fonts, label="Input Font")
        output_font_dropdown = gr.Dropdown(output_fonts, label="Output Font")
    
    with gr.Row():
        submit_button = gr.Button("Generate")
        download_button = gr.Button("Download")

    def generate_and_display(image, input_font, output_font):
        transformed_image = font_style_transfer(image, input_font, output_font)
        return transformed_image, transformed_image

    # Liaison des actions
    submit_button.click(generate_and_display, 
                        inputs=[image_input, input_font_dropdown, output_font_dropdown], 
                        outputs=[output_image, output_image])
    
    gr.Markdown("### Instructions")
    gr.Markdown("1. Upload a text image (PNG or JPG).\n2. Choose the input and output fonts.\n3. Click 'Generate' to view the transformed image.\n4. Click 'Download' to save the result.")
    
demo.launch()
