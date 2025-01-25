import gradio as gr
from PIL import Image, ImageDraw
import numpy as np

# Liste des polices disponibles
input_fonts = ["Times New Roman", "Lobster"]
output_fonts = ["Times New Roman", "Lobster"]

# Fonction simulée pour le transfert de style
def mock_font_style_transfer(image, input_font, output_font):
    # Redimensionner l'image au format attendu
    image = image.resize((256, 256)).convert("L")  # Conversion en niveaux de gris
    image_array = np.array(image)  # Convertir en tableau NumPy
    
    # Simuler une transformation en dessinant des lignes ou des motifs sur l'image
    simulated_image = Image.fromarray(image_array).convert("RGB")
    draw = ImageDraw.Draw(simulated_image)
    draw.rectangle([10, 10, 118, 118], outline="red", width=3)
    draw.text((20, 20), f"{output_font}", fill="blue")
    
    return simulated_image

# Configuration de l'interface Gradio
with gr.Blocks(css=".gradio-container {display: flex; justify-content: center; height: 100vh;}") as demo:
    gr.Markdown("<h1 style='text-align: center;'>Font Style Transfer (Mock Model)</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            # Left side: Upload Image + Dropdowns
            image_input = gr.Image(type="pil", label="Upload Your Image")
            input_font_dropdown = gr.Dropdown(input_fonts, label="Choose Input Font")
            output_font_dropdown = gr.Dropdown(output_fonts, label="Choose Output Font")
        with gr.Column(scale=1):
            # Right side: Transformed Image
            output_image = gr.Image(label="Transformed Image")

    with gr.Row():
        # Buttons below
        submit_button = gr.Button("Generate", elem_id="generate-button")

    def generate_and_display(image, input_font, output_font):
        transformed_image = mock_font_style_transfer(image, input_font, output_font)
        return transformed_image, transformed_image

    # Liaison des actions
    submit_button.click(generate_and_display, 
                        inputs=[image_input, input_font_dropdown, output_font_dropdown], 
                        outputs=[output_image, output_image])

demo.launch()
