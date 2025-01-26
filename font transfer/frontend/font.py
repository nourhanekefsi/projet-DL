import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load models
generator = load_model('/Users/ryham/Documents/study/hpc/M2/Deep Learning/musicgen/projet-DL/font transfer/frontend/generator.h5')
classifier = load_model('/Users/ryham/Documents/study/hpc/M2/Deep Learning/musicgen/projet-DL/font transfer/frontend/alphabet_number_classifier.h5')

# Set the input image size
image_size = (28, 28)

# Font lists (for now, these don't affect generation logic, so you can expand them as needed)
input_fonts = ["Lobster"]
output_fonts = ["algerian"]

def preprocess_image(image):
    """Resize and preprocess the input image for model predictions."""
    try:
        # Resize and convert to grayscale
        image = image.resize(image_size).convert("L")  
        # Normalize pixel values
        img_array = np.array(image) / 255.0  
        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=(0, -1))  
        return img_array
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

def font_style_transfer(image, input_font, output_font):
    """
    Main function to transfer font style using the generator and classifier models.
    """
    try:
        # Preprocess the input image
        img_array = preprocess_image(image)

        # Predict the class with the classifier
        predictions = classifier.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        print(f"Predicted class: {predicted_class}")

        # Generate new text with the generator
        label = np.array([predicted_class])  # Label must be in the correct shape
        noise = np.random.normal(0, 1, size=(1, 100))  # Random noise for generator
        generated_image = generator.predict([noise, label])[0]  # Generate the image
        generated_image = (generated_image * 255).astype(np.uint8)  # Rescale to 0-255
        
        # Convert to PIL image for display
        output_image = Image.fromarray(generated_image.squeeze(), mode="L")
        return output_image
    except Exception as e:
        raise ValueError(f"Error during font style transfer: {e}")

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Font Style Transfer")
    gr.Markdown("Upload a grayscale image and transform its font style using deep learning.")

    # Input and output sections
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an image")
        output_image = gr.Image(label="Generated Image")

    # Font style dropdowns (placeholders; not currently affecting generation logic)
    with gr.Row():
        input_font_dropdown = gr.Dropdown(input_fonts, label="Input Font")
        output_font_dropdown = gr.Dropdown(output_fonts, label="Output Font")
    
    # Buttons
    with gr.Row():
        submit_button = gr.Button("Generate")
        download_button = gr.File(label="Download Generated Image", file_types=[".png"])

    def generate_and_save(image, input_font, output_font):
        """
        Wrapper function to handle font style transfer and save the output image.
        """
        try:
            print("Starting font style transfer...")
            print(f"Input font: {input_font}, Output font: {output_font}")
            
            # Perform the style transfer
            transformed_image = font_style_transfer(image, input_font, output_font)
            
            # Save the generated image temporarily
            filepath = "/tmp/generated_image.png"
            transformed_image.save(filepath)
            
            print("Font style transfer completed successfully.")
            return transformed_image, filepath
        except Exception as e:
            print(f"Error during generation: {e}")
            raise e

    # Connect the button to the generation function
    submit_button.click(
        generate_and_save, 
        inputs=[image_input, input_font_dropdown, output_font_dropdown], 
        outputs=[output_image, download_button]
    )

    # Instructions
    gr.Markdown("### Instructions")
    gr.Markdown("""
    1. Upload a grayscale text image (e.g., PNG or JPG).
    2. Choose the input and output font styles.
    3. Click 'Generate' to create a transformed image.
    4. Use the 'Download' button to save the result.
    """)

# Launch the app
demo.launch()
