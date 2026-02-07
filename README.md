Run the following blocks of code in kaggle GPU and use the UI developed to measure the live metrics.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import gradio as gr
import torch # or import tensorflow as tf
import numpy as np
from PIL import Image

# This ensures Gradio can display inside Kaggle
gr.close_all()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

segmentation_training.py

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def colorize_mask(mask):
    """Converts the 0-9 integer mask into a color image for the UI."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        color_mask[mask == i] = color
    return color_mask

def predict_terrain(input_image):
    # 1. Prepare image to match training (448x224 and Normalized)
    w, h = 448, 224
    img = input_image.resize((w, h))
    
    # Standard DINOv2 normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 2. THE FIX: Pass through Backbone first, then the Head
    model.eval()
    backbone.eval()
    with torch.no_grad():
        # Get tokens from DINOv2
        features = backbone.forward_features(img_tensor)["x_norm_patchtokens"]
        
        # Pass tokens to your ConvNeXt head
        logits = model(features)
        
        # Upscale to original size
        preds = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        mask = torch.argmax(preds, dim=1).squeeze().cpu().numpy()

    # 3. Visuals and Robot Logic
    colored_mask = colorize_mask(mask)
    
    # Count Rocks (Class 7) and Logs (Class 6)
    rock_pct = (np.count_nonzero(mask == 7) / mask.size) * 100
    log_pct = (np.count_nonzero(mask == 6) / mask.size) * 100
    
    if rock_pct > 5 or log_pct > 5:
        status = f"⚠️ DANGER: Obstacles! (Rocks: {rock_pct:.1f}%, Logs: {log_pct:.1f}%)"
    else:
        status = "✅ PATH CLEAR: Safe to proceed."
    
    return colored_mask, status

    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    import gradio as gr

# 1. Create a Legend HTML string for the Judge
legend_html = """
<div style="display: flex; flex-wrap: wrap; gap: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 8px; border: 1px solid #ddd;">
    <div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: black; margin-right: 5px; border: 1px solid #000;"></div><span>Background</span></div>
    <div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: darkgreen; margin-right: 5px;"></div><span>Trees</span></div>
    <div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: forestgreen; margin-right: 5px;"></div><span>Lush Bushes</span></div>
    <div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: goldenrod; margin-right: 5px;"></div><span>Dry Grass</span></div>
    <div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: saddlebrown; margin-right: 5px;"></div><span>Dry Bushes</span></div>
    <div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: gray; margin-right: 5px;"></div><span>Clutter</span></div>
    <div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: #FF4500; margin-right: 5px;"></div><span>Logs</span></div>
    <div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: red; margin-right: 5px;"></div><span>Rocks</span></div>
    <div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: sandybrown; margin-right: 5px;"></div><span>Landscape</span></div>
    <div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: skyblue; margin-right: 5px;"></div><span>Sky</span></div>
</div>
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏜️ Mars Rover: Desert Terrain Navigator")
    gr.Markdown("This interface provides real-time semantic segmentation for off-road navigation.")
    
    # Add the Legend here
    gr.HTML(legend_html)
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Image(type="pil", label="Robot View")
            submit_btn = gr.Button("Analyze Terrain", variant="primary")
        
        with gr.Column():
            output_mask = gr.Image(label="AI Segmentation (Color-Coded)")
            analysis_box = gr.Textbox(label="Robot Navigation Decision", interactive=False)

    submit_btn.click(
        fn=predict_terrain, 
        inputs=input_box, 
        outputs=[output_mask, analysis_box]
    )

# Fix for the Error: We use max_threads and disable the "queue" if you aren't doing heavy batches
# This often silences the asyncio event loop errors in Kaggle.
demo.launch(share=True, max_threads=1)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

segmentation_testing.py
