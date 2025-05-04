import numpy as np
import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_obj
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import trimesh
from scipy.ndimage import gaussian_filter
from accelerate import Accelerator

# Initialize Hugging Face Accelerator for device management and mixed precision
accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device

# Converts a text prompt into a 3D model using the Shap-E pipeline
def text_to_3d_model(input_prompt: str, output_path: str = "output.obj"):
    # Load the Shap-E pipeline and move it to the correct device
    pipe = ShapEPipeline.from_pretrained(
        "openai/shap-e",
        torch_dtype=torch.float16
    )
    pipe.to(device)

    # Run the pipeline with specified parameters under mixed precision context
    with accelerator.autocast():
        result = pipe(
            input_prompt,
            guidance_scale=20.0,
            num_inference_steps=64,
            frame_size=512,
            output_type="mesh"
        )

    # Save the resulting mesh to an OBJ file
    export_to_obj(result["images"][0], output_path)
    print(f"3D model saved to {output_path}")


# Converts a 2D image into a 3D mesh by estimating depth and building a surface
def image_to_3d_model(input_image_path: str, output_path: str = "output_from_image.obj"):

    # Load image processor and depth estimation model
    depth_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large", use_fast=True)
    depth_model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)

    # Loads and converts image to numpy array (RGB format)
    def preprocess_image(image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        if image.mode == "P":
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")
        return np.array(image)

    # Estimates depth map from a given image using the depth model
    def estimate_depth(image: np.ndarray) -> np.ndarray:
        inputs = depth_processor(images=Image.fromarray(image), return_tensors="pt").to(device)
        with torch.no_grad(), accelerator.autocast():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Resize predicted depth to match input image and apply smoothing
        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

        # Convert to float32 to avoid scipy error with float16
        return gaussian_filter(depth_map.astype(np.float32), sigma=1.0)

    # Converts the RGB image and depth map into a 3D colored mesh using Trimesh
    def depth_to_colored_mesh(image: np.ndarray, depth_map: np.ndarray) -> trimesh.Trimesh:
        height, width = depth_map.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = x / width  # Normalize to [0,1]
        y = y / height
        z = depth_map / depth_map.max()  # Normalize depth

        # Combine normalized coordinates into vertex list
        vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        colors = image.reshape(-1, 3).astype(np.uint8)

        # Create face indices for the mesh grid
        faces = []
        for i in range(height - 1):
            for j in range(width - 1):
                v0 = i * width + j
                v1 = v0 + 1
                v2 = (i + 1) * width + j
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])

        # Create the Trimesh mesh with vertex colors
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = colors
        return mesh

    # Saves the generated mesh to an output file (e.g., .obj)
    def save_mesh(mesh: trimesh.Trimesh, path: str) -> None:
        mesh.export(path)
        print(f"Mesh saved to: {path}")

    # Full pipeline to convert an input image into a 3D model
    def generate_3d_from_image(image_path: str, path: str) -> None:
        try:
            image = preprocess_image(image_path)
            depth_map = estimate_depth(image)
            mesh = depth_to_colored_mesh(image, depth_map)
            save_mesh(mesh, path)
        except Exception as e:
            print(f"Error generating 3D model: {e}")

    generate_3d_from_image(input_image_path, output_path)
