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

# Converts a text prompt into a 3D model using the Shap-E pipeline
def text_to_3d_model(input_prompt):
    pipe = ShapEPipeline.from_pretrained(
        "openai/shap-e",
        torch_dtype=torch.float16)

    pipe.to(accelerator.device)

    prompt = input_prompt
    with accelerator.autocast():
        result = pipe(prompt, guidance_scale=20.0, num_inference_steps=64, frame_size=512, output_type="mesh")

    export_to_obj(result.images[0], "output.obj")
    print("3D model saved to output.obj")


# Converts a 2D image into a 3D mesh by estimating depth and building a surface
def image_to_3d_model(input_image_path, output_path="output_from_image.obj"):

    # Load image processor and depth estimation model
    depth_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
    depth_model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large").to(accelerator.device)

    # Loads and converts image to numpy array (RGB format)
    def preprocess_image(image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return np.array(image)

    # Estimates depth map from a given image using the depth model
    def estimate_depth(image: np.ndarray) -> np.ndarray:
        inputs = depth_processor(images=Image.fromarray(image), return_tensors="pt").to(accelerator.device)
        with torch.no_grad(), accelerator.autocast():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False).squeeze().cpu().numpy()
        return gaussian_filter(depth_map, sigma=1.0)

    # Converts the RGB image and depth map into a 3D colored mesh using Trimesh
    def depth_to_colored_mesh(image: np.ndarray, depth_map: np.ndarray) -> trimesh.Trimesh:
        height, width = depth_map.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = x / width
        y = y / height
        z = depth_map / depth_map.max()

        vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        colors = image.reshape(-1, 3).astype(np.uint8)

        faces = []
        for i in range(height - 1):
            for j in range(width - 1):
                v0 = i * width + j
                v1 = v0 + 1
                v2 = (i + 1) * width + j
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = colors
        return mesh

    # Saves the generated mesh to an output file (e.g., .obj)
    def save_mesh(mesh: trimesh.Trimesh, output_path: str) -> None:
        mesh.export(output_path)
        print(f"Mesh saved to: {output_path}")

    # Full pipeline to convert an input image into a 3D model
    def generate_3d_from_image(image_path: str, output_path: str) -> None:
        try:
            image = preprocess_image(image_path)
            depth_map = estimate_depth(image)
            mesh = depth_to_colored_mesh(image, depth_map)
            save_mesh(mesh, output_path)
        except Exception as e:
            print(f"Error generating 3D model: {e}")

    generate_3d_from_image(input_image_path, output_path)
