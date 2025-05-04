import trimesh
import main

choice = input('Choose:\n1.Text-to-3D\n2.Image-to-3D\n')
if choice=='1':
    prompt = input('Enter prompt: ')
    main.text_to_3d_model(prompt)
    mesh = trimesh.load("output.obj")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    mesh.show()
if choice=='2':
    main.image_to_3d_model("sedan.png")
    mesh = trimesh.load("output_from_image.obj")
    mesh = trimesh.smoothing.filter_laplacian(mesh)
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    mesh.fill_holes()
    mesh.show()