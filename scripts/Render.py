import bpy
import os
import random
import json
import math
import yaml
from math import radians
from mathutils import Vector

#=== LOAD CONFIG ===
# Get the absolute path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full path to config.yaml
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# Load config
with open(CONFIG_PATH, "r") as cf:
    cfg = yaml.safe_load(cf)


MODELS_DIR        = cfg["models_dir"]
OUTPUT_DIR        = cfg["output_dir"]
RENDERS_PER_MODEL = cfg["renders_per_model"]
IMAGE_RESOLUTION  = cfg["image_resolution"]
SAMPLES           = cfg["samples"]

HDRI_PATH         = cfg["hdri"]["path"]
HDRI_STRENGTH     = cfg["hdri"]["strength"]

RGB_DIR    = os.path.join(OUTPUT_DIR, cfg["outputs"]["rgb_subdir"])
DEPTH_DIR  = os.path.join(OUTPUT_DIR, cfg["depth"]["png_subdir"])
NORMAL_DIR = os.path.join(OUTPUT_DIR, cfg["outputs"]["normal_subdir"])

# Create output directories if they don't exist
for d in (RGB_DIR, DEPTH_DIR, NORMAL_DIR):
    os.makedirs(d, exist_ok=True)

#=== FUNCTIONS ===
#Set the GPU device as the default rendering device
def enable_gpu_rendering():
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    for device in prefs.devices:
        if device.type in {'CUDA', 'OPTIX'}:
            device.use = True
            print(f"Enabled GPU device: {device.name}")
    bpy.context.scene.cycles.device = 'GPU'
    for scene in bpy.data.scenes:
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'

# HDRI setup with white background
def setup_hdri_white_background(hdri_path, strength):
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    nt = world.node_tree
    nt.links.clear()

    nodes = nt.nodes
    links = nt.links
    for node in nodes: nodes.remove(node)

    env_tex = nodes.new('ShaderNodeTexEnvironment')
    env_tex.image = bpy.data.images.load(hdri_path)
    bg_hdri = nodes.new('ShaderNodeBackground')
    bg_hdri.inputs['Strength'].default_value = strength
    links.new(env_tex.outputs['Color'], bg_hdri.inputs['Color'])

    bg_white = nodes.new('ShaderNodeBackground')
    bg_white.inputs['Color'].default_value = (1, 1, 1, 1)
    bg_white.inputs['Strength'].default_value = 1.0

    lp = nodes.new('ShaderNodeLightPath')
    mix_node = nodes.new('ShaderNodeMixShader')
    links.new(lp.outputs['Is Camera Ray'], mix_node.inputs['Fac'])
    links.new(bg_white.outputs['Background'], mix_node.inputs[1])
    links.new(bg_hdri.outputs['Background'], mix_node.inputs[2])

    out_node = nodes.new('ShaderNodeOutputWorld')
    links.new(mix_node.outputs['Shader'], out_node.inputs['Surface'])

# Overhead light setup with minimal overhead
def setup_minimal_overhead_light():
    e_lo, e_hi = cfg["lighting"]["overhead_energy_range"]
    s_lo, s_hi = cfg["lighting"]["overhead_size_range"]
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    light = bpy.context.object
    light.data.energy = random.uniform(e_lo, e_hi)
    light.data.size = random.uniform(s_lo, s_hi)

# Clear the scene of all objects
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# Set up camera aimed at the object with random distance and angle
def setup_camera_aimed_at(obj):
    bpy.ops.object.camera_add()
    cam = bpy.context.object

    # radius & spherical coords
    rad_mul = cfg["camera"]["radius_multiplier"]
    radius = max(obj.dimensions) * rad_mul

    t_lo, t_hi = cfg["camera"]["theta_deg_range"]
    p_lo, p_hi = cfg["camera"]["phi_deg_range"]
    theta = radians(random.uniform(t_lo, t_hi))
    phi   = radians(random.uniform(p_lo, p_hi))

    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    cam.location = (x, y, z)

    # focal length
    l_lo, l_hi = cfg["camera"]["lens_mm_range"]
    cam.data.lens = random.uniform(l_lo, l_hi)

    direction = obj.location - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam

    randomize_camera_settings(cam, obj)

# Export camera intrinsics as sidecar JSON
def export_intrinsics_json(basename, cam, output_dir):
    # Ensure camera exists
    if cam is None:
        raise RuntimeError(
            "export_intrinsics_json: 'cam' is None – "
            "make sure to call this *after* setup_camera_aimed_at()"
        )

    # Read depth_scale from config
    depth_scale = cfg["depth"]["depth_scale"]

    # Scene & image dimensions
    scene = bpy.context.scene
    width  = scene.render.resolution_x
    height = scene.render.resolution_y

    # Compute fx, fy, cx, cy
    sensor_w = cam.data.sensor_width
    sensor_h = cam.data.sensor_height
    focal_mm = cam.data.lens

    fx = (focal_mm / sensor_w) * width
    fy = (focal_mm / sensor_h) * height
    cx = width  / 2.0
    cy = height / 2.0

    # Build JSON structure
    data = {
        "camera_matrix": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "depth_scale":   depth_scale,
        "image_width":   width,
        "image_height":  height
    }

    # Write out
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{basename}_intrinsics.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

# This function applies a small random positional jitter to the object
def apply_positional_jitter(obj, jitter_range):
    offset = Vector((
        random.uniform(-jitter_range, jitter_range),
        random.uniform(-jitter_range, jitter_range),
        random.uniform(-jitter_range, jitter_range)
    ))
    obj.location += offset

# Randomize camera settings for depth of field sensor size and varies the pixel aspect ratio for more realistic rendering
def randomize_camera_settings(cam, target_obj):
    # Enable depth of field
    cam.data.dof.use_dof = True
    # Focus at object distance ±10%
    cam.data.dof.focus_distance = (target_obj.location - cam.location).length * random.uniform(0.9, 1.1)

    # aperture, sensor & aspect ratio
    af_lo, af_hi   = cfg["camera"]["dof_aperture_range"]
    sw_lo, sw_hi   = cfg["camera"]["sensor_width_range"]
    sh_lo, sh_hi   = cfg["camera"]["sensor_height_range"]
    pax_lo, pax_hi = cfg["camera"]["pixel_aspect_x_range"]
    pay_lo, pay_hi = cfg["camera"]["pixel_aspect_y_range"]

    # Random aperture (f-stop)
    cam.data.dof.aperture_fstop = random.uniform(af_lo, af_hi)

    # Vary sensor size (mm)
    cam.data.sensor_width      = random.uniform(sw_lo, sw_hi)
    cam.data.sensor_height     = random.uniform(sh_lo, sh_hi)

    # Vary pixel aspect ratio
    scene = bpy.context.scene
    scene.render.pixel_aspect_x = random.uniform(pax_lo, pax_hi)
    scene.render.pixel_aspect_y = random.uniform(pay_lo, pay_hi)

# Import STL file and apply transformations
def import_stl(filepath):
    jitter = cfg["object"]["jitter_range"]
    rxy_lo, rxy_hi = cfg["object"]["rotation_range_xy"]
    rz_lo, rz_hi   = cfg["object"]["rotation_range_z"]
    sc_lo, sc_hi   = cfg["object"]["scale_range"]
    bpy.ops.import_mesh.stl(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = Vector((0, 0, 0))
    obj.rotation_euler = (
        radians(random.uniform(rxy_lo, rxy_hi)),
        radians(random.uniform(rxy_lo, rxy_hi)),
        radians(random.uniform(rz_lo, rz_hi))
    )
    scale = random.uniform(sc_lo, sc_hi)
    obj.scale = (scale, scale, scale)
    bpy.ops.object.shade_smooth()
    apply_positional_jitter(obj, jitter)

    mat = bpy.data.materials.new(name="SteelMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    steel_tex = nodes.new("ShaderNodeTexImage")
    mat_cfg   = cfg["material"]
    tex_path  = mat_cfg["steel_texture"]
    steel_tex.image = bpy.data.images.load(tex_path)
    links.new(steel_tex.outputs['Color'], bsdf.inputs['Base Color'])
    metallic  = mat_cfg["metallic"]
    roughness = mat_cfg["roughness"]
    specular  = mat_cfg["specular"]
    bsdf.inputs["Metallic"].default_value   = metallic
    bsdf.inputs["Roughness"].default_value  = roughness
    bsdf.inputs["Specular"].default_value   = specular
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    return obj

# Set up compositing nodes for RGB, Normal, and 16-bit PNG Depth
def enable_render_passes(scene):
    scene.use_nodes= True
    tree = scene.node_tree
    # Clear all existing nodes & links
    tree.nodes.clear()
    tree.links.clear()

    # Render Layers input
    rl = tree.nodes.new(type='CompositorNodeRLayers')
    rl.location = (0, 0)

    # NORMAL → PNG OUTPUT
    normal_out = tree.nodes.new(type='CompositorNodeOutputFile')
    normal_out.location = (400, 200)
    normal_out.base_path = os.path.join(OUTPUT_DIR, cfg["outputs"]["normal_subdir"])
    normal_out.format.file_format = 'PNG'
    normal_out.format.color_depth  = '16'
    tree.links.new(rl.outputs['Normal'], normal_out.inputs[0])

    # DEPTH → NORMALIZE → MAPRANGE → PNG OUTPUT
    # 3a. Normalize raw depth values
    norm = tree.nodes.new(type='CompositorNodeNormalize')
    norm.location = (200, -100)
    tree.links.new(rl.outputs['Depth'], norm.inputs[0])

    # 3b. Map normalized [0–1] to full 16-bit range using clip_start/end
    map_range = tree.nodes.new(type='CompositorNodeMapRange')
    map_range.location = (400, -100)
    # Use overrides from config or fall back to camera settings
    cam = scene.camera.data
    clip_start = cfg["depth"]["clip_start"] or cam.clip_start
    clip_end   = cfg["depth"]["clip_end"]   or cam.clip_end
    map_range.inputs['From Min'].default_value = clip_start
    map_range.inputs['From Max'].default_value = clip_end
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 1.0
    tree.links.new(norm.outputs[0], map_range.inputs['Value'])

    # 3c. Output to 16-bit PNG
    depth_out = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_out.location = (600, -100)
    depth_out.base_path = os.path.join(OUTPUT_DIR, cfg["depth"]["png_subdir"])
    depth_out.format.file_format = 'PNG'
    depth_out.format.color_depth  = '16'
    depth_out.file_slots[0].path  = ""  # Blender will append frame/model name
    tree.links.new(map_range.outputs['Value'], depth_out.inputs[0])

# Set up Render settings
def configure_renderer(scene):
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_ambient_occlusion = True
    scene.render.resolution_x = IMAGE_RESOLUTION
    scene.render.resolution_y = IMAGE_RESOLUTION
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'Filmic'
    exp_lo, exp_hi = cfg["camera"]["exposure_range"]
    scene.view_settings.exposure = random.uniform(exp_lo, exp_hi)

# Render the image and save it
def render_image(basename, index):
    scene = bpy.context.scene
    configure_renderer(scene)

    # Update filenames for each pass
    rgb_path = os.path.join(RGB_DIR, f"{basename}_{index:03d}.png")
    scene.render.filepath = rgb_path

    # Set output paths for depth and normal
    for node in scene.node_tree.nodes:
        if isinstance(node, bpy.types.CompositorNodeOutputFile):
            if "depth" in node.base_path:
                node.file_slots[0].path = f"{basename}_{index:03d}"
            if "normal" in node.base_path:
                node.file_slots[0].path = f"{basename}_{index:03d}"

    bpy.ops.render.render(write_still=True)

# === MAIN ===
enable_gpu_rendering()

# Loop over each model
model_files = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".stl")]
for model_file in model_files:
    clear_scene()

    # Set up lighting and HDRI
    setup_minimal_overhead_light()
    setup_hdri_white_background(HDRI_PATH, HDRI_STRENGTH)

    model_path = os.path.join(MODELS_DIR, model_file)
    obj = import_stl(model_path)

    setup_camera_aimed_at(obj)
    enable_render_passes(bpy.context.scene)

    base_name = os.path.splitext(model_file)[0]

    for i in range(RENDERS_PER_MODEL):
        # Export intrinsics JSON
        export_intrinsics_json(f"{base_name}_{i:03d}", bpy.context.scene.camera, DEPTH_DIR)
        render_image(base_name, i)
