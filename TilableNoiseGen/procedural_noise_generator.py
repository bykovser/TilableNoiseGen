bl_info = {
    "name": "Tilable NoiseGen",
    "author": "BykovSer",
    "version": (1, 6),
    "blender": (4, 3, 0),
    "location": "Image Editor > N Panel > Noise Tools",
    "description": "Generates procedural noise patterns and connects to shaders",
    "category": "Material",
}

import bpy
import math
import numpy as np
from bpy.types import Operator, Panel
from bpy.props import IntProperty, FloatProperty, BoolProperty, StringProperty

# Random Number Generator
class Random:
    def __init__(self):
        self.m = 2147483647  # 2^31 - 1
        self.a = 16807       # 7^5; primitive root of m
        self.q = 127773      # m / a
        self.r = 2836        # m % a
        self.seed = 1
    
    def set_seed(self, seed):
        if seed <= 0:
            seed = -(seed % (self.m - 1)) + 1
        if seed > self.m - 1:
            seed = self.m - 1
        self.seed = seed
    
    def next_long(self):
        res = self.a * (self.seed % self.q) - self.r * (self.seed // self.q)
        if res <= 0:
            res += self.m
        self.seed = res
        return res
    
    def next(self):
        return self.next_long() / self.m

# Perlin Noise Sampler
class PerlinSampler2D:
    def __init__(self, width, height, randseed):
        self.width = int(width)
        self.height = int(height)
        self.randseed = randseed
        
        rand = Random()
        rand.set_seed(randseed)
        angles = np.array([rand.next() * math.pi * 2 for _ in range(width * height)])
        self.gradients = np.column_stack([np.sin(angles), np.cos(angles)]).astype(np.float32)

    # ADD THESE STATIC METHODS
    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)
    
    @staticmethod
    def s_curve(t):
        return t * t * (3 - 2 * t)

    def dot(self, cell_x, cell_y, vx, vy):
        cell_x = cell_x % self.width
        cell_y = cell_y % self.height
        offsets = cell_x + cell_y * self.width
        return self.gradients[offsets, 0] * vx + self.gradients[offsets, 1] * vy

    def get_value_vectorized(self, x, y):
        x_floor = np.floor(x).astype(int)
        y_floor = np.floor(y).astype(int)
        x_frac = x - x_floor
        y_frac = y - y_floor

        x0 = x_floor % self.width
        y0 = y_floor % self.height
        x1 = (x0 + 1) % self.width
        y1 = (y0 + 1) % self.height

        v00 = self.dot(x0, y0, x_frac, y_frac)
        v10 = self.dot(x1, y0, x_frac - 1, y_frac)
        v01 = self.dot(x0, y1, x_frac, y_frac - 1)
        v11 = self.dot(x1, y1, x_frac - 1, y_frac - 1)

        sx = self.s_curve(x_frac)  # Now correctly references static method
        sy = self.s_curve(y_frac)
        return self.lerp(self.lerp(v00, v10, sx), self.lerp(v01, v11, sx), sy)

# Create Perlin Noise Image
def create_perlin_noise_image(name, width, height, period, randseed, overwrite, correct_aspect, use_color, use_alpha, absolute):
    # Image handling
    if overwrite and name in bpy.data.images:
        old_img = bpy.data.images[name]
        if old_img.size[0] == width and old_img.size[1] == height:
            img = old_img
        else:
            bpy.data.images.remove(old_img)
            img = bpy.data.images.new(name, width, height)
    else:
        img = bpy.data.images.new(name, width, height)

    # Channel setup
    num_channels = 3 if use_color else 1
    if use_alpha:
        num_channels += 1

    # Generate coordinate grid
    j, i = np.meshgrid(np.arange(width), np.arange(height))
    raster = np.zeros((height, width, num_channels), dtype=np.float32)

    # Generate noise per channel
    for k in range(num_channels):
        channel_seed = randseed + k * 1000  # Unique seed per channel
        sampler = PerlinSampler2D(
            math.ceil(width/period),
            math.ceil(height/period),
            channel_seed
        )
        
        x_coords = j / period
        y_coords = i / period
        noise = sampler.get_value_vectorized(x_coords, y_coords)
        raster[..., k] = noise

    # Post-processing
    if absolute:
        raster = np.abs(raster)
    else:
        raster = (raster + 1) / 2

    # Create pixel array
    pixels = np.zeros((height, width, 4), dtype=np.float32)
    if use_color:
        pixels[..., :3] = raster[..., :3]
        if num_channels > 3:
            pixels[..., 3] = raster[..., 3]
    else:
        pixels[..., :3] = raster[..., 0][..., np.newaxis]
    
    if not use_alpha:
        pixels[..., 3] = 1.0

    # Assign pixels
    img.pixels.foreach_set(pixels.ravel())
    img.update()

    # Aspect ratio
    if correct_aspect:
        img.display_aspect = (1.0, height/width) if width > height else (width/height, 1.0)
    else:
        img.display_aspect = (1.0, 1.0)
    
    return img

def create_turbulence_image(name, width, height, period, randseed, depth, atten, use_color, use_alpha, absolute, overwrite, correct_aspect):
    # Image handling (same as Perlin)
    if overwrite and name in bpy.data.images:
        old_img = bpy.data.images[name]
        if old_img.size[0] == width and old_img.size[1] == height:
            img = old_img
        else:
            bpy.data.images.remove(old_img)
            img = bpy.data.images.new(name, width, height)
    else:
        img = bpy.data.images.new(name, width, height)

    # Channel setup
    num_channels = 3 if use_color else 1
    if use_alpha:
        num_channels += 1

    # Generate coordinate grid
    j, i = np.meshgrid(np.arange(width), np.arange(height))
    raster = np.zeros((height, width, num_channels), dtype=np.float32)
    weight_total = 0.0

    # Multi-octave generation
    for lvl in range(depth):
        freq = 2 ** lvl
        amplitude = (1.0/freq) ** atten
        local_period = period / freq
        
        for k in range(num_channels):
            channel_seed = randseed + k * 1000 + lvl * 10000  # Unique per channel/octave
            sampler = PerlinSampler2D(
                math.ceil(width / local_period),
                math.ceil(height / local_period),
                channel_seed
            )
            
            x_coords = j / local_period
            y_coords = i / local_period
            noise = sampler.get_value_vectorized(x_coords, y_coords)
            raster[..., k] += noise * amplitude
        
        weight_total += amplitude

    # Normalize and process
    raster /= weight_total
    if absolute:
        raster = np.abs(raster)
    else:
        raster = (raster + 1) / 2

    # Pixel packaging (same as Perlin)
    pixels = np.zeros((height, width, 4), dtype=np.float32)
    if use_color:
        pixels[..., :3] = raster[..., :3]
        if num_channels > 3:
            pixels[..., 3] = raster[..., 3]
    else:
        pixels[..., :3] = raster[..., 0][..., np.newaxis]
    
    if not use_alpha:
        pixels[..., 3] = 1.0

    img.pixels.foreach_set(pixels.ravel())
    img.update()

    # Aspect ratio
    if correct_aspect:
        img.display_aspect = (1.0, height/width) if width > height else (width/height, 1.0)
    else:
        img.display_aspect = (1.0, 1.0)
    
    return img
# Operator to Generate Perlin Noise
class NOISE_OT_generate_perlin(Operator):
    bl_idname = "noise.generate_perlin"
    bl_label = "Generate Perlin Noise"
    bl_options = {'REGISTER', 'UNDO'}

    # Operator properties
    image_name: StringProperty(
        name="Image Name",
        default="PerlinNoise",
    )
    overwrite: BoolProperty(
        name="Overwrite Existing",
        default=True,
        description="Replace existing image with the same name"
    )
    depth: IntProperty(
        name="Depth",
        default=4,
        min=1,
        max=8,
        description="Number of noise layers"
    )
    atten: FloatProperty(
        name="Attenuation",
        default=0.5,
        min=0.1,
        max=1.0,
        description="Amplitude reduction per layer"
    )
    use_color: BoolProperty(
        name="RGB",
        default=False,
        description="Generate separate noise for each color channel"
    )
    use_alpha: BoolProperty(
        name="Alpha",
        default=False,
        description="Generate alpha channel noise"
    )
    absolute: BoolProperty(
        name="Absolute",
        default=False,
        description="Use absolute values for contrast"
    )
    turbulence: BoolProperty(
        name="Turbulence",
        default=False,
        description="Enable multi-layer turbulence"
    )
    correct_aspect: BoolProperty(
        name="Correct Aspect Ratio",
        default=True,
        description="Adjust display aspect ratio based on image dimensions"
    )

    width: IntProperty(default=512, min=64, max=8192)
    height: IntProperty(default=512, min=64, max=8192)
    period: FloatProperty(default=64.0, min=1.0, max=1000.0)
    seed: IntProperty(default=1, min=0)

    def execute(self, context):
        if self.turbulence:
            image = create_turbulence_image(
                self.image_name,
                self.width,
                self.height,
                self.period,
                self.seed,
                self.depth,
                self.atten,
                self.use_color,
                self.use_alpha,
                self.absolute,
                self.overwrite,
                self.correct_aspect
            )
        else:
            image = create_perlin_noise_image(
                self.image_name,
                self.width,
                self.height,
                self.period,
                self.seed,
                self.overwrite,
                self.correct_aspect,
                self.use_color,
                self.use_alpha,
                self.absolute
            )
        
        # Set the active image in the Image Editor
        if context.space_data and context.space_data.type == 'IMAGE_EDITOR':
            context.space_data.image = image
        
        self.report({'INFO'}, f"Image updated: {image.name}")
        context.scene.noise_generator_last_image = image.name
        context.scene.noise_image_name = image.name
        context.scene.noise_overwrite = True
        image.pack()
        image.colorspace_settings.name = 'Non-Color'
        return {'FINISHED'}

# Operator to Add Noise to Shader
class NOISE_OT_add_to_shader(Operator):
    bl_idname = "noise.add_to_shader"
    bl_label = "Add to Active Shader"
    bl_description = "Connect generated image to active material"
    bl_options = {'REGISTER', 'UNDO'}

    def get_absolute_location(self, node):
        """Calculate absolute location considering all parent frames/groups"""
        abs_location = (node.location)
        parent = node.parent
        while parent:
            abs_location += (parent.location)
            parent = parent.parent
        return abs_location

    def find_parent_tree(self, node):
        """Find the root node tree through parent groups"""
        if node.id_data.users == 1:  # Check if it's a node group
            for mat in bpy.data.materials:
                if mat.node_tree:
                    for n in mat.node_tree.nodes:
                        if n.type == 'GROUP' and n.node_tree == node.id_data:
                            return self.find_parent_tree(n)
        return node.id_data

    def get_active_node(self, context):
        """Safely get the active node from context"""
        # Try to get from node editor first
        for area in context.screen.areas:
            if area.type == 'NODE_EDITOR':
                if area.spaces.active and area.spaces.active.node_tree:
                    return area.spaces.active.node_tree.nodes.active
        
        # Fallback to material's active node
        if context.object and context.object.active_material:
            return context.object.active_material.node_tree.nodes.active
        
        return None

    def execute(self, context):
        if not hasattr(context.scene, 'noise_generator_last_image'):
            self.report({'ERROR'}, "No generated image exists")
            return {'CANCELLED'}
        
        image = bpy.data.images.get(context.scene.noise_generator_last_image)
        if not image:
            self.report({'ERROR'}, "Image not found")
            return {'CANCELLED'}

        obj = context.object
        if not obj or not obj.active_material:
            self.report({'ERROR'}, "No active object or material")
            return {'CANCELLED'}

        mat = obj.active_material
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Get the active node safely
        active_node = self.get_active_node(context)
        node_tree = mat.node_tree

        # If inside a node group, find the parent tree
        if active_node and active_node.id_data != mat.node_tree:
            node_tree = self.find_parent_tree(active_node)

        # Create texture node in the correct tree
        tex_node = node_tree.nodes.new('ShaderNodeTexImage')
        tex_node.image = image

        # Positioning logic
        if active_node:
            # Direct relative positioning
            if active_node.parent and active_node.parent.type == 'FRAME':
                frame = active_node.parent
                tex_node.parent = frame  # Parent first
                # Use frame-relative coordinates directly
                tex_node.location = (active_node.location.x - 300, 
                                    active_node.location.y)
            else:
                # Standard positioning
                tex_node.location = (active_node.location.x - 300,
                                    active_node.location.y)
        else:
            # Fallback positioning
            output_node = next((n for n in node_tree.nodes 
                              if isinstance(n, bpy.types.ShaderNodeOutputMaterial)), None)
            if output_node:
                tex_node.location = (output_node.location.x - 300, 
                                   output_node.location.y)

        # Ensure frame expansion
        if tex_node.parent and tex_node.parent.type == 'FRAME':
            tex_node.parent.update()

        # Force UI update
        context.area.tag_redraw()
        
        return {'FINISHED'}

# Panel in Image Editor
class NOISE_PT_main_panel(Panel):
    bl_label = "Tilable NoiseGen"
    bl_idname = "NOISE_PT_main_panel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Noise Tools"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Image Name and Overwrite
        box = layout.box()
        box.prop(scene, "noise_image_name", text="Name")
        box.prop(scene, "noise_overwrite", toggle=True)
        
        #Image settings
        box = layout.box()
        box.label(text="Image settings")
        col = box.column(align=True)
        col.prop(scene, "noise_correct_aspect", text="display as 1x1")
        col.prop(scene, "noise_width", text="Width")
        col.prop(scene, "noise_height", text="Height")

        # Noise Settings
        box = layout.box()
        box.label(text="Noise Settings")
        col = box.column(align=True)
        col.prop(scene, "noise_seed", text = "Seed")
        col.prop(scene, "noise_period", text = "Scale")

        col = box.column(align=True)
        col.prop(scene, "noise_turbulence", text = "Use depth")
        col.prop(scene, "noise_depth", text = "Depth details")
        col.prop(scene, "noise_atten", text = "Mix details")


        box = layout.box()
        box.label(text="Other")
        col = box.column(align=True)
        col.prop(scene, "noise_use_color", text = "RGB")
        col.prop(scene, "noise_use_alpha", text = "Alpha")
        col.prop(scene, "noise_absolute", text = "Groovy")
        
        # Generate Button
        op = layout.operator("noise.generate_perlin")
        op.image_name = scene.noise_image_name
        op.overwrite = scene.noise_overwrite

        op.correct_aspect = scene.noise_correct_aspect
        op.width = scene.noise_width
        op.height = scene.noise_height

        op.seed = scene.noise_seed
        op.period = scene.noise_period

        op.turbulence = scene.noise_turbulence
        op.depth = scene.noise_depth
        op.atten = scene.noise_atten

        op.use_color = scene.noise_use_color
        op.use_alpha = scene.noise_use_alpha
        op.absolute = scene.noise_absolute
        
        # Add to Shader Button
        layout.operator("noise.add_to_shader")

# Register and Unregister
def register():
    bpy.utils.register_class(NOISE_OT_generate_perlin)
    bpy.utils.register_class(NOISE_OT_add_to_shader)
    bpy.utils.register_class(NOISE_PT_main_panel)
    
    # Scene properties for UI
    bpy.types.Scene.noise_image_name = StringProperty(
        name="Image Name",
        default="PerlinNoise",
    )
    bpy.types.Scene.noise_overwrite = BoolProperty(
        name="Overwrite",
        default=True
    )
    bpy.types.Scene.noise_width = IntProperty(default=512, min=64, max=8192)
    bpy.types.Scene.noise_height = IntProperty(default=512, min=64, max=8192)
    bpy.types.Scene.noise_period = FloatProperty(default=64.0, min=1.0, max=1000.0)
    bpy.types.Scene.noise_seed = IntProperty(default=1, min=0)
    bpy.types.Scene.noise_generator_last_image = StringProperty()
    bpy.types.Scene.noise_depth = IntProperty(default=4, min=1, max=8)
    bpy.types.Scene.noise_atten = FloatProperty(default=0.5, min=0.1, max=1.0)
    bpy.types.Scene.noise_use_color = BoolProperty(default=False)
    bpy.types.Scene.noise_use_alpha = BoolProperty(default=False)
    bpy.types.Scene.noise_absolute = BoolProperty(default=False)
    bpy.types.Scene.noise_turbulence = BoolProperty(default=False)
    bpy.types.Scene.noise_correct_aspect = BoolProperty(default=True)

def unregister():
    bpy.utils.unregister_class(NOISE_OT_generate_perlin)
    bpy.utils.unregister_class(NOISE_OT_add_to_shader)
    bpy.utils.unregister_class(NOISE_PT_main_panel)
    
    # Remove scene properties
    del bpy.types.Scene.noise_image_name
    del bpy.types.Scene.noise_overwrite
    del bpy.types.Scene.noise_width
    del bpy.types.Scene.noise_height
    del bpy.types.Scene.noise_period
    del bpy.types.Scene.noise_seed
    del bpy.types.Scene.noise_generator_last_image
    del bpy.types.Scene.noise_depth    
    del bpy.types.Scene.noise_attenmin
    del bpy.types.Scene.noise_use_color
    del bpy.types.Scene.noise_use_alpha
    del bpy.types.Scene.noise_absolute
    del bpy.types.Scene.noise_turbulence
    del bpy.types.Scene.noise_correct_aspect


if __name__ == "__main__":
    register()