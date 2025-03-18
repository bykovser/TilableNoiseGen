bl_info = {
    "name": "Tilable NoiseGen",
    "author": "BykovSer",
    "version": (1, 4),
    "blender": (4, 3, 0),
    "location": "Image Editor > N Panel > Noise Tools",
    "description": "Generates procedural noise patterns and connects to shaders",
    "category": "Material",
}

import bpy
import math
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
        self.width = width
        self.height = height
        self.randseed = randseed
        self.gradients = [0.0] * (width * height * 2)
        
        rand = Random()
        rand.set_seed(randseed)
        for i in range(0, len(self.gradients), 2):
            angle = rand.next() * math.pi * 2
            self.gradients[i] = math.sin(angle)
            self.gradients[i + 1] = math.cos(angle)
    
    def dot(self, cell_x, cell_y, vx, vy):
        offset = (cell_x + cell_y * self.width) * 2
        return self.gradients[offset] * vx + self.gradients[offset + 1] * vy
    
    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)
    
    @staticmethod
    def s_curve(t):
        return t * t * (3 - 2 * t)
    
    def get_value(self, x, y):
        x_floor = math.floor(x)
        y_floor = math.floor(y)
        x_frac = x - x_floor
        y_frac = y - y_floor
        
        x0 = int(x_floor)
        y0 = int(y_floor)
        x1 = x0 + 1 if x0 < self.width - 1 else 0
        y1 = y0 + 1 if y0 < self.height - 1 else 0
        
        v00 = self.dot(x0, y0, x_frac, y_frac)
        v10 = self.dot(x1, y0, x_frac - 1, y_frac)
        v01 = self.dot(x0, y1, x_frac, y_frac - 1)
        v11 = self.dot(x1, y1, x_frac - 1, y_frac - 1)
        
        sx = self.s_curve(x_frac)
        sy = self.s_curve(y_frac)
        
        vx0 = self.lerp(v00, v10, sx)
        vx1 = self.lerp(v01, v11, sx)
        return self.lerp(vx0, vx1, sy)

# Create Perlin Noise Image
def create_perlin_noise_image(name, width, height, period, randseed, overwrite, correct_aspect):
    # Delete existing image if overwrite is enabled
    if overwrite and name in bpy.data.images:
        old_img = bpy.data.images[name]
        if old_img.size[0] == width and old_img.size[1] == height:
            # Reuse existing image
            img = old_img
        else:
            # Remove and create new
            bpy.data.images.remove(old_img)
            img = bpy.data.images.new(name, width, height)
    else:
        # Create new image
        img = bpy.data.images.new(name, width, height)
    
    # Generate noise pixels
    sampler = PerlinSampler2D(math.ceil(width / period), math.ceil(height / period), randseed)
    pixels = [0.0] * (width * height * 4)
    
    for j in range(height):
        for i in range(width):
            val = sampler.get_value(i / period, j / period)
            grayscale = (val + 1) / 2
            idx = (j * width + i) * 4
            pixels[idx:idx+3] = [grayscale] * 3
            pixels[idx + 3] = 1.0
    
    img.pixels = pixels
    img.update()

    # Set display aspect ratio
    if correct_aspect:
        if width > height:
            img.display_aspect[0] = height / width
            img.display_aspect[1] = 1.0
        else:
            img.display_aspect[0] = 1.0
            img.display_aspect[1] = width / height
    else:
        img.display_aspect[0] = 1.0
        img.display_aspect[1] = 1.0
    
    return img

def create_turbulence_image(name, width, height, period, randseed, depth, atten, use_color, use_alpha, absolute, overwrite, correct_aspect):
    if overwrite and name in bpy.data.images:
        old_img = bpy.data.images[name]
        if old_img.size[0] == width and old_img.size[1] == height:
            img = old_img
        else:
            bpy.data.images.remove(old_img)
            img = bpy.data.images.new(name, width, height)
    else:
        img = bpy.data.images.new(name, width, height)

    num_channels = 3 if use_color else 1
    if use_alpha:
        num_channels += 1

    raster = [0.0] * (width * height * num_channels)

    for k in range(num_channels):
        freq_inv = 1.0
        local_period_inv = 1.0 / period
        weight = 0.0
        
        for lvl in range(depth):
            sampler = PerlinSampler2D(
                math.ceil(width * local_period_inv),
                math.ceil(height * local_period_inv),
                randseed + k + lvl
            )
            
            for j in range(height):
                for i in range(width):
                    val = sampler.get_value(i * local_period_inv, j * local_period_inv)
                    idx = (j * width + i) * num_channels + k
                    raster[idx] += val * (freq_inv ** atten)
            
            weight += freq_inv ** atten
            freq_inv *= 0.5
            local_period_inv *= 2
        
        if weight > 0:
            weight_inv = 1.0 / weight
            for j in range(height):
                for i in range(width):
                    idx = (j * width + i) * num_channels + k
                    raster[idx] *= weight_inv

    pixels = [0.0] * (width * height * 4)
    for j in range(height):
        for i in range(width):
            idx = (j * width + i) * num_channels
            if use_color:
                r = raster[idx]
                g = raster[idx+1] if num_channels > 1 else raster[idx]
                b = raster[idx+2] if num_channels > 2 else raster[idx]
                a = raster[idx+3] if use_alpha else 1.0
            else:
                r = g = b = raster[idx]
                a = raster[idx+1] if use_alpha else 1.0

            if absolute:
                r, g, b, a = abs(r), abs(g), abs(b), abs(a)
            else:
                r = (r + 1) / 2
                g = (g + 1) / 2 if use_color else r
                b = (b + 1) / 2 if use_color else r
                a = (a + 1) / 2 if use_alpha else 1.0

            pixel_idx = (j * width + i) * 4
            pixels[pixel_idx:pixel_idx+4] = [r, g, b, a]

    img.pixels = pixels
    img.update()

    # Set display aspect ratio
    if correct_aspect:
        if width > height:
            img.display_aspect[0] = height / width
            img.display_aspect[1] = 1.0
        else:
            img.display_aspect[0] = 1.0
            img.display_aspect[1] = width / height
    else:
        img.display_aspect[0] = 1.0
        img.display_aspect[1] = 1.0
    
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
                self.correct_aspect
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

    def execute(self, context):
        if not hasattr(context.scene, 'noise_generator_last_image'):
            self.report({'ERROR'}, "No generated image exists")
            return {'CANCELLED'}
        
        image = bpy.data.images.get(context.scene.noise_generator_last_image)
        if not image:
            self.report({'ERROR'}, "Image not found")
            return {'CANCELLED'}
        
        # Get or create material
        obj = context.active_object
        if not obj:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}
        
        mat = obj.active_material
        if not mat:
            mat = bpy.data.materials.new(name="Procedural Material")
            obj.active_material = mat
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf_node = next((n for n in nodes if isinstance(n, bpy.types.ShaderNodeBsdfPrincipled)), None)
        
        # Create image texture node
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.image = image
        if bsdf_node:
            tex_node.location = (bsdf_node.location[0] - 300, bsdf_node.location[1] - 25)
        else:
            tex_node.location = (-300, 300)
        
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