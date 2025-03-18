# Noise Generator Add-on for Blender

This Blender add-on generates procedural Perlin noise and turbulence textures, which can be used for creating materials, textures, or other visual effects. The generated textures can be directly applied to the active material or saved as images.

## Features

- **Perlin Noise Generation**: Create seamless Perlin noise textures with customizable dimensions, scale, and seed.
- **Turbulence Noise**: Generate multi-layered turbulence noise with adjustable depth, attenuation, and color options.
- **Aspect Ratio Correction**: Automatically adjust the display aspect ratio for non-square textures.
- **Overwrite Existing Images**: Option to replace existing images with the same name.
- **Shader Integration**: Easily connect generated textures to the active material's shader nodes.
- **Customizable Parameters**: Control noise properties such as scale, depth, attenuation, color, alpha, and absolute values.

## Installation

1. Download the `.py` file containing the script.
2. Open Blender and go to `Edit > Preferences > Add-ons`.
3. Click `Install...` and select the downloaded `.py` file.
4. Enable the add-on by checking the box next to its name.

## Usage

### Generating Noise Textures

1. Open the `Image Editor` in Blender.
2. Navigate to the `Noise Tools` panel in the sidebar (press `N` to open the sidebar if it's not visible).
3. Configure the noise settings:
   - **Image Name**: Name of the generated image.
   - **Overwrite Existing**: Replace an existing image with the same name.
   - **Width/Height**: Dimensions of the generated texture.
   - **Correct Aspect Ratio**: Adjust the display aspect ratio for non-square textures.
   - **Seed**: Random seed for noise generation.
   - **Scale**: Scale of the noise pattern.
   - **Use Depth**: Enable turbulence noise with multiple layers.
   - **Depth Details**: Number of noise layers for turbulence.
   - **Mix Details**: Attenuation factor for turbulence layers.
   - **RGB**: Generate separate noise for each color channel.
   - **Alpha**: Generate an alpha channel for the texture.
   - **Groovy**: Use absolute values for higher contrast.
4. Click the `Generate Perlin Noise` button to create the texture.

### Adding Noise to Shader

1. After generating a texture, select an object with a material in the 3D Viewport.
2. In the `Noise Tools` panel, click the `Add to Active Shader` button.
3. The generated texture will be connected to the active material's shader nodes.

## Parameters

### Image Settings
- **Image Name**: Name of the generated image.
- **Overwrite Existing**: Replace an existing image with the same name.
- **Width/Height**: Dimensions of the generated texture.
- **Correct Aspect Ratio**: Adjust the display aspect ratio for non-square textures.

### Noise Settings
- **Seed**: Random seed for noise generation.
- **Scale**: Scale of the noise pattern.
- **Use Depth**: Enable turbulence noise with multiple layers.
- **Depth Details**: Number of noise layers for turbulence.
- **Mix Details**: Attenuation factor for turbulence layers.
- **RGB**: Generate separate noise for each color channel.
- **Alpha**: Generate an alpha channel for the texture.
- **Groovy**: Use absolute values for higher contrast.

## Example Use Cases

- **Procedural Textures**: Create seamless textures for materials like clouds, marble, or wood.
- **Displacement Maps**: Generate height maps for displacement in shaders.
- **Backgrounds**: Use noise textures as dynamic backgrounds in animations.

## Notes

- The generated textures are saved as packed data within the Blender file. To save them externally, use the `Image > Save As` option in the Image Editor.
- The add-on is designed for Blender's built-in shader system and may require adjustments for use with external render engines.

## License

This add-on is provided under the MIT License. Feel free to modify and distribute it as needed.

---

For questions or feedback, please open an issue on the repository or contact the developer directly. Enjoy creating procedural textures with Blender!
