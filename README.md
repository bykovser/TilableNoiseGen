This Blender add-on generates procedural Perlin noise and turbulence textures, which can be used for creating materials, textures, or other visual effects. The generated textures can be directly applied to the active material or saved as images.

# Important
Use powers of 2 in **scale** and **image size** to get tilable texture

2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384

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

![image](https://github.com/user-attachments/assets/7676f5fc-9d64-4566-88e9-0c69796be543)

### Adding Noise to Shader

1. After generating a texture, select an object with a material in the 3D Viewport.
2. In the `Noise Tools` panel, click the `Add to Active Shader` button.
3. The generated texture will be connected to the active node.

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

## Notes

- The generated textures are saved as packed data within the Blender file. To save them externally, use the `Image > Save As` option in the Image Editor.
- The add-on is designed for Blender's built-in shader system and may require adjustments for use with external render engines.

## License

This add-on is provided under the MIT License. Feel free to modify and distribute it as needed.

---

For questions or feedback, please open an issue on the repository or contact the developer directly. Enjoy creating procedural textures with Blender!

## Changelog

- **v1.6.1**: fixed Aspect ratio equalizer
- **v1.6**: Generator works 20x faster. `Add to Active Shader` dont ignores frames and group
- **v1.5**: `Add to Active Shader` button connects image texture to active node in shader editor, added some warnings
- **v0 - v1.4**: it was born
 
