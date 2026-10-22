# Bundled skin reference

`skin_reference_cc0.jpg` is the unmodified original photograph **Image skin
texture.jpg**, anonymous author, dated 13 March 2017. It is a photograph of human
skin, not a generated noise map.

- Original source: https://pixnio.com/textures-and-patterns/human-skin
- File and license record: https://commons.wikimedia.org/wiki/File:Image_skin_texture.jpg
- Original file: https://upload.wikimedia.org/wikipedia/commons/7/75/Image_skin_texture.jpg
- License: CC0 1.0 Universal Public Domain Dedication.
- Original dimensions: 3693 x 2462 pixels.

The node derives local luminance texture from this photo at runtime. It removes
broad colour/illumination, varies patch placement and orientation, and transfers
the resulting relief through the skin mask. It does not copy the photographed
person's identity or claim to recover the subject's true pores. A custom close-up
skin crop can be connected to `skin_reference` to replace this source.

The reference includes natural fine creases and occasional fine hairs; it is
not a facial-pore-only scan. Choose a reference suited to the desired skin area.
The default scale reduces macrostructure, and patch filtering rejects unusually
strong or blank regions, but it is not a semantic hair/crease remover.
