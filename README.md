# URL_His_TissueSeg
The source code of Unsupervised Representation Learning for Tissue Segmentation in Histopathological Images: From Global to Local Contrast

![URL_TS](./Tasks_Intro.pdf)

Our framework is enlightened by a domain-specific cue: different tissues are composed by different cells and extracellular matrices.
Thus, we design three contrastive learning tasks with multi-granularity views (from global to local) for encoding necessary features into representations without accessing annotations.

- (1) an image-level task to capture the difference between tissue components, i.e., encoding the component discrimination; 
- (2) a superpixel-level task to learn discriminative representations of local regions with different tissue components, i.e., encoding the prototype discrimination;
- (3) a pixel-level task to encourage similar representations of different tissue components within a local region, i.e., encoding the spatial smoothness.
