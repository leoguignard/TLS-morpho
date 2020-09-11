# Computing some morphological metrics

This code is related to the article [Mouse embryonic stem cells self-organize into trunk-like structures with neural tube and somites]([https://dx.doi.org/10.1126/science.aba4937](https://dx.doi.org/10.1126/science.aba4937).

Globally, it allows to compute some morphological metrics from binary images in 2D and 3D.

Provided a isotropic masked image, its distance transform (see this [scipy module](https://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html#distance-transforms) for example), a voxel size and potentially an anterior and posterior position (only for 2D images), it allows to compute:

- length

- width

- aspect ratio: $\frac{length}{width}$

- volume ($V$) or area ($A$) in 2D

- surface ($S$) or perimeter ($p$) in 2D

- solidity: $\frac{V}{V_{conv-hull}}$

- the sphericity (circularity in 2D): $\frac{\pi^\frac{1}{3}(6V)^{\frac{2}{3}}}{S}$  (or $\frac{4\pi A}{p^2}$ in 2D)

# Dependencies

- [numpy](numpy.org)

- [scipy](scipy.org)

- [tifffile](https://pypi.org/project/tifffile/)

- [scikit-image](scikit-image.org)


