[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.21904575.svg)](https://doi.org/10.5281/zenodo.21904575)
# iconspy

## ICON Sections in PYthon
[`iconspy`](https://github.com/eerie-project/iconspy) is a python package for constructing sections on the [ICON model's](https://gitlab.dkrz.de/icon/icon-model) native grid.
It offers the ability to create sections that approximate great circles, follow lines of constant latitude or longitude,
and follow contours of an arbitrary field.

The functionality builds upon that provided  by the [`pyicon`](https://gitlab.dkrz.de/m300602/pyicon) package;
however, `iconspy` offers some extra functionality —
in particular the ability to join together multiple sections and construct sections which follow contours.

Documentation is hosted [here](https://eerie-project.github.io/iconspy/) and the package code [here](https://github.com/eerie-project/iconspy).

## Citation
If you use this software in your work, please cite it with either an in-text citation (preferred) or in the acknowledgements. 

For the text I recommend something along the lines of
> Sections were constructed on the native ICON grid using `iconspy` (Goldsworth, 2026).

Followed by a bibliography entry of the form
> Goldsworth, F. W. (2026). eerie-project/iconspy: vX (Version X) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21904575

If you use LaTeX you may want to use the bibtex entry:
```
@software{goldsworth2026iconspy,
  author  = {Goldsworth, Fraser William},
  title   = {{eerie-project/iconspy: vX}},
  year    = {2026},
  publisher = {Zenodo},
  version = {X},
  doi     = {10.5281/zenodo.21904575},
}
```

## Issues and feature requests
If you encounter any problems with the code please raise an issue on [github](https://github.com/eerie-project/iconspy).
If you have feature requests feel free to reach out there also.

## Acknowledgements
This software was developed as part of the EERIE project (grant agreement no 101081383) funded by the European Union.

This software contains modified code from the [`pyicon`](https://gitlab.dkrz.de/m300602/pyicon) package, namely:
  - function iconspy.core.Section._get_pyic_orientation_along_path
  - function iconspy.utils._pyicon_convert_tgrid_data
An MIT license applies to these parts of the code (see also the `LICENSE` file).
I am grateful to Nils Brüggemann for his ongoing development and maintenance of the [`pyicon`](https://gitlab.dkrz.de/m300602/pyicon) package.