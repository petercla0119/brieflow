from tifffile import imread, imwrite

from lib.phenotype.identify_cytoplasm_cellpose import (
    identify_cytoplasm_cellpose,
)

# load nuclei and cell segmentation data
nuclei = imread(snakemake.input[0])
cells = imread(snakemake.input[1])

# identify cytoplasms with cellpose
cytoplasms = identify_cytoplasm_cellpose(nuclei, cells)

# if identify_cytoplasm_cellpose couldn’t find usable mask, create a blank one
if cytoplasms is None or cytoplasms.size == 0:
    # keep dtype small; uint16 lets you store >65 k labels if needed
    cytoplasms = np.zeros_like(cells, dtype=np.uint16)

# save cytoplasms data
imwrite(snakemake.output[0], cytoplasms)
