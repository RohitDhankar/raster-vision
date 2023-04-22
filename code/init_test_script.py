
## https://github.com/azavea/raster-vision/issues/1306


import os
from subprocess import check_output

os.environ['GDAL_DATA'] = check_output('pip show rasterio | grep Location | awk \'{print $NF"/rasterio/gdal_data/"}\'', shell=True).decode().strip()
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Read Raster Files -- https://rasterio.readthedocs.io/en/stable/

# (env_raster_vis) dhankar@dhankar-1:~/temp$ find . -type f -name "*.tif"
# ./04_22/a______MAIN__old_2020_cv20/GitsDoneDown_OpenCV2020/farm-field.tif
# ./02_23/raster_vision/raster-vision/integration_tests/chip_classification/scene/image.tif
# ./02_23/raster_vision/raster-vision/integration_tests/chip_classification/scene/image2.tif
# ./02_23/raster_vision/raster-vision/integration_tests/object_detection/scene/image.tif
# ./02_23/raster_vision/raster-vision/integration_tests/object_detection/scene/image2.tif
# ./02_23/raster_vision/raster-vision/integration_tests/semantic_segmentation/scene/labels.tif
# ./02_23/raster_vision/raster-vision/integration_tests/semantic_segmentation/scene/image.tif
# ./02_23/raster_vision/raster-vision/integration_tests/semantic_segmentation/scene/image2.tif
# ./02_23/raster_vision/raster-vision/integration_tests/semantic_segmentation/scene/labels2.tif
# ./02_23/raster_vision/raster-vision/tests/data_files/ones.tif
# ./02_23/raster_vision/raster-vision/tests/data_files/small-rgb-tile.tif
# ./02_23/raster_vision/raster-vision/tests/data_files/small-uint16-tile.tif
# ./02_23/raster_vision/raster-vision/tests/data_files/3857.tif
# ./02_23/raster_vision/raster-vision/tests/data_files/evaluator/cc-label-img-blank.tif
# (env_raster_vis) dhankar@dhankar-1:~/temp$ 

## Ok files below --- https://github.com/azavea/raster-vision/discussions/1504 
# https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/spacenet/RGB-PanSharpen_AOI_2_Vegas_img25.tif
# https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/spacenet/buildings_AOI_2_Vegas_img25.geojson

# https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/spacenet/RGB-PanSharpen_AOI_2_Vegas_img205.tif
# https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/spacenet/buildings_AOI_2_Vegas_img205.geojson


from rastervision.core.data import RasterioSource

# img_uri = 's3://azavea-research-public-data/raster-vision/examples/spacenet/RGB-PanSharpen_AOI_2_Vegas_img205.tif'
# raster_source = RasterioSource(img_uri, allow_streaming=True)
# raster_source.shape

from matplotlib import pyplot as plt


import rasterio
import rasterio.features
import rasterio.warp

# path_raster = "/home/dhankar/temp/02_23/raster_vision/raster-vision/tests/data_files/ones.tif"
# path_raster1 = "/home/dhankar/temp/02_23/raster_vision/raster-vision/integration_tests/chip_classification/scene/image.tif"

path_raster2 = "/home/dhankar/temp/02_23/raster_vision/input_img/RGB-PanSharpen_AOI_2_Vegas_img205.tif"
#  RGB-PanSharpen_AOI_2_Vegas_img25.tif
path_raster3 = "/home/dhankar/temp/02_23/raster_vision/input_img/RGB-PanSharpen_AOI_2_Vegas_img25.tif"
#Rionegro_RGB_8_2
path_raster4 = "/home/dhankar/temp/02_23/raster_vision/input_img/Rionegro_RGB_8_2.tif"
ls_rast = [path_raster2,path_raster3,path_raster4]


with rasterio.open(path_raster2) as dataset:
    print(type(dataset)) #<class 'rasterio.io.DatasetReader'>
    print("--dataset--")


def get_rastFiles(ls_rast):
    for img_rast in ls_rast:

        raster_source = RasterioSource(img_rast)#, allow_streaming=True)
        shape_of_ras_file = raster_source.shape    
        print("---shape_of_ras_file----",shape_of_ras_file) ##(256, 256, 1)

        # get a smaller CHIP -- from the RASTER File 
        if "Rionegro" in str(img_rast):
            chip = raster_source[:2000, :2000]
            shape_of_chip_file = chip.shape
            print("---shape_of_chipped_file--_raster4--",shape_of_chip_file) ##(256, 256, 1)

        else:
            chip = raster_source[:600, :600]
            shape_of_chip_file = chip.shape
            print("---shape_of_chipped_file----",shape_of_chip_file) ##(256, 256, 1)

        # The returned chip is a numpy array which we can plot using matplotlib. 
        # Note that since the values are uint16, we first normalize them before plotting.


        colors_mins = chip.reshape(-1, chip.shape[-1]).min(axis=0)
        colors_maxs = chip.reshape(-1, chip.shape[-1]).max(axis=0)
        chip_normalized = (chip - colors_mins) / (colors_maxs - colors_mins)
        print(type(chip_normalized)) #<class 'numpy.ndarray'>
        shape_of_chipNorm_file = chip_normalized.shape
        print("---shape_of_Normalized--chipped_file----",shape_of_chipNorm_file) ##(256, 256, 1)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(chip_normalized)
        #plt.imshow(chip_normalized,interpolation='nearest')
        plt.show()


get_rastFiles(ls_rast)



