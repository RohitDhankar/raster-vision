.PHONY: 0 1 2 3

all: 0 1 2 3

0: 
	python -m rastervision.pipeline.cli run_command /home/dhankar/temp/02_23/raster_vision/raster-vision/output/pipeline-config.json train

1: 0
	python -m rastervision.pipeline.cli run_command /home/dhankar/temp/02_23/raster_vision/raster-vision/output/pipeline-config.json predict

2: 1
	python -m rastervision.pipeline.cli run_command /home/dhankar/temp/02_23/raster_vision/raster-vision/output/pipeline-config.json eval

3: 2
	python -m rastervision.pipeline.cli run_command /home/dhankar/temp/02_23/raster_vision/raster-vision/output/pipeline-config.json bundle

