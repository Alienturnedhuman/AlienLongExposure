import AlienLongExposure
import AlienLongExposureVideo
from AlienImageSequenceReader import VideoSequenceReader as VideoReader,ListOfImagePaths as ImagesReader
from AlienImageSequenceGenerator import ImageSequenceOutput , VideoOutput
from cv2 import resize as cv_resize , imwrite as cv_imwrite , VideoWriter_fourcc as cv_VideoWriter_fourcc
import os
from math import sin as math_sin, pi as math_pi

print("testing LongExposureVideo")

# details of the input file (eg for ~/Documents/long_exposure_src/input_video.mp4)
name = "input_video"
input_extension = "mp4"
folder = "~/Documents/long_exposure_src"

# frames you wish to extract (if you want to extract all the frames, for now just enter a very high number for end frame)
start_frame = 0
end_frame = 600000



# detaisl of the output file
# this will output the file to: ~/Documents/long_exposure_src/output/input_video_descriptive_suffix.mp4

# details of the output folder
output_folder = folder + "output/"

if not os.path.isdir(output_folder):
	os.mkdir(output_folder)

# modify the output name as you desire here - for example adding a descriptive suffix
output_name = name+"_descriptive_suffix"
output_codec = cv_VideoWriter_fourcc(*'mp4v')
output_extension = "mp4"

output_resolution = (1080,1920,3)

# final transformation settings flip = v / h / vh / none (for vertical flip / horizontal flip / both flips (180 degree rotation) / no flip)
transform_settings = {"flip":"none"}

# create some iput filters
square_input_filter = AlienLongExposure.LongExposureBasicInputFilters.square()
sqrt_input_filter = AlienLongExposure.LongExposureBasicInputFilters.square_root()
power_input_filter = AlienLongExposure.LongExposureBasicInputFilters.power(1.5)

# create post process fitlers
minor_clipping_max_range = AlienLongExposure.LongExposureBasicProcessAnalysers.maximise_range(.5,.5)
bottom_clipping_max_range = AlienLongExposure.LongExposureBasicProcessAnalysers.maximise_range(2.5,.5)
naive_reduce_contrast = AlienLongExposure.LongExposureProcessFilters.fast_reduce_contrast()

# set up filters for input / post_process / output
input_filters = [power_input_filter]
input_post = []
output_filters = [minor_clipping_max_range]


# output a moving exposure (video of successive minor long exposures rather than one single image)
moving_exposure = True

# for moving expusores:
# step between frames
step = 10
# number of frames to blend (note, be careful not to run into memory problems here)
exposure_frame_count = 30
# weighting pattern to use (this is specific to this script, generates a list of values)
weighting = "NONE"
# output a video rather than an image sequence
do_video = True

#
#		End of customisation here
#
###############################################################################



exposure_frame_weight = None

if weighting == "INVERSE":
	exposure_frame_weight = [1/(exposure_frame_count-i) for i in range(exposure_frame_count)]
elif weighting == "SIN":
	exposure_frame_weight = [math_sin(math_pi*i/exposure_frame_count) for i in range(exposure_frame_count)]




src = folder+name+"."+input_extension
input_src = VideoReader(src,start_frame=start_frame, end_frame=end_frame)


if moving_exposure:
	if do_video:
		output_generator = VideoOutput(dims=(1080,1920,3),name=output_name,folder=output_folder,video_type=output_extension,codec=output_codec,frame_rate=30)
	else:
		output_generator = ImageSequenceOutput(dims=(1080,1920,3),name=output_name,folder=output_folder,image_type="png")
	if step == 1:
		long_exposure = AlienLongExposureVideo.LongExposureVideo(output_resolution,input_filters,input_post,output_filters,output_generator=output_generator,exposure_frame_count=exposure_frame_count,transform_settings=transform_settings,exposure_frame_weight=exposure_frame_weight)
	else:
		long_exposure = AlienLongExposureVideo.LongExposureVideo(output_resolution,input_filters,input_post,output_filters,output_generator=output_generator,exposure_frame_count=exposure_frame_count,transform_settings=transform_settings,exposure_frame_weight=exposure_frame_weight,step=step)
else:
	long_exposure = AlienLongExposure.LongExposure(output_resolution,input_filters,input_post,output_filters,transform_settings=transform_settings)


counter = 1



long_exposure.initialize()

print("LongExposureVideo initialised")
while not input_src.has_finished():
	frame = input_src.get_next_frame()
	long_exposure.add_frame(frame) 
	counter += 1
	if counter%25 == 0:
		print("adding frames: ",counter)

print("total frames: ",counter)

if not moving_exposure:
	output_img = long_exposure.create()
	cv_imwrite(output_folder+output_name+".png",output_img)