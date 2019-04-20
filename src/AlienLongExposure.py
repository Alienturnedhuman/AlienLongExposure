from cv2 import resize as cv_resize , imwrite as cv_imwrite , flip as cv_flip
from os import listdir as os_listdir, path as os_path
from AlienImageSequenceReader import VideoSequenceReader as VideoReader,ListOfImagePaths as ImagesReader
import numpy as np
from collections import deque
from math import log as math_log


'''
	LongExposureStandard

	a class for containing general values and methods for use elsewhere in the code, to avoid the use of magic numbers
'''
class LongExposureStandard:
	'''
		constants

		class containing numerical constants
	'''
	class constants:
		mid_log = math_log(0.5)
		one_byte = 256
		two_bytes = 256*256
		one_byte_percentile = 100/one_byte
		two_bytes_percentile = 100/two_bytes

	'''
		dtype_of

		class containing dtypes of different canvases
	'''
	class dtype_of:
		input_buffer = np.double
		process_buffer = np.double
		output_16bit = np.uint16
		output_8bit = np.uint8
		output_default = np.uint8


	'''
		calculate_gamma

		function for calculating the gamma value to adjust one point to another. If no 'to_value' is provided, an adjustment to 0.5 will be used
	'''
	def calculate_gamma(from_value,to_value=None):
		return math_log(from_value) / LongExposureStandard.constants.mid_log if to_value	is None else math_log(from_value)/math_log(to_value)

'''
	LongExposureFilter

	* abstract class for all the filter types

	Filters process pixels from one value to another. Typically - for speed - these are done using numpy functions.

'''
class LongExposureFilter:
	def __init__(self):
		pass
	def process_algorithm(self,frame,out):
		return output_to
	def process_frame(self,frame,out=None):
		if out is None:
			out = frame
		elif not isinstance(out,np.ndarray) or not out.shape == frame.shape:
			out = LongExposureCanvas(frame.shape)
		return self.process_algorithm(frame,out=out)

'''
	LongExposureInputFilter

	* abstract class for Input Filters
	* extends LongExposureFilter

	Input filters deal with unsigned ints
'''
class LongExposureInputFilter(LongExposureFilter):
	pass


'''
	LongExposureBasicInputFilters

	* container class for all the basic input filters
'''
class LongExposureBasicInputFilters:
	'''
		LongExposureBasicInputFilters.square

		squares all the pixel values
	'''
	class square(LongExposureInputFilter):
		def process_algorithm(self,frame,out):
			return np.multiply(frame,frame,out=out)
	'''
		LongExposureBasicInputFilters.square

		squares all the pixel values
	'''
	class square_scaled(LongExposureInputFilter):
		def __init__(self,scalar = 256):
			self.scalar = scalar
		def process_algorithm(self,frame,out):
			np.multiply(frame,frame,out=out)
			return np.floor_divide(out,self.scalar,out=out)

	'''
		LongExposureBasicInputFilters.square_root

		square roots all the pixel values
	'''
	class square_root(LongExposureInputFilter):
		def process_algorithm(self,frame,out):
			return np.sqrt(frame,out=out)
	'''
		LongExposureBasicInputFilters.square_root_curve

		square roots all the pixel values (multiplies by 256 first to maximise data resolution)
	'''
	class square_root_scaled(LongExposureInputFilter):
		def __init__(self,scalar = 4096,buffer=None):
			self.scalar = scalar
		def process_algorithm(self,frame,out):

			np.multiply(self.frame,self.scalar,out=out)
			return np.sqrt(out,out=out)


	'''
		LongExposureBasicInputFilters.log

		does a log1p operation to all pixel values
	'''
	class log(LongExposureInputFilter):
		def process_algorithm(self,frame,out):
			return np.log1p(frame,out=out)

	'''
		LongExposureBasicInputFilters.exp

		does a expm1 operation on all pixel values
	'''
	class exp(LongExposureInputFilter):
		def process_algorithm(self,frame,out):
			return np.expm1(frame,1,out=out)
	'''
		LongExposureBasicInputFilters.exp

		does a expm1 operation on all pixel values
	'''
	class power(LongExposureInputFilter):
		def __init__(self,n):
			self.n = n
		def process_algorithm(self,frame,out):
			return np.power(frame,self.n,out=out)

'''
	LongExposureProcessFilter

	* abstract class for Process Filters
	* extends LongExposureFilter

	Process filters deal with floats, pixel values should always output from 0 to 1
'''
class LongExposureProcessFilter(LongExposureFilter):
	pass


'''
	LongExposureProcessFilters

	* container class for all the basic process filters
'''
class LongExposureProcessFilters:
	'''
		LongExposureProcessFilters.gamma

		applies the gamma value set in the constructor
	'''
	class gamma(LongExposureProcessFilter):
		'''
			Constructor

			* gamma_value (double) : the gamma value of the filter
		'''
		def __init__(self,gamma_value):
			self.gamma_value = gamma_value

		def process_algorithm(self,frame,out):
			# gamma correction is just raising the value to the power of the gamma value
			return np.power(frame,self.gamma_value,out=out)

	'''
		LongExposureProcessFilters.arithmetic_sequence

		applies the sequence of of numpy arithmetic processes.

	'''
	class arithmetic_sequence(LongExposureProcessFilter):
		'''
			Constructor

			* sequence (list of (numpy operation,value)) : Aritimetic operations to follow
		'''
		def __init__(self,sequence):
			self.sequence = sequence
		def process_algorithm(self,frame,out):
			if len(self.sequence) < 1:
				return frame if frame == out else np.add(frame,0,out=out)
			else:
				sequence = deque(self.sequence)
				rule = sequence.popleft()
				rule[0](frame,rule[1],out=out)
				while len(sequence) > 0:
					rule = sequence.popleft()
					rule[0](out,rule[1],out=out)
				return np.clip(out,0,1,out=out)

		

	'''
		LongExposureProcessFilters.fast_reduce_contrast

		Does a 'fast' contrast reduction to the image data

		remaps data from 0,0 -> 1,1 to -1,-1 -> 1,1 and then cubes it, before remapping back to 0,0 -> 1,1
	'''
	class fast_reduce_contrast(LongExposureProcessFilter):
		def process_algorithm(self,frame,out):
			np.multiply(frame,2,out=out)
			np.subtract(out,1,out=out)
			np.power(out,3,out=out)
			np.add(out,1,out=out)
			return np.divide(out,2,out=out)
		

	'''
		LongExposureProcessFilters.fast_increase_contrast

		Does a 'fast' contrast increase to the image data

		remaps data from 0,0 -> 1,1 to -1,-1 -> 1,1 and then cuberoots it, before remapping back to 0,0 -> 1,1
	'''
	class fast_increase_contrast(LongExposureProcessFilter):
		def process_algorithm(self,frame,out):
			np.multiply(frame,2.0,out=out)
			np.subtract(out,1.0,out=out)
			np.cbrt(out,out=out)
			np.add(out,1.0,out=out)
			return np.divide(out,2.0,out=out)


'''
	LongExposureCanvas

	class containing methods for generating / handling canvases
'''
class LongExposureCanvas:
	'''
		generate_canvas

		generic canvas creator, returns a canvas of desired size

		arguments:

		shape:		Tuple containing canvas dimensions (height,width,channels)
		type:		numpy data type
		value:		initial value to generate the canvas containing (typically zero)
	'''
	def generate_canvas(shape,type,value=0):
		print(shape)
		return np.zeros(shape,dtype=type) if value == 0 else np.full(shape,value,dtype=type)

	def generate_input(shape , value = 0):
		return LongExposureCanvas.generate_canvas(shape,LongExposureStandard.dtype_of.input_buffer,value)
	def generate_process(shape , value = 0):
		return LongExposureCanvas.generate_canvas(shape,LongExposureStandard.dtype_of.process_buffer,value)
	def generate_output_8bit(shape , value = 0):
		return LongExposureCanvas.generate_canvas(shape,LongExposureStandard.dtype_of.output_8bit,value)
	def generate_output_16bit(shape , value = 0):
		return LongExposureCanvas.generate_canvas(shape,LongExposureStandard.dtype_of.output_16bit,value)
	
	'''
		convert_to_process_frame

		converts a canvas from being an input frame to being a process frame, and returns the new frame.

		An input can have very high values to each pixel (typically up to 256 * number of frames)
		A process frame is scaled from 0.0 to 1.0

		arguments:

		input_frame (required)				the image data for the input frame
		process_frame (optional)			the process frame to output to, if none is provided then a new frame will be generated
		manual_clip_lower (optional)		the lower value to clip the input data to before standardisation (will override auto_clip_lower if provided)
		manual_clip_upper (optional)		the upper value to clip the input data to before standardisation (will override auto_clip_higher if provided)
		auto_clip_lower (optional)			the lower percentile to automatically clip the input data to before standardisation (defaults to 100/256)
		auto_clip_upper (optional)			the upper percentile to automatically clip the input data to before standardisation (defaults to 100/256)


	'''
	def convert_to_process_frame(input_frame,process_frame = None,manual_clip_lower=None , manual_clip_upper = None , auto_clip_lower = LongExposureStandard.constants.one_byte_percentile , auto_clip_upper = LongExposureStandard.constants.one_byte_percentile):
		if process_frame is None:
			process_frame = LongExposureCanvas.generate_process(input_frame.shape)
		clip_lower = np.percentile(input_frame,auto_clip_lower) if manual_clip_lower is None else manual_clip_lower
		clip_upper = np.percentile(input_frame,100-auto_clip_upper) if manual_clip_upper is None else manual_clip_upper
		clip_bounds = clip_upper - clip_lower
		process_frame = np.clip(input_frame,clip_lower,clip_upper,out=process_frame)
		return np.divide(process_frame,clip_bounds,out=process_frame)

	'''
		convert_to_output_frame

		converts a process frame to an output frame and returns the output frame

		process frames are floating point value pixel scaled from 0.0 to 1.0
		output frames are integer value pixels, either uint8 (default) or uint16

		arguments

		process_frame (required)		the image data for the output frame
		output_frame (optional)			the output frame for the final image data (if no output frame is provided, one will be generated)
		output_dtyle (optional)			the numpy dtype used for the output frame

	'''
	def convert_to_output_frame(process_frame,output_frame=None,output_dtype = np.uint8):
		if output_frame is None or output_frame.dtype != output_dtype or output_frame.shape != process_frame.shape:
			if output_dtype == LongExposureStandard.dtype_of.output_16bit:
				output_frame = LongExposureCanvas.generate_output_16bit(process_frame.shape)
			elif output_dtype == LongExposureStandard.dtype_of.output_8bit:
				output_frame = LongExposureCanvas.generate_output_8bit(process_frame.shape)
			else:
				output_dtype = LongExposureStandard.dtype_of.output_default
				output_frame = LongExposureCanvas.generate_canvas(process_frame.shape,output_dtype,0)

		if output_dtype == np.uint16:
			multiple = LongExposureStandard.constants.two_bytes
		else:
			multiple = LongExposureStandard.constants.one_byte
		np.multiply(process_frame,multiple,out=process_frame)
		np.clip(process_frame,0,multiple-1,out=process_frame)
		output_frame = process_frame.astype(output_dtype)
		return output_frame


	'''
		reset_to_value

		resets and returns a canvas to a desired value (default 0)

		arguments

		canvas (required)		the canvas to reset
		value (optional)		the value to reset to (default = 0)
	'''
	def reset_to_value(canvas,value=0):
		np.multiply(canvas,0,out=canvas)
		if value!=0:
			np.add(canvas,value,out=canvas)
		return canvas


'''
	LongExposureInputAnalyser

	* abstract class

	InputAnalysers to used to analyse the current state of an input frame and can then add filters to the deque
'''
class LongExposureInputAnalyser:
	def __init__(self):
		pass
	def analyse(self,img_data,analysis_data,post_processors):
		pass

class LongExposureBasicInputAnalysers:
	class mean(LongExposureInputAnalyser):
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_data["mean"] = np.mean(img_data)
	class median(LongExposureInputAnalyser):
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_data["median"] = np.median(img_data)
	class std(LongExposureInputAnalyser):
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_data["std"] = np.std(img_data)
	class min(LongExposureInputAnalyser):
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_data["min"] = np.min(img_data)
	class max(LongExposureInputAnalyser):
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_data["max"] = np.max(img_data)

	class clip_to_3std(LongExposureInputAnalyser):
		def __init__(self):
			self.std = LongExposureBasicInputAnalysers.std()
			self.std = LongExposureBasicInputAnalysers.std()
			self.max = LongExposureBasicInputAnalysers.max()
			self.mean = LongExposureBasicInputAnalysers.mean()
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			if "std" not in analysis_data:
				self.std.analyse(img_data,post_processors,analysis_data,analysis_log)
			std = analysis_data["std"]
			if "min" not in analysis_data:
				self.min.analyse(img_data,post_processors,analysis_data,analysis_log)
			min_val = analysis_data["min"]
			if "max" not in analysis_data:
				self.max.analyse(img_data,post_processors,analysis_data,analysis_log)
			max_val = analysis_data["max"]
			if "mean" not in analysis_data:
				self.mean.analyse(img_data,post_processors,analysis_data,analysis_log)
			mean = analysis_data["mean"]

			lower_std = mean - (3 * std)
			upper_std = mean + (3 * std)

			analysis_data["manual_clip_upper"] = upper_std if upper_std < max_val else max_val
			analysis_data["manual_clip_lower"] = lower_std if lower_std > min_val else min_val


'''
	LongExposureProcessAnalyser

	* abstract class

	Long Exposure Process Analyser analyses the image data and can add more image processors into the queue
'''
class LongExposureProcessAnalyser:
	def __init__(self):
		pass
	'''
		analyse

		* required method

		this method analyses the data, and based on its findings can:
			* add more processors to the start or end of the post_processors deque
			* add/change values to the analysis_data dictionary
			* add/change values to the analysis_log dictionary

		arguments:

		img_data:			np.array of the img_data
		post_processors:	deque of the post_processors remaining to be processed
		analysis_data:		a dictionary of the current analysis data (results will be put here) 
								...this gets emptied after the image data is modified by a filter
		analysis_log: 		a dictionary of the current analysis log (logging / counters of )
	'''
	def analyse(self,img_data,post_processors,analysis_data,analysis_log):
		pass

'''
	LongExposureBasicProcessAnalysers

	* collection of sub classes containing basic process analysers


'''
class LongExposureBasicProcessAnalysers:
	'''
		LongExposureBasicProcessAnalysers.mean

		* extends LongExposureProcessAnalyser

		calculates the mean and records it
	'''
	class mean(LongExposureProcessAnalyser):
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_log["mean"] = analysis_log["mean"] + 1 if "mean" in analysis_log else 1
			analysis_data["mean"] = np.mean(img_data)

	'''
		LongExposureBasicProcessAnalysers.median

		* extends LongExposureProcessAnalyser

		finds the median and records it
	'''
	class median(LongExposureProcessAnalyser):
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_log["median"] = analysis_log["median"] + 1 if "median" in analysis_log else 1
			analysis_data["median"] = np.median(img_data)
			if analysis_data["median"] < 0.01:
				analysis_data["median"] = 0.01
			elif analysis_data["median"] > 0.99:
				analysis_data["median"] = 0.99

	'''
		LongExposureBasicProcessAnalysers.std

		* extends LongExposureProcessAnalyser

		calculates the standard deviation and records it
	'''
	class std(LongExposureProcessAnalyser):
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_log["std"] = analysis_log["std"] + 1 if "std" in analysis_log else 1
			analysis_data["std"] = np.std(img_data)


	'''
		LongExposureBasicProcessAnalysers.min

		* extends LongExposureProcessAnalyser

		finds the lowest value and records it
	'''
	class min(LongExposureProcessAnalyser):
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_data["min"] = np.min(img_data)

	'''
		LongExposureBasicProcessAnalysers.max

		* extends LongExposureProcessAnalyser

		finds the highest value and records it
	'''
	class max(LongExposureProcessAnalyser):
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_data["max"] = np.max(img_data)

	'''
		LongExposureBasicProcessAnalysers.clipped_min

		* extends LongExposureProcessAnalyser

		finds the minimum value, clipped to a percentile point and records it as "min"
	'''
	class clipped_min(LongExposureProcessAnalyser):
		def __init__(self,bound):
			self.bound = bound
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_data["clipped_min"] = np.percentile(img_data,self.bound)

	'''
		LongExposureBasicProcessAnalysers.clipped_max

		* extends LongExposureProcessAnalyser

		finds the maximum value, clipped to a percentile point and records it as "max"
	'''
	class clipped_max(LongExposureProcessAnalyser):
		def __init__(self,bound):
			self.bound = 100-bound
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_data["clipped_max"] = np.percentile(img_data,self.bound)

	'''
		LongExposureBasicProcessAnalysers.maximise_range

		* extends LongExposureProcessAnalyser

		transforms the data so the minimum value is 0.0 and the maximum value is 1.0
	'''
	class maximise_range(LongExposureProcessAnalyser):
		def __init__(self,lower_bound=0,upper_bound=0):
			self.clipped_max = upper_bound != 0
			self.clipped_min = lower_bound != 0
			self.max = LongExposureBasicProcessAnalysers.max() if not self.clipped_max else LongExposureBasicProcessAnalysers.clipped_max(upper_bound)
			self.min = LongExposureBasicProcessAnalysers.min() if not self.clipped_min else LongExposureBasicProcessAnalysers.clipped_min(lower_bound)
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			if self.clipped_max:
				self.max.analyse(img_data,post_processors,analysis_data,analysis_log)
				max_val = analysis_data["clipped_max"]
			else:
				if "max" not in analysis_data:
					self.max.analyse(img_data,post_processors,analysis_data,analysis_log)
				max_val = analysis_data["max"]


			if self.clipped_min:
				self.min.analyse(img_data,post_processors,analysis_data,analysis_log)
				min_val = analysis_data["clipped_min"]
			else:
				if "min" not in analysis_data:
					self.min.analyse(img_data,post_processors,analysis_data,analysis_log)
				min_val = analysis_data["min"]

			difference = max_val - min_val
			inv_difference = 1/difference

			post_processors.appendleft(LongExposureProcessFilters.arithmetic_sequence([(np.subtract,min_val),(np.divide,difference)]))


	'''
		LongExposureBasicProcessAnalysers.gamma_adjust_to_median

		* extends LongExposureProcessAnalyser

		adjusts the gamma so the median value is 0.5
	'''
	class gamma_adjust_to_median(LongExposureProcessAnalyser):
		def __init__(self):
			self.calc_median = LongExposureBasicProcessAnalysers.median()
		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_log["gamma_adjust_to_median"] = analysis_log["gamma_adjust_to_median"] + 1 if "gamma_adjust_to_median" in analysis_log else 1
			if "median" not in analysis_data:
				self.calc_median.analyse(img_data,post_processors,analysis_data,analysis_log)
			gamma_value = LongExposureStandard.calculate_gamma(analysis_data["median"])
			post_processors.appendleft(LongExposureProcessFilters.gamma(gamma_value))

	'''
		LongExposureBasicProcessAnalysers.smart_contrast_adjuster

		* extends LongExposureProcessAnalyser

		analyses the spread of the data and attempts to adjust the contrast to the image is more 'flat'
	'''
	class smart_contrast_adjuster(LongExposureProcessAnalyser):
		def __init__(self,lower_bound=0.45,upper_bound=.75,max_iterations=5):
			self.lower_bound = lower_bound
			self.upper_bound = upper_bound
			self.max_iterations = max_iterations
			self.calc_std = LongExposureBasicProcessAnalysers.std()



		def analyse(self,img_data,post_processors,analysis_data,analysis_log):
			analysis_log["smart_contrast_adjuster"] = analysis_log["smart_contrast_adjuster"] + 1 if "smart_contrast_adjuster" in analysis_log else 1
			if analysis_log["smart_contrast_adjuster"] < self.max_iterations:
				post_processors.appendleft(self)

			if "std" not in analysis_data:
				self.calc_std.analyse(img_data,post_processors,analysis_data,analysis_log)
			std = analysis_data["std"]


			if std < self.lower_bound:
				post_processors.appendleft(LongExposureBasicProcessAnalysers.gamma_adjust_to_median())
				post_processors.appendleft(LongExposureProcessFilters.fast_reduce_contrast())
			elif std > self.upper_bound:
				post_processors.appendleft(LongExposureBasicProcessAnalysers.gamma_adjust_to_median())
				post_processors.appendleft(LongExposureProcessFilters.fast_increase_contrast())






'''
	LongExposure

	class for handling the long exposure

	This is the main class. Create one of these for handling the long exposure. 
'''
class LongExposure:
	# FLIP_OPTIONS is used for configuring the clip method from openCV2
	FLIP_OPTIONS = {"v" : 1 ,"h" : 0  ,"vh" : -1 , "hv" : -1}

	'''
		CONSTRUCTOR

		arguments:

		canvas:						None / Tuple / np.array

									None		- canvas to be set up later
									Tuple 		- tuple of three integers that match the shape of the image (height,width,channels)
									np.array 	- image data of first frame

		input_frame_filters:		list of filters to apply to the image data before it is added to the canvas

		input_data_post_processors:	list of post_processors that get applied to the data before it gets normalised to 0.0 - 1.0

		process_frame_filters:		list of post_processors that get applied to the data after it has been normalised to 0.0 - 1.0

		output_dtype:				the numpy data type that the image is outputted as (either uint8 or uint16)

		transform_settings:			tranformation settings to apply to the image after it has been generated (for flipping the image)

	'''
	def __init__(self,canvas=None,input_frame_filters=[],input_data_post_processors=[],process_frame_filters=[],output_dtype=LongExposureStandard.dtype_of.output_default,transform_settings=None,**kwargs):
		self.setup_canvas(canvas,**kwargs)
		self.setup_output(output_dtype=output_dtype,**kwargs)
		self.setup_filters(input_frame_filters,input_data_post_processors,process_frame_filters,output_dtype,**kwargs)
		self.setup_transform(transform_settings,**kwargs)

	'''
		setup_canvas

		Sets up the canvas to be used to create the final image

		arguments:

		canvas:		None / Tuple / np.array

					None		- canvas to be set up later
					Tuple 		- tuple of three integers that match the shape of the image (height,width,channels)
					np.array 	- image data of first frame
	'''
	def setup_canvas(self,canvas = None, **kwargs):
		self.canvas = canvas if isinstance(canvas,np.ndarray) and canvas.dtype == LongExposureStandard.dtype_of.input_buffer else (LongExposureCanvas.generate_input(canvas) if isinstance(canvas,tuple) and len(canvas)==3 else None)
		self.canvas_dims = self.canvas.shape if self.canvas is not None else None
		self.initialised = canvas is not None	
	

	'''
		setup_output

		Sets up the output settings for the final image at the end

		arguments:

		outtype_dtype:	numpy datatype to be used for the final image
	'''
	def setup_output(self,output_dtype=LongExposureStandard.dtype_of.output_default,**kwargs):
		self.output_dtype = output_dtype
		self.output_frame = None


	def setup_filters(self,input_frame_filters=[],input_data_post_processors=[],process_frame_filters=[],output_dtype=LongExposureStandard.dtype_of.output_default,**kwargs):
		# filters (input)
		self.input_frame_filters = [f for f in input_frame_filters if isinstance(f,LongExposureInputFilter)]

		self.input_post_processors = [f for f in input_data_post_processors if isinstance(f,LongExposureInputFilter) or isinstance(f,LongExposureInputAnalyser)]
		self.input_analysis_data = {}
		self.input_analysis_log = {}

		self.process_frame_filters = [f for f in process_frame_filters if isinstance(f,LongExposureProcessFilter) or isinstance(f,LongExposureProcessAnalyser)]
		self.process_analysis_data = {}
		self.process_analysis_log = {}

		self.manual_clip_lower = None
		self.manual_clip_upper = None
		self.auto_clip_lower = LongExposureStandard.constants.two_bytes_percentile if self.output_dtype == np.uint16 else LongExposureStandard.constants.one_byte_percentile
		self.auto_clip_upper = LongExposureStandard.constants.two_bytes_percentile if self.output_dtype == np.uint16 else LongExposureStandard.constants.one_byte_percentile

	def setup_transform(self,transform_settings=None,**kwargs):
		self.transform_settings = {}
		if transform_settings is None:
			self.transform_settings["flip"] = ""
		else:
			self.transform_settings["flip"] = transform_settings["flip"] if transform_settings["flip"] in self.FLIP_OPTIONS else ""


	def initialize(self,first_frame=None):
		if self.canvas is None:
			return False
		self.input_frame_buffer = LongExposureCanvas.generate_input(self.canvas.shape)
		self.frame_counter = 0
		if first_frame is not None:
			self.add_frame(first_frame)


	def add_image_data(self,img_data):
		np.add(self.canvas,img_data,out=self.canvas)

	def add_frame(self,img_data):
		if img_data.shape != self.canvas_dims:
			img_data = cv_resize(img_data,(self.canvas_dims[1],self.canvas_dims[0]))
		if self.transform_settings["flip"] in self.FLIP_OPTIONS:
			img_data = cv_flip(img_data,self.FLIP_OPTIONS[self.transform_settings["flip"]])

		if len(self.input_frame_filters) < 1:
			self.add_image_data(img_data)

		np.add(img_data,0,self.input_frame_buffer,dtype=LongExposureStandard.dtype_of.input_buffer)

		for input_filter in self.input_frame_filters:
			input_filter.process_frame(self.input_frame_buffer)

		self.add_image_data(self.input_frame_buffer)
		self.frame_counter += 1

	def set_process_frame(self,process_frame):
		if process_frame.shape == self.canvas.shape:
			self.process_frame = process_frame

	def create_new_process_frame(self):
		self.process_frame = LongExposureCanvas.generate_process(self.canvas.shape)

	def convert_input_to_process(self):
		if "manual_clip_lower" in self.input_analysis_data:
			manual_clip_lower = self.input_analysis_data["manual_clip_lower"]
		if "manual_clip_upper" in self.input_analysis_data:
			manual_clip_upper = self.input_analysis_data["manual_clip_upper"]

		LongExposureCanvas.convert_to_process_frame(self.canvas,self.canvas,manual_clip_lower=self.manual_clip_lower , manual_clip_upper = self.manual_clip_upper , auto_clip_lower = self.auto_clip_lower , auto_clip_upper = self.auto_clip_upper)

	def run_input_post_processors(self):
		self.run_image_processor(self.canvas,self.input_post_processors,self.input_analysis_data,self.input_analysis_log)
	def run_process_frame_filters(self):
		self.run_image_processor(self.canvas,self.process_frame_filters,self.process_analysis_data,self.process_analysis_log)
	def run_image_processor(self,image_data,processor_queue,analysis_data,analysis_log):
		if len(processor_queue) < 1:
			return
		post_processors = deque(processor_queue)
		while len(post_processors) > 0:
			processor = post_processors.popleft()
			if isinstance(processor,LongExposureProcessFilter):
				processor.process_frame(image_data)
				# remove all analysis data after processing
				analysis_data_keys = list(analysis_data.keys())
				for key in analysis_data_keys:
					analysis_data.pop(key)
			elif isinstance(processor,LongExposureProcessAnalyser):
				processor.analyse(image_data,post_processors,analysis_data,analysis_log)

	def output_process_data(self,output_frame=None):
		return LongExposureCanvas.convert_to_output_frame(self.canvas,output_frame=output_frame,output_dtype=self.output_dtype)

	def create(self,output_frame=None):
		output_frame = self.output_frame if output_frame is None else output_frame
		self.run_input_post_processors()
		self.convert_input_to_process()
		self.run_process_frame_filters()
		return self.output_process_data(output_frame=output_frame)





