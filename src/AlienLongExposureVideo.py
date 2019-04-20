from AlienLongExposure import LongExposure,LongExposureStandard,LongExposureCanvas
from AlienImageSequenceGenerator import ImageSequenceOutput , VideoOutput
import numpy as np
from math import ceil as math_ceil

'''
	LongExposureVideo

	extends AlienLongExposure.LongExposure

	This creates moving long exposures from image sequences. It essentially creates multiple long exposures sequentially.

	For CPU and memory optimisation, this extends the LongExposure class as processed input data can be used in multiple frames
'''
class LongExposureVideo(LongExposure):
	'''
		setup_canvas

		overrides setup_canvas from LongExposure

		This sets up the multiple canvases used for processing the multiple long exposures that are generated simultaneously.
		It determines the maximum number of canvases needed for this.

		arguments:

		canvas:					None / Tuple / np.array

								None - no details of image data provided, will be initialized yet (do not recommend doing this)
								Tuple - tuple describing the shape of the image data (height,width,channels)
								np.array - image data of first frame

		exposure_frame_count:	number of frames to be added to each frame exposure

		exposure_frame_weight:	None / list

								None - all frames will be added with equal weight
								list - list of floats, from -1.0 to 1.0, describing the weight each frame will add, depending
											on the point it was added.

		step:					number of steps to start of next output frame (1 or greater)
	'''
	def setup_canvas(self,canvas,exposure_frame_count=12,exposure_frame_weight=None,step=1,**kwargs):
		
		self.canvas_dims = canvas if isinstance(canvas,tuple) else (canvas.shape if canvas is not None else None)
		self.initialised = canvas is not None


		self.exposure_frame_count = exposure_frame_count
		
		# exposure_frame_weight is a weighting added to the frame, dependent on when it was added in the exposure period
		if exposure_frame_weight is None or len(exposure_frame_weight) < exposure_frame_count:
			print("no weights provided" if exposure_frame_weight is None else "not enough weights provided" )
			self.exposure_frame_weight = None
		else:
			e_max = max(exposure_frame_weight)
			e_min = min(exposure_frame_weight)
			if e_max == e_min:
				print("min/max weights the same")
				self.exposure_frame_weight = None
			else:
				e_mod = max(e_max,abs(e_min))
				if e_mod>1:
					self.exposure_frame_weight = [e/e_mod for e in exposure_frame_weight]
				else:
					self.exposure_frame_weight = [e for e in exposure_frame_weight]
		

		self.step = 1 if step < 1 else int(step)
		self.step_counter = -1
		self.current_canvas = -1

		#self.required_canvases = 1 + self.exposure_frame_count - self.step
		self.required_canvases = math_ceil(self.exposure_frame_count / self.step)
		print("required_canvases",self.required_canvases)

		self.canvases = [None] * self.required_canvases
		self.canvas_tracker = [0] * self.required_canvases

		if self.initialised:
			self.initialize_canvases()

	def initialize_canvases(self):
		for c in range(self.required_canvases):
			self.canvases[c] = LongExposureCanvas.generate_input(self.canvas_dims)
		self.canvas = self.canvases[0]
		self.exposure_weight_buffer = LongExposureCanvas.generate_input(self.canvas_dims) if self.exposure_frame_weight is not None else None


	def add_image_data(self,img_data):
		new_canvas = -1
		self.step_counter +=1
		self.step_counter %= self.step
		if self.step_counter == 0:
			self.current_canvas += 1
			self.current_canvas %= self.required_canvases
			new_canvas = self.current_canvas

			if self.canvas_tracker[self.current_canvas] > 0:
				self.canvas = self.canvases[self.current_canvas]
				rendered_frame = self.create()
				self.output_generator.add_frame(rendered_frame)

			# reset to zero
			np.multiply(self.canvases[self.current_canvas],0,out=self.canvases[self.current_canvas])
			self.canvas_tracker[self.current_canvas] = 0

		for i in range(0,self.required_canvases):
			if self.canvas_tracker[i] > 0 or new_canvas == i:
				if self.exposure_frame_weight:
					np.multiply(self.exposure_frame_weight[self.canvas_tracker[i]],img_data,self.exposure_weight_buffer)
					np.add(self.canvases[i],self.exposure_weight_buffer,out=self.canvases[i])
				else:
					np.add(self.canvases[i],img_data,out=self.canvases[i])
				self.canvas_tracker[i]+=1


	def setup_output(self,output_dtype=LongExposureStandard.dtype_of.output_default,output_generator=None,**kwargs):
		self.output_dtype = output_dtype
		self.output_generator = output_generator
		self.output_frame = LongExposureCanvas.generate_canvas(self.canvas_dims,output_dtype,0) if self.initialised else None


	def initialize(self,first_frame=None):
		self.input_frame_buffer = LongExposureCanvas.generate_input(self.canvas.shape)
		self.output_generator.change_dims(self.canvas.shape)
		self.output_generator.initialize()
		self.frame_counter = 0
		if first_frame is not None:
			self.add_frame(first_frame)

	def finalize(self):
		self.output_generator.finalize()



