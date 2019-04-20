import os
from cv2 import resize as cv_resize , imwrite as cv_imwrite , VideoWriter as cv_VideoWriter


class ImageSequenceGenerator:
	initialized = False
	def __init__(self,dims=(720,1280,3),name="ImageSequence"):
		self.setup_dims(dims)
	def setup_dims(self,dims):
		self.dims = dims
	def change_dims(self,dims):
		if not self.initialized:
			self.dims = dims

	def setup_output(self,folder,name):
		self.name = name
		folder = folder.replace("\\","/")
		self.folder = folder
		if not os.path.isdir(folder):
			subfolders = folder.split("/")
			breadcrumbs = subfolders[0]
			if not os.path.isdir(breadcrumbs):
				os.mkdir(breadcrumbs)
			subfolders = subfolders[1:]
			for s in subfolders:
				breadcrumbs = breadcrumbs + "/" + s
				if not os.path.isdir(breadcrumbs):
					print("creating folder:"+breadcrumbs)
					os.mkdir(breadcrumbs)

	def add_frame(self,data):
		if data.shape == self.dims:
			self.save_frame(data)
		else:
			self.save_frame(self.resize_data(data))

	def resize_data(self,data):
		self.save_frame(cv_resize(data,(self.dims[1],self.dims[0])))

	def save_frame(self,data):
		pass

	def initialize(self):
		self.initialized = True

	def finalize(self):
		pass




class ImageSequenceOutput(ImageSequenceGenerator):
	def __init__(self,dims=(720,1280,3),name="ImageSequence",folder="",image_type="jpg",start_at=0,padding = 4):
		self.setup_dims(dims)
		self.setup_output(folder,name)
		self.setup_image(image_type)
		self.padding = -padding
		self.padding_string = "0" * padding
		self.current_frame = start_at -1
		self.saved_frames = []

	def setup_image(self,image_type):
		self.extension = image_type

	def save_frame(self,data):
		self.current_frame += 1
		filename = self.folder + "/" + self.name + "_" + (self.padding_string+str(self.current_frame))[self.padding:]+"."+self.extension
		cv_imwrite(filename,data)
		self.saved_frames.append(filename)


	def finalize(self,save_list=True):
		if save_list:
			filename = self.folder + "/" + self.name + ".txt"
			with open(filename,"w") as output_text:
				output_text.write("\n".join(self.saved_frames))
				output_text.close()



class VideoOutput(ImageSequenceGenerator):
	def __init__(self,dims=(720,1280,3),name="ImageSequence",folder="",video_type="jpg",codec="x264",frame_rate=30):
		self.setup_dims(dims)
		self.setup_output(folder,name)
		self.setup_codec(video_type,codec,frame_rate)


	def setup_codec(self,video_type,codec,frame_rate):
		self.extension = video_type
		self.codec = codec 
		self.frame_rate = frame_rate

	def save_frame(self,data):
		self.video.write(data)

	def initialize(self):
		print("initializing video: "+self.name+" at location "+self.folder)
		self.video = cv_VideoWriter(self.folder+"/"+self.name+"."+self.extension, # Filename
            self.codec, # Negative 1 denotes manual codec selection. You can make this automatic by defining the "fourcc codec" with "cv2.VideoWriter_fourcc"
            self.frame_rate, # 10 frames per second is chosen as a demo, 30FPS and 60FPS is more typical for a YouTube video
            (self.dims[1],self.dims[0]) # The width and height come from the stats of image1
		)
		self.initialized = True

	def finalize(self):
		print("finalizing video")
		self.video.release()
		print("video saved")


