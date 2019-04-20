from cv2 import VideoCapture as cv_VideoCapture , resize as cv_resize , imwrite as cv_imwrite , imread as cv_imread
from os import listdir as os_listdir, path as os_path

class ImageSequenceReader:
	def __init__(self,src,start_frame=0,end_frame=None):
		pass
	def get_next_frame(self):
		pass
	def get_dimensions(self):
		pass
	def has_finished(self):
		pass


class VideoSequenceReader(ImageSequenceReader):
	def __init__(self,src,start_frame=0,end_frame=None):
		self.video_capture = cv_VideoCapture(src)
		self.end_frame = end_frame
		self.counter = -1
		self.buffer = None
		self.complete = False
		img = self.advance_n_frames(start_frame + 1)
		height, width, channels = img.shape
		self.width = width
		self.height = height
		self.channels = channels

	def get_dimensions(self):
		return self.height,self.width,self.channels

	def get_next_frame(self):
		img = self.buffer
		self.advance_n_frames()
		return img

	def has_finished(self):
		return self.complete

	def advance_n_frames(self,n=1):
		success = True
		img = None
		counter = 0
		while counter < n and success:
			success , img = self.video_capture.read()
			counter += 1
		self.counter += counter
		success = success and (self.end_frame is None or self.end_frame>self.counter)
		self.buffer = img
		if img is None or not success:
			self.complete = True
			self.buffer = None
		return img


class ImagePathHandler:
	def __init__(self,src,extension="jpg",recurse=0,img_list=[]):
		self.img_list = img_list
		self.extension = (extension.lower() if extension[0] == "." else  "."+extension) if extension else ".jpg"

		ext_len = len(self.extension)
		neg_ext_len = -ext_len
		if isinstance(src,list):
			 
			self.img_list = [img_path for img_path in src if isinstance(img_path,str) and len(img_path)>ext_len and img_path[neg_ext_len:].lower() == self.extension]
		elif os_path.isdir(src):
			if recurse and recurse > 0:
				subfolders = [src+"/"+sub for sub in os_listdir(src) if os_path.isdir(src+"/"+sub) and sub != "." and sub != ".."]
				subfolders.sort()
				for subfolder in subfolders:
					ImagePathHandler(subfolder,extension,recurse-1,self.img_list)
			else:
				files = [src+"/"+file for file in os_listdir(src) if len(file)>ext_len and file[neg_ext_len:].lower() == self.extension]
				files.sort()
				self.img_list.extend(files)
		elif src[-4:].lower() == ".txt":
			if os_path.isfile(src):
				file_list = open(src,"r")
				self.img_list.extend(file_list)
		self.counter = 0
		self.size = len(self.img_list)

	def __len__(self):
		return len(self.img_list)

	def get_next_path(self):
		if self.has_next():
			r = self.img_list[self.counter]
			self.counter+=1
			return r
		return None

	def has_next(self):
		return self.counter<self.size

	def jump_to(self,frame):
		self.counter = frame




class ListOfImagePaths(ImageSequenceReader):
	def __init__(self,src,start_frame=0,end_frame=None):
		self.frame_list = ImagePathHandler(src)
		self.total_frames = len(self.frame_list)
		self.current_frame = start_frame - 1
		self.end_frame = end_frame if end_frame is not None and end_frame < self.frame_list.size else self.frame_list.size
		self.buffer = None
		self.frame_list.counter = self.current_frame
		self.get_next_frame()
		if self.buffer is not None:
			height , width , channels = self.buffer.shape()
		else:
			width = 0
			height = 0
			channels = 0
		self.width = width
		self.height = height
		self.channels = channels
	def get_next_frame(self):
		self.current_frame += 1
		frame_path = self.frame_list.get_next_path()
		if frame_path is None and (self.end_frame is None or  self.current_frame<self.end_frame):
			r = self.buffer
			self.buffer =  cv_imread(frame_path)
			return r
		else:
			return None
	def get_dimensions(self):
		return self.height,self.width,self.channels
	def has_finished(self):
		return not self.current_frame < self.end_frame	

