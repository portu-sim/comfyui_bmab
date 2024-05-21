import os
import time
import folder_paths


def get_output_directory():
	dd = time.strftime('%Y-%m-%d', time.localtime(time.time()))
	full_output_folder = os.path.join(folder_paths.output_directory, dd)
	print(full_output_folder)
	if not os.path.exists(full_output_folder):
		os.mkdir(full_output_folder)
	return full_output_folder


folder_paths.get_output_directory = get_output_directory
