import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from preprocessing.predict import predict

sign_dict = {
1: "Stop Ahead",
2:	"Maximum 80",
3:	"Maximum 40",
4:	"Yield Sign",
5:	"Maximum 50",
6:	'Right Curve Sign',
7:	'Left Curve Sign',
8:	'Maximum 60 ahead',
9:	'Construction ahead',
10:	'Stop Sign',
11:	'Maximum 30',
12:	'Maximum 30',
13:	'Yield Ahead Sign',
14:	'Signal Ahead Sign',
15:	'Maximum 50 ahead',
16:	'Maximum 60',
17:	'Maximum 70',
18:	'Guide Sign'
}

def video_to_frames(vedio_path):
	vidcap = cv2.VideoCapture(vedio_path)
	success, image = vidcap.read()
	print("image shape:", image.shape)
	img_set = []
	count = 0
	while success:
		# Save frame as JPEG file
		cv2.imwrite("frame%d.jpg" % count, image)
		# Get 1 frame each 1 second:
		vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
		success, image = vidcap.read()
		img_set.append(image)
		print('Read a new frame: ', success)
		count += 1
	print("length of img_set", len(img_set))
	return img_set

def video_to_stream(video_path):
	out = cv2.VideoWriter('sample_images/test_video.mov', cv2.VideoWriter_fourcc(
		'm', 'p', '4', 'v'), 1, (640, 360))
	cap = cv2.VideoCapture(video_path)
	frame_i = 0
	classes = None
	while (cap.isOpened()):
		# Read the frame
		ret, frame = cap.read()
		if not frame_i % 5:
			classes = process_frame(frame)
		if classes:
			for pos, pred in classes:
				if int(pred) > 18:
					continue
				x, y, w, h = pos[0]
				color, contour_thickness = (0, 255, 0), 2
				frame = np.asarray(frame)
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, contour_thickness)
				frame = Image.fromarray(frame)
				draw = ImageDraw.Draw(frame)
				draw.text((x, y), str(sign_dict[int(pred)]), fill=color)
				frame = np.asarray(frame)
		cv2.imshow('frame', frame)
		output_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		out.write(output_rgb)
		cv2.waitKey(33)
		frame_i += 1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	out.release()
	cap.release()
	cv2.destroyAllWindows()

def frames_to_video(input_img_list, outputpath, fps):
	image_array = []
	for img in input_img_list:
		h = img.shape[1]
		w = img.shape[0]
		size = (h, w)
		img = cv2.resize(img, size)
		image_array.append(img)
	fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
	out = cv2.VideoWriter(outputpath, fourcc, fps, size)
	for i in range(len(image_array)):
		out.write(image_array[i])
	out.release()


def load_prep_img(sample_img):
	size = sample_img.shape
	blank_canvas_red = np.zeros(size, dtype=np.uint8)
	blank_canvas_yellow_orange = np.zeros(size, dtype=np.uint8)
	blank_canvas_white = np.zeros(size, dtype=np.uint8)
	# Convert to RGB color space
	rgb_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
	# Convert to HSV color space
	hsv_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)
	# Blur image to reduce unrelated noise
	blur_img = cv2.bilateralFilter(hsv_img, 9, 75, 75)
	return blank_canvas_red, blank_canvas_yellow_orange, blank_canvas_white, sample_img, rgb_img, hsv_img, blur_img


def get_red_mask(img, lower_color1, upper_color1, lower_color2, upper_color2):
	masked_img1 = cv2.inRange(img, lower_color1, upper_color1)
	masked_img2 = cv2.inRange(img, lower_color2, upper_color2)
	loose_red_mask = masked_img1 + masked_img2
	return loose_red_mask


def get_other_colors_mask(img, lower_color, upper_color):
	masked_img = cv2.inRange(img, lower_color, upper_color)
	other_colors_mask = masked_img
	return other_colors_mask


def dilate_erode(img, kernel_size):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	dilated_img = cv2.dilate(img, kernel, iterations=1)
	cleaned_img = cv2.erode(dilated_img, kernel, iterations=1)
	return cleaned_img


def find_contour(img, lower_area_thres, upper_area_thres):
	_, all_contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	contours = [cnt for cnt in all_contours if lower_area_thres < cv2.contourArea(cnt) < upper_area_thres]
	return contours


def draw_contour_2(img, canvas, contour_thickness):
	_, all_contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	all_contours = sorted(all_contours, key=cv2.contourArea)
	if all_contours is not None:
		cv2.drawContours(canvas, all_contours[-1], -1, [0, 255, 0], contour_thickness)


def fix_convex_defect(cropped_img):
	_, contours, _ = cv2.findContours(cropped_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	contour = sorted(contours, key=cv2.contourArea)[-1]
	# Detect convexity defects base on the contour.
	hull_before = cv2.convexHull(contour, returnPoints=False)
	# print(type(hull_before))
	defects_before = cv2.convexityDefects(contour, hull_before)
	if defects_before is not None:
		for i in range(defects_before.shape[0]):
			s_before, e_before, f_before, d_before = defects_before[i, 0]
			start_before = tuple(contour[s_before][0])
			end_before = tuple(contour[e_before][0])
			far_before = tuple(contour[f_before][0])
			cv2.line(cropped_img, start_before, end_before, [255, 0, 0], 1)
		convex_fixed_img = dilate_erode(cropped_img, 3)
		_, fixed_contours, _ = cv2.findContours(convex_fixed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		fixed_contour = sorted(fixed_contours, key=cv2.contourArea)[-1]
		return fixed_contour
	else:
		return contour


def load_template(template_path):
	template = cv2.imread(template_path)
	blurred_template = cv2.bilateralFilter(template, 9, 75, 75)
	gray_scale_template = cv2.cvtColor(blurred_template, cv2.COLOR_BGR2GRAY)
	ret, thresh2 = cv2.threshold(gray_scale_template, 127, 255, 0)
	_, template_contours, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	template_cnt = template_contours[0]
	return gray_scale_template, template_cnt


def shape_compare(contours, template_cnt):
	similarity_level = cv2.matchShapes(contours, template_cnt, 1, 0.0)
	return similarity_level


def draw_box(img, canvas, contour_thickness):
	pos = []
	_, all_sorted_cnt = find_contour(img, 10, 100)
	for i, cnt in enumerate(all_sorted_cnt):
		colour = (0, 255, 0)
		x, y, w, h = cv2.boundingRect(cnt)
		# cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, contour_thickness)
		pos.append([x, y, w, h])
	# print(pos)
	return pos, canvas


def crop_and_hist(original_img, target_region_pos, border):
	[img_h, img_w] = original_img.shape[:2]
	[x, y, w, h] = target_region_pos[0]
	# Original position: (should not chage this!)
	box_x = [x, x + w]
	box_y = [y, y + h]
	crop_x0 = box_x[0] - border*img_w*0.01
	crop_y0 = box_y[0] - border*img_h*0.01
	crop_x1 = box_x[1] + border*img_w*0.01
	crop_y1 = box_y[1] + border*img_h*0.01
	if crop_x0 < 0:
		crop_x0 = 0
	else:
		crop_x0 = crop_x0
	if crop_y0 < 0:
		crop_y0 = 0
	else:
		crop_y0 = crop_y0
	if crop_x1 > img_w:
		crop_x1 = img_w
	else:
		crop_x1 = crop_x1
	if crop_y1 > img_h:
		crop_y1 = img_h
	else:
		crop_y1 = crop_y1
	# New position:
	x_exp = [crop_x0, crop_x1]
	y_exp = [crop_y0, crop_y1]
	w_exp = x_exp[1] - x_exp[0]
	h_exp = y_exp[1] - y_exp[0]
	if w_exp > h_exp:
		diff = w_exp - h_exp
		h_exp = w_exp
		if y_exp[0] - round(diff / 2) < 0:
			y_exp_new0 = 0
			remains_h = round(diff / 2) - y_exp[0]
			y_exp_new1 = y_exp[1] + round(diff / 2) + remains_h
		else:
			y_exp_new0 = y_exp[0] - round(diff / 2)
			if y_exp[1] + round(diff / 2) > img_h:
				y_exp_new1 = img_h
				remains_h = y_exp[1] + round(diff / 2) - img_h
				y_exp_new0 = y_exp[0] - round(diff / 2) - remains_h
			else:
				y_exp_new1 = y_exp[1] + round(diff / 2)
		y_exp_new = [y_exp_new0, y_exp_new1]
		x_exp_new = x_exp
	else:
		diff = h_exp - w_exp
		w_exp = h_exp
		if x_exp[0] - round(diff / 2) < 0:
			x_exp_new0 = 0
			remains_w = round(diff / 2) - x_exp[0]
			x_exp_new1 = x_exp[1] + round(diff / 2) + remains_w
		else:
			x_exp_new0 = x_exp[0] - round(diff / 2)
			if x_exp[1] + round(diff / 2) > img_w:
				x_exp_new1 = img_w
				remains_w = x_exp[1] + round(diff / 2) - img_w
				x_exp_new0 = x_exp[0] - round(diff / 2) - remains_w
			else:
				x_exp_new1 = x_exp[1] + round(diff / 2)
		y_exp_new = y_exp
		x_exp_new = [x_exp_new0, x_exp_new1]

	target_region_img = original_img[y_exp_new[0]:y_exp_new[1], x_exp_new[0]:x_exp_new[1], :].copy()
	print("target_region_img shape:", target_region_img.shape)
	return target_region_img


def crop_possible_signs(similarity_level_ori, similarity_level_fix, final_mask, rgb_img):
	if similarity_level_ori < 0.8 or similarity_level_fix < 0.8:
		target_region_pos, box_on_img = draw_box(final_mask, rgb_img, 1)
		#print(target_region_pos)
		#         plt.imshow(box_on_img)
		#         plt.show()
		x, y, w, h = target_region_pos[0]
		if w*h < 1000:
			return None, None
		border = 8
		target_region_img_before_eq = crop_and_hist(rgb_img, target_region_pos, border)
		output = target_region_img_before_eq
	else:
		return None, None
	return target_region_pos, output


def draw_box_text(final_mask, img_draw_box, contour_thickness):
	pos = []
	_, all_sorted_cnt = find_contour(final_mask)
	for i, cnt in enumerate(all_sorted_cnt):
		colour = (255, 0, 0)
		x, y, w, h = cv2.boundingRect(cnt)
		cv2.rectangle(img_draw_box, (x, y), (x + w, y + h), colour, contour_thickness)
		pos.append([x, y, w, h])
	return pos, img_draw_box


def text_box_on_img(final_mask, img_draw_box, text):
	target_region_pos, img_draw_box = draw_box_text(final_mask, img_draw_box, contour_thickness=2)
	li = target_region_pos[0]
	x = li[0]
	y = li[1]
	color = (0, 255, 0)
	img_draw_box = Image.fromarray(img_draw_box)
	draw = ImageDraw.Draw(img_draw_box)
	draw.text((x, y), text, fill=color)
	img_draw_box = np.asarray(img_draw_box)
	return img_draw_box


# Load in the Octagon template and get its contours for red canvas
octagon_template_loc = 'red_octagon.jpeg'
gray_octagon_template, octagon_template_cnt = load_template(octagon_template_loc)
# plt.imshow(gray_octagon_template,cmap='gray')
# plt.show()

# Load in the Rectagule template and get its contours for white canvas
rectangle_template_loc = 'rectangle.png'
gray_rectangle_template, rectangle_template_cnt = load_template(rectangle_template_loc)
# plt.imshow(gray_rectangle_template,cmap='gray')
# plt.show()

# Load in the Dimond shape template and get its contours for red canvas
dimond_shape_template_loc = 'dimond_shape.jpeg'
gray_dimond_shape_template, dimond_shape_template_cnt = load_template(dimond_shape_template_loc)
# plt.imshow(gray_dimond_shape_template,cmap='gray')
# plt.show()


# Read image(whatever color) as sample_img and display it.
video_path = '../sample_images/video_1.mov'
# input_frames_set = video_to_frames(video_path)
# print(len(input_frames_set))
# frame_i = 0

processed_frames = []
c = 0
#for frame in input_frames_set[:len(input_frames_set) - 1]:
def process_frame(frame):
	output_labeled_img_list = []
	output_red_list = []
	output_yellow_orange_list = []
	output_white_list = []
	output_green_list = []

	#print("start processing frame:", frame_i)
	blank_canvas_red, blank_canvas_yellow_orange, blank_canvas_white, sample_img, rgb_img, hsv_img, blur_img = load_prep_img(
		frame)
	r, c = frame.shape[:-1]
	lower_area_thres, upper_area_thres = r*c*0.0007, r*c*0.05

	# Red:
	lower_red1, upper_red1, lower_red2, upper_red2 = (0, 70, 0), (10, 255, 255), (165, 70, 0), (180, 255, 255)
	red_loose_mask = get_red_mask(blur_img, lower_red1, upper_red1, lower_red2, upper_red2)
	red_cleaned_img_1 = dilate_erode(red_loose_mask, 3)

	lower_red3, upper_red3, lower_red4, upper_red4 = (0, 100, 50), (10, 255, 255), (170, 100, 50), (180, 255, 255)
	red_strict_mask = get_red_mask(blur_img, lower_red3, upper_red3, lower_red4, upper_red4)
	red_cleaned_img_2 = dilate_erode(red_strict_mask, 15)

	final_red_mask = cv2.bitwise_and(red_cleaned_img_1, red_cleaned_img_2)

	# Yellow:
	# lower_yellow_orange, upper_yellow_orange = (5, 70, 0), (15, 255, 200)
	# yellow_orange_loose_mask = get_other_colors_mask(blur_img, lower_yellow_orange, upper_yellow_orange)
	# yellow_orange_cleaned_img_1 = dilate_erode(yellow_orange_loose_mask, 3)

	lower_yellow_orange2, upper_yellow_orange2 = (5, 150, 30), (35, 255, 200)
	yellow_orange_strict_mask = get_other_colors_mask(blur_img, lower_yellow_orange2, upper_yellow_orange2)
	kernel = np.ones((3, 3), np.uint8)
	erosion = cv2.erode(yellow_orange_strict_mask, kernel, iterations=2)
	final_yellow_orange_mask = erosion
	yellow_orange_cleaned_img_2 = dilate_erode(yellow_orange_strict_mask, 15)

	# final_yellow_orange_mask = cv2.bitwise_and(yellow_orange_cleaned_img_1, yellow_orange_cleaned_img_2)

	# Green:
	lower_green, upper_green = (45, 90, 30), (95, 255, 200)
	green_loose_mask = get_other_colors_mask(blur_img, lower_green, upper_green)
	green_cleaned_img_1 = dilate_erode(green_loose_mask, 3)
	final_green_mask = green_loose_mask

	# lower_green2, upper_green2 = (50, 100, 50), (80, 255, 255)
	# green_strict_mask = get_other_colors_mask(blur_img, lower_green2, upper_green2)
	# green_cleaned_img_2 = dilate_erode(green_strict_mask, 15)
	#
	# final_green_mask = cv2.bitwise_and(green_cleaned_img_1, green_cleaned_img_2)
	# cv2.imshow('Yellow Mask', final_green_mask)
	# cv2.waitKey(0)

	# Red:
	# Draw contours on red_canvas before fixing convex defects.
	red_contours = find_contour(final_red_mask, lower_area_thres, upper_area_thres)
	if red_contours is not None:
		# Compare similarity:
		# similarity_level_ori_red = shape_compare(largest_ori_red_cnt, octagon_template_cnt)

		# Re-draw contour after fixing convex defect
		for cnt in red_contours:
			x, y, w, h = cv2.boundingRect(cnt)
			if w < 0.45*h or h < 0.45*w:
				continue
			fixed_shape = fix_convex_defect(final_red_mask[y:y+h, x:x+w])
			similarity_level_fix_red = shape_compare(fixed_shape, octagon_template_cnt)
			if similarity_level_fix_red < 0.2:
				output_red_list.append(([[x, y, w, h]], frame[y:y+h, x:x+w]))

	# Yellow:
	# Draw contours on yellow_orange_canvas before fixing convex defects.
	yellow_contours = find_contour(final_yellow_orange_mask, lower_area_thres, upper_area_thres)
	if yellow_contours is not None:
		for cnt in yellow_contours:
			x, y, w, h = cv2.boundingRect(cnt)
			# if w < 0.45*h or h < 0.45*w:
			# 	continue
			fixed_shape = fix_convex_defect(final_yellow_orange_mask[y:y+h, x:x+w])
			similarity_level_fix_yellow = shape_compare(fixed_shape, dimond_shape_template_cnt)
			if similarity_level_fix_yellow < 0.2:
				output_yellow_orange_list.append(([[x, y, w, h]], frame[y:y+h, x:x+w]))

	# Green:
	# Draw contours on yellow_orange_canvas before fixing convex defects.
	green_contours = find_contour(final_green_mask, lower_area_thres, upper_area_thres)
	if green_contours is not None:
		for cnt in green_contours:
			x, y, w, h = cv2.boundingRect(cnt)
			# if w < 0.45*h or h < 0.45*w:
			# 	continue
			fixed_shape = fix_convex_defect(final_green_mask[y:y+h, x:x+w])
			similarity_level_fix_green = shape_compare(fixed_shape, rectangle_template_cnt)
			if similarity_level_fix_green < 0.2:
				output_green_list.append(([[x, y, w, h]], frame[y:y+h, x:x+w]))

	# White:
	kernel = np.ones((3, 3), np.uint8)
	erosion = cv2.erode(frame, kernel, iterations=1)
	#cv2.imshow('mask', erosion)
	#cv2.waitKey(0)
	dilation = cv2.dilate(erosion, kernel, iterations=1)
	erosion = cv2.erode(dilation, kernel, iterations=1)
	retval, threshold = cv2.threshold(erosion, 120, 240, cv2.THRESH_BINARY)

	threshold = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
	#cv2.imshow('mask', threshold)
	#cv2.waitKey(0)
	_, cnts, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cnts = sorted(cnts, key=cv2.contourArea)[::-1]
	for cnt in cnts:
		x, y, w, h = cv2.boundingRect(cnt)
		#cv2.imshow('cropped', frame[y:y+h, x:x+w])
		#cv2.waitKey(0)
		if w * h < r * c * 0.0008 or w * h > r * c * 0.05 or w > h or w < h * 0.55:
			continue
		output_white_list.append(([[x, y, w, h]], frame[y:y + h, x:x + w]))

	#data_dir = "D:\Yanxi\MMGRAD\MM803\Project/new dataset1/train_negative/"

	cropped_img_input_list = output_red_list + output_yellow_orange_list + output_green_list + output_white_list
	if not cropped_img_input_list:
		pass
	#file_i = 0
	classes = []
	# for pos, img in cropped_img_input_list:
	# 	pred, prob, prob_std = predict(img)
	# 	if prob_std < 0.1 or prob < 0.7:
	# 		#classes.append((None, None))
	# 		#cv2.imwrite(filename=data_dir + 'missed' + '/' + str(video_path.split('/')[-1]) + str(frame_i) + '_' + str(
	# 			#file_i) + '.jpg', img=img)
	# 		continue
	# 	#cv2.imwrite(filename=data_dir+str(pred)+'/'+str(video_path.split('/')[-1])+str(frame_i)+'_'+str(file_i)+'.jpg', img=img)
	# 	classes.append((pos, pred))
	for pos, img in output_red_list:
		pred, prob, prob_std = predict(img)
		if int(pred) not in [4, 10] or prob_std < 0.1 or prob < 0.7:
			# cv2.imwrite(
			# 	filename='D:\Yanxi\MMGRAD\MM803\Project\Cropped/' + str(pred)+ '_' + str(prob) + str(video_path.split('/')[-1]) + str(
			# 		c) + '.jpg', img=img)
			# c += 1
			continue
		# cv2.imwrite(
		# 	filename='D:\Yanxi\MMGRAD\MM803\Project\Cropped/' + 'positives/'+ str(pred)+ '_'+ str(prob) + str(video_path.split('/')[-1]) + str(
		# 		c) + '.jpg', img=img)
		# c += 1
		classes.append((pos, pred))

	for pos, img in output_yellow_orange_list:
		pred, prob, prob_std = predict(img)

		if int(pred) not in [0, 1, 6, 7, 9, 13, 14] or prob_std < 0.1 or prob < 0.7:
			# cv2.imwrite(
			# 	filename='D:\Yanxi\MMGRAD\MM803\Project\Cropped/' + str(pred)+ '_'+ str(prob) + str(video_path.split('/')[-1]) + str(
			# 		c) + '.jpg', img=img)
			# c += 1
			continue
		# cv2.imwrite(
		# 	filename='D:\Yanxi\MMGRAD\MM803\Project\Cropped/' + 'positives/'+ str(pred)+ '_'+ str(prob) + str(video_path.split('/')[-1]) + str(
		# 		c) + '.jpg', img=img)
		# c += 1
		classes.append((pos, pred))

	for pos, img in output_green_list:
		pred, prob, prob_std = predict(img)

		if int(pred) != 18 or prob_std < 0.1 or prob < 0.7:
			# cv2.imwrite(
			# 	filename='D:\Yanxi\MMGRAD\MM803\Project\Cropped/' + str(pred)+ '_'+ str(prob) + str(video_path.split('/')[-1]) + str(
			# 		c) + '.jpg', img=img)
			# c += 1
			continue
		# cv2.imwrite(
		# 	filename='D:\Yanxi\MMGRAD\MM803\Project\Cropped/' + 'positives/'+ str(pred)+ '_'+ str(prob) + str(video_path.split('/')[-1]) + str(
		# 		c) + '.jpg', img=img)
		# c += 1
		classes.append((pos, pred))

	for pos, img in output_white_list:
		pred, prob, prob_std = predict(img)

		if int(pred) not in [2, 3, 5, 8, 11, 12, 15, 16, 17] or prob_std < 0.1 or prob < 0.7:
			# cv2.imwrite(
			# 	filename='D:\Yanxi\MMGRAD\MM803\Project\Cropped/' + str(pred)+ '_'+ str(prob) + str(video_path.split('/')[-1]) + str(
			# 		c) + '.jpg', img=img)
			# c += 1
			continue
		# cv2.imwrite(
		# 	filename='D:\Yanxi\MMGRAD\MM803\Project\Cropped/' + 'positives/'+ str(pred)+ '_'+ str(prob) + str(video_path.split('/')[-1]) + str(
		# 		c) + '.jpg', img=img)
		# c += 1
		classes.append((pos, pred))

	return classes


print('finish all frames!')
#for f in processed_frames:
	#plt.imshow(f)
	#plt.show()


# if len(output_red_list) != 0:
# 	a = 0
# 	for crop_red_img in output_red_list:
# 		Image.fromarray(crop_red_img).save('sample_images/cropped_video_8/output_output_red_%d.jpg' % a)
# 		a += 1
#
# if len(output_yellow_orange_list) != 0:
# 	b = 0
# 	for crop_yellow_orange_img in output_yellow_orange_list:
# 		Image.fromarray(crop_yellow_orange_img).save('sample_images/cropped_video_8/output_output_yellow_%d.jpg' % b)
# 		b += 1
#
# if len(output_green_list) != 0:
# 	c = 0
# 	for crop_green_img in output_green_list:
# 		Image.fromarray(crop_green_img).save('sample_images/cropped_video_8/output_output_green_%d.jpg' % c)
# 		c += 1
#
# if len(output_white_list) != 0:
# 	d = 0
# 	for crop_white_img in output_white_list:
# 		Image.fromarray(crop_white_img).save('sample_images/cropped_video_8/output_output_white_%d.jpg' % d)
# 		d += 1

video_path = '../sample_images/2.mp4'
stream = 0
video_to_stream(video_path)
#process_frame(cv2.imread('Capture2.jpg'))

# test_img = 'Capture.jpg'
# test_img = cv2.imread(test_img)
# cv2.imshow('img', test_img)
# cv2.waitKey(0)
#
# lower_yellow_orange, upper_yellow_orange = (11, 70, 0), (65, 255, 255)
# yellow_orange_loose_mask = get_other_colors_mask(test_img, lower_yellow_orange, upper_yellow_orange)
# yellow_orange_cleaned_img_1 = dilate_erode(yellow_orange_loose_mask, 3)
#
# cv2.imshow('mask', yellow_orange_cleaned_img_1)
# cv2.waitKey(0)
#
# lower_yellow_orange2, upper_yellow_orange2 = (11, 70, 30), (65, 255, 255)
# yellow_orange_strict_mask = get_other_colors_mask(test_img, lower_yellow_orange2, upper_yellow_orange2)
# yellow_orange_cleaned_img_2 = dilate_erode(yellow_orange_strict_mask, 15)
#
# final_yellow_orange_mask = cv2.bitwise_and(yellow_orange_cleaned_img_1, yellow_orange_cleaned_img_2)
# cv2.imshow('img', final_yellow_orange_mask)
# cv2.waitKey(0)
#
# contours = find_contour(final_yellow_orange_mask, 0, 10000000)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)
# for cnt in contours:
# 	x, y, w, h = cv2.boundingRect(cnt)
# 	cv2.imshow('cnt', test_img[y:y+h, x:x+w])
