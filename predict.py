from keras.models import model_from_json
import cv2
import numpy as np
#import os
# import pickle
# from sklearn.utils import shuffle
#import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score

json_file = open("final_model1_read.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("final_model1_read.h5")


def contrast_norm(image):
	img = cv2.resize(image, (32, 32))
	img = img.astype(np.uint8)
	for c in range(3):
		img[:, :, c] = cv2.equalizeHist(img[:, :, c])
	img = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]*3, axis=-1)
	for i in range(3):
		img[:, :, i] = img[:, :, i] * (255.0 / img[:, :, i].max())
	img = np.moveaxis(img, 2, 0)

	return np.expand_dims(img, axis=0)


def predict(img):
	#img = cv2.imread(img)
	img = contrast_norm(img)
	out = model.predict(img)
	pred = np.argmax(out)
	prob = out[0][np.argmax(out)]
	prob_std = np.std(out)
	# print(out)
	# print('std = ' + str(np.std(out)))
	return pred, prob, prob_std


PROB_THRESH = 0.7  # classified sign vs else
STD_THRESH = 0.1  # non-sign vs non-classified sign

# data_dir = '../test_images/'
# X, Y, Y_test = [], [], []
# for cls in os.listdir(data_dir):
# 	img_dir = data_dir+cls+'/'
# 	for img_name in os.listdir(img_dir):
# 		img = cv2.imread(img_dir+img_name)
# 		img = img.astype(np.uint8)
# 		for c in range(3):
# 			img[:, :, c] = cv2.equalizeHist(img[:, :, c])
# 		out = cv2.resize(img, (32, 32))
# 		#out = np.moveaxis(out, 2, 0)
# 		X.append(np.asarray(out))
# 		y = np.eye(37, dtype='uint8')[int(cls)]
# 		Y.append(y)
# 		Y_test.append(int(cls))
# #
# # with open(data_dir + 'X_test.p', 'wb') as f:
# # 		pickle.dump(np.array(X), f)
# #
# # with open(data_dir + 'y_test.p', 'wb') as f:
# # 	pickle.dump(np.array(Y), f)
#
# X, Y, Y_test = shuffle(X, Y, Y_test, random_state=0)
#
# X = []
# for imname in os.listdir(data_dir):
# 	img = cv2.imread(data_dir+imname)
# 	out = cv2.resize(img, (200, 100))
# 	X.append(np.asarray(out))
#
#
# fig = plt.figure(figsize=(150, 100))
# for i in range(6):
# 	img = X[i]
# 	fig.add_subplot(2, 3, i+1)
# 	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
#
# y_pred = [np.argmax(model.predict(np.asarray([np.moveaxis(img, 2, 0)]))) for img in X]
# print(y_pred)
# print(Y_test)
# count, correct = 0, 0
# for i in range(len(y_pred)):
# 	if y_pred[i] == Y_test[i]:
# 		correct += 1
# 	count += 1
# print(correct/count)
# print(accuracy_score(y_pred, Y_test))




### Non-classified sign
# [[6.1113016e-05 4.5888345e-03 6.8805598e-02 4.4876698e-01 1.7901228e-03
#   3.3214014e-02 2.6507974e-03 3.4297723e-02 3.9078292e-02 3.8747936e-03
#   2.1514152e-03 8.9929655e-02 3.3995940e-03 1.5297895e-03 8.7913591e-03
#   1.8616934e-01 2.8141001e-03 2.1035725e-02 4.7050726e-02]]
# std = 0.1033

# [[9.1467331e-09 4.7717581e-08 4.1715931e-02 3.2967636e-03 1.3071583e-07
#   2.7517331e-01 7.8874479e-08 2.2461718e-06 5.8744531e-06 4.0667236e-04
#   9.9949553e-08 1.9305566e-02 4.5803529e-07 1.3164685e-05 4.4981243e-06
#   2.4230172e-05 2.5342679e-05 1.0701948e-03 6.5895545e-01]]
# std = 0.1555

# [[1.5883229e-06 1.4103388e-03 5.6657554e-03 3.0434239e-01 2.1447986e-03
#   1.0842450e-02 1.6830593e-03 2.0200463e-01 1.3732511e-03 3.8703111e-01
#   1.5420929e-04 2.8035441e-02 1.6544030e-04 5.8133275e-05 3.9225370e-03
#   5.3342683e-03 1.2573402e-03 3.4907248e-02 9.6659940e-03]]
# std = 0.110714436


### Non sign
# [[1.3360115e-04 1.0159642e-03 6.1950557e-02 1.0293668e-02 8.7458314e-03
#   7.4194513e-02 3.2765123e-03 1.4079451e-03 1.2026933e-02 1.3473975e-02
#   1.2597493e-03 7.8805439e-02 2.2546302e-03 2.4088223e-03 1.8441371e-03
#   1.8615869e-01 2.2553621e-02 2.7273563e-01 2.4545984e-01]]
# std = 0.08374733
