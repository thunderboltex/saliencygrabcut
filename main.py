# coding: UTF-8
#from __future__ import division
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input
from keras.models import Model
import os, cv2, sys
import numpy as np
from config import *
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions
from models import sam_vgg, sam_resnet, kl_divergence, correlation_coefficient, nss


def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        fixs = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith('.mat')]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        fixs = [fixs_val_path + f for f in os.listdir(fixs_val_path) if f.endswith('.mat')]
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()
    fixs.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        Y = preprocess_maps(maps[counter:counter+b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps(fixs[counter:counter + b_s], shape_r_out, shape_c_out)
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian], [Y, Y, Y_fix]
        counter = (counter + b_s) % len(images)


def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian]
        counter = (counter + b_s) % len(images)

def save_frame_camera_key(device_num, dir_path, basename, ext='bmp', delay=1, window_name='frame'):
    cap = cv2.VideoCapture(device_num)

    if not cap.isOpened():
        return

    try:
        os.makedirs(dir_path)
    except WindowsError:
        pass

    base_path = os.path.join(dir_path, basename)

    n = 0
    while True:
        # retは画像を取得成功フラグ
        ret, frame = cap.read()
        size = (320, 240)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('c'):
            frame = cv2.resize(frame, size)
            cv2.imwrite('{}_{}.{}'.format(base_path, n, ext), frame)
            n += 1
        elif key == ord('q'):
            break

    cap.release       
    cv2.destroyWindow(window_name)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise NotImplementedError
    else:
        phase = sys.argv[1]
        x = Input((3, shape_r, shape_c))
        x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

        if version == 0:
            m = Model(input=[x, x_maps], output=sam_vgg([x, x_maps]))
            print("Compiling SAM-VGG")
            m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
        elif version == 1:
            m = Model(input=[x, x_maps], output=sam_resnet([x, x_maps]))
            print("Compiling SAM-ResNet")
            m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
        else:
            raise NotImplementedError

        if phase == 'train':
            if nb_imgs_train % b_s != 0 or nb_imgs_val % b_s != 0:
                print("The number of training and validation images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
                exit()

            if version == 0:
                print("Training SAM-VGG")
                m.fit_generator(generator(b_s=b_s), nb_imgs_train, nb_epoch=nb_epoch,
                                validation_data=generator(b_s=b_s, phase_gen='val'), nb_val_samples=nb_imgs_val,
                                callbacks=[EarlyStopping(patience=3),
                                           ModelCheckpoint('weights.sam-vgg.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True)])
            elif version == 1:
                print("Training SAM-ResNet")
                m.fit_generator(generator(b_s=b_s), nb_imgs_train, nb_epoch=nb_epoch,
                                validation_data=generator(b_s=b_s, phase_gen='val'), nb_val_samples=nb_imgs_val,
                                callbacks=[EarlyStopping(patience=3),
                                           ModelCheckpoint('weights.sam-resnet.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True)])

        elif phase == "test":
            #testの引数を受けたときに前景検出を行う
            # Output Folder Path
            while True:
                save_frame_camera_key(0, 'inputimage/', 'camera_capture')#カメラ画像を取得，保存

                imgs_test_path = 'inputimage/'#入力画像ディレクトリ
                output_folder = 'predictions/'#顕著性マップ出力ディレクトリ
                thresh_folder = 'threshold/'#2値化画像保存ディレクトリ
                grabcut_folder = 'grabcut/'#前景抽出保存ディレクトリ

                if len(sys.argv) < 1:
                    raise SyntaxError
           
                imgs_test_path = 'inputimage/'


                file_names = [f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                file_names.sort()
                nb_imgs_test = len(file_names)

                if nb_imgs_test % b_s != 0:
                    print("The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
                    exit()

                if version == 0:
                    print("Loading SAM-VGG weights")
                    m.load_weights('weights/sam-vgg_salicon_weights.pkl')
                elif version == 1:
                    print("Loading SAM-ResNet weights")
                    m.load_weights('weights/sam-resnet_salicon_weights.pkl')

                print(("Predicting saliency maps for " + os.getcwd() +  "\\" + imgs_test_path))
                predictions = m.predict_generator(generator_test(b_s=b_s, imgs_test_path=imgs_test_path), nb_imgs_test)[0]
            
            
            

                for pred, name in zip(predictions, file_names):
                    original_image = cv2.imread(imgs_test_path + name, cv2.IMREAD_COLOR)
                    res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
                    cv2.imwrite(output_folder + '%s' % name, res.astype(int))
                    img = cv2.imread(output_folder + '%s' % name, cv2.IMREAD_GRAYSCALE)
                    mean = np.mean(img)
                    threshold = mean * 1
                    img_thresh = np.zeros(img.shape[:2],dtype = np.uint8)
                    ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
                    img_mask = cv2.merge((img_thresh, img_thresh, img_thresh))
                    mask_rows, mask_cols, mask_channel = img_mask.shape
                    min_x = mask_cols
                    min_y = mask_rows
                    max_x = 0
                    max_y = 0

                    for y in range(mask_rows):
                        for x in range(mask_cols):
                            if all(img_mask[y, x] == 255):
                                if x < min_x:
                                    min_x = x
                                elif x > max_x:
                                    max_x = x
                                if y < min_y:
                                    min_y = y
                                elif y > max_y:
                                    max_y = y

                        
                    rect_x = min_x
                    rect_y = min_y
                    rect_w = max_x - min_x
                    rect_h = max_y - min_y
                    bgdmodel = np.zeros((1,65),np.float64)
                    fgdmodel = np.zeros((1,65),np.float64)
                    mask = np.zeros(original_image.shape[:2], dtype = np.uint8)
                    rect = (rect_x, rect_y, rect_w, rect_h)
                    cv2.imwrite(thresh_folder + '%s' % name, img_thresh)
                    cv2.cvtColor(original_image, original_image, cv2.COLOR_BGRA2BGR)
                    cv2.cvtColor(img_mask, img_mask, cv2.COLOR_GRAY2BGR)
                    cv2.grabCut(original_image,mask,rect,bgdmodel,fgdmodel,5,cv2.GC_INIT_WITH_RECT)
                    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                    original_image = original_image*mask2[:,:,np.newaxis]
                    cv2.imshow('output_image', original_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    cv2.imwrite(grabcut_folder + '%s' % name, original_image)
                    


                        
        else:
            raise NotImplementedError
