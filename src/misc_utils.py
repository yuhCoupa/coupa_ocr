import os
import cv2


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def read_image(input_image):
    '''
    function to read an image and return OpenCV object
    '''
    try:
        image = cv2.imread(input_image)
    except AttributeError:
        print(f"Input file '{input_image}' is not valid.")
    except Exception as e:
        print(e)

    return image


def show_image_opencv(image_instance, name="Image in OpenCV"):
    '''
    function to show an image in OpenCV popup
    '''
    try:
        cv2.imshow(name, image_instance)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)


def save_image_opencv(image_instance, img_name):
    '''
    save a file from OpenCV image instance.
    '''
    create_folder('img_output')
    target_name = os.path.join('img_output',
                               "{}.jpg".format(img_name))
    try:
        cv2.imwrite(target_name, image_instance)
    except Exception as e:
        print(e)
