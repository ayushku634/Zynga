import certifi
from paddleocr import PaddleOCR
import os
import re
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

os.environ['SSL_CERT_FILE'] = certifi.where()

class Solution:
    def __init__(self):
        self.base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.output)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def extract_features(self, img_path):
        img_array = self.preprocess_image(img_path)
        features = self.model.predict(img_array)
        return features

    def calculate_similarity(self, main_image_path, test_image_path1, test_image_path2):
        main_features = self.extract_features(main_image_path)
        test1_features = self.extract_features(test_image_path1)
        test2_features = self.extract_features(test_image_path2)
        similarity1 = cosine_similarity(main_features, test1_features)[0][0]
        similarity2 = cosine_similarity(main_features, test2_features)[0][0]
        percentage1 = similarity1 * 100
        percentage2 = similarity2 * 100
        return [percentage1, percentage2]

    def process_image_sets(self, base_path, set_number):
        main_image_path = f'{base_path}/Set{set_number}/Image.png'
        test_image_path1 = f'{base_path}/Set{set_number}/Test1.png'
        test_image_path2 = f'{base_path}/Set{set_number}/Test2.png'
        return self.calculate_similarity(main_image_path, test_image_path1, test_image_path2)

    def extract_numerical_value(self, text):
        pattern = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'
        match = re.search(pattern, text)
        if match:
            return match.group().replace(',', '')  # Remove commas from the extracted number
        else:
            return None

    def find_nearest_numerical_value(self, target_word, ocr_result, max_distance):
        target_box = None
        for (box, text) in zip(ocr_result['boxes'], ocr_result['texts']):
            if target_word in text:
                target_box = box
                break
        if target_box is None:
            return ""
        target_x, target_y = target_box[0][0], target_box[0][1]
        target_center = np.array([target_x + (target_box[1][0] - target_x) / 2,
                                  target_y + (target_box[2][1] - target_y) / 2])
        closest_value = None
        closest_distance = float('inf')
        for box, text in zip(ocr_result['boxes'], ocr_result['texts']):
            numerical_value = self.extract_numerical_value(text)
            if numerical_value is not None:
                box_x, box_y = box[0][0], box[0][1]
                box_center = np.array([box_x + (box[1][0] - box_x) / 2,
                                       box_y + (box[2][1] - box_y) / 2])
                distance = np.linalg.norm(target_center - box_center)
                if distance < closest_distance and distance <= max_distance:
                    closest_distance = distance
                    closest_value = numerical_value
        return closest_value if closest_value is not None else ""

    def find_nearest_values_for_targets(self, target_words, ocr_result, max_distance):
        results = {}
        for word in target_words:
            results[word] = self.find_nearest_numerical_value(word, ocr_result, max_distance)
        return results

    def process_images_for_sets(self, base_path, set_number, target_words, max_distance):
        results = {word: [] for word in target_words}
        image_files = ['Test1.png', 'Test2.png']
        for image_file in image_files:
            img_path = os.path.join(base_path, f'Set{set_number}', image_file)
            if not os.path.exists(img_path):
                print(f"File {img_path} not found. Skipping.")
                continue
            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
            ocr_result = self.ocr.ocr(img_path, cls=True)
            parsed_result = {'boxes': [], 'texts': [], 'scores': []}
            for line in ocr_result:
                for word_info in line:
                    parsed_result['boxes'].append(word_info[0])
                    parsed_result['texts'].append(word_info[1][0])
                    parsed_result['scores'].append(word_info[1][1])
            nearest_values = self.find_nearest_values_for_targets(target_words, parsed_result, max_distance)
            for word in target_words:
                results[word].append(nearest_values[word])
        return results

    def get_answer(self, problem):
        base_path = './Problems'
        if problem == 'Set8':
            target_words = ["TOTAL WIN"]
            return self.process_images_for_sets(base_path, 8, target_words, 150)["TOTAL WIN"]
        elif problem == 'Set9':
            target_words = ["BET"]
            return self.process_images_for_sets(base_path, 9, target_words, 150)["BET"]
        else:
            set_number = int(problem.split('Set')[1])
            return self.process_image_sets(base_path, set_number)

