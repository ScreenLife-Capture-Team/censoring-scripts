import filecmp
import math
import os
from os import path
import PIL
from PIL.Image import Image
from decouple import Config
import numpy as np
import cv2
import re
from facenet_pytorch import MTCNN
import pytesseract
import spacy
import torch
from tesserocr import PyTessBaseAPI, RIL
import threading
from decouple import config


class Censor:
    def __init__(self, dest_dir, num_threads = 2):

        self.TESSDATA_PATH = config("TESSDATA_PATH")
        self.SKIP_IDENTICAL = config("SKIP_IDENTICAL", default=True, cast=bool)
        self.TEXT_CONF = config("TEXT_CONF", default=80, cast=int)
        self.dest_dir = dest_dir
        self.num_threads = num_threads

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.tessbases = []

        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_sm")
        self.nlp = nlp

        self.prev_filepath = None
        self.prev_censored = None

        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709

        self.skipped = 0

    def hasNumbers(self, inputString):
        return bool(re.search(r"\d", inputString))

    def apply_blur(self, image, x1, y1, x2, y2):
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)

    def drawBoxes(self, im, boxes):
        for (x1, y1, x2, y2) in boxes:
            self.apply_blur(im, x1, y1, x2, y2)

    def censor_files(self, images):
        portion = math.ceil(len(images) / self.num_threads)
        threads = []
        for i in range(self.num_threads):
            tessbase = PyTessBaseAPI(path=self.TESSDATA_PATH)
            thread_images = images[portion * i: portion * (i+1)]
            t = threading.Thread(target=self.censor_multiple, args=(thread_images, tessbase))
            t.start()
            threads.append(t)
            self.tessbases.append(tessbase)

        for t in threads:
            t.join()

    def censor_multiple(self, images, tessbase):
        for image in images:
            self.censor(image, tessbase)

    def end(self):
        for tb in self.tessbases:
            tb.End()

    def censor(self, image_path, tessbase):

        if self.SKIP_IDENTICAL and self.prev_filepath is not None:
            if filecmp.cmp(self.prev_filepath, image_path, shallow=False):
                dest_path = os.path.join(self.dest_dir, os.path.basename(image_path))
                self.skipped += 1
                if not cv2.imwrite(dest_path, self.prev_censored):
                    raise Exception("Could not write image")
                return
        try:
            image = cv2.imread(image_path)
            img_matlab = image.copy()
        except AttributeError:
            fileobj = open('error_logs.txt', 'a')
            fileobj.write('could not read file ', + image_path)
            return

        tmp = img_matlab[:, :, 2].copy()
        img_matlab[:, :, 2] = img_matlab[:, :, 0]
        img_matlab[:, :, 0] = tmp

        # Detect face
        boxes, *_ = self.mtcnn.detect(image, landmarks=False)
        if boxes is not None:
            self.drawBoxes(image, boxes)

        # Image processing before text detection
        image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.threshold(
            cv2.GaussianBlur(gray, (5, 5), 0),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
        cv2.threshold(
            cv2.bilateralFilter(gray, 5, 75, 75),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
        cv2.threshold(
            cv2.medianBlur(gray, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (5, 5), 0),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )
        cv2.adaptiveThreshold(
            cv2.bilateralFilter(gray, 9, 75, 75),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )
        cv2.adaptiveThreshold(
            cv2.medianBlur(gray, 3),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )

        # Text detection
        mg = PIL.Image.fromarray(gray)
        tessbase.SetImage(mg)
        boxes = tessbase.GetComponentImages(RIL.BLOCK, True)

        # for i, (im, box, _, _) in enumerate(boxes):
        #     # im is a PIL image object
        #     # box is a dict with x, y, w and h keys
        #     tessbase.SetRectangle(box['x'], box['y'], box['x']+box['w'], box['y']+box['h'])
        #     text = tessbase.GetUTF8Text()
        #     conf = tessbase.MeanTextConf()

        #     # extract the bounding box coordinates of the text region from
        #     # the current result
        #     x, y, w, h = box["x"], box["y"], box["w"], box["h"]

        #     # extract the OCR text itself along with the confidence of the
        #     # text localization
        #     # filter out weak confidence text localizations
        #     if conf > 80:
        #         ## replacing the non-alphanumeric characters with empty char
        #         text = re.sub(r"[^\w\s]", "", text)
        #         if text != "":
        #             ## applying SPACY
        #             doc = self.nlp(text)

        #             if len(doc.ents) == 0:
        #                 if self.hasNumbers(text) and len(text) >= 5:
        #                     self.apply_blur(image, x, y, x + w, y + h)
        #                     # cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #                     # 	1, (0, 0, 255), 2)
        #             else:
        #                 for ent in doc.ents:
        #                     # print(ent.text, ent.label_)
        #                     if ent.label_ == "PERSON" or (
        #                         self.hasNumbers(ent.text) and len(ent.text) >= 5
        #                     ):
        #                         self.apply_blur(image, x, y, x + w, y + h)
        #                         # cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                # 	1, (0, 0, 255), 2)

        # # text detection
        results = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        # iterating through the text detection results
        for i in range(0, len(results["text"])):
            text = results["text"][i]
            text = re.sub(r'[^\w\s]','', text.strip())
            if len(text) <= 2: continue

            # extract the bounding box coordinates of the text region from
            # the current result
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]

            # extract the OCR text itself along with the confidence of the
            conf = int(results["conf"][i])

            # filter out weak confidence text localizations
            if conf > self.TEXT_CONF:
                if self.hasNumbers(text) and len(text) >= 5:
                    self.apply_blur(image, x, y, x+w, y+h)
                    continue

                if '@' in text:
                    self.apply_blur(image, x, y, x+w, y+h)
                    continue

                # Apply Spacy to identify names
                doc = self.nlp(text)
                if doc.ents and doc.ents[0].label_ == "PERSON":
                    self.apply_blur(image, x, y, x+w, y+h)

        dest_path = os.path.join(self.dest_dir, os.path.basename(image_path))
        if not cv2.imwrite(dest_path, image):
            raise Exception("Could not write image")
        self.prev_filepath = image_path
        self.prev_censored = image
