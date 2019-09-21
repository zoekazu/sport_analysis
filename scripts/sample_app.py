import os

import cv2
import sport_analysis as sp


def main():
    coat_img_path = os.path.join(
        os.path.dirname(__file__), "..",
        "assets", "tennis", "tennis_sample.jpg")
    coat_img = cv2.imread(coat_img_path)


if __name__ == "__main__":
    main()
