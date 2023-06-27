from pathlib import Path
import os
from typing import List, Tuple

import cv2
import numpy as np


def binarize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return dst


def save_image(image, filename):
    ret = cv2.imwrite(filename, image)
    print(f"Saving {filename} was {'NOT ' if not ret else ''}successful")


def random_bright_rgb_color():
    # Generate a random HSV color
    hsv = np.array([np.random.randint(0, 180), 255, 255])
    # Convert the color to RGB
    rgb = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0]
    # Return the color as a tuple
    color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return color


class Line:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
    
    def __repr__(self):
        return f"Line(x={self.x}, y={self.y}, angle={self.angle})"
    
    def __str__(self):
        return self.__repr__()
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.angle == other.angle
    
    def __hash__(self):
        return hash((self.x, self.y, self.angle))
    
    def intersection(self, other: "Line"):
        """Find the intersection of two lines."""
        m1 = np.tan(self.angle)
        m2 = np.tan(other.angle)
        if m1 == m2:
            return None
        x = (m1 * self.x - m2 * other.x + other.y - self.y) / (m1 - m2)
        y = m1 * (x - self.x) + self.y
        return (x, y)
    
    def distance_to_point(self, point: Tuple[float, float]):
        """Find the distance between a point and a line."""
        e1 = np.cos(self.angle + np.pi/2)
        e2 = np.sin(self.angle + np.pi/2)
        dx = point[0] - self.x
        dy = point[1] - self.y
        return abs(e1 * dx + e2 * dy)


def find_rotated_rectangles(image):
    # Find contours in the image
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the rotated rectangles
    rectangles = []
    for contour in contours:
        rectangles.append(cv2.minAreaRect(contour))
    # Return the rotated rectangles
    return rectangles


def filter_rectangles(rectangles):
    filtered_rectangles = []
    for rectangle in rectangles:
        # Calculate the area of the rectangle
        area = rectangle[1][0] * rectangle[1][1]
        if area < 100 or area > 2000:
            continue
        shorter = min(rectangle[1][0], rectangle[1][1])
        longer = max(rectangle[1][0], rectangle[1][1])
        aspect_ratio = longer / shorter
        if aspect_ratio < 2 or aspect_ratio > 4:
            continue
        print(f"area: {area}, aspect_ratio: {aspect_ratio}")
        filtered_rectangles.append(rectangle)
    return filtered_rectangles


def render_rectangles(image, rectangles):
    # Make a copy of the image
    image = image.copy()
    # Draw the rectangles
    lines = []
    for rectangle in rectangles:
        box = np.intc(cv2.boxPoints(rectangle))
        # box = np.int0(box)
        color = random_bright_rgb_color()
        cv2.drawContours(image, [box], 0, color, 2)
        # Draw a line from the center of the rectangle in the direction of the angle
        center = (int(rectangle[0][0]), int(rectangle[0][1]))
        angle = rectangle[2]
        # Add 90 degress if the other side of the rectangle is longer
        if rectangle[1][0] < rectangle[1][1]:
            angle += 90
        angle_radians = angle * np.pi / 180
        radius = 200
        center = (int(rectangle[0][0]), int(rectangle[0][1]))
        end = (int(rectangle[0][0] + radius * np.cos(angle_radians)), int(rectangle[0][1] + radius * np.sin(angle_radians)))
        other_end = (int(rectangle[0][0] - radius * np.cos(angle_radians)), int(rectangle[0][1] - radius * np.sin(angle_radians)))
        cv2.line(image, end, other_end, color, 2)
        lines.append(Line(center[0], center[1], angle_radians))
    # Return the image
    return image, lines


def distance_to_nth_closest_line(point: Tuple[float, float], lines: List[Line], n: int) -> float:
    """Find the distance between a point and the nth closest line."""
    distances_to_lines = [line.distance_to_point(point) for line in lines]
    distances_to_lines.sort()
    return distances_to_lines[n-1]


def render_center(image, guess: Tuple[float, float]):
    # Make a copy of the image
    image = image.copy()
    # Draw the guess
    cv2.circle(image, (int(guess[0]), int(guess[1])), 5, (0, 255, 255), 2)
    # Return the image
    return image


def main(directory, filename, out_dir):
    image = cv2.imread(str(Path(directory) / filename))
    image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
    bin_image = binarize_image(image)
    filename_without_extension = filename.split(".")[0]
    save_image(bin_image, str(Path(out_dir) / f"{filename_without_extension}_1_bin.jpg"))
    rectangles = find_rotated_rectangles(bin_image)
    rectangles = filter_rectangles(rectangles)
    rect_image, lines = render_rectangles(image, rectangles)
    save_image(rect_image, str(Path(out_dir) / f"{filename_without_extension}_2_rect.jpg"))
    print(f"Found {len(lines)} lines")
    if len(lines) < 8:
        return
    intersection_points = [lines[i].intersection(lines[j]) 
                           for i in range(len(lines)) 
                           for j in range(i+1, len(lines))
                           if lines[i].intersection(lines[j]) is not None]
    intersection_points.sort(key=lambda point: distance_to_nth_closest_line(point, lines, 8))
    center_guess = intersection_points[0]
    center_image = render_center(image, center_guess)
    save_image(center_image, str(Path(out_dir) / f"{filename_without_extension}_3_guess.jpg"))


if __name__ == "__main__":
    import sys

    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "out"
    os.makedirs(out_dir, exist_ok=True)
    for filename in os.listdir(directory):
        if not filename.endswith(".jpg"):
            continue
        print(f"Processing {filename}")
        # image = cv2.imread(str(Path(directory) / filename))
        main(directory, filename, out_dir)

