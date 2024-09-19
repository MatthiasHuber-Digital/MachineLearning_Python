# Standard Library
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int


class Detector:
    def sort_components_bookwise(self, stats: np.ndarray) -> list[np.ndarray]:
        """
        Sorts the connected components in a book-wise manner,
        grouping them into rows and sorting within each row.

        Args:
            stats: The stats from the connected components analysis.

        Returns:
            A list of the sorted detections.
        """
        median_component_height = np.median(stats[:, cv2.CC_STAT_HEIGHT])
        sorting_window = median_component_height
        rows = []
        already_in_row = np.zeros(stats.shape[0], dtype=bool)

        for i, stat in enumerate(stats[stats[:, cv2.CC_STAT_TOP].argsort()]):
            if already_in_row[i]:
                continue

            matched_inds = (
                np.abs(stats[:, cv2.CC_STAT_TOP] - stat[cv2.CC_STAT_TOP]) < sorting_window
            )

            if np.any(already_in_row):
                matched_inds = (already_in_row & matched_inds) ^ matched_inds
                matched_inds = matched_inds.astype(bool)
                already_in_row = already_in_row | matched_inds
            else:
                already_in_row = matched_inds

            rows.append(stats[matched_inds, :])

        sorted_stats = [
            stat
            for row in rows
            for stat in sorted(row, key=lambda x: x[cv2.CC_STAT_LEFT])
        ]
        return sorted_stats

    def predict(self, img: np.ndarray) -> list[BBox]:
        """
        Detects on image and returns bounding boxes.

        Args:
            img: The image to detect on in.

        Returns:
            A list of BBox objects representing the bounding boxes of detected items.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )

        # Skip the first component (background)
        stats = stats[1:]

        # Sort detections
        sorted_stats = self.sort_components_bookwise(stats)

        # Create bounding boxes
        bboxes = [
            BBox(
                x=s[cv2.CC_STAT_LEFT],
                y=s[cv2.CC_STAT_TOP],
                width=s[cv2.CC_STAT_WIDTH],
                height=s[cv2.CC_STAT_HEIGHT],
            )
            for s in sorted_stats
        ]

        return bboxes

    def visualize_bboxes(self, img: np.ndarray, bboxes: list[BBox]) -> np.ndarray:
        """
        Visualizes the bounding boxes on the image.

        Args:
            img: The original image.
            bboxes: A list of BBox objects representing the bounding boxes to be drawn.

        Returns:
            The image with bounding boxes drawn on it.
        """
        img_with_bboxes = img.copy()
        for bbox in bboxes:
            top_left = (bbox.x, bbox.y)
            bottom_right = (bbox.x + bbox.width, bbox.y + bbox.height)
            cv2.rectangle(img_with_bboxes, top_left, bottom_right, (0, 255, 0), 4)
        return img_with_bboxes
