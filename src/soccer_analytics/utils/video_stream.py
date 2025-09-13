from __future__ import annotations
import cv2
from typing import Iterator, Union

class VideoStream:
    def __init__(self, source: Union[int,str] = 0):
        if isinstance(source,str) and source.isdigit():
            source = int(source)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open video source: {source}')

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.release()
            raise StopIteration
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
