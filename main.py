import cv2
import numpy as np
from collections import deque


class BackgroundExtraction:
    def __init__(self, width, height, scale, maxlen=10):
        self.maxlen = maxlen
        self.scale = scale
        self.width = width // scale
        self.height = height // scale
        self.buffer = deque(maxlen=maxlen)
        self.background = None

    def calculate_background(self):
        self.background = np.zeros((self.height, self.width), dtype='float32')
        for item in self.buffer:
            self.background += item
        self.background /= len(self.buffer)

    def update_background(self, old_frame, new_frame):
        self.background -= old_frame / self.maxlen
        self.background += new_frame / self.maxlen

    def update_frame(self, frame):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(frame)
            self.calculate_background()
        else:
            old_frame = self.buffer.popleft()
            self.buffer.append(frame)
            self.update_background(old_frame, frame)

    def get_background(self):
        return self.background.astype('uint8')

    def apply(self, frame):
        down_scale = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(down_scale, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        self.update_frame(gray)
        abs_diff = cv2.absdiff(bg_buffer.get_background(), gray)
        _, ad_mask = cv2.threshold(abs_diff, 15, 255, cv2.THRESH_BINARY)
        return ad_mask


class Game:
    def __init__(self, width, height, size=50):
        self.width = width
        self.height = height
        self.size = size

        self.bomb = cv2.imread('blast.jpg')
        self.bomb = cv2.resize(self.bomb, (self.size, self.size))
        grey = cv2.cvtColor(self.bomb, cv2.COLOR_BGR2GRAY)

        # masking - take all the color codes which are equal or above 1, and convert them to value 255
        _, self.mask = cv2.threshold(grey, 1, 255, cv2.THRESH_BINARY)

        self.x = np.random.randint(0, self.width - self.size)
        self.y = 0
        self.speed = 10

    def update_frame(self,frame):
        self.update_position()
        # inserting the bomb
        roi = frame[self.y: self.y+self.size,  self.x: self.x+self.size]
        roi[np.where(self.mask)] = 0  # all the places where mask values are non zero, put those as 0 in roi
        roi += self.bomb

    def update_position(self):
        self.y += self.speed
        if self.y+self.size == self.height:
            self.y = 0
            self.x = np.random.randint(0, self.width - self.size)


width = 640
height = 480
scale = 2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

bg_buffer = BackgroundExtraction(width, height, scale, maxlen=5)

game = Game(width=width,height=height)

while True:
    # Reading, resizing, and flipping the frame
    _, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 1)

    # Processing the frame
    # fg_mask = bg_buffer.apply(frame)

    game.update_frame(frame)

    # cv2.imshow("FG Mask", fg_mask)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord('q'):
        break