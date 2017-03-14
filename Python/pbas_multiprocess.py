import cv2
import numpy as np
from numpy import pi
from multiprocessing import Process, Queue
import time

class PBAS(Process):

    img = 0
    channel = 1
    foreground = 0
    background_pixel = 0
    background_grad = 0
    d_min = 0
    d_min_arr = 0
    d_min_avg = 0
    pixel_probabilities = 0
    R_arr = 0
    downsample = .5
    dist = 0

    # INITIALIZATION ===============================================================
    def __init__(self, channel=1, N=10, nmbr_min=2, R_inc_dec=0.05, R_lower=18, R_scale=5, T_dec=0.1, T_inc=1, T_lower=2, T_upper=150, alpha=1):
        Process.__init__(self)

        # Initialize Parameters
        self.channel = channel
        self.N = N
        self.nmbr_min = nmbr_min
        self.R_inc_dec = R_inc_dec
        self.R_lower = R_lower
        self.R_scale = R_scale
        self.T_dec = T_dec
        self.T_inc = T_inc
        self.T_lower = T_lower
        self.T_upper = T_upper
        self.alpha = alpha

        # Variable for checking if background models are initialized
        self.init = 0
    # INITIALIZATION ===============================================================


    # PRINT_PARAMS =================================================================
    def print_params(self):
        """ Prints out Parameters for this instance of class PBAS

        :self: instance of the class PBAS which is calling the function

        """
        print 'Parameters:'
        print 'nmbr_min: ', self.nmbr_min
        print 'R_inc_dec: ', self.R_inc_dec
        print 'R_lower: ', self.R_lower
        print 'R_scale: ', self.R_scale
        print 'T_dec: ', self.T_dec
        print 'T_inc: ', self.T_inc
        print 'T_lower: ', self.T_lower
        print 'T_upper: ', self.T_upper
        print 'alpha: ', self.alpha
    # PRINT_PARAMS =================================================================


    # DISTANCE =====================================================================
    def distance(self):
        """ Returns an distance for the given channel and gradient value

        :returns: distance

        """
        # calculate the distance
        return self.alpha/self.avg_grad*np.abs(self.grad[:,:,np.newaxis]-self.background_grad)\
                + np.abs(self.img[:,:,np.newaxis] - self.background_pixel)

    # DISTANCE =====================================================================


    # GRADIENT =====================================================================
    def gradient(self):
        """ Calculates the gradient magnitude an image

        """
        sobelx = cv2.Sobel(self.img,cv2.CV_64F,dx=1,dy=0,ksize=1)
        sobely = cv2.Sobel(self.img,cv2.CV_64F,dx=0,dy=1,ksize=1)
        # sobelx = cv2.Sobel(self.img,ddepth=-1,dx=1,dy=0,ksize=5)
        # sobely = cv2.Sobel(self.img,ddepth=-1,dx=0,dy=1,ksize=5)
        self.grad = np.sqrt(sobelx**2+sobely**2)
        # self.avg_grad = np.average(self.grad)
        self.avg_grad = np.sum(self.grad)/(self.grad.shape[0] * self.grad.shape[1])
    # GRADIENT =====================================================================


    # BACKGROUND_DECISION ==========================================================
    def decision(self):
        self.foreground = self.img*0
        d = self.distance()
        self.dist = d
        self.d_min = np.amin(d, axis=2)
        comp = d<self.R_arr[:,:,np.newaxis]
        self.foreground[(comp != False).sum(2) < self.nmbr_min] = 255
    # BACKGROUND_DECISION ==========================================================


    # BACKGROUND_UPDATE ============================================================
    def background_update(self, n):
        # Random pixels with probability 1/T
        rand_array = 100.*np.random.random(self.img.shape)
        update_array = np.logical_and(((100/self.T_rate_arr) > rand_array), self.foreground == 0)
        # Choose adjacent pixels
        ind_x, ind_y = np.nonzero(update_array)
        rand_coords = [(-1,-1), (-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        # print 'loops: ', len(ind_x)
        for i in range(len(ind_x)):
            rand_x, rand_y = rand_coords[np.uint8(np.random.rand()*8)]
            new_x = ind_x[i]+rand_x
            new_y = ind_y[i]+rand_y
            if new_x < 0 or new_x >= update_array.shape[0]:
                new_x = ind_x[i]-rand_x
            if new_y < 0 or new_y >= update_array.shape[1]:
                new_y = ind_y[i]-rand_y
            update_array[new_x, new_y] = 1

        # Update background pixels in plane n
        self.background_pixel[update_array,n] = self.img[update_array]
        self.background_grad[update_array,n] = self.grad[update_array]
    # BACKGROUND_UPDATE ============================================================


    # DISTANCE_UPDATE ==============================================================
    def distance_update(self, n):
        # Update minimum distance array
        self.d_min_arr[:,:,n] = self.d_min
        # Calculate average minimum distances
        self.d_min_avg = self.d_min_arr.sum(2)/self.N
    # DISTANCE_UPDATE ==============================================================


    # THRESHOLD_UPDATE =============================================================
    def threshold_update(self):
        th_update = self.R_arr > self.d_min_avg * self.R_scale

        self.R_arr[th_update] *= (1 - self.R_inc_dec)
        self.R_arr[~th_update] *= (1 + self.R_inc_dec)
    # THRESHOLD_UPDATE =============================================================


    # LEARNING_RATE_UPDATE =========================================================
    def learn_update(self):
        update_inc = (self.foreground > 0)
        update_dec = (self.foreground == 0)
        self.T_rate_arr[update_inc & (self.d_min > 0)] += self.T_inc/self.d_min[update_inc & (self.d_min > 0)]
        self.T_rate_arr[update_dec & (self.d_min > 0)] -= self.T_dec/self.d_min[update_dec & (self.d_min > 0)]
        self.T_rate_arr[self.T_rate_arr < self.T_lower] = self.T_lower
        self.T_rate_arr[self.T_rate_arr > self.T_upper] = self.T_upper
    # LEARNING_RATE_UPDATE =========================================================


    # PROBABILITY_UPDATE ===========================================================
    def probability_update(self):
        self.pixel_probabilities = 1/self.T_rate_arr
    # PROBABILITY_UPDATE ===========================================================

    ################################################################################
    # Main function =========================================================
    def run(self):
        while True:
            if self.channel == 0:
                self.img = image_queue_r.get()
            elif self.channel == 1:
                self.img = image_queue_g.get()
            else:
                self.img = image_queue_b.get()

            self.img = cv2.resize(self.img,None,fx=self.downsample, fy=self.downsample, interpolation = cv2.INTER_AREA)

            if self.init == 0:
                self.init = 1

                rows, cols = self.img.shape
                self.foreground = np.zeros((rows,cols))
                self.gradient()

                self.pixel_probabilities = np.ones((rows, cols)) * 50

                self.background_pixel = np.uint8(np.ones((rows, cols, self.N)) * self.img[:,:,np.newaxis])
                self.background_grad = np.ones((rows, cols, self.N)) * self.grad[:,:,np.newaxis]

                self.d_min = np.ones((rows, cols))
                self.d_min_arr = np.zeros((rows, cols, self.N))
                self.d_min_avg = np.zeros((rows, cols))

                # R_arr = np.zeros((rows, cols))
                self.R_arr = np.ones((rows, cols))*self.R_lower

                self.T_rate_arr = np.ones((rows,cols))*2

            # Random plane
            n = np.uint8(np.floor(self.N * np.random.random()))
            # Update minimum distance array
            self.distance_update(n)
            # Update decision threshold
            self.threshold_update()
            # Update learning rate
            self.learn_update()
            # Update pixel probability
            self.probability_update()
            # Update background model
            self.background_update(n)

            self.gradient()
            self.decision()
            self.foreground = cv2.medianBlur(self.foreground, 9)
            # self.foreground = cv2.medianBlur(cv2.resize(self.foreground,None,fx=1/self.downsample, fy=1/self.downsample, interpolation = cv2.INTER_LINEAR), 15)

            if self.channel == 0:
                foreground_queue_r.put(cv2.resize(self.foreground,None,fx=1/self.downsample, \
                                    fy=1/self.downsample, interpolation = cv2.INTER_LINEAR))
                # temp_queue.put(cv2.resize(self.dist[:,:,0],None,fx=1/self.downsample,\
                temp_queue.put(cv2.resize(self.grad,None,fx=1/self.downsample,\
                                    fy=1/self.downsample, interpolation = cv2.INTER_LINEAR))
                temp_queue.put(cv2.resize(self.background_grad[:,:,0],None,fx=1/self.downsample,\
                                    fy=1/self.downsample, interpolation = cv2.INTER_LINEAR))
            elif self.channel == 1:
                foreground_queue_g.put(cv2.resize(self.foreground,None,fx=1/self.downsample, fy=1/self.downsample, interpolation = cv2.INTER_LINEAR))
            else:
                foreground_queue_b.put(cv2.resize(self.foreground,None,fx=1/self.downsample, fy=1/self.downsample, interpolation = cv2.INTER_LINEAR))

    # Main function =========================================================
    ################################################################################

if __name__ == '__main__':
    # Channel Variables
    ch_r = 0
    ch_g = 1
    ch_b = 2

    # Set desired parameters
    N = 35
    nmbr_min = 2
    R_inc_dec = 0.05
    R_lower = 18
    R_scale = 5
    T_dec = 0.1
    T_inc = 1
    T_lower = 2
    T_upper = 150
    alpha = 10

    # Queues
    image_queue_r = Queue()
    image_queue_g = Queue()
    image_queue_b = Queue()
    foreground_queue_r = Queue()
    foreground_queue_g = Queue()
    foreground_queue_b = Queue()

    temp_queue = Queue()

    # Create an instance per channel
    pbas_r = PBAS(ch_r, N, nmbr_min, R_inc_dec, R_lower, R_scale, T_dec, T_inc, T_lower, T_upper, alpha)
    pbas_g = PBAS(ch_g, N, nmbr_min, R_inc_dec, R_lower, R_scale, T_dec, T_inc, T_lower, T_upper, alpha)
    pbas_b = PBAS(ch_b, N, nmbr_min, R_inc_dec, R_lower, R_scale, T_dec, T_inc, T_lower, T_upper, alpha)

    # Open VideoCapture; here i.e. 'highway.avi'
    cap = cv2.VideoCapture('highway.avi')
    once = 0
    n_im = 0

    while True:
        start = time.time()
        # Read one frame of the video and break when error occurs (i.e. video ended)
        ret, img = cap.read()
        if ret == False:
            break

        image_queue_r.put(img[:,:,ch_r])
        image_queue_g.put(img[:,:,ch_g])
        image_queue_b.put(img[:,:,ch_b])

        if once == 0:
            once = 1
            pbas_r.start()
            pbas_g.start()
            pbas_b.start()

        foreground_r = foreground_queue_r.get()
        foreground_g = foreground_queue_g.get()
        foreground_b = foreground_queue_b.get()

        foreground = np.logical_or(np.logical_or(foreground_r, foreground_g), foreground_b) * 1.

        cv2.imshow('orig', img)
        cv2.imshow('foreground', foreground)

        print 'number: ', n_im
        cv2.imwrite('./pbas_bilder/foreground{:04d}.jpg'.format(n_im), foreground*255)
        temp = temp_queue.get()
        cv2.imshow('temp_queue', np.uint8(255*(temp - temp.min())/(temp.max()-temp.min())))
        cv2.imwrite('./pbas_bilder/grad{:04d}.jpg'.format(n_im), temp)
        temp = temp_queue.get()
        cv2.imwrite('./pbas_bilder/back_grad{:04d}.jpg'.format(n_im), temp)
        cv2.imshow('temp_queue', np.uint8(255*(temp - temp.min())/(temp.max()-temp.min())))
        n_im += 1

        # Quit program pressing key 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        print 'FPS: ', 1/(time.time() - start)

    foreground_queue_r.close()
    foreground_queue_g.close()
    foreground_queue_b.close()

    image_queue_r.close()
    image_queue_g.close()
    image_queue_b.close()

    pbas_r.terminate()
    pbas_g.terminate()
    pbas_b.terminate()

    cap.release()
    cv2.destroyAllWindows()
