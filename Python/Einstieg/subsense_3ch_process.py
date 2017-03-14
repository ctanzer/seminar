import cv2
import numpy as np
from numpy import pi
import multiprocessing
from multiprocessing import Process, Queue
import time


class SUBSENSE(Process):

    channel = 0
    img = 0
    rows = 0
    cols = 0
    
    background = 0
    large_background = 0
    color_pictures = 0
    lbsp_background_model = 0
    
    n = 0
    update_array = 0
    
    d_min_new = 0
    d_min_arr = 0
    
    grid_pictures= 0

    R_arr = 0
    R_color_arr = 0
    R_lbsp_arr = 0
    
    T_arr = 0

    v_arr = 0

    old_background = 0

    MAX_hamming_weight = 0

    color_decision = 0
    color_decision_blured = 0
    lbsp_decision = 0

    
    # INITIALIZATION ===============================================================
    def __init__(self,channel=0, T_r=0.003, N_grid=16, nmbr_min_lbsp=12, N_color=50, nmbr_min_color=2, R_color=30,R_lbsp=3, T_lower=2, T_upper=256, alpha=0.03, v_incr=1, v_decr=0.1,downsample = 1):
        Process.__init__(self)

        # Initialize Parameters
        self.channel = channel
        self.T_r = T_r
        self.N_grid = N_grid
        self.nmbr_min_lbsp = nmbr_min_lbsp
        self.N_color = N_color
        self.nmbr_min_color = nmbr_min_color
        self.R_color = R_color
        self.R_lbsp = R_lbsp
        self.T_lower = T_lower
        self.T_upper = T_upper
        self.alpha = alpha
        self.v_incr = v_incr
        self.v_decr = v_decr
        self.downsample = downsample

        # Variable for checking if background models are initialized
        self.init = 0
    # INITIALIZATION ===============================================================


    # PRINT_PARAMS =================================================================
    def print_params(self):
        """ Prints out Parameters for this instance of class PBAS

        :self: instance of the class PBAS which is calling the function

        """
        print 'Parameters:'
        print 'channel', channel
        print 'T_r', self.T_r
        print 'N_grid', self.N_grid 
        print 'nmbr_min_lbsp', self.nmbr_min_lbsp
        print 'N_color', self.N_color
        print 'nmbr_min_color', self.nmbr_min_color
        print 'R_color', self.R_color
        print 'R_lbsp', self.R_lbsp
        print 'T_lower', self.T_lower
        print 'T_upper', self.T_upper
        print 'alpha', self.alpha
        print 'v_incr', self.v_incr
        print 'v_decr', self.v_decr
        print 'downsample', self.downsample
    # PRINT_PARAMS =================================================================


    # IMAGE_FETCH ==================================================================
    def image_fetch(self, chan):
        """ Fetches one color channel of an image
        """
        self.img = chan
    # IMAGE_FETCH ==================================================================


    # LBSP-UPDATE GRID PICTURES ==========================================================
    def update_grid_pictures(self):
        #    0    x   1   x   2
        #    x    3   4   5   x 
        #    6    7   x   8   9
        #    x   10  11  12   x
        #   13    x  14   x   15
        
        # First row
        self.grid_pictures[:self.rows-2,:self.cols-2,0]  = self.img[2:self.rows,2:self.cols]
        self.grid_pictures[:self.rows-2,:self.cols,1]    = self.img[2:self.rows,:self.cols]
        self.grid_pictures[:self.rows-2,2:self.cols,2]   = self.img[2:self.rows,:self.cols-2]

        # Second row
        self.grid_pictures[:self.rows-1,:self.cols-1,3]  = self.img[1:self.rows,1:self.cols]
        self.grid_pictures[:self.rows-1,:self.cols,4]    = self.img[1:self.rows,:self.cols]
        self.grid_pictures[:self.rows-1,1:self.cols,5]   = self.img[1:self.rows,:self.cols-1]

        # Third row
        self.grid_pictures[:self.rows,:self.cols-2,6]    = self.img[:self.rows,2:self.cols]
        self.grid_pictures[:self.rows,:self.cols-1,7]    = self.img[:self.rows,1:self.cols]
        self.grid_pictures[:self.rows,1:self.cols,8]     = self.img[:self.rows,:self.cols-1]
        self.grid_pictures[:self.rows,2:self.cols,9]     = self.img[:self.rows,:self.cols-2]

        # Fourth row
        self.grid_pictures[1:self.rows,:self.cols-1,10]  = self.img[:self.rows-1,1:self.cols]
        self.grid_pictures[1:self.rows,:self.cols,11]    = self.img[:self.rows-1,:self.cols]
        self.grid_pictures[1:self.rows,1:self.cols,12]   = self.img[:self.rows-1,:self.cols-1]

        # Fifth row
        self.grid_pictures[2:self.rows,:self.cols-2,13]  = self.img[:self.rows-2,2:self.cols]
        self.grid_pictures[2:self.rows,:self.cols,14]    = self.img[:self.rows-2,:self.cols]
        self.grid_pictures[2:self.rows,2:self.cols,15]   = self.img[:self.rows-2,:self.cols-2]

    # LBSP-UPDATE GRID PICTURES ==========================================================
    
    # LBSP DECISION =================================================================
    def lbsp(self):
        grid_decision = self.grid_pictures - self.img[:,:,np.newaxis]
        self.lbsp_decision = self.img*0+255
        T = self.T_r * self.img
        actual_bin = grid_decision<T[:,:,np.newaxis]
        bin_comp = (self.lbsp_background_model == actual_bin[:,:,np.newaxis,:])
        summe = (bin_comp == True).sum(3)
        comp = summe > self.nmbr_min_lbsp
        self.lbsp_decision[((comp == True).sum(2) >= self.R_lbsp_arr)] = 0
    # LBSP DECISION =================================================================
    
    # COLOR DECISION ====================================================================
    def color(self):
        self.color_decision = self.img*0
        comp = np.absolute(self.img[:,:,np.newaxis] - self.color_pictures) < self.R_color_arr[:,:,np.newaxis]
        self.color_decision[(comp != False).sum(2) <= self.nmbr_min_color] = 255;
    # COLOR DECISION ====================================================================

    # BACKGROUND_UPDATE ======================================================================
    def background_update(self):
        self.n = np.uint8(np.floor(self.N_color*np.random.random()))
        # Random pixels with probability 1/T_arr
        rand_array = 100.*np.random.random(self.img.shape)
        self.update_array = np.logical_and(((100/self.T_arr) > rand_array), self.background == 0)
        # Choose adjacent pixels
        ind_x, ind_y = np.nonzero(self.update_array)
        rand_coords = [(-1,-1), (-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        # print 'loops: ', len(ind_x)
        for i in range(len(ind_x)):
            rand_x, rand_y = rand_coords[np.uint8(np.random.rand()*8)]
            new_x = ind_x[i]+rand_x
            new_y = ind_y[i]+rand_y
            if new_x < 0 or new_x >= self.update_array.shape[0]:
                new_x = ind_x[i]-rand_x
            if new_y < 0 or new_y >= self.update_array.shape[1]:
                new_y = ind_y[i]-rand_y
            self.update_array[new_x, new_y] = True
        # Update color pictures in plane n
        self.color_pictures[self.update_array,self.n] = self.img[self.update_array]
                
        # Update the lbsp background model
        model_decision = self.grid_pictures - self.img[:,:,np.newaxis]
        T = self.T_r * self.img
        model = model_decision<T[:,:,np.newaxis]
        self.lbsp_background_model[self.update_array,self.n,:] = model[self.update_array,:]

        # T_r update ------ Grenzen noch anpassen, eventuell auch Geschwindigkeit der Aenderung
        hammingweight=model.sum()*1.0/self.MAX_hamming_weight
        #print(hammingweight)
        if(self.T_r > 0.001 and self.T_r < 1):
            if(hammingweight < 0.6):
                self.T_r = self.T_r + 0.0001
            if(hammingweight > 0.61):
               self.T_r = self.T_r - 0.0001
    # BACKGROUND_UPDATE ======================================================================

    # DISTANCE_UPDATE ========================================================================
    def distance_update(self):
         # Get the new minimum distance array
        self.d_min_new = np.amin((np.absolute(self.img[:,:,np.newaxis] - self.color_pictures)), axis = 2)/255.0
     
        self.d_min_arr = self.d_min_arr*(1-self.alpha) + self.d_min_new * self.alpha
        # bound it from 0 to 1
        self.d_min_arr[self.d_min_arr < 0] = 0
        self.d_min_arr[self.d_min_arr > 1] = 1
    # DISTANCE_UPDATE ========================================================================

    # BLINKING PIXELS ========================================================================
    def recognize_blinking_pixels(self):
        # Find blinking pixels
        blinking_pixels = (self.background ^ self.old_background)/255
        self.old_background = self.background
        # Increment or Decrement the v_arr, which shows how dynamic the region is
        self.v_arr[blinking_pixels == 1] = self.v_arr[blinking_pixels == 1] + self.v_incr
        self.v_arr[blinking_pixels == 0] = self.v_arr[blinking_pixels == 0] - self.v_decr
        # v_arr must be greater than zero and should be zero for foreground
        self.v_arr[self.v_arr < 0] = 0
        self.v_arr[self.color_decision_blured == 255] = 0 
    # BLINKING PIXELS ========================================================================

    # THRESHOLD UPDATE =======================================================================
    def threshold_update(self):
        incr = (self.R_arr < np.square(1+2*self.d_min_arr))
        self.R_arr[incr] = self.R_arr[incr] + self.v_arr[incr]
        decr = (incr == False) & (self.v_arr != 0)
        self.R_arr[decr] = self.R_arr[decr] - (1/self.v_arr[decr])
        self.R_arr[self.R_arr < 1] = 1
        self.R_arr[(incr | decr) == False] = 1

        self.R_color_arr = self.R_arr * self.R_color
        self.R_lbsp_arr = np.power(2, self.R_arr) + self.R_lbsp
    # THRESHOLD UPDATE =======================================================================

    # PROBABILITY_UPDATE =====================================================================
    def probability_update(self):
        incr = (self.v_arr != 0) & (self.d_min_arr != 0) & (self.background != 0)
        self.T_arr[incr] = self.T_arr[incr] + (1/(self.v_arr[incr] * self.d_min_arr[incr]))
        maxi = ((self.v_arr == 0) | (self.d_min_arr == 0)) & (self.background != 0)
        self.T_arr[maxi] = self.T_upper
        
        decr = (self.d_min_arr != 0) & (self.background == 0)
        self.T_arr[decr] = self.T_arr[decr] - (self.v_arr[decr]/self.d_min_arr[decr])
        mini = (self.d_min_arr == 0) & (self.background == 0)
        self.T_arr[mini] = self.T_lower

        self.T_arr[self.T_arr < self.T_lower] = self.T_lower
        self.T_arr[self.T_arr > self.T_upper] = self.T_upper
    # PROBABILITY_UPDATE =====================================================================
    
    ################################################################################
    # Main function =========================================================
    def run(self):
        while True:
            if( self.channel == 1):
                self.img = image_queue_r.get()
            elif (self.channel == 2):
                self.img = image_queue_g.get()
            else:
                self.img = image_queue_b.get()
                
            self.init
            self.img = cv2.resize(self.img,None,fx=self.downsample, fy=self.downsample, interpolation = cv2.INTER_AREA) 
            if self.init == 0:
                self.init = 1

                self.rows, self.cols = self.img.shape
                
                self.MAX_hamming_weight = self.rows*self.cols*self.N_grid

                self.background = np.zeros((self.rows,self.cols))
                self.large_background = np.zeros((np.int(1/self.downsample*self.rows),np.int(1/self.downsample*self.cols)))


                self.grid_pictures = np.ones((self.rows,self.cols,self.N_grid))*255
                self.update_grid_pictures()
                
                model_decision = self.grid_pictures - self.img[:,:,np.newaxis]
                T = self.T_r * self.img
                model = model_decision<T[:,:,np.newaxis]
                self.lbsp_background_model = np.ones((self.rows,self.cols,self.N_color,self.N_grid), dtype=bool)*model[:,:,np.newaxis,:]
                
                self.d_min_new = self.img*255
                self.d_min_arr = np.ones((self.rows, self.cols))

                self.R_arr = np.ones((self.rows, self.cols))
                self.R_color_arr = self.R_arr
                self.R_lbsp_arr = self.R_arr

                self.T_arr = np.ones((self.rows, self.cols)) * 2

                self.v_arr = self.img*0.0

                self.threshold_update()

                self.color_pictures = np.uint8(np.ones((self.rows, self.cols, self.N_color)) * self.img[:,:,np.newaxis])
                self.color()
                self.old_background = self.color_decision
                self.color_decision_blured = cv2.medianBlur(self.color_decision, 5)

                
               
            self.update_grid_pictures()
            self.lbsp()
            self.background = self.lbsp_decision&self.color_decision;
            self.background_update()
            self.color()
            self.color_decision_blured = cv2.medianBlur(self.color_decision, 5)
                    
            self.distance_update()
            self.recognize_blinking_pixels()
            self.threshold_update()
            self.probability_update()
            self.background = cv2.medianBlur(self.background, 3)
            
            self.large_background = cv2.medianBlur(cv2.resize(self.background,None,fx=1/self.downsample, fy=1/self.downsample, interpolation = cv2.INTER_LINEAR),3)
            
            if( self.channel == 1):
                background_queue_r.put(self.background)
                background_queue_r.put(self.large_background)
            elif (self.channel == 2):
                background_queue_g.put(self.background)
                background_queue_g.put(self.large_background)
            else:
                background_queue_b.put(self.background)
                background_queue_b.put(self.large_background)
                
# Main function =========================================================
################################################################################

if __name__ == '__main__':
    # Set desired parameters
    # Constants
    channel_r = 1
    channel_g = 2
    channel_b = 3

    T_r = 0.003
    N_grid = 16
    nmbr_min_lbsp = 12
    N_color = 50
    nmbr_min_color = 2

    R_color = 30
    R_lbsp = 3

    T_update = 2
    T_lower = 2
    T_upper = 256

    alpha = 0.03
    v_incr = 1
    v_decr = 0.1
    
    background_queue_r = Queue()
    background_queue_g = Queue()
    background_queue_b = Queue()

    image_queue_r = Queue()
    image_queue_g = Queue()
    image_queue_b = Queue()


    downsample = 1  
    
    # Create one instance per channel
    subsense_r = SUBSENSE(channel_r, T_r, N_grid, nmbr_min_lbsp, N_color, nmbr_min_color, R_color,R_lbsp, T_lower, T_upper, alpha, v_incr, v_decr, downsample)
    subsense_g = SUBSENSE(channel_g, T_r, N_grid, nmbr_min_lbsp, N_color, nmbr_min_color, R_color,R_lbsp, T_lower, T_upper, alpha, v_incr, v_decr, downsample)
    subsense_b = SUBSENSE(channel_b, T_r, N_grid, nmbr_min_lbsp, N_color, nmbr_min_color, R_color,R_lbsp, T_lower, T_upper, alpha, v_incr, v_decr, downsample)
    
    # Open VideoCapture; here i.e. 'highway.avi'
    cap = cv2.VideoCapture('2.mp4')
    once = 0

    ret, img = cap.read()
    rows, cols = img[:,:,0].shape
            
    while True:
        # Start time
        start = time.time()

        # Read one frame of the video and break when error occurs (i.e. video ended)
        ret, img = cap.read()
        #print(img.shape)
        if ret == False:
            break

        image_queue_r.put(img[:,:,0])
        image_queue_g.put(img[:,:,1])
        image_queue_b.put(img[:,:,2])
    
        

        if once == 0:
            once = 1
            subsense_r.start()
            subsense_g.start()
            subsense_b.start()
        
        background = np.logical_or(np.logical_or(background_queue_r.get(), background_queue_g.get()), background_queue_b.get()) * 1.
        large_background = np.logical_or(np.logical_or(background_queue_r.get(), background_queue_g.get()), background_queue_b.get()) * 1.
        
        cv2.imshow('orig', img)
        cv2.imshow('background',background)
        cv2.imshow('large_background', large_background)
        
        video.write(background)
        # Quit program pressing key 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            background_queue_r.close()
            background_queue_g.close()
            background_queue_b.close()

            image_queue_r.close()
            image_queue_g.close()
            image_queue_b.close()

            subsense_r.terminate()
            subsense_g.terminate()
            subsense_b.terminate()
            cap.release()
            cv2.destroyAllWindows()
            subsense_r.join()
            subsense_g.join()
            subsense_b.join()
            
            break
        # End time
        end = time.time()
        # Time elapsed
        seconds = end - start
        # Calculate frames per second
        fps  = 1 / seconds;
        print "Estimated frames per second : {0}".format(fps);
    cap.release()
    cv2.destroyAllWindows()

