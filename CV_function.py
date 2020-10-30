import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


##############################################################################################
#---------------------------CALIBRATION CAMERA + DISTORSION CORRECTION------------------------

def calculation_undistort(image, objpoints, imgpoints):
    # get image size
    img_size = (image.shape[1], image.shape[0])
    
    # Camera Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        
    # Distorsion Correction
    undist = cv2.undistort(image,mtx,dist,None,mtx)
    
    return undist



################################################################################################
#--------------------------------------GRADIENT------------------------------------------------

def image_computing(image_undistorted):    
    
 
    # Grayscale image to compute the gradient 
    gray = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2GRAY)
    
    #  HLS color space and separate the S channel for the color
    hls = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
#------------------------------------------------------------------------------------------- 
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) 
    abs_sobelx = np.absolute(sobelx) # x derivative accentuates lines away from horizontal
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Sobel y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1) 
    abs_sobely = np.absolute(sobely) # y derivative accentuates lines away from horizontal
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
#--------------------------------------------------------------------------------------------- 
    # x-gradient Thresholding
    thresh_min = 40
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1
 
    # y-gradient Thresholding
    thresh_min = 40
    thresh_max = 100
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= thresh_min) & (scaled_sobely <= thresh_max)] = 1
#----------------------------------------------------------------------------------------------
    # color channel Thresholding
    s_thresh_min = 150
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    l_thresh_min = 150
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1
    
   
    # Gradient direction thresholding
    thresh=(0.7, 1.3)
    direction = np.arctan2 (abs_sobely, abs_sobelx) # Calculate the direction
    dir_binary = np.zeros_like (direction)
    dir_binary[(direction >= thresh[0]) & (direction<=thresh[1])] =1
    
    
    # Gradient magnitude thresholding
    mag_thresh=(30, 100) 
    magnitude = np.sqrt((sobelx)**2 + (sobely)**2) # Calculate the magnitude   
    scaled_magnitude  = np.uint8 (255* magnitude/np.max(magnitude))  # Scale to 8-bit (0 - 255) and convert to type = np.uint8      
    mag_binary = np.zeros_like (scaled_magnitude) # Create a binary mask where mag thresholds are met
    mag_binary[(scaled_magnitude >= mag_thresh[0]) & (scaled_magnitude<=mag_thresh[1])] =1
                
#-----------------------------------------------------------------------------------------------  
#-----------------------------------------------------------------------------------------------    
    # Combine the two binary thresholds : Only color and sobel x 
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (l_binary == 1)) |(sxbinary == 1)] = 1
#-----------------------------------------------------------------------------------------------
     # Combine the two binary thresholds : color and sobel x + magnitude/direction gradient        
    combined_binary_b = np.zeros_like(dir_binary)
    combined_binary_b[(sxbinary == 1 | ((mag_binary == 1) & (dir_binary == 1))) | s_binary == 1] = 1
#-----------------------------------------------------------------------------------------------    
    
    # return the combinaison color and sobelx
    return sxbinary, sybinary, mag_binary, dir_binary, s_binary, l_binary, combined_binary, combined_binary_b



#########################################################################################################
# ----------------------------PERPECTIVE TRANSFORM-------------------------------------------------------
def warper(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped,Minv




##################################################################################################
#---------------------------------FIND LANE PIXEL - Methodic : Histrogramm and Window ------------

def find_lanes_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    # creation de l'histogramme avec les 2 pics, image coupé en 2
    # et prendre en consideration que la demie-partie d'en bas --> shape[0] = axe y
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    # Create an output image to draw on and visualize the result
    # creation d'une image resultat nommé out_img
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # analyse de d'histogramme : 
    # 1) graphe histogramme coupe en 2 en x (partie droite et partie gauche)
    # 2) prise de la partie gauche en x  et prendre le point le plus haut
    # 3) prise de la partie droite en x  et prendre le point le plus haut
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
#--------------------------------------------------------
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    # nombre de fenetres qui vont se succeder
    nwindows = 9
    # Set the width of the windows +/- margin
    # tolerance en x
    margin = 50
    # Set minimum number of pixels found to recenter window
    # nombre minimum de pixel trouve pour recentrer la fenetre
    minpix = 100
#------------------------------------------------------------
    # DERNIERES PARAMETRAGES
    # definition de la hauteur des 9 fenetre : taille de la fenetre total / 9
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    
    # position x et y de tous les pixels actives dans l'image
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0]) # axe y
    nonzerox = np.array(nonzero[1]) # axe x
    
    # Current positions to be updated later for each window in nwindows
    #position courante des deux fenetres (gauche et droite), ici initalisation
    # donc fenetre (ligne) de base = fenetre (ligne) courante, qui va evoluer 
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    # creation de liste vide pour recevoir les indices des pixels des lignes (droite et gauche)
    left_lane_inds = []
    right_lane_inds = []

#---------------------------------------------------------------

    # Step through the windows one by one
    for window in range(nwindows):
        # iterer de 1 a 9 :
        
        # limites en y pour les  fenetres : 
        #hauteur de l'image/ ((hauteur_fenetre)*fenetre (0->9)), 
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        # limite en x des fenetres : avec les tolerances margin en x a partir
        # du positionnement de la courbe
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin  
        win_xright_low = rightx_current - margin 
        win_xright_high = rightx_current + margin  
        
# ---------------------        
        # Draw the windows on the visualization image
        # tracage de la fenetre dans l'image de sortie avec les coodonnées donnés ci-dessus
        
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
#------------------------        
       
        #  identification des pixels activés en x et y dans la fenetre
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        # ajout des indices des pixels activés
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
      
        # si le nombre de pixel dans la fentre est >50
        # on calcule la moyenne des indices pour donner leftx_current
        # et recentrer la fenetre
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        
#----------------------------PREPARATION A LA FONCION POLYNOMIAL---------
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    # extraction des posisitions (x et y ) des pixels (cote droit et gauche)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds 
# on retourne l'image de sortie, 
# les 2 listes avec les pixels ligne droit avec leurs coordonnees
# les 2 listes avec les pixels ligne gauche avec leur corronnées    
   

#-----------------------------------------------------------------------------   
#----------------FONCTION POLYNOMIALE-----------------------------------------
def fit_polynomial(binary_warped, leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds) :
    
#------------------------------------------
    #SECOND ORDER POLYNOMIAL for the right and the left line
    left_fit = np.polyfit (lefty,leftx,2)
    right_fit = np.polyfit (righty, rightx,2)
#-----------------------------------------

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return left_fit, right_fit, ploty, left_fitx, right_fitx, out_img

#------------------------------------------------------------------------------------

def calculation_curvature (leftx, lefty, rightx, righty) :
    # radius of curvature in meters
    y_eval = 719  # 720 image in reality, the lowest on screen y index is 719 (720-1)

    # conversions in x and y from pixels space (in pixel) to real world (in meters)
    ym_per_pix = 30/720 # meters for one pixel  y dim
    xm_per_pix = 3.7/700 # meters for one pixel in x dim

    # new polynomials to x,y in world space (in meters)
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # new radius of curvature Calculation (in meters)
    left_radius_curve = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_radius_curve = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Radius of curvature is now in meters    
    return left_radius_curve, right_radius_curve
    
#---------------------------------------------------------------------------------------
#------------------VEHICLE POSITION FROM THE LANE CENTER--------------------------------

def calculation_vehicle_position(undist, left_fit, right_fit):

    # vehicle center position (in pixels)  
    bottom_y = undist.shape[0] - 1
    bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
    bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
    vehicle_position = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

    # Conversion : from pixel positison to meters
    xm_per_pix = 3.7/700 # meters per pixel in x
    vehicle_position *= xm_per_pix
    
    return vehicle_position
    
#----------------------VISUALIZATION FINALE (original image with the news informations)-----------
#---------------------Warp the detected lane boundaries back onto the original image--------------

def vizualization (undist, left_fit, right_fit, Minv, left_radius_curve, right_radius_curve, vehicle_position):

    # x and y values generation
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # image to draw the lines
    color_warp = np.zeros((720, 1280, 3), dtype='uint8')

    # Recast the x and y points into usable format
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # inverse perspective matrix (Minv) to write the disgnostic information on the image before the processing
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    # result combinaison with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Diagnostic infomation on the image : Lane curvature values and vehicle position from the center 
    avg_curve = (left_radius_curve + left_radius_curve)/2
    label_str = 'Radius of curvature: %.1f m' % avg_curve
    result = cv2.putText(result, label_str, (30,60), 0, 1, (0,0,0), 2, cv2.LINE_AA) 
    
    label_str = 'Deviation from lane center: %.1f m' % vehicle_position
    result = cv2.putText(result, label_str, (30,100), 0, 1, (0,0,0), 2, cv2.LINE_AA)
      
    return result 

