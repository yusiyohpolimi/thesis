import argparse
import numpy as np
import json
import itertools
import utm
import osrm
import folium
import glob
import pymap3d as pm
import dtld_parsing.driveu_dataset

from dtld_parsing.calibration import CalibrationData
from dtld_parsing.three_dimensional_position import ThreeDimensionalPosition
from folium import plugins  
from scipy.cluster.hierarchy import fclusterdata
from copy import deepcopy
from pykalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class CustomDatabase:
    """
    Class describing the Custom DriveU Dataset containing a list of images
    Attributes:
        images (List of DriveuImage)  All images of the dataset
        file_path (string):           Path of the dataset (.json)
    """

    def __init__(self, images):
        self.images = []
        self.input_images = images

    def open(self, database_dir: str = ""):
        """
        Method loading the dataset
        Args:
            database_dir(str): Base path where images are stored, optional
            if image paths in json are outdated
        """
        
        for image_dict in self.input_images:
            # parse and store image
            image = DriveuImage()
            image.parse_image_dict(image_dict, database_dir)
            self.images.append(image)

        return True    


def positions_3D(args):
    """
    Function from dtld_parsing repo which calculates the position of the 
    objects with respect to the camera frame in meters.
    """
    # Load database
    if None: #args.images:
        DriveuDatabase = CustomDatabase        
        database = DriveuDatabase(args.images)
    else:
        DriveuDatabase = dtld_parsing.driveu_dataset.DriveuDatabase
        database = DriveuDatabase(args.label_file)
        
    if not database.open(args.database_dir):
        return False
    # Load calibration
    calibration_left = CalibrationData()
    intrinsic_left = calibration_left.load_intrinsic_matrix(
        args.calib_dir + "/intrinsic_left.yml"
    )
    rectification_left = calibration_left.load_rectification_matrix(
        args.calib_dir + "/rectification_left.yml"
    )
    projection_left = calibration_left.load_projection_matrix(
        args.calib_dir + "/projection_left.yml"
    )
    extrinsic = calibration_left.load_extrinsic_matrix(
        args.calib_dir + "/extrinsic.yml"
    )
    distortion_left = calibration_left.load_distortion_matrix(
        args.calib_dir + "/distortion_left.yml"
    )
    calibration_right = CalibrationData()
    projection_right = calibration_right.load_projection_matrix(
        args.calib_dir + "/projection_right.yml"
    )
    
    threed_position = ThreeDimensionalPosition(
        calibration_left=calibration_left,
        calibration_right=calibration_right,
        binning_x=2,
        binning_y=2,
        roi_offset_x=0,
        roi_offset_y=0,
    )
    
    TL_coords = []
    class_list = []
    no_pred_id = []
    
    for idx_d, img in enumerate(database.images):       

        try:
            disparity_image = img.get_disparity_image()
        except:
            print('Disparity images are not available from this point: {}'.format(idx_d))
            break
        
        TL_per_img = []
        classes_img = []
        # Get 3D position
        for o in img.objects:    
            aspect = o.attributes['aspects']
            direction = o.attributes['direction']
            orientation = o.attributes['orientation']
            occ = o.attributes['occlusion']
            reflection = o.attributes['reflection']
            x =  o.x
            y =  o.y
            w =  o.width
            h =  o.height
            if x < 0:
                w = w + x
                x = 0                       
            elif y < 0:
                h = h + y
                y = 0
            elif x + w > 2048:
                w = 2048 - x
            elif y + h > 1024:
                h = 1024 - y 
            
            o.x = x
            o.y = y
            o.width = w
            o.height = h        
            if aspect == '':
                # TL_per_img.append(np.array([]))
                # classes_img.append(['', ''])
                # print('hi')
                continue
            
            if reflection == 'reflected' or occ == 'occluded' or aspect == 'four_aspects' \
                    or orientation == 'horizontal' or aspect == 'unknown':                
                continue

            # Camera Coordinate System: X right, Y down, Z front
            threed_pos = threed_position.determine_three_dimensional_position(
                o.x, o.y, o.width, o.height, disparity_image=disparity_image
            )
            # Get in vehicle coordinates: X front, Y left and Z up
            threed_pos_numpy = np.array([threed_pos.get_pos()[0], 
                                         threed_pos.get_pos()[1], 
                                         threed_pos.get_pos()[2], 
                                         1])
            threed_pos_vehicle_coordinates = extrinsic.dot(threed_pos_numpy)
            classes = [aspect, direction]
            TL_per_img.append(threed_pos_vehicle_coordinates[:2])   # take only x, y
            classes_img.append(classes)
        
        # if len(TL_per_img):   
        TL_coords.append(np.array(TL_per_img))
        class_list.append(classes_img)

    TL_coords = list(filter(lambda x: x != np.array([]), TL_coords))    
    class_list = list(filter(lambda x: x != np.array([]), class_list))    

    return TL_coords, class_list


def get_GPS(json_path):
    """
    From DTLD v2.0 label file, GPS measurements are taken and 
    grouped by sequences.
    Returns grouped longitude latitude list, velocity and yaw
    rate list, timestamp list and indices to split raw data
    into sequence lists.
    """    
    with open(json_path) as f:
        parsed = json.load(f)    
    data = parsed['images']

    LLA = []
    vel_list = []
    time_list = []
    sequence = []
    indices = []
    failed_ind = []
    prev_seq = None

    for idx, image in enumerate(data): 
        img_path = image['image_path']
        path_split = img_path.split('/')   
        curr_seq = path_split[3]
        lat = image["latitude"]
        lon = image["longitude"]
        time_stamp = image["time_stamp"]
        vel = image["velocity"]
        yaw_rate = image["yaw_rate"]
        vel_list.append([vel, yaw_rate])
        time_list.append(time_stamp)
        
        # to filter out the measurements which are not in Germany
        # Better to save idx so that I can mask it as failed measurement
        # while giving into KF
        if (lat > 60.0 or lat < 40.0) or (lon > 20.0 or lon < 2.5):
            failed_ind.append(idx)
            continue
            
        if idx == 0:
            sequence.append([lat, lon])         
                
        if curr_seq == prev_seq:
            sequence.append([lat, lon])      
            
        if (curr_seq != prev_seq and idx > 0):       # checking the sequence folder
            LLA.append(sequence)
            indices.append(len(sequence))            
            sequence = []
            sequence.append([lat, lon])  # append first element of the new sequence            
            
        prev_seq = curr_seq
        
    LLA.append(sequence)   # adding last sequence to the list
    indices = np.cumsum(indices)
    failed_ind = np.cumsum(failed_ind)
    
    return LLA, vel_list, time_list, [indices, failed_ind]


def get_UTM(LLA):
    """
    Taking input as (latitude, longitude) and return UTM list as
    (Easting, Northing, Zone Number, Zone Letter)
    """    
    UTM_list = []
    if np.shape(LLA[0]) == (2,):
        lats = np.array(LLA).T[0]
        longs = np.array(LLA).T[1]
        UTM = utm.from_latlon(lats, longs)
        UTM_list.append(UTM)

    else:
        for lla in LLA:
            lats = np.array(lla).T[0]
            longs = np.array(lla).T[1]
            UTM = utm.from_latlon(lats, longs)
            UTM_list.append(UTM)

    return UTM_list


def match_GPS(coords, radius=20.0):
    """
    Taking longitude latitude coordinates and snap them onto the
    nearest road segment.
    """
    client = osrm.Client(host='http://0.0.0.0:5000')
    if len(coords) == 1:
        response = client.nearest(
            coordinates=coords,
            radiuses=[radius]*len(coords)
        )
        loc = response['waypoints'][0]['location']
        locs_map = [list(reversed(loc))]
        matchings_map = locs_map        
    
    else:
        response = client.match(
            coordinates=coords,
            overview=osrm.overview.full,
            radiuses=[radius]*len(coords)
        )

        locs = [idx['location'] for idx in response['tracepoints'] if idx]
        locs_map = [list(reversed(loc)) for loc in locs]
        matchings = response['matchings'][0]['geometry']['coordinates']
        matchings_map = [list(reversed(loc)) for loc in matchings]
   
    return locs_map, matchings_map   
    

def mark_map(loc, grouped=False, show_popup=False, classes=False):
    """
    Visualization of the positions of the traffic lights,
    and vehicle. OSM France map is used since it has max 
    zoom 20 instead of 19. 
    'grouped' is used to visualize sequences. 
    'classes' is used for final map with class information.
    """
    color_list = ['black', 'blue', 'green', 'purple', 'orange', 'darkgreen',
                  'pink', 'lightred', 'black', 'lightgray', 'beige', 'lightblue']
    shape_list = sorted(glob.glob('osrm/shapes/*'))[:3] \
                 + sorted(glob.glob('osrm/shapes/*'))[6:]
    cls_list = []
    tile_layer = folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png',
        attr='&copy; OpenStreetMap France | &copy; \
    <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        zoom_start=18,
        max_zoom=20,
        name='Frmap',
        control=False,        
    )
    
    shadow = 'https://www.nicepng.com/png/full/409-4092997_lifestyle-for-mobile-shadow-png.png'
    if classes:
        cls_list = loc[1]
        loc = loc[0]

    if grouped:
        max_len = max(max(len(loc_groups) for loc_groups in loc), len(loc))
        color_list = color_list * np.ceil(max_len / len(color_list)).astype(np.int8)
        bulb_list = ['three_aspects', 'one_aspect', 'two_aspects']
        dir_list = ['back', 'front', 'left', 'right']
        gt_cls_list = [list(p) for p in itertools.product(dir_list, bulb_list)]
        
        try:
            # If multiple sequences with multiple images are given as input
            my_map = folium.Map(location=loc[0][0][0], zoom_start=18, max_zoom=20, tiles=None)
            tile_layer.add_to(my_map)  # add France map tile
            
            for seq_id, loc_groups in enumerate(loc):
                # Creating feature group for each sequence to add corresponding images
                # to the groups.
                seq_group = folium.FeatureGroup(name=str(seq_id+1), show=show_popup)
                my_map.add_child(seq_group)                
                
                for img_id ,coords in enumerate(loc_groups):
                    # Creating feature subgroups for each image to add traffic lights
                    # to the images.
                    img_group = plugins.FeatureGroupSubGroup(
                        seq_group, 
                        name="seq "+str(seq_id+1)+"img "+str(img_id+1),
                        show=show_popup
                    ) 
                    my_map.add_child(img_group)
                    for idx, lla in enumerate(coords):
                        # Choosing colors for different images
                        clr_idx = gt_cls_list.index(cls_list[img_id][idx]) if cls_list else img_id 
                        popup_name = ','.join(curr_cls) if classes else str(idx)
                        
                        marker = folium.Marker(
                            location=lla, 
                            popup=folium.Popup(popup_name, show=True),
                            icon=folium.map.Icon(color=color_list[clr_idx])
                        )                         
                        marker.add_to(img_group)
                        
#             my_map.add_child(MeasureControl()) # optional distance measure        
            folium.LayerControl(collapsed=False).add_to(my_map)
            
        except: 
            # If a sequence with multiple images is given
            max_len = max(max(len(coords) for coords in loc), len(loc))
            color_list = color_list * np.ceil(max_len / len(color_list)).astype(np.int8)
            my_map = folium.Map(location=loc[0][0], zoom_start=18, max_zoom=20, tiles=None)
            tile_layer.add_to(my_map)
            for group_id ,coords in enumerate(loc):
                group = folium.map.FeatureGroup(name=str(group_id+1), show=show_popup)
                for idx, lla in enumerate(coords):
                    if classes:
                        # If classes are given with coordinates,
                        # custom icons are used to visualized 
                        # the traffic lights.
                        curr_cls = cls_list[group_id][idx]
                        curr_cls = [curr_cls[1], curr_cls[0]]                        
                        clr_idx = gt_cls_list.index(curr_cls)
                        popup_name = ','.join(curr_cls)
                        # different sizes for different bulb numbers
                        icon_sizes = [(12, 24), (12, 36), (12, 12)]
                        icon = folium.features.CustomIcon(
                            shape_list[clr_idx], 
                            icon_size=icon_sizes[(clr_idx+1)%3],
                            shadow_image=shadow,
                            shadow_size=icon_sizes[(clr_idx+1)%3],
                            shadow_anchor=(-2, 17)
                        )

                        marker = folium.Marker(
                            location=lla, 
                            popup=folium.Popup(popup_name, show=False),
                            icon=icon
                        )
                        
                    else:
                        # If there is no class information, choose colors
                        # for circular markers instead of custom icons.
                        clr_idx = group_id
                        popup_name = str(idx)
                        icon = folium.map.Icon(color=color_list[clr_idx%12])
                        marker = folium.CircleMarker(location=lla, radius=8, 
                                                     fill_color=color_list[clr_idx%12], 
                                                     color='white', fill_opacity=0.75,
                                                     popup=folium.Popup(popup_name, show=False),
                                                     show=True)
            

                    marker.add_to(group)
                my_map.add_child(group)
#             my_map.add_child(MeasureControl())       
            folium.LayerControl(collapsed=False).add_to(my_map)                 
    
    elif len(loc) == 2 and isinstance(loc[0], float) == 1:  
        # only 1 coordinate is given
        my_map = folium.Map(location=loc, zoom_start=18, max_zoom=19)
        marker = folium.Marker(location=loc, 
                               popup=folium.Popup(str(1), show=False))
        marker.add_to(my_map)

    else:        
        my_map = folium.Map(location=loc[0], zoom_start=18, max_zoom=19)
        for idx, lla in enumerate(loc):
            marker = folium.Marker(location=lla, popup=folium.Popup(str(idx),
                                        show=False))
            marker.add_to(my_map)
        
    return my_map
    

def to_enu(LLA):
    """
    Converting longitude latitude to local ENU coordinates
    using pymap3d package. Altitude is taken as 0 for all 
    sequences since the difference in altitude is negligible
    for frames in a sequence.
    """
    lat0 = [seq[0][0] for seq in LLA if len(seq[0])]
    lon0 = [seq[0][1] for seq in LLA if len(seq[0])]
    h0 = np.zeros(len(lat0))
    lats = [np.transpose(seq)[0] for seq in LLA if len(seq[0])]
    lons = [np.transpose(seq)[1] for seq in LLA if len(seq[0])]
    hs = np.zeros(len(lats))

    ENU = []
    for idx in range(len(lats)):
        e, n, u = pm.geodetic2enu(lats[idx], lons[idx], hs[idx],
                                  lat0[idx], lon0[idx], h0[idx])
        
        ENU.append(np.array([e, n, u]))             

    return ENU


def angle_from_gps(gps_seq):           
    """
    Calculates angle between two consecutive 
    ENU coordinates in the whole array of
    sequence.
    """    
    gps_x = gps_seq[0]
    gps_y = gps_seq[1]
    dy = gps_y[1:] - gps_y[:-1]
    dx = gps_x[1:] - gps_x[:-1]
    dy = np.append(dy, dy[-1])
    dx = np.append(dx, dx[-1]) 
    angles = np.arctan2(dy, dx)
    
    return angles


def project_TL(LLA, TL_splitted):
    """
    Takes coordinate measurements of the vehicle
    and sequence-wise grouped traffic light 
    positions with respect to the camera frame.
    Returns projected traffic light coordinates
    in latitude longitude, ENU, 
    """
    tl_latlon_list = []
    tl_enu_list = []
    angle_list = []
    gps_tl_list = []
    
    if len(LLA[0][0]) == 2:    # [Lat, Lon] given
        gps = to_enu(LLA)    
        lat0 = [seq[0][0] for seq in LLA if len(seq[0])]
        lon0 = [seq[0][1] for seq in LLA if len(seq[0])]
        h0 = np.zeros(len(lat0))
        # hs = np.zeros(len(LLA))
        
    else:        
        gps = LLA
    
    for seq_id, (gps_seq, tl_seq) in enumerate(zip(gps, TL_splitted)):
        if tl_seq.size:
            gps_arr = np.transpose(gps_seq[:2])
            angles = angle_from_gps(gps_seq)  # using ENU coordinates         
            R = np.array([[np.cos(angles), -np.sin(angles)],
                          [np.sin(angles),  np.cos(angles)]])
            R = np.transpose(R, (2,0,1))
            
            rotated_tl = [((R[idx] @ tl_img.T).T + gps_arr[idx]).tolist()
                          for idx, tl_img in enumerate(tl_seq)]
            
            if len(gps[0]) == 4:      # UTM measurements given
                zone_number = gps_seq[-2]
                zone_letter = gps_seq[-1]
                tl_latlon = [np.transpose(utm.to_latlon(tl.T[0], tl.T[1], 
                                                        zone_number, 
                                                        zone_letter)
                                         ).tolist()
                             for tl in rotated_tl]
                
            
            else:          # ENU measurements given      
                tl_latlon = [np.transpose(
                                pm.enu2geodetic(
                                    np.transpose(tl)[0], np.transpose(tl)[1], 0,
                                    lat0[seq_id], lon0[seq_id], h0[seq_id] 
                                )[:2]
                             ).tolist()
                             for idx, tl in enumerate(rotated_tl)]
                
            tl_enu_list.append(rotated_tl)    
            tl_latlon_list.append(tl_latlon)
            angle_list.append(angles)
#             gps_tl_list.append(rotated_tl)
    
        else:
            tl_enu_list.append([[]])           
            tl_latlon_list.append([[]])
            angle_list.append([])
#             gps_tl_list.append([[]])
            
    return tl_latlon_list, tl_enu_list, angle_list #, gps_tl_list


def vel2velxy(gps_angle_init, vel_list, split_ids):
    """
    Projecting the velocity measurements of the vehicle 
    onto the East (x) and North (y) directions.
    """
    dt = 0.66   # sampling rate of the data
    seq_idx = split_ids
    yaw_rates = np.transpose(vel_list)[1]
    yaw_rates_splitted = np.split(yaw_rates, seq_idx)
    theta_splitted = []
    for seq, theta0 in zip(yaw_rates_splitted, gps_angle_init):
        theta_seq = []
        theta_seq.append(theta0)
        for idx, yaw_rate in enumerate(seq):    
            theta_seq.append(theta_seq[idx] + yaw_rate * dt)
        theta_seq.pop(-1)
        theta_splitted.append(theta_seq)  

    velocities = np.transpose(vel_list)[0]
    vel_splitted = np.split(velocities, seq_idx)
    vel_xy_splitted = [vel_seq[...,None] * np.array([np.cos(theta_seq), 
                                                     np.sin(theta_seq)]).T 
                       for vel_seq, theta_seq in zip(vel_splitted, theta_splitted)]
    
    return [vel_xy_splitted, yaw_rates_splitted]


def get_meas_kf(snapped, vel_splitted):
    """
    Constructing measurement vector for 
    Kalman Filter which is in the form of
    [x, x_dot, y, y_dot, theta_dot] 
    from the ENU coordinates, and splitted
    and projected velocity data.
    """
    lat0 = [seq[0][0] for seq in snapped]
    lon0 = [seq[0][1] for seq in snapped]
    h0 = np.zeros(len(lat0))
    lats = [np.transpose(seq)[0] for seq in snapped]
    lons = [np.transpose(seq)[1] for seq in snapped]
    hs = np.zeros(len(lats))
    xe, yn, zu = [], [], []

    for idx in range(len(lats)):
        e, n, u = pm.geodetic2enu(lats[idx], lons[idx], hs[idx],
                                  lat0[idx], lon0[idx], h0[idx])
        xe.append(e), yn.append(n), zu.append(u)  

    vel_xy_splitted = vel_splitted[0]
    yaw_rates_splitted = vel_splitted[1]    
    measurements_enu = [
         np.array(
         [np.asarray([xe, vel_xy[0], yn, vel_xy[1], yaw_rate])
         for xe, yn, vel_xy, yaw_rate   
         in zip(xe_seq, yn_seq, vel_xy_seq, yaw_rate_seq)]
                 ) 
         for xe_seq, yn_seq, vel_xy_seq, yaw_rate_seq 
         in zip(xe, yn, vel_xy_splitted, yaw_rates_splitted)
        ]

    meas_kf = np.array(measurements_enu)
    return meas_kf, [lat0, lon0, h0]


def kf_car(snapped, vel_splitted, plot=False):
    """
    Kalman Filter with Expectation-Maximization
    algorithm for better localization of the 
    vehicle. Snapped GPS measurments, splitted 
    and projected velocity data is converted
    KF measurements with get_meas_kf().
    """    
    dt = 0.66  # calculated from the timestamps
    
    # states = x x_dot y y_dot theta theta_dot
    transition_matrix = [[1, dt, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, dt, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, dt],
                         [0, 0, 0, 0, 0, 1]]
    
    # observed states = x x_dot y y_dot theta_dot
    observation_matrix = [[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],                     
                          [0, 0, 0, 0, 0, 1]]
   
    # get the measurements in appropriate format
    meas_kf, lla = get_meas_kf(snapped, vel_splitted)
    
    # initial localization for each sequence
    lat0, lon0, h0 = lla[0], lla[1], lla[2]    
    smoothed_LLA = []
    
    for idx, meas_seq in enumerate(meas_kf):
        if len(meas_seq):
            
            # Initializing states 
            initial_state_mean = [0.0,
                                  meas_seq[0][1],
                                  0.0,
                                  meas_seq[0][3],
                                  0.0,
                                  meas_seq[0][-1]]
            
            kf1 = KalmanFilter(transition_matrices=transition_matrix,
                               observation_matrices=observation_matrix,
                               initial_state_mean=initial_state_mean,
                               n_dim_obs=5)
            
            # Expectation-Maximization with 3 iterations on observation covariance
            kf1 = kf1.em(meas_seq, 
                         n_iter=3, 
                         em_vars=['observation_covariance'])
            
            (smoothed_state_means, smoothed_state_covariances) = kf1.filter(meas_seq)
                
            # plot the smoothed states and state covariances 
            if plot:    

                # %matplotlib inline
                import matplotlib                
                import matplotlib.pyplot as plt
                matplotlib.rcParams['figure.dpi'] = 600                
                
                times = range(meas_seq.shape[0])
                fig, axs = plt.subplots(2)
                fig.suptitle('vs Time and XY Graphs')
                axs[0].plot(times, meas_seq[:, 0], 'bo',
#                             times, meas_seq[:, 2], 'ro',
#                             times, meas_seq[:, 4], 'go',         
                            times, smoothed_state_means[:, 0], 'ro',
#                             times, smoothed_state_means[:, 2], 'r--',
#                             times, smoothed_state_means[:, 4], 'g--',
                           )
                
                for i, txt in enumerate(meas_seq[:, 0]):
                    axs[0].annotate(i, (times[i], txt+1), fontsize=5)
                    axs[0].annotate(i, 
                                    (times[i], smoothed_state_means[:, 0][i]), 
                                    fontsize=5)
                    
                axs[1].plot(meas_seq[:, 0], meas_seq[:, 2], 'bo')
                axs[1].plot(smoothed_state_means[:, 0], smoothed_state_means[:, 2], 'ro')
                for i, txt in enumerate(meas_seq[:, 0]):
#                     axs[1].annotate(round(txt), (times[i], txt+1), fontsize=5)
                    axs[1].annotate(
                        i, 
                        (smoothed_state_means[:, 0][i], smoothed_state_means[:, 2][i]-1.2), 
                        fontsize=5
                    )
                
                plt.show()
#                 _ = input('Press enter to continue')

            lat_smooth, lon_smooth, h_smooth = pm.enu2geodetic(
                smoothed_state_means[1:, 0], 
                smoothed_state_means[1:, 2], 
                smoothed_state_means[1:, 4],
                lat0[idx], lon0[idx], h0[idx]
            )        
            lat_smooth = np.concatenate(([lat0[idx]], lat_smooth)) 
            lon_smooth = np.concatenate(([lon0[idx]], lon_smooth)) 
            smoothed_LLA.append(np.column_stack((lat_smooth, lon_smooth)).tolist())
        else:
            smoothed_LLA.append([[]])
                            
    return smoothed_LLA


class Track(object):

    def __init__(self, prediction, attributes, trackIdCount, first_seen):
        """
        prediction: predicted xy of the object 
        attributes: classes from the model
        trackIdCount: id for each track
        first_seen: the first frame when the detection
                    is declared as a track
        """
        self.track_id = trackIdCount 
        # states are [x, y] <=> [E, N], both are measured
        transition_matrix = [[1, 0], 
                             [0, 1]]    
        observation_matrix = [[1, 0],
                              [0, 1]] 
        
        # first element of the trace of the track is initial state
        initial_state_mean = [prediction[0],
                              prediction[1]] 
        
        # initialazing observation covariance matrix
        observation_covariance = np.diag((10.0, 5.0))        
        # initialazing state covariance matrix
        state_covar_init = np.diag((30.0, 15.0))        
        self.KF = KalmanFilter(transition_matrices=transition_matrix,
                               observation_matrices=observation_matrix,
                               initial_state_mean=initial_state_mean,
                               observation_covariance=observation_covariance,
                               n_dim_obs=2)  
        # keeping last state and covariance information
        self.last_mean = initial_state_mean
        self.last_covar = state_covar_init        
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path of the track
        self.att = np.array(attributes)  # two-labeled class information
        self.P_list = []  # keeping state covariance matrix
        self.P_list.append(state_covar_init)
        self.trace.append(self.prediction)  # constructing trace of the track
        self.first_seen = first_seen


class Tracker(object):
    """
    Tracker class that updates track-related vectors 
    of the objects which are tracked.
    """

    def __init__(self, dist_thresh, max_frames_to_skip):
        """
        dist_thresh: distance threshold. When exceeds the threshold,
                     track will be deleted and new track is created.
        max_frames_to_skip: maximum allowed frames to be skipped for
                            the track undetected.
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.tracks = []  
        self.trackIdCount = 0  # trackIdCount: identification of each track
                
    def calculate_cost(self, detections, classes):
        """
        Calculating the Euclidean distance between the
        detections and the tracks with the same class
        information, and constructing cost matrix. 
        """
        N = len(self.tracks)
        M = len(detections)
        self.cost = np.zeros((N, M))
        for track_id, track in enumerate(self.tracks):
            for det_id, detection in enumerate(detections):
                if all(track.att == classes[det_id]):
                    # same class information 
                    diff = track.prediction - np.array(detection)
                    # Euclidean distance
                    distance = np.sqrt(diff.data[0]**2
                                       + diff.data[1]**2) 
                    self.cost[track_id][det_id] = distance
                else: 
                    # classes are not the same, put high cost
                    self.cost[track_id][det_id] = 1e8
                
    def hungarian(self, detections, classes, det_idx):
        """
        Deploying Hungarian Algorithm to assign the detections
        to the correct tracks. 
        """
        N = len(self.tracks)
        assignment = [-1] * N  # initializing with no matches
        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(self.cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        for i in range(len(assignment)):
            if assignment[i] != -1:
                # check for cost distance threshold.
                # If cost is very high then un_assign the track
                if (self.cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1

        # If tracks are not detected for a long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames >= self.max_frames_to_skip:
                del_tracks.append(i)                

        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            self.tracks = np.delete(self.tracks, del_tracks, axis=0).tolist()
            assignment = np.delete(assignment, del_tracks, axis=0).tolist()
        
        # Looking for unassigned detections
        for i in range(len(detections)):
            if (i not in assignment):
                # create new track for unassigned detections
                track = Track(detections[i], 
                              classes[i],
                              self.trackIdCount,                              
                              det_idx)
                self.trackIdCount +=1
                self.tracks.append(track)        

        return assignment

    def update(self, detections, det_idx):
        """
        - Initialize tracks
        - Calculate cost between predicted and
          detected locations
        - Assign detections to tracks using Hungarian
          Algorithm
        - Kalman Filter update step
        detections: model predictions in the form of
                    [xy, classes].        
        det_idx: it is used as the first seen frame        
        """
        classes = detections[1]
        detections = detections[0]
        
        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):          
            for i in range(len(detections)):
                track = Track(detections[i], classes[i], 
                              self.trackIdCount, 
                              det_idx)                
                self.trackIdCount += 1
                self.tracks.append(track)
        
        # No need for any action at first frame
        # of the sequence.
        if det_idx == 0:
            return
        
        # calculate the cost using both xy and classes
        self.calculate_cost(detections, classes)
        
        # assigning detections to tracks with Hungarian Algorithm,
        # or creating new tracks.        
        assignment = self.hungarian(detections, classes, det_idx)
        
        for i in range(len(assignment)):
            mean_curr = self.tracks[i].last_mean
            covar_curr = self.tracks[i].last_covar            
            if assignment[i] != -1:
                # since there is assignment, set skipped frames to 0
                self.tracks[i].skipped_frames = 0
                
                mean_curr, covar_curr = self.tracks[i].KF.filter_update(
                    mean_curr, 
                    covar_curr, 
                    detections[assignment[i]],
                )
                
                mean_curr = mean_curr.data  # data of masked array 
            else:
                # use covar_curr as observation covariance since there is no 
                # observation
                mean_curr, covar_curr = self.tracks[i].KF.filter_update(
                    mean_curr, 
                    covar_curr,
                    observation=None,
                    observation_covariance=covar_curr
                )
                # since no assignment for the track, it is skipped
                self.tracks[i].skipped_frames += 1                
           
            self.tracks[i].prediction = mean_curr
            self.tracks[i].trace.append(self.tracks[i].prediction)                
            self.tracks[i].last_mean = mean_curr
            self.tracks[i].last_covar = covar_curr
            self.tracks[i].P_list.append(covar_curr)            


def thres_filter(smoothed, heading_angles, thres, indices):
    """
    Outlier rejection for the initial map, using elliptic
    search window considering the moving direction of the 
    vehicle.
    """
    img_mean_list = [np.mean(img, axis=0) for seq in smoothed[0] for img in seq]
    seq_mean_list = np.split(img_mean_list, indices) 
    filtered_smooth = deepcopy(smoothed[0])
    classes = deepcopy(smoothed[1])
    for seq_id, (seq_mean, tl_seq, angle) in enumerate(zip(seq_mean_list, filtered_smooth, heading_angles)):
        if seq_mean.size and len(tl_seq[0]) and len(angle):
            R = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]])
            R = np.transpose(R, (2, 0, 1))

            for img_id, (img_mean, tl_img) in enumerate(zip(seq_mean, tl_seq)):            
                diff = tl_img - img_mean
                diff = (R[img_id] @ diff.T).T   # projecting according to heading angle of the car
                dist2mean = np.sqrt(diff.T[0] ** 2 + 10 * diff.T[1] ** 2).T   # elliptic window 

                if (dist2mean > thres).any():
                    del_id = np.nonzero(dist2mean > thres)
                    filtered_smooth[seq_id][img_id] = np.delete(filtered_smooth[seq_id][img_id], 
                                                                del_id, axis=0).tolist()
                    classes[seq_id][img_id] = np.delete(classes[seq_id][img_id], 
                                                        del_id, axis=0).tolist()
    
    return [filtered_smooth, classes], seq_mean_list


def tl_tracker(filtered_smoothed, smoothed_LLA):
    """
    Utilizing the location information of the vehicle,
    the projected traffic lights are localized and 
    tracked. Track-related vectors are returned to
    map, plot covariances, observe the traces.

    filtered_smoothed: [xy, classes] of the projected 
                       traffic lights
    smoothed_LLA: latitude longitude coordinates of the 
                  vehicle to get initial position for
                  each sequence
    """
    results = []
    trace_list = []
    att_list = []
    covar_list = []
    first_seen_list = []
    del_traces_list, del_covars_list, del_seen_list = [], [], []
    
    coords = filtered_smoothed[0]
    classes = filtered_smoothed[1]
    # a traffic light should be detected atleast 2 frames 
    min_trace_len = 2  
    search_dist = 10.0  # distance threshold between the lights

    # remove empty sequences due to missing disparity images
    if [[]] in coords:
        coords = [seq for seq in coords if seq[0]]

    for seq_id, det_seq in enumerate(coords):
        # set allowed skipped frames for the sequence 
        max_frames_to_skip = np.ceil(len(det_seq) / 2).astype(np.int8)
        # initialize the Tracker for the sequence
        tracker = Tracker(search_dist, max_frames_to_skip) 
        # get initial position of the vehicle to set it as the origin of
        # local ENU frame
        lat0 = smoothed_LLA[seq_id][0][0]
        lon0 = smoothed_LLA[seq_id][0][1]
        h0 = 0

        for det_idx, det_img in enumerate(det_seq):
            # update Tracker for each frame            
            tracker.update([det_img, 
                            classes[seq_id][det_idx]], 
                            det_idx)
        
        tracks = tracker.tracks
        traces, att, covars, first_seen, skipped \
            = np.transpose([[track.trace, track.att, track.P_list, 
                             track.first_seen, track.skipped_frames] 
                             for track in tracks]).tolist()
        del_idx = []
        for idx, trace in enumerate(traces):
            trace = np.array(trace)
            _, index= np.unique(trace, return_index=True,
                                axis=0)
            # find the frames where an assignment occurs     
            uniq = trace[np.sort(index)]       
            if (len(uniq) <= min_trace_len): 
                del_idx.append(idx)
        
        # keep deleted vectors 
        if del_idx:
            del_traces = np.array(traces)[np.array(del_idx)].tolist()
            del_covars = np.array(covars)[np.array(del_idx)].tolist()
            del_seen = np.array(first_seen)[np.array(del_idx)].tolist()
        else:
            del_traces = del_covars = del_seen = None
        
        traces = np.delete(traces, del_idx, axis=0).tolist()
        att = np.delete(att, del_idx, axis=0).tolist()
        covars = np.delete(covars, del_idx, axis=0).tolist()
        first_seen = np.delete(first_seen, del_idx, axis=0).tolist()
        last_coords = [group[-1] for group in traces if group]
        
        # group the last coordinates since different tracks can 
        # start from different positions but end up very close
        # to each other
        last_coords, att = group_last(last_coords, att)
        # transform from ENU to latitude longitue
        tl_latlon = [np.transpose(
                     pm.enu2geodetic(tl.T[0], tl.T[1], h0,
                                     lat0, lon0, h0
                                    )[:2]
                                 ).tolist()
                     for idx, tl in enumerate( last_coords)] 
        
        results.append(tl_latlon)
        trace_list.append(traces)
        att_list.append(att)
        covar_list.append(covars)
        first_seen_list.append(first_seen)
        if del_traces:
            del_traces_list.append(del_traces)
            del_covars_list.append(del_covars)
            del_seen_list.append(del_seen)
        
    return results, trace_list, att_list, covar_list, first_seen_list, \
           [del_traces_list, del_covars_list, del_seen_list]


def group_last(detections, classes):
    """
    Groups the last coordinates of each traces if
    they are closer than 5.0 m and have same class
    information.
    """
    cls = np.array(classes)
    try:
        srt = np.lexsort((cls[:,1], cls[:,0]))
    except:
        return detections, classes
    
    cls_srt= np.array(cls)[srt]
    grouped = []
    cls_grp = []
    # grouping sorted classes
    for _, group in itertools.groupby(cls_srt, lambda x: (x[0], x[1])):
        group = list(group)
        cls_grp.append(group)
        grouped.append(len(group))
    
    # grouping xy of the detections according to classes
    group_ind = np.cumsum(grouped)
    det_srt = np.array(detections)[srt]    
    det_grp = np.split(det_srt, group_ind[:-1])
    det_list = []
    
    for idx, dets in enumerate(det_grp):
        if len(dets) == 1:  # only one element in the group
            det_list.append(dets[0])
            continue

        # clustering the xys with 5.0m distance threshold
        cluster_ind = fclusterdata(dets, 5.0, criterion='distance') 
        det_splits = [dets[cluster_ind==idx+1, :] 
                      for idx in range(max(cluster_ind ))]
        for split in det_splits:
            if len(split) > 1:
                del_cnt = len(split) - 1      
                # get mean of the cluster, since they are the same object                
                split = [np.mean(split, axis=0)]
                # only keep one class information, delete others                
                cls_grp[idx] = cls_grp[idx][:-del_cnt]
            
            det_list.append(split[0])

    detections = np.concatenate([det_list], axis=0)
    classes = np.concatenate(cls_grp[:0] + cls_grp[:], axis=0)

    return detections, classes


def main(args):

    TL_coords, classes = positions_3D(args)  # 2D positions with respecto the car
    LLA, vel_list, times, indices = get_GPS(args.label_file)  # GPS data from json file
    LLA_rev = [[list(reversed(loc)) for loc in seq] for seq in LLA]  # reverse to use OSRM
    snapped = []
    # Snapping the GPS data sequence-wise
    for seq in LLA_rev:    
        locs_map, _ = match_GPS(seq)
        snapped.append(locs_map)
    # import pdb;pdb.set_trace()
    # Splitting the TL coordinates into sequences
    TL_splitted = np.split(TL_coords, indices[0])
    # Projecting the velocity measurements onto East and North directions
    # and using Kalman Filter with Expectation-Maximization for better
    # localization of the vehicle.
    enu_car = to_enu(snapped)
    gps_angle_init = [angle_from_gps(seq)[0] for seq in enu_car]
    vel_splitted = vel2velxy(gps_angle_init, vel_list, indices[0])
    smoothed_snapped = kf_car(snapped, vel_splitted)

    # Projecting the TL positions from camera frame to the local ENU frame
    smoothed, smoothed_enu, angles = project_TL(smoothed_snapped, TL_splitted)
    classes_split = np.split(classes, indices[0])
    # Outlier rejection
    smoothed_preds = [smoothed_enu, classes_split]
    filtered_smoothed, seq_mean_list = thres_filter(smoothed_preds, angles, 150.0, indices[0])
    
    # Tracking and obtaining track-related vectors
    track_results_ca = tl_tracker(filtered_smoothed, smoothed_snapped)
    res_smooth = track_results_ca[0]
    # traces = track_results_ca[1]
    classes = track_results_ca[2]
    # covar_list = track_results_ca[3]
    # first_seen_list = track_results_ca[4]
    # del_traces = track_results_ca[5][0]
    # del_covars = track_results_ca[5][1]
    # del_seen = track_results_ca[5][2]

    # Mapping the tracked TLs with class information
    tracked = [res_smooth, classes]
    mark_map(tracked, classes=True, grouped=True).save('tracked_map_last.html') 
    # Mapping the video sequence
    # car_tl = [tl + car for tl, car in zip(smoothed[7], smoothed_snapped[7])]
    # mark_map([smoothed[7], classes_split[7]], classes=True, grouped=True).save('seq7_nokf.html') 


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--calib-dir', type=str, 
        default='/dtld_parsing/calibration/', 
        help='calibration data directory'
    )
    parser.add_argument(
        '--database-dir', type=str, 
        default='/multiverse/datasets/shared/DTLD/Berlin_disp', 
        help='if all disparity images are extracted from the archive, no need to use this'
    )                        
    parser.add_argument(
        '--gt-file', type=str, 
        default='/multiverse/datasets/shared/DTLD/v2.0/v2.0/DTLD_test.json', 
        help='ground-truth json file'
    )    
    parser.add_argument(
        '--label-file', type=str, 
        default='det2/2022_04_12_1346_berlin1_inp_detection.json',
        help='json file of labels (gt or detection)'
    )
      
    args = parser.parse_args()    
    main(args)
