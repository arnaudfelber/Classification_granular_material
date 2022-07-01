
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import *
from sympy.physics.wigner import wigner_3j

# Get distances between one node and the other
def get_dist_3D (seed: np.array, point):
    number = int(point[4])
    dist =[]
    dist = np.append(dist, number)
    x0=point[0]
    y0=point[1]
    z0=point[2]
    for i in range (len(seed)):
        x1 = seed[i,0]
        y1 = seed[i,1]
        z1 = seed[i,2]
        dist_i = np.sqrt((y1-y0)**2+(x1-x0)**2+(z1-z0)**2)
        if (dist_i<=(0.007+10**(-6))):#ajout d'une petite tolérance
            dist = np.append(dist, dist_i)
        else:
            dist = np.append(dist, 0)
    return dist

# Get all the distances between all nodes
def get_dist_all_nodes_3D (seed: np.array):
    dist_all_nodes_3D =[]
    
    for i in range (len(seed)):
        dist_all_nodes_3D = np.append(dist_all_nodes_3D, get_dist_3D(seed,seed[i,:]))
    dist_all_nodes_3D = np.resize(dist_all_nodes_3D, (len(seed),len(seed)+1))
    
    for j in range (len(dist_all_nodes_3D)):
        dist_all_nodes_3D[i,0] = int(dist_all_nodes_3D[i,0])
        
    return dist_all_nodes_3D #each line represent the distance between the same line node and the other



# Get the nearest neighbour
def get_all_NN (dist_all: np.array):
    all_NN = {} #changer en matrice
    a_NN = np.empty((0,8),int)
    for i in range (len(dist_all)):
        NN=[]
        NN = np.append(NN, dist_all[i,0]) #add departure point
        for j in range (len(dist_all[0])): #len(dist_all[0]) 
            NN_num=[]
            #NN_dist=[]
            if(dist_all[i,j]!=0 and dist_all[i,j]<1):
                NN_num = np.append(NN_num,dist_all[j-1,0])
                #NN_dist = np.append(NN_dist,dist_all[i,j])
            NN = np.append(NN,NN_num)
            #NN = np.append(NN,NN_dist)
            
        all_NN[dist_all[i,0]] = NN # changer en matrice
    return all_NN #dictionnary

# Coodrination number
def get_CN (NN):
    CN = len(NN)
    return CN

# Get all the coordination number
def get_all_CN (all_NN):
    CNs= []
    for i in range (len(all_NN)):
        NN = get_CN(all_NN[i+1])
        CNs = np.append(CNs,NN)     
    return CNs    

# Get the voronoi area of a seed
def get_voronoi_volumes(seed):
    v = Voronoi(seed[:,:3])
    vol = np.zeros(v.npoints)
    points = v.point_region
    
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    
    volume = []
    point = []
    for i in range (len(vol)):    
        if (vol[i]<1):
            volume = np.append(volume,vol[i])
            point = np.append(point,i)
        else:
            pass
    
    return volume, point

# Calculate the volume fraction
def get_local_volume_fraction(seed):
    volume, point = get_voronoi_volumes(seed)
    lva = []
    for i in range (len(volume)):
        r = seed[int(point[i]),3]
        V = volume[i]
        lva = np.append(lva, ((4/3)*r**3*np.pi)/V)
        
    return lva

# Get the two angles between two neighbour and the x axes
def get_angle (seed:np.array, NN):
    n_xo = NN[0]#ajouter point de départ 
    angle1 = []
    angle2 = []
    for i in range (len(NN)-1):
        for j in range (len(seed)):
            if (n_xo == seed[j,4]):
                x0 = seed[j,0]
                y0 = seed[j,1]
                z0 = seed[j,2]
            elif (NN[i+1]== seed[j,4]):
                x1 = seed[j,0]
                y1 = seed[j,1]
                z1 = seed[j,2]
            
        adj = x1-x0
        hyp1 = np.sqrt((y1-y0)**2+(x1-x0)**2)
        hyp2 = np.sqrt((z1-z0)**2+(x1-x0)**2)
       
    
        if (hyp1 !=0):
            if (y1 < y0):
                angle1 = np.append(angle1,180+(360/(2*np.pi)) * np.arccos(adj/hyp1))            
            else:
                angle1 = np.append(angle1,(360/(2*np.pi)) * np.arccos(adj/hyp1))
            
            
        
        if (hyp2 !=0):    
            if (z1 < z0):
                angle2 = np.append(angle2,180+(360/(2*np.pi)) * np.arccos(adj/hyp2))            
            else:
                angle2 = np.append(angle2,(360/(2*np.pi)) * np.arccos(adj/hyp2))
            
            
    return angle1,angle2 

# Get all the angles
def get_all_angle (seed, all_NN):
    all_angle1= {}
    all_angle2= {}
    for i in range (len(all_NN)):
        angles1, angles2 = get_angle(seed,all_NN[i+1])
        all_angle1[seed[i,4]] = angles1
        all_angle2[seed[i,4]] = angles2
    return all_angle1, all_angle2


# Get the total volume fraction
def get_total_volume_fraction(seed:np.array):
    V_tot = 0.03**3 # get a larger cube to be sure to have all the grain inside  
    V_sub = 0
    
    for i in range (len(seed)):
        V_sub = V_sub + 4/3*np.pi * seed[i,3]**3
    t_v = V_sub/V_tot
    
    return t_v

#angle in 3D
def calculate_angle_3D(df, index1, index2, index3):
    # Angle between the vectors BA and BC.
    # B is the point given by index 1
    a = df.iloc[index2][['x', 'y', 'z']]
    b = df.iloc[index1][['x', 'y', 'z']]
    c = df.iloc[index3][['x', 'y', 'z']]

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return angle

#Enclidean distances
def calculate_distance_3D(df, index1, index2):
    atom1 = df.iloc[index1]
    atom2 = df.iloc[index2]
    dist = np.linalg.norm(atom1[['x', 'y', 'z']] - atom2[['x', 'y', 'z']])
    return dist



#Parameters for symmetry function
def fc(R, Rc, alpha_c, epsilon_c):
    if R < Rc:
        value = 1 / (1 + np.exp(alpha_c * (R - Rc + epsilon_c)))
        return value
    return 0

def fa(R, eta, mu):
    if R < mu + np.pi / (2 * eta) and R > mu - np.pi / (2 * eta):
        value = np.cos(eta * (R - mu)) ** 2
        return value
    return 0

def fb(R, nu, a1, ar):
    if R < a1 and R > a1 - np.pi / (2 * nu):
        value = np.cos(nu * (R - a1)) ** 2
    elif R < ar and R >= a1:
        value = 1
    elif R < ar + np.pi / (2 * nu) and R >= ar:
        value = np.cos(nu * (R - ar)) ** 2
    else:
        value = 0
    return value


# The 8 symmetry functions
def get_G1(df, i, neighbours, Rc=0.004, alpha_c=100, epsilon_c=0):
    sum = 0
    for index in neighbours:
        if index != i:
            R = calculate_distance_3D(df, i, index)
            value = fc(R, Rc, alpha_c, epsilon_c)
            sum += value
    return sum


def get_G2(df, i, neighbours, Rc=0.004, alpha_c=100, epsilon_c=0, eta=3, Rs=0.003):
    sum = 0
    for index in neighbours:
        if index != i:
            R = calculate_distance_3D(df, i, index)
            value = np.exp(-eta * (R - Rs) ** 2) * fc(R, Rc, alpha_c, epsilon_c)
            sum += value
    return sum
    

def get_G3(df, i, neighbours, Rc=0.004, alpha_c=100, epsilon_c=0, kappa=2):
    sum = 0
    for index in neighbours:
        if index != i:
            R = calculate_distance_3D(df, i, index)
            value = np.cos(kappa * R) * fc(R, Rc, alpha_c, epsilon_c)
            sum += value
    return sum

def get_G1_2_3(df, i, neighbours, Rc=0.004, alpha_c=100, epsilon_c=0, eta=3, Rs=0.003, kappa=2):
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for index in neighbours:
        if index != i:
            R = calculate_distance_3D(df, i, index)
            value_1 = fc(R, Rc, alpha_c, epsilon_c)
            value_2 = np.exp(-eta * (R - Rs) ** 2) * fc(R, Rc, alpha_c, epsilon_c)
            value_3 = np.cos(kappa * R) * fc(R, Rc, alpha_c, epsilon_c)
            sum_1 += value_1
            sum_2 += value_2
            sum_3 += value_3
    return sum_1, sum_2, sum_3


def get_G4(df, i, neighbours, Rc=0.004, alpha_c=100, epsilon_c=0, eta=3, lam=1, zeta=1):
    sum = 0
    for index2 in neighbours:
        for index3 in neighbours:
            if i != index2 and i != index3 and index2 != index3:
                theta = calculate_angle_3D(df, i, index2, index3)
                R12 = calculate_distance_3D(df, i, index2)
                R13 = calculate_distance_3D(df, i, index3)
                R23 = calculate_distance_3D(df, index2, index3)

                value = ((1 + lam * np.cos(theta)) ** zeta) * \
                    np.exp(-eta * (R12**2 + R13**2 + R23**2)) * \
                        fc(R12, Rc, alpha_c, epsilon_c) * \
                            fc(R13, Rc, alpha_c, epsilon_c) * \
                                fc(R23, Rc, alpha_c, epsilon_c)
                
                sum += value
    return (1 / (2 ** zeta)) * sum



def get_G5(df, i, neighbours, Rc=0.004, alpha_c=100, epsilon_c=0, eta=3, lam=1, zeta=1):
    sum = 0
    for index2 in neighbours:
        for index3 in neighbours:
            if i != index2 and i != index3 and index2 != index3:
                theta = calculate_angle_3D(df, i, index2, index3)
                R12 = calculate_distance_3D(df, i, index2)
                R13 = calculate_distance_3D(df, i, index3)

                value = ((1 + lam * np.cos(theta)) ** zeta) * \
                    np.exp(-eta * (R12**2 + R13**2)) * \
                        fc(R12, Rc, alpha_c, epsilon_c) * \
                            fc(R13, Rc, alpha_c, epsilon_c)
                
                sum += value
    return (1 / (2 ** zeta)) * sum


def get_G6(df, i, neighbours, eta=3, mu=0.003, lam=1, zeta=1):
    sum = 0
    for index2 in neighbours:
        for index3 in neighbours:
            if i != index2 and i != index3 and index2 != index3:
                theta = calculate_angle_3D(df, i, index2, index3)
                R12 = calculate_distance_3D(df, i, index2)
                R13 = calculate_distance_3D(df, i, index3)

                value = ((1 + lam * np.cos(theta)) ** zeta) * \
                    fa(R12, eta, mu) * fa(R13, eta, mu)
                
                sum += value
    return (1 / (2 ** zeta)) * sum



def get_G7(df, i, neighbours, Rc=0.004, alpha_c=100, epsilon_c=0, eta=3, alpha=50):
    sum = 0
    for index2 in neighbours:
        for index3 in neighbours:
            if i != index2 and i != index3 and index2 != index3:
                theta = calculate_angle_3D(df, i, index2, index3)
                R12 = calculate_distance_3D(df, i, index2)
                R13 = calculate_distance_3D(df, i, index3)

                value = np.sin(eta*(theta - alpha)) * \
                    fc(R12, Rc, alpha_c, epsilon_c) * \
                        fc(R13, Rc, alpha_c, epsilon_c)

                sum += value
    return sum / 2


def get_G8(df, i, neighbours, nu=1, a1=0.003, ar=0.003, eta=3, alpha=50):
    sum = 0
    for index2 in neighbours:
        for index3 in neighbours:
            if i != index2 and i != index3 and index2 != index3:
                theta = calculate_angle_3D(df, i, index2, index3)
                R12 = calculate_distance_3D(df, i, index2)
                R13 = calculate_distance_3D(df, i, index3)

                value = np.sin(eta*(theta - alpha)) * \
                    fb(R12, nu, a1, ar) * \
                        fb(R13, nu, a1, ar)

                sum += value
    return sum / 2

def get_G4_5_6_7_8(df, i, neighbours, Rc=0.004, alpha_c=100, epsilon_c=0, eta=3, lam=1, zeta=1, mu=0.003, alpha=50, nu=1, a1=0.003, ar=0.003):
    sum_4 = 0
    sum_5 = 0
    sum_6 = 0
    sum_7 = 0
    sum_8 = 0
    for index2 in neighbours:
        for index3 in neighbours:
            if i != index2 and i != index3 and index2 != index3:
                theta = calculate_angle_3D(df, i, index2, index3)
                R12 = calculate_distance_3D(df, i, index2)
                R13 = calculate_distance_3D(df, i, index3)
                R23 = calculate_distance_3D(df, index2, index3)

                value_4 = ((1 + lam * np.cos(theta)) ** zeta) * \
                    np.exp(-eta * (R12**2 + R13**2 + R23**2)) * \
                        fc(R12, Rc, alpha_c, epsilon_c) * \
                            fc(R13, Rc, alpha_c, epsilon_c) * \
                                fc(R23, Rc, alpha_c, epsilon_c)

                value_5 = ((1 + lam * np.cos(theta)) ** zeta) * \
                    np.exp(-eta * (R12**2 + R13**2)) * \
                        fc(R12, Rc, alpha_c, epsilon_c) * \
                            fc(R13, Rc, alpha_c, epsilon_c)

                value_6 = ((1 + lam * np.cos(theta)) ** zeta) * \
                    fa(R12, eta, mu) * fa(R13, eta, mu)

                value_7 = np.sin(eta*(theta - alpha)) * \
                    fc(R12, Rc, alpha_c, epsilon_c) * \
                        fc(R13, Rc, alpha_c, epsilon_c)
                        
                value_8 = np.sin(eta*(theta - alpha)) * \
                    fb(R12, nu, a1, ar) * \
                        fb(R13, nu, a1, ar)
                
                sum_4 += value_4
                sum_5 += value_5
                sum_6 += value_6
                sum_7 += value_7
                sum_8 += value_8

                sum_4 *= (1 / (2 ** zeta))
                sum_5 *= (1 / (2 ** zeta))
                sum_6 *= (1 / (2 ** zeta))
                sum_7 *= (1 / 2)
                sum_8 *= (1 / 2)

    return sum_4, sum_5, sum_6, sum_7, sum_8

#Parameter of local bon orientational parameter
def get_vector3D(df, index1, index2):
    a = df.iloc[index2][['x', 'y', 'z']]
    b = df.iloc[index1][['x', 'y', 'z']]
    return b-a

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def get_qlm(df, i, neighbours, l, m):
    sum = 0
    Nb = len(neighbours)
    for index in neighbours:
        x, y, z = get_vector3D(df = df, index1 = i, index2 = index)
        az, el, r = cart2sph(x, y, z)
        harm = scipy.special.sph_harm(m, l, el, az)
        sum += harm
    return 1/Nb * sum

def get_list_m(max):
    list_m = []
    for m1 in range(-max,max+1):
        for m2 in range(-max,max+1):
            for m3 in range(-max,max+1):
                if m1 + m2 + m3 == 0:
                    list_m.append([m1, m2, m3])
    return list_m

#local bon orientational parameter q4 and q6
def get_ql(df, i, neighbours, l):
    sum = 0
    for m in range(-l, l+1):
        qlm = get_qlm(df, i, neighbours, l, m)
        sum += np.absolute(qlm)**2
    return (4*np.pi / (2*l + 1) * sum) ** (1/2) 


#local bon orientational parameter w4 and w6
def get_wl(df, i, neighbours, l, max):
    denom = 0
    sum = 0
    for m in range(-l, l+1):
        qlm = get_qlm(df, i, neighbours, l, m)
        denom += np.absolute(qlm)**2
    denom = denom ** (3/2)
    list_m = get_list_m(max)
    for m1, m2, m3 in list_m:
        wigner = wigner_3j(l, l, l, m1, m2, m3)
        Qlm1 = get_qlm(df, i, neighbours, l, m1)
        Qlm2 = get_qlm(df, i, neighbours, l, m2)
        Qlm3 = get_qlm(df, i, neighbours, l, m3)
        sum += wigner * Qlm1 * Qlm2 * Qlm3
    return complex(sum / denom)

#Parameter for the overlap force
def get_E_star(E1 = 50E9, E2 = 50E9, nu1 = 0.3, nu2 = 0.3):
    return ((1 - nu1**2) / E1 + (1 - nu2**2) / E2) ** (-1)

#get the overlap force with hertz law
def force(df, index1, index2, E1 = 50E9, E2 = 50E9, nu1 = 0.3, nu2 = 0.3):
    E_star = get_E_star(E1, E2, nu1, nu2)
    dist = calculate_distance_3D(df, index1, index2)
    R1 = df.iloc[index2]['r']
    R2 = df.iloc[index1]['r']
    delta = dist - R1 - R2
    if delta < 0:
        R = R1 * R2 / (R1 + R2)
        return 4/3 * E_star * R ** (1/2) * np.abs(delta) ** (3/2)
    return 0
    
            
#get the centroymmetry paramter
def centrosymmetry(df, index, neighbours):
    list_bonds = []
    Nb = len(neighbours)
    sum = 0
    for i in neighbours:
        for j in neighbours:
            if i != j:
                a = get_vector3D(df, index, i)
                b = get_vector3D(df, index, j)
                value = np.linalg.norm(a + b) ** 2
                list_bonds.append(value)
    list_bonds.sort()
    if Nb != 0:
        for i in range(Nb // 2):
            sum += list_bonds[i]
    return sum


# extansion of the sample to allow periodic conditions
def extend_periodic(df, xyz, dc):
    ext_df = df.copy() 

    x_low = xyz['x_low']
    x_high = xyz['x_high']
    y_low = xyz['y_low']
    y_high = xyz['y_high']
    z_low = xyz['z_low']
    z_high = xyz['z_high']

    x_d = x_high - x_low
    y_d = y_high - y_low
    z_d = z_high - z_low

    core = [1 for l in range(len(ext_df))]
    ext_df['core'] = core

    for i in range(3):
        for j in range(3):
            for k in range(3):
                part_df = df.copy()
                if i == 1:
                    part_df.x -= x_d
                if i == 2:
                    part_df.x += x_d
                if j == 1:
                    part_df.y -= y_d
                if j == 2:
                    part_df.y += y_d
                if k == 1:
                    part_df.z -= z_d
                if k == 2:
                    part_df.z += z_d

                outer = [0 for l in range(len(part_df))]
                part_df['core'] = outer
                if (i, j, k) != (0, 0, 0):
                    ext_df = pd.concat([ext_df, part_df], axis=0, ignore_index=True)

    list_cut = []
    for index, row in ext_df.iterrows():
        if row.x < x_low - dc or row.x > x_high + dc or \
            row.y < y_low - dc or row.y > y_high + dc or \
                row.z < z_low - dc or row.z > z_high + dc:
                list_cut.append(index)
    ext_df = ext_df.drop(list_cut).reset_index()

    return ext_df