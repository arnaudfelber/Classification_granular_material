
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Extract xyz
def extract_data_3D(path):

    # Entrée : chemin vers le fichier .data
    # Sortie : dataf, xyz
    # où
    # dataf = DataFrame contenant les données des atomes | Numéro, position en x, position en y, rayon.
    # xyz = dictionnaire contenant les caractéristiques de la boîte | Bornes en x, bornes en y, bornes en z.

    d = pd.read_csv(path)
    data = d['LAMMPS data file via write_data']

    number_of_atoms = int(data[0][:4])

    x_low = float(data[2].split(" ")[0])
    x_high = float(data[2].split(" ")[1])

    y_low = float(data[3].split(" ")[0])
    y_high = float(data[3].split(" ")[1])

    z_low = float(data[4].split(" ")[0])
    z_high = float(data[4].split(" ")[1])

    xyz = {'n_atoms' : number_of_atoms, 'x_low' : x_low, 'x_high' : x_high, 'y_low' : y_low, 'y_high' : y_high, 'z_low' : z_low, 'z_high' : z_high}

    dict_atoms = {}

    for i in range(number_of_atoms):
        atom_data = data[6+i].split(" ") # Voir le format des données. Les données concernant les atomes individuels commencent à la ligne 6.
        number = atom_data[0]
        radius = atom_data[2]

        x_pos = atom_data[4]
        y_pos = atom_data[5]
        z_pos = float(atom_data[6])

        dict_atoms[number] = [x_pos, y_pos,z_pos, radius, number]

    dataf = pd.DataFrame(dict_atoms).transpose().set_axis(['x','y','z','r','n'], axis=1)

    return dataf, xyz


#Extract extanded set from the pickle
def extract_data_pickles(path):

    # Entrée : chemin vers le fichier .data
    # Sortie : dataf, xyz
    # où
    # dataf = DataFrame contenant les données des atomes | Numéro, position en x, position en y, rayon.
    # xyz = dictionnaire contenant les caractéristiques de la boîte | Bornes en x, bornes en y, bornes en z.

    d = pd.read_pickle(path)
    data = d['LAMMPS data file via write_data']

    number_of_atoms = int(data[0][:4])

    x_low = float(data[2].split(" ")[0])
    x_high = float(data[2].split(" ")[1])

    y_low = float(data[3].split(" ")[0])
    y_high = float(data[3].split(" ")[1])

    z_low = float(data[4].split(" ")[0])
    z_high = float(data[4].split(" ")[1])

    xyz = {'n_atoms' : number_of_atoms, 'x_low' : x_low, 'x_high' : x_high, 'y_low' : y_low, 'y_high' : y_high, 'z_low' : z_low, 'z_high' : z_high}

    dict_atoms = {}

    for i in range(number_of_atoms):
        atom_data = data[6+i].split(" ") # Voir le format des données. Les données concernant les atomes individuels commencent à la ligne 6.
        number = atom_data[0]
        radius = atom_data[2]

        x_pos = atom_data[4]
        y_pos = atom_data[5]
        z_pos = float(atom_data[6])

        dict_atoms[number] = [x_pos, y_pos,z_pos, radius, number]

    dataf = pd.DataFrame(dict_atoms).transpose().set_axis(['x','y','z','r','n'], axis=1)

    return dataf, xyz



#Extract data  3D    
def extract_data_list(path):

    # Entrée : chemin vers le fichier .data
    # Sortie : dataf, xyz
    # où
    # dataf = DataFrame contenant les données des atomes | Numéro, position en x, position en y, rayon.
    # xyz = dictionnaire contenant les caractéristiques de la boîte | Bornes en x, bornes en y, bornes en z.

    d = pd.read_csv(path)
    data = d['LAMMPS data file via write_data']

    number_of_atoms = int(data[0][:4])

    x_low = float(data[2].split(" ")[0])
    x_high = float(data[2].split(" ")[1])

    y_low = float(data[3].split(" ")[0])
    y_high = float(data[3].split(" ")[1])

    z_low = float(data[4].split(" ")[0])
    z_high = float(data[4].split(" ")[1])

    xyz = {'n_atoms' : number_of_atoms, 'x_low' : x_low, 'x_high' : x_high, 'y_low' : y_low, 'y_high' : y_high, 'z_low' : z_low, 'z_high' : z_high}

    list_atoms_radii = []

    for i in range(number_of_atoms):
        atom_data = data[6+i].split(" ") # Voir le format des données. Les données concernant les atomes individuels commencent à la ligne 6.
        number = atom_data[0]
        radius = float(atom_data[2])

        x_pos = float(atom_data[4])
        y_pos = float(atom_data[5])
        #z_pos = 0
        z_pos = float(atom_data[6])

        list_atoms_radii.append(x_pos)
        list_atoms_radii.append(y_pos)
        list_atoms_radii.append(z_pos)
        list_atoms_radii.append(radius)

    return list_atoms_radii, xyz

    # dataf = pd.DataFrame(dict_atoms).transpose().set_axis(['x', 'y', 'r'], axis=1)


#Transform in list of float
def get_list(train_set):
    for i in range (len(train_set)):
        train_set = np.array(train_set)
        for j in range (len(train_set[0])):
            if (j+1 == len(train_set[0])):
                train_set [i,j]= int(train_set [i,j])
            else:
                train_set [i,j]= float(train_set [i,j])
    return train_set
#same for the pickle extraction
def get_list_pickle(train_set):
    for i in range (len(train_set)):
        train_set = np.array(train_set)
        for j in range (len(train_set[0])):
            if (j+1 == len(train_set[0])):
                train_set [i,j]= train_set [i,j]
            else:
                train_set [i,j]= train_set [i,j]
    return train_set

#Adapt to the format required for the DB scan
def get_DBscan_data_3D(Data,n_atom_max,no_path):
    n=0
    DB_test = {}
    for i in range (len(Data)):
        if (i == no_path[n]):
            n = n + 1
            pass
        else:
            line = Data["seed_{}".format(i)]
            sample = np.zeros(4*n_atom_max)
            for l in range (len(line)):
                for c in range (4):
                    sample[4*l+c] = line[l,c]       
            DB_test[i] = sample
            DB_frame = pd.DataFrame(DB_test).transpose()
            DB_frame = get_list(DB_frame)
    return DB_frame

#flatten the pickle
def get_flatten_pickle(train_set):
    DB = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        for j in range (len(train_set[0])):
            li = train_set[i,j]
            if (type(li)==list):
                for z in range (len(li)):
                    r = li[z].real
                    DB = np.append(DB, r)
            else:
                r = li.real
                DB = np.append(DB, r)
            
    return DB

#transform the pickle in DB and k-means format
def get_DB_pickle(n):
    trainset = {}
    for i in range (n):
        if(i==183):
            pass
        else:
            train = pd.read_pickle('C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\Données\\sample_ML\\sample_ML\\extracted_pickles\\input_data_{}.pkl'.format(i))
            train = get_flatten_pickle(train)    
            for j in range (30100-len(train)):
                if (len(train) != 30100):
                    train = np.append(train,0)
                else:
                    pass
            trainset[i] = train
            
    DB_frame = pd.DataFrame(trainset).transpose()
    DB = get_list_pickle(DB_frame)
    return DB


#Find the max length of a sample
def max_length():
    max_len = 0
    for i in range (50):
        DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8 = cut_sample('C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\3D\\sample_students_ML\\box_0.036000000000000004\\seed_1\\pressure_5.0\\confined_5.0_MPa.data','C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\Données\\sample_ML\\sample_ML\\extracted_pickles\\input_data_{}.pkl'.format(i))
        if (len(DB1)>max_len):
            max_len = len(DB1)
        if (len(DB2)>max_len):
            max_len = len(DB2)
        if (len(DB3)>max_len):
            max_len = len(DB3)
        if (len(DB4)>max_len):
            max_len = len(DB4)
        if (len(DB5)>max_len):
            max_len = len(DB5)
        if (len(DB6)>max_len):
            max_len = len(DB6)
        if (len(DB7)>max_len):
            max_len = len(DB7)
        if (len(DB8)>max_len):
            max_len = len(DB8)
    return (max_len)

#Cut the sample to a medium size 6d0 ( in 8 parts)
def cut(path):
    seed, info = extract_data_3D(path)
    seed = get_list(seed)
    B1=[]
    B2=[]
    B3=[]
    B4=[]
    B5=[]
    B6=[]
    B7=[]
    B8=[]
    for i in range(len(seed)):
        if (seed[i,0]<=0 and seed[i,1]<=0 and seed[i,2]<=0):
            B1 = np.append(B1,i)
        elif (seed[i,0]>=0 and seed[i,1]<=0 and seed[i,2]<=0):
            B2 = np.append(B2,i)
        elif (seed[i,0]<=0 and seed[i,1]>=0 and seed[i,2]<=0):
            B3 = np.append(B3,i)
        elif (seed[i,0]>=0 and seed[i,1]>=0 and seed[i,2]<=0):
            B4 = np.append(B4,i)
        elif (seed[i,0]<=0 and seed[i,1]<=0 and seed[i,2]>=0):
            B5 = np.append(B5,i)
        elif (seed[i,0]>=0 and seed[i,1]<=0 and seed[i,2]>=0):
            B6 = np.append(B6,i)
        elif (seed[i,0]<=0 and seed[i,1]>=0 and seed[i,2]>=0):
            B7 = np.append(B7,i)
        elif (seed[i,0]>=0 and seed[i,1]>=0 and seed[i,2]>=0):
            B8 = np.append(B8,i)
    return (B1,B2,B3,B4,B5,B6,B7,B8)        

#Cut the sample to a medium size 6d0 ( in 8 parts)
def cut_sample(path1,path2):
    B1,B2,B3,B4,B5,B6,B7,B8=cut(path1)
    train_set = pd.read_pickle(path2)
    
    DB1 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B1):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB1 = np.append(DB1, r)
                else:
                    r = li.real
                    DB1 = np.append(DB1, r)
            else:
                pass
    DB2 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B2):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB2 = np.append(DB2, r)
                else:
                    r = li.real
                    DB2 = np.append(DB2, r)
            else:
                pass
    DB3 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B3):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB3 = np.append(DB3, r)
                else:
                    r = li.real
                    DB3 = np.append(DB3, r)
            else:
                pass
    DB4 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B4):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB4 = np.append(DB4, r)
                else:
                    r = li.real
                    DB4 = np.append(DB4, r)
            else:
                pass
            
    DB5 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B5):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB5 = np.append(DB5, r)
                else:
                    r = li.real
                    DB5 = np.append(DB5, r)
            else:
                pass
    DB6 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B6):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB6 = np.append(DB6, r)
                else:
                    r = li.real
                    DB6 = np.append(DB6, r)
            else:
                pass
    DB7 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B7):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB7 = np.append(DB7, r)
                else:
                    r = li.real
                    DB7 = np.append(DB7, r)
            else:
                pass
            
    DB8 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B8):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB8 = np.append(DB8, r)
                else:
                    r = li.real
                    DB8 = np.append(DB8, r)
            else:
                pass
    return DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8

# Transform medium sample to DB and K-means format
def DB_cut(n):
    DB_tot = {}
    for i in range (n):
        DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8 = cut_sample('C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\3D\\sample_students_ML\\box_0.036000000000000004\\seed_{}\\pressure_5.0\\confined_5.0_MPa.data'.format(i),'C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\Données\\sample_ML\\sample_ML\\extracted_pickles\\input_data_{}.pkl'.format(i))    
        while (len(DB1) != 4200):
                DB1 = np.append(DB1,0)
        DB_tot[i] = DB1
        
        while (len(DB2) != 4200):
            DB2 = np.append(DB2,0)
        DB_tot[n+i] = DB2
        
        while (len(DB3) != 4200):
            DB3 = np.append(DB3,0)
        DB_tot[2*n+i] = DB3
        
        while (len(DB4) != 4200):
            DB4 = np.append(DB4,0)
        DB_tot[3*n+i] = DB4
        
        while (len(DB5) != 4200):
            DB5 = np.append(DB5,0)
        DB_tot[4*n+i] = DB5
        
        while (len(DB6) != 4200):
            DB6 = np.append(DB6,0)
        DB_tot[5*n+i] = DB6
        
        while (len(DB7) != 4200):
            DB7 = np.append(DB7,0)
        DB_tot[6*n+i] = DB7
        
        while (len(DB8) != 4200):
            DB8 = np.append(DB8,0)
        DB_tot[7*n+i] = DB8
        DB_tot = pd.DataFrame(DB_tot)
    return DB_tot

#Cut in small samples 3d0 (8x8 = 64), There are 8 functions
def cut_mini_1(path, lim, lim1):
    seed, info = extract_data_3D(path)
    seed = get_list(seed)
    B1=[]
    B2=[]
    B3=[]
    B4=[]
    B5=[]
    B6=[]
    B7=[]
    B8=[]
    
    for i in range(len(seed)):
        if(seed[i,0]<=lim1 and seed[i,1]<=lim1 and seed[i,2]<=lim1):
            if (seed[i,0]<= -lim and seed[i,1]<= -lim and seed[i,2]<=lim):
                B1 = np.append(B1,i)
            elif (seed[i,0]>= -lim and seed[i,1]<= -lim and seed[i,2]<= -lim):
                B2 = np.append(B2,i)
            elif (seed[i,0]<= -lim and seed[i,1]>= -lim and seed[i,2]<= -lim):
                B3 = np.append(B3,i)
            elif (seed[i,0]>= -lim and seed[i,1]>= -lim and seed[i,2]<= -lim):
                B4 = np.append(B4,i)
            elif (seed[i,0]<= -lim and seed[i,1]<= -lim and seed[i,2]>= -lim):
                B5 = np.append(B5,i)
            elif (seed[i,0]>= -lim and seed[i,1]<= -lim and seed[i,2]>= -lim):
                B6 = np.append(B6,i)
            elif (seed[i,0]<= -lim and seed[i,1]>= -lim and seed[i,2]>= -lim):
                B7 = np.append(B7,i)
            elif (seed[i,0]>= -lim and seed[i,1]>= - -lim and seed[i,2]>= -lim):
                B8 = np.append(B8,i)
        else:
            pass
    return (B1,B2,B3,B4,B5,B6,B7,B8) 

def cut_mini_2(path, lim, lim1):
    seed, info = extract_data_3D(path)
    seed = get_list(seed)
    B1=[]
    B2=[]
    B3=[]
    B4=[]
    B5=[]
    B6=[]
    B7=[]
    B8=[]
    
    for i in range(len(seed)):
        if(seed[i,0]>=lim1 and seed[i,1]<=lim1 and seed[i,2]<=lim1):
            if (seed[i,0]<=lim and seed[i,1]<= -lim and seed[i,2]<= -lim):
                B1 = np.append(B1,i)
            elif (seed[i,0]>=lim and seed[i,1]<= -lim and seed[i,2]<= -lim):
                B2 = np.append(B2,i)
            elif (seed[i,0]<=lim and seed[i,1]>= -lim and seed[i,2]<= -lim):
                B3 = np.append(B3,i)
            elif (seed[i,0]>=lim and seed[i,1]>= -lim and seed[i,2]<= -lim):
                B4 = np.append(B4,i)
            elif (seed[i,0]<=lim and seed[i,1]<= -lim and seed[i,2]>= -lim):
                B5 = np.append(B5,i)
            elif (seed[i,0]>=lim and seed[i,1]<= -lim and seed[i,2]>= -lim):
                B6 = np.append(B6,i)
            elif (seed[i,0]<=lim and seed[i,1]>= -lim and seed[i,2]>= -lim):
                B7 = np.append(B7,i)
            elif (seed[i,0]>=lim and seed[i,1]>= -lim and seed[i,2]>= -lim):
                B8 = np.append(B8,i)
        else:
            pass
    return (B1,B2,B3,B4,B5,B6,B7,B8) 

def cut_mini_3(path, lim, lim1):
    seed, info = extract_data_3D(path)
    seed = get_list(seed)
    B1=[]
    B2=[]
    B3=[]
    B4=[]
    B5=[]
    B6=[]
    B7=[]
    B8=[]
    
    for i in range(len(seed)):
        if(seed[i,0]<=lim1 and seed[i,1]>=lim1 and seed[i,2]<=lim1):
            if (seed[i,0]<= -lim and seed[i,1]<=lim and seed[i,2]<= -lim):
                B1 = np.append(B1,i)
            elif (seed[i,0]>= -lim and seed[i,1]<=lim and seed[i,2]<= -lim):
                B2 = np.append(B2,i)
            elif (seed[i,0]<= -lim and seed[i,1]>=lim and seed[i,2]<= -lim):
                B3 = np.append(B3,i)
            elif (seed[i,0]>= -lim and seed[i,1]>=lim and seed[i,2]<= -lim):
                B4 = np.append(B4,i)
            elif (seed[i,0]<= -lim and seed[i,1]<=lim and seed[i,2]>= -lim):
                B5 = np.append(B5,i)
            elif (seed[i,0]>= -lim and seed[i,1]<=lim and seed[i,2]>= -lim):
                B6 = np.append(B6,i)
            elif (seed[i,0]<= -lim and seed[i,1]>=lim and seed[i,2]>= -lim):
                B7 = np.append(B7,i)
            elif (seed[i,0]>= -lim and seed[i,1]>=lim and seed[i,2]>= -lim):
                B8 = np.append(B8,i)
        else:
            pass
    return (B1,B2,B3,B4,B5,B6,B7,B8) 

def cut_mini_4(path, lim, lim1):
    seed, info = extract_data_3D(path)
    seed = get_list(seed)
    B1=[]
    B2=[]
    B3=[]
    B4=[]
    B5=[]
    B6=[]
    B7=[]
    B8=[]
    
    for i in range(len(seed)):
        if(seed[i,0]>=lim1 and seed[i,1]>=lim1 and seed[i,2]<=lim1):
            if (seed[i,0]<=lim and seed[i,1]<=lim and seed[i,2]<= -lim):
                B1 = np.append(B1,i)
            elif (seed[i,0]>=lim and seed[i,1]<=lim and seed[i,2]<= -lim):
                B2 = np.append(B2,i)
            elif (seed[i,0]<=lim and seed[i,1]>=lim and seed[i,2]<= -lim):
                B3 = np.append(B3,i)
            elif (seed[i,0]>=lim and seed[i,1]>=lim and seed[i,2]<= -lim):
                B4 = np.append(B4,i)
            elif (seed[i,0]<=lim and seed[i,1]<=lim and seed[i,2]>= -lim):
                B5 = np.append(B5,i)
            elif (seed[i,0]>=lim and seed[i,1]<=lim and seed[i,2]>= -lim):
                B6 = np.append(B6,i)
            elif (seed[i,0]<=lim and seed[i,1]>=lim and seed[i,2]>= -lim):
                B7 = np.append(B7,i)
            elif (seed[i,0]>=lim and seed[i,1]>=lim and seed[i,2]>= -lim):
                B8 = np.append(B8,i)
        else:
            pass
    return (B1,B2,B3,B4,B5,B6,B7,B8) 

def cut_mini_5(path, lim, lim1):
    seed, info = extract_data_3D(path)
    seed = get_list(seed)
    B1=[]
    B2=[]
    B3=[]
    B4=[]
    B5=[]
    B6=[]
    B7=[]
    B8=[]
    
    for i in range(len(seed)):
        if(seed[i,0]<=lim1 and seed[i,1]<=lim1 and seed[i,2]>=lim1):
            if (seed[i,0]<= -lim and seed[i,1]<= -lim and seed[i,2]>=lim):
                B1 = np.append(B1,i)
            elif (seed[i,0]>= -lim and seed[i,1]<= -lim and seed[i,2]<=lim):
                B2 = np.append(B2,i)
            elif (seed[i,0]<= -lim and seed[i,1]>= -lim and seed[i,2]<=lim):
                B3 = np.append(B3,i)
            elif (seed[i,0]>= -lim and seed[i,1]>= -lim and seed[i,2]<=lim):
                B4 = np.append(B4,i)
            elif (seed[i,0]<= -lim and seed[i,1]<= -lim and seed[i,2]>=lim):
                B5 = np.append(B5,i)
            elif (seed[i,0]>= -lim and seed[i,1]<= -lim and seed[i,2]>=lim):
                B6 = np.append(B6,i)
            elif (seed[i,0]<= -lim and seed[i,1]>= -lim and seed[i,2]>=lim):
                B7 = np.append(B7,i)
            elif (seed[i,0]>= -lim and seed[i,1]>= -lim and seed[i,2]>=lim):
                B8 = np.append(B8,i)
        else:
            pass
    return (B1,B2,B3,B4,B5,B6,B7,B8) 

def cut_mini_6(path, lim, lim1):
    seed, info = extract_data_3D(path)
    seed = get_list(seed)
    B1=[]
    B2=[]
    B3=[]
    B4=[]
    B5=[]
    B6=[]
    B7=[]
    B8=[]
    
    for i in range(len(seed)):
        if(seed[i,0]>=lim1 and seed[i,1]<=lim1 and seed[i,2]>=lim1):
            if (seed[i,0]<=lim and seed[i,1]<= -lim and seed[i,2]<=lim):
                B1 = np.append(B1,i)
            elif (seed[i,0]>=lim and seed[i,1]<= -lim and seed[i,2]<=lim):
                B2 = np.append(B2,i)
            elif (seed[i,0]<=lim and seed[i,1]>= -lim and seed[i,2]<=lim):
                B3 = np.append(B3,i)
            elif (seed[i,0]>=lim and seed[i,1]>= -lim and seed[i,2]<=lim):
                B4 = np.append(B4,i)
            elif (seed[i,0]<=lim and seed[i,1]<= -lim and seed[i,2]>=lim):
                B5 = np.append(B5,i)
            elif (seed[i,0]>=lim and seed[i,1]<= -lim and seed[i,2]>=lim):
                B6 = np.append(B6,i)
            elif (seed[i,0]<=lim and seed[i,1]>= -lim and seed[i,2]>=lim):
                B7 = np.append(B7,i)
            elif (seed[i,0]>=lim and seed[i,1]>= -lim and seed[i,2]>=lim):
                B8 = np.append(B8,i)
        else:
            pass
    return (B1,B2,B3,B4,B5,B6,B7,B8) 

def cut_mini_7(path, lim, lim1):
    seed, info = extract_data_3D(path)
    seed = get_list(seed)
    B1=[]
    B2=[]
    B3=[]
    B4=[]
    B5=[]
    B6=[]
    B7=[]
    B8=[]
    
    for i in range(len(seed)):
        if(seed[i,0]<=lim1 and seed[i,1]>=lim1 and seed[i,2]>=lim1):
            if (seed[i,0]<= -lim and seed[i,1]<=lim and seed[i,2]<=lim):
                B1 = np.append(B1,i)
            elif (seed[i,0]>= -lim and seed[i,1]<=lim and seed[i,2]<=lim):
                B2 = np.append(B2,i)
            elif (seed[i,0]<= -lim and seed[i,1]>=lim and seed[i,2]<=lim):
                B3 = np.append(B3,i)
            elif (seed[i,0]>= -lim and seed[i,1]>=lim and seed[i,2]<=lim):
                B4 = np.append(B4,i)
            elif (seed[i,0]<= -lim and seed[i,1]<=lim and seed[i,2]>=lim):
                B5 = np.append(B5,i)
            elif (seed[i,0]>= -lim and seed[i,1]<=lim and seed[i,2]>=lim):
                B6 = np.append(B6,i)
            elif (seed[i,0]<= -lim and seed[i,1]>=lim and seed[i,2]>=lim):
                B7 = np.append(B7,i)
            elif (seed[i,0]>= -lim and seed[i,1]>=lim and seed[i,2]>=lim):
                B8 = np.append(B8,i)
        else:
            pass
    return (B1,B2,B3,B4,B5,B6,B7,B8) 

def cut_mini_8(path, lim, lim1):
    seed, info = extract_data_3D(path)
    seed = get_list(seed)
    B1=[]
    B2=[]
    B3=[]
    B4=[]
    B5=[]
    B6=[]
    B7=[]
    B8=[]
    
    for i in range(len(seed)):
        if(seed[i,0]>=lim1 and seed[i,1]>=lim1 and seed[i,2]>=lim1):
            if (seed[i,0]<=lim and seed[i,1]<=lim and seed[i,2]<=lim):
                B1 = np.append(B1,i)
            elif (seed[i,0]>=lim and seed[i,1]<=lim and seed[i,2]<=lim):
                B2 = np.append(B2,i)
            elif (seed[i,0]<=lim and seed[i,1]>=lim and seed[i,2]<=lim):
                B3 = np.append(B3,i)
            elif (seed[i,0]>=lim and seed[i,1]>=lim and seed[i,2]<=lim):
                B4 = np.append(B4,i)
            elif (seed[i,0]<=lim and seed[i,1]<=lim and seed[i,2]>=lim):
                B5 = np.append(B5,i)
            elif (seed[i,0]>=lim and seed[i,1]<=lim and seed[i,2]>=lim):
                B6 = np.append(B6,i)
            elif (seed[i,0]<=lim and seed[i,1]>=lim and seed[i,2]>=lim):
                B7 = np.append(B7,i)
            elif (seed[i,0]>=lim and seed[i,1]>=lim and seed[i,2]>=lim):
                B8 = np.append(B8,i)
        else:
            pass
    return (B1,B2,B3,B4,B5,B6,B7,B8) 


#Cut in small samples 3d0 (8x8 = 64), There are 8 functions
def cut_mini_sample_1(path1,path2,lim,lim1):
    B1,B2,B3,B4,B5,B6,B7,B8=cut_mini_1(path1,lim,lim1)
    train_set = pd.read_pickle(path2)
    
    DB1 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B1):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB1 = np.append(DB1, r)
                else:
                    r = li.real
                    DB1 = np.append(DB1, r)
            else:
                pass
    DB2 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B2):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB2 = np.append(DB2, r)
                else:
                    r = li.real
                    DB2 = np.append(DB2, r)
            else:
                pass
    DB3 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B3):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB3 = np.append(DB3, r)
                else:
                    r = li.real
                    DB3 = np.append(DB3, r)
            else:
                pass
    DB4 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B4):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB4 = np.append(DB4, r)
                else:
                    r = li.real
                    DB4 = np.append(DB4, r)
            else:
                pass
            
    DB5 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B5):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB5 = np.append(DB5, r)
                else:
                    r = li.real
                    DB5 = np.append(DB5, r)
            else:
                pass
    DB6 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B6):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB6 = np.append(DB6, r)
                else:
                    r = li.real
                    DB6 = np.append(DB6, r)
            else:
                pass
    DB7 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B7):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB7 = np.append(DB7, r)
                else:
                    r = li.real
                    DB7 = np.append(DB7, r)
            else:
                pass
            
    DB8 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B8):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB8 = np.append(DB8, r)
                else:
                    r = li.real
                    DB8 = np.append(DB8, r)
            else:
                pass
    return DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8

def cut_mini_sample_2(path1,path2,lim,lim1):
    B1,B2,B3,B4,B5,B6,B7,B8=cut_mini_2(path1,lim,lim1)
    train_set = pd.read_pickle(path2)
    
    DB1 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B1):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB1 = np.append(DB1, r)
                else:
                    r = li.real
                    DB1 = np.append(DB1, r)
            else:
                pass
    DB2 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B2):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB2 = np.append(DB2, r)
                else:
                    r = li.real
                    DB2 = np.append(DB2, r)
            else:
                pass
    DB3 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B3):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB3 = np.append(DB3, r)
                else:
                    r = li.real
                    DB3 = np.append(DB3, r)
            else:
                pass
    DB4 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B4):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB4 = np.append(DB4, r)
                else:
                    r = li.real
                    DB4 = np.append(DB4, r)
            else:
                pass
            
    DB5 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B5):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB5 = np.append(DB5, r)
                else:
                    r = li.real
                    DB5 = np.append(DB5, r)
            else:
                pass
    DB6 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B6):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB6 = np.append(DB6, r)
                else:
                    r = li.real
                    DB6 = np.append(DB6, r)
            else:
                pass
    DB7 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B7):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB7 = np.append(DB7, r)
                else:
                    r = li.real
                    DB7 = np.append(DB7, r)
            else:
                pass
            
    DB8 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B8):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB8 = np.append(DB8, r)
                else:
                    r = li.real
                    DB8 = np.append(DB8, r)
            else:
                pass
    return DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8

def cut_mini_sample_3(path1,path2,lim,lim1):
    B1,B2,B3,B4,B5,B6,B7,B8=cut_mini_3(path1,lim,lim1)
    train_set = pd.read_pickle(path2)
    
    DB1 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B1):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB1 = np.append(DB1, r)
                else:
                    r = li.real
                    DB1 = np.append(DB1, r)
            else:
                pass
    DB2 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B2):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB2 = np.append(DB2, r)
                else:
                    r = li.real
                    DB2 = np.append(DB2, r)
            else:
                pass
    DB3 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B3):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB3 = np.append(DB3, r)
                else:
                    r = li.real
                    DB3 = np.append(DB3, r)
            else:
                pass
    DB4 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B4):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB4 = np.append(DB4, r)
                else:
                    r = li.real
                    DB4 = np.append(DB4, r)
            else:
                pass
            
    DB5 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B5):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB5 = np.append(DB5, r)
                else:
                    r = li.real
                    DB5 = np.append(DB5, r)
            else:
                pass
    DB6 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B6):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB6 = np.append(DB6, r)
                else:
                    r = li.real
                    DB6 = np.append(DB6, r)
            else:
                pass
    DB7 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B7):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB7 = np.append(DB7, r)
                else:
                    r = li.real
                    DB7 = np.append(DB7, r)
            else:
                pass
            
    DB8 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B8):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB8 = np.append(DB8, r)
                else:
                    r = li.real
                    DB8 = np.append(DB8, r)
            else:
                pass
    return DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8

def cut_mini_sample_4(path1,path2,lim,lim1):
    B1,B2,B3,B4,B5,B6,B7,B8=cut_mini_4(path1,lim,lim1)
    train_set = pd.read_pickle(path2)
    
    DB1 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B1):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB1 = np.append(DB1, r)
                else:
                    r = li.real
                    DB1 = np.append(DB1, r)
            else:
                pass
    DB2 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B2):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB2 = np.append(DB2, r)
                else:
                    r = li.real
                    DB2 = np.append(DB2, r)
            else:
                pass
    DB3 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B3):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB3 = np.append(DB3, r)
                else:
                    r = li.real
                    DB3 = np.append(DB3, r)
            else:
                pass
    DB4 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B4):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB4 = np.append(DB4, r)
                else:
                    r = li.real
                    DB4 = np.append(DB4, r)
            else:
                pass
            
    DB5 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B5):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB5 = np.append(DB5, r)
                else:
                    r = li.real
                    DB5 = np.append(DB5, r)
            else:
                pass
    DB6 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B6):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB6 = np.append(DB6, r)
                else:
                    r = li.real
                    DB6 = np.append(DB6, r)
            else:
                pass
    DB7 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B7):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB7 = np.append(DB7, r)
                else:
                    r = li.real
                    DB7 = np.append(DB7, r)
            else:
                pass
            
    DB8 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B8):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB8 = np.append(DB8, r)
                else:
                    r = li.real
                    DB8 = np.append(DB8, r)
            else:
                pass
    return DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8

def cut_mini_sample_5(path1,path2,lim,lim1):
    B1,B2,B3,B4,B5,B6,B7,B8=cut_mini_5(path1,lim,lim1)
    train_set = pd.read_pickle(path2)
    
    DB1 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B1):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB1 = np.append(DB1, r)
                else:
                    r = li.real
                    DB1 = np.append(DB1, r)
            else:
                pass
    DB2 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B2):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB2 = np.append(DB2, r)
                else:
                    r = li.real
                    DB2 = np.append(DB2, r)
            else:
                pass
    DB3 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B3):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB3 = np.append(DB3, r)
                else:
                    r = li.real
                    DB3 = np.append(DB3, r)
            else:
                pass
    DB4 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B4):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB4 = np.append(DB4, r)
                else:
                    r = li.real
                    DB4 = np.append(DB4, r)
            else:
                pass
            
    DB5 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B5):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB5 = np.append(DB5, r)
                else:
                    r = li.real
                    DB5 = np.append(DB5, r)
            else:
                pass
    DB6 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B6):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB6 = np.append(DB6, r)
                else:
                    r = li.real
                    DB6 = np.append(DB6, r)
            else:
                pass
    DB7 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B7):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB7 = np.append(DB7, r)
                else:
                    r = li.real
                    DB7 = np.append(DB7, r)
            else:
                pass
            
    DB8 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B8):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB8 = np.append(DB8, r)
                else:
                    r = li.real
                    DB8 = np.append(DB8, r)
            else:
                pass
    return DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8

def cut_mini_sample_6(path1,path2,lim,lim1):
    B1,B2,B3,B4,B5,B6,B7,B8=cut_mini_6(path1,lim,lim1)
    train_set = pd.read_pickle(path2)
    
    DB1 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B1):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB1 = np.append(DB1, r)
                else:
                    r = li.real
                    DB1 = np.append(DB1, r)
            else:
                pass
    DB2 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B2):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB2 = np.append(DB2, r)
                else:
                    r = li.real
                    DB2 = np.append(DB2, r)
            else:
                pass
    DB3 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B3):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB3 = np.append(DB3, r)
                else:
                    r = li.real
                    DB3 = np.append(DB3, r)
            else:
                pass
    DB4 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B4):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB4 = np.append(DB4, r)
                else:
                    r = li.real
                    DB4 = np.append(DB4, r)
            else:
                pass
            
    DB5 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B5):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB5 = np.append(DB5, r)
                else:
                    r = li.real
                    DB5 = np.append(DB5, r)
            else:
                pass
    DB6 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B6):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB6 = np.append(DB6, r)
                else:
                    r = li.real
                    DB6 = np.append(DB6, r)
            else:
                pass
    DB7 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B7):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB7 = np.append(DB7, r)
                else:
                    r = li.real
                    DB7 = np.append(DB7, r)
            else:
                pass
            
    DB8 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B8):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB8 = np.append(DB8, r)
                else:
                    r = li.real
                    DB8 = np.append(DB8, r)
            else:
                pass
    return DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8

def cut_mini_sample_7(path1,path2,lim,lim1):
    B1,B2,B3,B4,B5,B6,B7,B8=cut_mini_7(path1,lim,lim1)
    train_set = pd.read_pickle(path2)
    
    DB1 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B1):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB1 = np.append(DB1, r)
                else:
                    r = li.real
                    DB1 = np.append(DB1, r)
            else:
                pass
    DB2 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B2):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB2 = np.append(DB2, r)
                else:
                    r = li.real
                    DB2 = np.append(DB2, r)
            else:
                pass
    DB3 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B3):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB3 = np.append(DB3, r)
                else:
                    r = li.real
                    DB3 = np.append(DB3, r)
            else:
                pass
    DB4 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B4):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB4 = np.append(DB4, r)
                else:
                    r = li.real
                    DB4 = np.append(DB4, r)
            else:
                pass
            
    DB5 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B5):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB5 = np.append(DB5, r)
                else:
                    r = li.real
                    DB5 = np.append(DB5, r)
            else:
                pass
    DB6 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B6):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB6 = np.append(DB6, r)
                else:
                    r = li.real
                    DB6 = np.append(DB6, r)
            else:
                pass
    DB7 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B7):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB7 = np.append(DB7, r)
                else:
                    r = li.real
                    DB7 = np.append(DB7, r)
            else:
                pass
            
    DB8 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B8):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB8 = np.append(DB8, r)
                else:
                    r = li.real
                    DB8 = np.append(DB8, r)
            else:
                pass
    return DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8

def cut_mini_sample_8(path1,path2,lim,lim1):
    B1,B2,B3,B4,B5,B6,B7,B8=cut_mini_8(path1,lim,lim1)
    train_set = pd.read_pickle(path2)
    
    DB1 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B1):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB1 = np.append(DB1, r)
                else:
                    r = li.real
                    DB1 = np.append(DB1, r)
            else:
                pass
    DB2 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B2):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB2 = np.append(DB2, r)
                else:
                    r = li.real
                    DB2 = np.append(DB2, r)
            else:
                pass
    DB3 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B3):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB3 = np.append(DB3, r)
                else:
                    r = li.real
                    DB3 = np.append(DB3, r)
            else:
                pass
    DB4 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B4):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB4 = np.append(DB4, r)
                else:
                    r = li.real
                    DB4 = np.append(DB4, r)
            else:
                pass
            
    DB5 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B5):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB5 = np.append(DB5, r)
                else:
                    r = li.real
                    DB5 = np.append(DB5, r)
            else:
                pass
    DB6 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B6):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB6 = np.append(DB6, r)
                else:
                    r = li.real
                    DB6 = np.append(DB6, r)
            else:
                pass
    DB7 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B7):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB7 = np.append(DB7, r)
                else:
                    r = li.real
                    DB7 = np.append(DB7, r)
            else:
                pass
            
    DB8 = []
    train_set = np.array(train_set)
    for i in range (len(train_set)):
        if (i in B8):
            for j in range (len(train_set[0])):
                li = train_set[i,j]
                if (type(li)==list):
                    for z in range (len(li)):
                        r = li[z].real
                        DB8 = np.append(DB8, r)
                else:
                    r = li.real
                    DB8 = np.append(DB8, r)
            else:
                pass
    return DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8

# Transform small sample in DB and K-means format (8 functions)
def DB_cut_mini_1(n,lim,lim1):
    DB_tot = {}
    for i in range (n):
        DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8 = cut_mini_sample_1('C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\3D\\sample_students_ML\\box_0.036000000000000004\\seed_{}\\pressure_5.0\\confined_5.0_MPa.data'.format(i),'C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\Données\\sample_ML\\sample_ML\\extracted_pickles\\input_data_{}.pkl'.format(i),lim,lim1)    
        while (len(DB1) != 1100):
                DB1 = np.append(DB1,0)
        DB_tot[i] = DB1
        
        while (len(DB2) != 1100):
            DB2 = np.append(DB2,0)
        DB_tot[n+i] = DB2
        
        while (len(DB3) != 1100):
            DB3 = np.append(DB3,0)
        DB_tot[2*n+i] = DB3
        
        while (len(DB4) != 1100):
            DB4 = np.append(DB4,0)
        DB_tot[3*n+i] = DB4
        
        while (len(DB5) != 1100):
            DB5 = np.append(DB5,0)
        DB_tot[4*n+i] = DB5
        
        while (len(DB6) != 1100):
            DB6 = np.append(DB6,0)
        DB_tot[5*n+i] = DB6
        
        while (len(DB7) != 1100):
            DB7 = np.append(DB7,0)
        DB_tot[6*n+i] = DB7
        
        while (len(DB8) != 1100):
            DB8 = np.append(DB8,0)
        DB_tot[7*n+i] = DB8
        DB_tot = pd.DataFrame(DB_tot)
    return DB_tot


def DB_cut_mini_2(n,lim,lim1):
    DB_tot = {}
    for i in range (n):
        DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8 = cut_mini_sample_2('C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\3D\\sample_students_ML\\box_0.036000000000000004\\seed_{}\\pressure_5.0\\confined_5.0_MPa.data'.format(i),'C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\Données\\sample_ML\\sample_ML\\extracted_pickles\\input_data_{}.pkl'.format(i),lim,lim1)    
        while (len(DB1) != 1100):
                DB1 = np.append(DB1,0)
        DB_tot[i] = DB1
        
        while (len(DB2) != 1100):
            DB2 = np.append(DB2,0)
        DB_tot[n+i] = DB2
        
        while (len(DB3) != 1100):
            DB3 = np.append(DB3,0)
        DB_tot[2*n+i] = DB3
        
        while (len(DB4) != 1100):
            DB4 = np.append(DB4,0)
        DB_tot[3*n+i] = DB4
        
        while (len(DB5) != 1100):
            DB5 = np.append(DB5,0)
        DB_tot[4*n+i] = DB5
        
        while (len(DB6) != 1100):
            DB6 = np.append(DB6,0)
        DB_tot[5*n+i] = DB6
        
        while (len(DB7) != 1100):
            DB7 = np.append(DB7,0)
        DB_tot[6*n+i] = DB7
        
        while (len(DB8) != 1100):
            DB8 = np.append(DB8,0)
        DB_tot[7*n+i] = DB8
        DB_tot = pd.DataFrame(DB_tot)
    return DB_tot

def DB_cut_mini_3(n,lim,lim1):
    DB_tot = {}
    for i in range (n):
        DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8 = cut_mini_sample_3('C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\3D\\sample_students_ML\\box_0.036000000000000004\\seed_{}\\pressure_5.0\\confined_5.0_MPa.data'.format(i),'C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\Données\\sample_ML\\sample_ML\\extracted_pickles\\input_data_{}.pkl'.format(i),lim,lim1)    
        while (len(DB1) != 1100):
                DB1 = np.append(DB1,0)
        DB_tot[i] = DB1
        
        while (len(DB2) != 1100):
            DB2 = np.append(DB2,0)
        DB_tot[n+i] = DB2
        
        while (len(DB3) != 1100):
            DB3 = np.append(DB3,0)
        DB_tot[2*n+i] = DB3
        
        while (len(DB4) != 1100):
            DB4 = np.append(DB4,0)
        DB_tot[3*n+i] = DB4
        
        while (len(DB5) != 1100):
            DB5 = np.append(DB5,0)
        DB_tot[4*n+i] = DB5
        
        while (len(DB6) != 1100):
            DB6 = np.append(DB6,0)
        DB_tot[5*n+i] = DB6
        
        while (len(DB7) != 1100):
            DB7 = np.append(DB7,0)
        DB_tot[6*n+i] = DB7
        
        while (len(DB8) != 1100):
            DB8 = np.append(DB8,0)
        DB_tot[7*n+i] = DB8
        DB_tot = pd.DataFrame(DB_tot)
    return DB_tot

def DB_cut_mini_4(n,lim,lim1):
    DB_tot = {}
    for i in range (n):
        DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8 = cut_mini_sample_4('C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\3D\\sample_students_ML\\box_0.036000000000000004\\seed_{}\\pressure_5.0\\confined_5.0_MPa.data'.format(i),'C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\Données\\sample_ML\\sample_ML\\extracted_pickles\\input_data_{}.pkl'.format(i),lim,lim1)    
        while (len(DB1) != 1100):
                DB1 = np.append(DB1,0)
        DB_tot[i] = DB1
        
        while (len(DB2) != 1100):
            DB2 = np.append(DB2,0)
        DB_tot[n+i] = DB2
        
        while (len(DB3) != 1100):
            DB3 = np.append(DB3,0)
        DB_tot[2*n+i] = DB3
        
        while (len(DB4) != 1100):
            DB4 = np.append(DB4,0)
        DB_tot[3*n+i] = DB4
        
        while (len(DB5) != 1100):
            DB5 = np.append(DB5,0)
        DB_tot[4*n+i] = DB5
        
        while (len(DB6) != 1100):
            DB6 = np.append(DB6,0)
        DB_tot[5*n+i] = DB6
        
        while (len(DB7) != 1100):
            DB7 = np.append(DB7,0)
        DB_tot[6*n+i] = DB7
        
        while (len(DB8) != 1100):
            DB8 = np.append(DB8,0)
        DB_tot[7*n+i] = DB8
        DB_tot = pd.DataFrame(DB_tot)
    return DB_tot

def DB_cut_mini_5(n,lim,lim1):
    DB_tot = {}
    for i in range (n):
        DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8 = cut_mini_sample_5('C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\3D\\sample_students_ML\\box_0.036000000000000004\\seed_{}\\pressure_5.0\\confined_5.0_MPa.data'.format(i),'C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\Données\\sample_ML\\sample_ML\\extracted_pickles\\input_data_{}.pkl'.format(i),lim,lim1)    
        while (len(DB1) != 1100):
                DB1 = np.append(DB1,0)
        DB_tot[i] = DB1
        
        while (len(DB2) != 1100):
            DB2 = np.append(DB2,0)
        DB_tot[n+i] = DB2
        
        while (len(DB3) != 1100):
            DB3 = np.append(DB3,0)
        DB_tot[2*n+i] = DB3
        
        while (len(DB4) != 1100):
            DB4 = np.append(DB4,0)
        DB_tot[3*n+i] = DB4
        
        while (len(DB5) != 1100):
            DB5 = np.append(DB5,0)
        DB_tot[4*n+i] = DB5
        
        while (len(DB6) != 1100):
            DB6 = np.append(DB6,0)
        DB_tot[5*n+i] = DB6
        
        while (len(DB7) != 1100):
            DB7 = np.append(DB7,0)
        DB_tot[6*n+i] = DB7
        
        while (len(DB8) != 1100):
            DB8 = np.append(DB8,0)
        DB_tot[7*n+i] = DB8
        DB_tot = pd.DataFrame(DB_tot)
    return DB_tot

def DB_cut_mini_6(n,lim,lim1):
    DB_tot = {}
    for i in range (n):
        DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8 = cut_mini_sample_6('C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\3D\\sample_students_ML\\box_0.036000000000000004\\seed_{}\\pressure_5.0\\confined_5.0_MPa.data'.format(i),'C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\Données\\sample_ML\\sample_ML\\extracted_pickles\\input_data_{}.pkl'.format(i),lim,lim1)    
        while (len(DB1) != 1100):
                DB1 = np.append(DB1,0)
        DB_tot[i] = DB1
        
        while (len(DB2) != 1100):
            DB2 = np.append(DB2,0)
        DB_tot[n+i] = DB2
        
        while (len(DB3) != 1100):
            DB3 = np.append(DB3,0)
        DB_tot[2*n+i] = DB3
        
        while (len(DB4) != 1100):
            DB4 = np.append(DB4,0)
        DB_tot[3*n+i] = DB4
        
        while (len(DB5) != 1100):
            DB5 = np.append(DB5,0)
        DB_tot[4*n+i] = DB5
        
        while (len(DB6) != 1100):
            DB6 = np.append(DB6,0)
        DB_tot[5*n+i] = DB6
        
        while (len(DB7) != 1100):
            DB7 = np.append(DB7,0)
        DB_tot[6*n+i] = DB7
        
        while (len(DB8) != 1100):
            DB8 = np.append(DB8,0)
        DB_tot[7*n+i] = DB8
        DB_tot = pd.DataFrame(DB_tot)
    return DB_tot

def DB_cut_mini_7(n,lim,lim1):
    DB_tot = {}
    for i in range (n):
        DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8 = cut_mini_sample_7('C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\3D\\sample_students_ML\\box_0.036000000000000004\\seed_{}\\pressure_5.0\\confined_5.0_MPa.data'.format(i),'C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\Données\\sample_ML\\sample_ML\\extracted_pickles\\input_data_{}.pkl'.format(i),lim,lim1)    
        while (len(DB1) != 1100):
                DB1 = np.append(DB1,0)
        DB_tot[i] = DB1
        
        while (len(DB2) != 1100):
            DB2 = np.append(DB2,0)
        DB_tot[n+i] = DB2
        
        while (len(DB3) != 1100):
            DB3 = np.append(DB3,0)
        DB_tot[2*n+i] = DB3
        
        while (len(DB4) != 1100):
            DB4 = np.append(DB4,0)
        DB_tot[3*n+i] = DB4
        
        while (len(DB5) != 1100):
            DB5 = np.append(DB5,0)
        DB_tot[4*n+i] = DB5
        
        while (len(DB6) != 1100):
            DB6 = np.append(DB6,0)
        DB_tot[5*n+i] = DB6
        
        while (len(DB7) != 1100):
            DB7 = np.append(DB7,0)
        DB_tot[6*n+i] = DB7
        
        while (len(DB8) != 1100):
            DB8 = np.append(DB8,0)
        DB_tot[7*n+i] = DB8
        DB_tot = pd.DataFrame(DB_tot)
    return DB_tot

def DB_cut_mini_8(n,lim,lim1):
    DB_tot = {}
    for i in range (n):
        DB1,DB2,DB3,DB4,DB5,DB6,DB7,DB8 = cut_mini_sample_8('C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\3D\\sample_students_ML\\box_0.036000000000000004\\seed_{}\\pressure_5.0\\confined_5.0_MPa.data'.format(i),'C:\\Users\\arnau\\OneDrive\\Documents\\Bachelor semestre 6\\Projet de Bachelor\\Données\\sample_ML\\sample_ML\\extracted_pickles\\input_data_{}.pkl'.format(i),lim,lim1)    
        while (len(DB1) != 1100):
                DB1 = np.append(DB1,0)
        DB_tot[i] = DB1
        
        while (len(DB2) != 1100):
            DB2 = np.append(DB2,0)
        DB_tot[n+i] = DB2
        
        while (len(DB3) != 1100):
            DB3 = np.append(DB3,0)
        DB_tot[2*n+i] = DB3
        
        while (len(DB4) != 1100):
            DB4 = np.append(DB4,0)
        DB_tot[3*n+i] = DB4
        
        while (len(DB5) != 1100):
            DB5 = np.append(DB5,0)
        DB_tot[4*n+i] = DB5
        
        while (len(DB6) != 1100):
            DB6 = np.append(DB6,0)
        DB_tot[5*n+i] = DB6
        
        while (len(DB7) != 1100):
            DB7 = np.append(DB7,0)
        DB_tot[6*n+i] = DB7
        
        while (len(DB8) != 1100):
            DB8 = np.append(DB8,0)
        DB_tot[7*n+i] = DB8
        DB_tot = pd.DataFrame(DB_tot)
    return DB_tot


# Get all the small samples inn the DB and K-means format
def get_DB_mini_final(DB_mini_tot_fin1,DB_mini_tot_fin2,DB_mini_tot_fin3,DB_mini_tot_fin4,DB_mini_tot_fin5,DB_mini_tot_fin6,DB_mini_tot_fin7,DB_mini_tot_fin8):
    DB_mini_final = {}
    for i1 in range (80):
        DB_mini_final[i1+1] = DB_mini_tot_fin1[i1,:]
    for i2 in range (80):
        DB_mini_final[i2+81] = DB_mini_tot_fin2[i2,:]
    for i3 in range (80):
        DB_mini_final[i3+161] = DB_mini_tot_fin3[i3,:]
    for i4 in range (80):
        DB_mini_final[i4+241] = DB_mini_tot_fin4[i4,:]
    for i5 in range (80):
        DB_mini_final[i5+321] = DB_mini_tot_fin5[i5,:]
    for i6 in range (80):
        DB_mini_final[i6+401] = DB_mini_tot_fin6[i6,:]
    for i7 in range (80):
        DB_mini_final[i7+481] = DB_mini_tot_fin7[i7,:]
    for i8 in range (80):
        DB_mini_final[i8+561] = DB_mini_tot_fin8[i8,:]
    DBminifinal = pd.DataFrame(DB_mini_final)
    DBminifinal = DBminifinal.transpose()
    DBminifinal = get_list(DBminifinal)
    
    return DBminifinal   

#Find the distribution of a sample for K-Means
def get_repartition_24_KM(labels):
    
    C_1 = 0
    C_2 = 0
    C_3 = 0
    C_4 = 0
    C_5 = 0
    C_6 = 0
    C_7 = 0
    C_8 = 0
    C_9 = 0
    C_10 = 0
    C_11 = 0
    C_12 = 0
    C_13 = 0
    C_14 = 0
    C_15 = 0
    C_16 = 0
    C_17 = 0
    C_18 = 0
    C_19 = 0
    C_20 = 0
    C_21 = 0
    C_22 = 0
    C_23 = 0
    C_24 = 0

    C_tot = []

    for i in range (len(labels)):
        if (labels[i]==0):
            C_1 = C_1+1
        elif (labels[i]==1):
            C_2 = C_2+1
        elif (labels[i]==2):
            C_3 = C_3+1
        elif (labels[i]==3):
            C_4 = C_4+1
        elif (labels[i]==4):
            C_5 = C_5+1
        elif (labels[i]==5):
            C_6 = C_6+1
        elif (labels[i]==6):
            C_7 = C_7+1
        elif (labels[i]==7):
            C_8 = C_8+1
        elif (labels[i]==8):
            C_9 = C_9+1
        elif (labels[i]==9):
            C_10 = C_10+1
        elif (labels[i]==10):
            C_11 = C_11+1
        elif (labels[i]==11):
            C_12 = C_12+1
        elif (labels[i]== 12):
            C_13 = C_13+1 
        elif (labels[i]==13):
            C_14 = C_14+1
        elif (labels[i]==14):
            C_15 = C_15+1
        elif (labels[i]==15):
            C_16 = C_16+1
        elif (labels[i]==16):
            C_17 = C_17+1
        elif (labels[i]==17):
            C_18 = C_18+1
        elif (labels[i]==18):
            C_19 = C_19+1
        elif (labels[i]==19):
            C_20 = C_20+1
        elif (labels[i]==20):
            C_21 = C_21+1
        elif (labels[i]==21):
            C_22 = C_22+1
        elif (labels[i]==22):
            C_23 = C_23+1
        elif (labels[i]==23):
            C_24 = C_24+1

    C_tot = np.append(C_tot,C_1)
    C_tot = np.append(C_tot,C_2)
    C_tot = np.append(C_tot,C_3)
    C_tot = np.append(C_tot,C_4)
    C_tot = np.append(C_tot,C_5)
    C_tot = np.append(C_tot,C_6)
    C_tot = np.append(C_tot,C_7)
    C_tot = np.append(C_tot,C_8)
    C_tot = np.append(C_tot,C_9)
    C_tot = np.append(C_tot,C_10)
    C_tot = np.append(C_tot,C_11)
    C_tot = np.append(C_tot,C_12)
    C_tot = np.append(C_tot,C_13)
    C_tot = np.append(C_tot,C_14)
    C_tot = np.append(C_tot,C_15)
    C_tot = np.append(C_tot,C_16)
    C_tot = np.append(C_tot,C_17)
    C_tot = np.append(C_tot,C_18)
    C_tot = np.append(C_tot,C_19)
    C_tot = np.append(C_tot,C_20)
    C_tot = np.append(C_tot,C_21)
    C_tot = np.append(C_tot,C_22)
    C_tot = np.append(C_tot,C_23)
    C_tot = np.append(C_tot,C_24)
    
    return C_tot

#Find the distribution of a sample for DBscan
def get_repartition_24_DB(labels):
    
    C_0 = 0
    C_1 = 0
    C_2 = 0
    C_3 = 0
    C_4 = 0
    C_5 = 0
    C_6 = 0
    C_7 = 0
    C_8 = 0
    C_9 = 0
    C_10 = 0
    C_11 = 0
    C_12 = 0
    C_13 = 0
    C_14 = 0
    C_15 = 0
    C_16 = 0
    C_17 = 0
    C_18 = 0
    C_19 = 0
    C_20 = 0
    C_21 = 0
    C_22 = 0
    C_23 = 0
    C_24 = 0

    C_tot = []

    for i in range (len(labels)):
        if (labels[i]==0):
            C_1 = C_1+1
        elif (labels[i]==1):
            C_2 = C_2+1
        elif (labels[i]==2):
            C_3 = C_3+1
        elif (labels[i]==3):
            C_4 = C_4+1
        elif (labels[i]==4):
            C_5 = C_5+1
        elif (labels[i]==5):
            C_6 = C_6+1
        elif (labels[i]==6):
            C_7 = C_7+1
        elif (labels[i]==7):
            C_8 = C_8+1
        elif (labels[i]==8):
            C_9 = C_9+1
        elif (labels[i]==9):
            C_10 = C_10+1
        elif (labels[i]==10):
            C_11 = C_11+1
        elif (labels[i]==11):
            C_12 = C_12+1
        elif (labels[i]== 12):
            C_13 = C_13+1 
        elif (labels[i]==13):
            C_14 = C_14+1
        elif (labels[i]==14):
            C_15 = C_15+1
        elif (labels[i]==15):
            C_16 = C_16+1
        elif (labels[i]==16):
            C_17 = C_17+1
        elif (labels[i]==17):
            C_18 = C_18+1
        elif (labels[i]==18):
            C_19 = C_19+1
        elif (labels[i]==19):
            C_20 = C_20+1
        elif (labels[i]==20):
            C_21 = C_21+1
        elif (labels[i]==21):
            C_22 = C_22+1
        elif (labels[i]==22):
            C_23 = C_23+1
        elif (labels[i]==23):
            C_24 = C_24+1
        elif (labels[i]== -1):
            C_0 = C_0+1 

    C_tot = np.append(C_tot,C_0)
    C_tot = np.append(C_tot,C_1)
    C_tot = np.append(C_tot,C_2)
    C_tot = np.append(C_tot,C_3)
    C_tot = np.append(C_tot,C_4)
    C_tot = np.append(C_tot,C_5)
    C_tot = np.append(C_tot,C_6)
    C_tot = np.append(C_tot,C_7)
    C_tot = np.append(C_tot,C_8)
    C_tot = np.append(C_tot,C_9)
    C_tot = np.append(C_tot,C_10)
    C_tot = np.append(C_tot,C_11)
    C_tot = np.append(C_tot,C_12)
    C_tot = np.append(C_tot,C_13)
    C_tot = np.append(C_tot,C_14)
    C_tot = np.append(C_tot,C_15)
    C_tot = np.append(C_tot,C_16)
    C_tot = np.append(C_tot,C_17)
    C_tot = np.append(C_tot,C_18)
    C_tot = np.append(C_tot,C_19)
    C_tot = np.append(C_tot,C_20)
    C_tot = np.append(C_tot,C_21)
    C_tot = np.append(C_tot,C_22)
    C_tot = np.append(C_tot,C_23)
    C_tot = np.append(C_tot,C_24)
    
    return C_tot

# Find location of the small samples inside the large one
def get_class_repartition(labels):    
    center = []
    i_center = []
    for i in range (len(labels)):
        if (i+1<=80 and (i+1)%8==0):
            center = np.append(center, labels[i])
            i_center = np.append(i_center,i)
        elif (80<=(i+1)<=160 and (i+1)%8==7):
            center = np.append(center, labels[i])
            i_center = np.append(i_center,i)
        elif (160<=(i+1)<=240 and (i+1)%8==6):
            center = np.append(center, labels[i])
            i_center = np.append(i_center,i)
        elif (240<=(i+1)<=320 and (i+1)%8==5):
            center = np.append(center, labels[i])
            i_center = np.append(i_center,i)
        elif (320<=(i+1)<=400 and (i+1)%8==4):
            center = np.append(center, labels[i])
            i_center = np.append(i_center,i)
        elif (400<=(i+1)<=480 and (i+1)%8==3):
            center = np.append(center, labels[i])
            i_center = np.append(i_center,i)
        elif (480<=(i+1)<=560 and (i+1)%8==2):
            center = np.append(center, labels[i])
            i_center = np.append(i_center,i)
        elif (560<=(i+1)<=640 and (i+1)%8==1):
            center = np.append(center, labels[i])
            i_center = np.append(i_center,i)

    corner = []
    i_corner = []
    for i in range (len(labels)):
        if (i+1<=80 and (i+1)%8==1):
            corner = np.append(corner, labels[i])
            i_corner = np.append(i_corner,i)
        elif (80<=(i+1)<=160 and (i+1)%8==2):
            corner = np.append(corner, labels[i])
            i_corner = np.append(i_corner,i)
        elif (160<=(i+1)<=240 and (i+1)%8==3):
            corner = np.append(corner, labels[i])
            i_corner = np.append(i_corner,i)
        elif (240<=(i+1)<=320 and (i+1)%8==4):
            corner = np.append(corner, labels[i])
            i_corner = np.append(i_corner,i)
        elif (320<=(i+1)<=400 and (i+1)%8==5):
            corner = np.append(corner, labels[i])
            i_corner = np.append(i_corner,i)
        elif (400<=(i+1)<=480 and (i+1)%8==6):
            corner = np.append(corner, labels[i])
            i_corner = np.append(i_corner,i)
        elif (480<=(i+1)<=560 and (i+1)%8==7):
            corner = np.append(corner, labels[i])
            i_corner = np.append(i_corner,i)
        elif (560<=(i+1)<=640 and (i+1)%8==0):
            corner = np.append(corner, labels[i])
            i_corner = np.append(i_corner,i)

    edge = []
    i_edge = []
    for i in range (len(labels)):
        if (i+1<=80 and ((i+1)%8==2 or (i+1)%8==3 or (i+1)%8==5)):
            edge = np.append(edge, labels[i])
            i_edge = np.append(i_edge,i)
        elif (80<=(i+1)<=160 and ((i+1)%8==1 or (i+1)%8==4 or (i+1)%8==6)):
            edge = np.append(edge, labels[i])
            i_edge = np.append(i_edge,i)
        elif (160<=(i+1)<=240 and ((i+1)%8==1 or (i+1)%8==4 or (i+1)%8==7)):
            edge = np.append(edge, labels[i])
            i_edge = np.append(i_edge,i)
        elif (240<=(i+1)<=320 and ((i+1)%8==2 or (i+1)%8==8 or (i+1)%8==7)):
            edge = np.append(edge, labels[i])
            i_edge = np.append(i_edge,i)
        elif (320<=(i+1)<=400 and ((i+1)%8==1 or (i+1)%8==6 or (i+1)%8==7)):
            edge = np.append(edge, labels[i])
            i_edge = np.append(i_edge,i)
        elif (400<=(i+1)<=480 and ((i+1)%8==2 or (i+1)%8==8 or (i+1)%8==5)):
            edge = np.append(edge, labels[i])
            i_edge = np.append(i_edge,i)
        elif (480<=(i+1)<=560 and ((i+1)%8==8 or (i+1)%8==6 or (i+1)%8==5)):
            edge = np.append(edge, labels[i])
            i_edge = np.append(i_edge,i)
        elif (560<=(i+1)<=640 and ((i+1)%8==6 or (i+1)%8==7 or (i+1)%8==4)):
            edge = np.append(edge, labels[i])
            i_edge = np.append(i_edge,i)

    outer = []
    for i in range (len(labels)):
        if (i in i_center or i in i_corner or i in i_edge ):
            pass
        else:
            outer = np.append(outer, labels[i])
    return center, corner, edge, outer       

# Adapt the class distribution to the right class size
def class_adapt(labels):
    label_adapt = []
    j = 0
    for i in range (len(labels)):
        if (labels[i] != 0):
            label_adapt = np.append(label_adapt,labels[i])
            j = j+1
    return label_adapt  