#Script writen entirely by Arturo Sirvent Fresneda (2020-2021) as part of its final physics bachelor thesis in the University of Granada (UGR).   
#All the code is available free in GitHub. Its use is only conditioned by a proper authorship acknowledgment and citation.  
#https://github.com/ArturoSirvent/CTA-data-analisis-library

#FUNCTIONS FOR CTA DATA WRANGLING ANALISIS AND MODEL CREATION/TRAINING/TESTING.

#The functions are separated into blocks according to its purpose.

#comments for further improvement are marked as comments but with double questions --> #??
#in that way it is easier to find the spots where I find the code could be improved.

#The requiered libraries are:

#for directory managing
import os 
#array use
import numpy as np 
#tar files
import tarfile 
#gunzip files
import gzip
#files names listing
import glob
#¿?
import re
#unnecesary library used for having progress bars in for loops
#?? this is for notebook, it should be changed or omitted
#from tqdm.notebook import tqdm



#FIRST STAGE, FROM DATA FILES INTO PYTHON

#the most important/relevant thing here is the directory structure

#the data came as compressed files .tar and inside of it we had a .txt file with the details of the simulation in CORSIKA, 
#and the results of the simulation as a .dt.gz (gunzip compressed .dt file).

#the .tar compresion was for the elements: iron.tar, gamma.tar, etc.
#inside iron.tar we had: iron_tel_1_run_1.dt.gz, iron_tel_1_run_1.txt, iron_tel_1_run_2.dt.gz, iron_tel_1_run_2.txt, etc.


def extract_single_tar(dir_in,dir_out,final_folder=True):
    #this function receives de directory of a .tar file (dir_in) an unzips it into into 
    #dir_out in a folder (if final_folder=True) with the same name.
    if tarfile.is_tarfile(dir_in)==False:
        print(f"ERROR, {os.path.basename(dir_in)} IT IS NOT A TAR FILE")
        return
    else:
        with tarfile.open(dir_in) as aux_tar:
            if final_folder:
                nombre_aux=os.path.basename(dir_in).replace(".tar","")
                dir_out=f"{dir_out}/{nombre_aux}"
                os.mkdir(dir_out)
            else:
                #if not dir_out stays the same
                pass
            aux_tar.extractall(dir_out)



def extract_multiple_tar(dir_folder_with_tars,dir_final_folder):
    #this is just a function that implements extract_single_tar to a folder with all the elements compressed as .tar
    list_tar=os.listdir(dir_folder_with_tars)
    for i in list_tar:
        #?? it could be good to add some robust "error" management for not .tar elements
        extract_single_tar(f"{dir_folder_with_tars}/{i}",dir_final_folder,final_folder=True)


#Once we hace the folders with the .txt and .dt.gz , we need to unzip the .gz

#??
def descomprimir_gunzip(base_dir,final_dir=None,elements=None,folders=True):
    #this function attempts to unzip all the .dt.gz files, no matter the element.
    #the structure for using it should be the one recieved from extract_multiple_tar()
    #folders with the names of the elements. Then
    #es necesaria una carpeta con carpetas para los elementos y se crearan nuevas como extract_element
    #funcion para descomprimir los elementos gunzip
    #base_dir es la carpeta padre de las otras carpetas o la contenedora de los datos talcual.
    #elementos son los [gamma,electron,silicium...]
    #folders true nos indica que hay carpetas contenedoras para cada elemento. I false, estan todos los datos
    #en el ground directory.
    if folders:
        for i in elements:
            new_folder=f"{base_dir}/extract_{i}"
            os.mkdir(new_folder)
            element_dir=f"{base_dir}/{i}"
            os.chdir(element_dir)
            files_names_dt=glob.glob("*.dt.gz")
            files_names_txt=[h.replace(".dt.gz",".txt.gz") for h in files_names_dt]

            for j in range(len(files_names_dt)): #tqdm(range(len(files_names_dt))): ?? how to use the tqdm for progress bars if it is not in notebook?
                try:
                    if (os.path.isfile(f"{element_dir}/{files_names_dt[j]}")) and (os.path.isfile(f"{element_dir}/{files_names_txt[j]}")):
                        #print(files_names_dt[j].replace(".dt.gz",""))
                        with gzip.open(f"{element_dir}/{files_names_dt[j]}","rb") as f:
                            new_name_dt=files_names_dt[j].replace(".dt.gz",".dt")
                            fp=open(f"{new_folder}/{new_name_dt}","wb")
                            aux=f.read()
                            fp.write(aux)
                            fp.close()
                            f.close()
                        with gzip.open(f"{element_dir}/{files_names_txt[j]}","rb") as f:
                            new_name_txt=files_names_txt[j].replace(".txt.gz",".txt")
                            fp=open(f"{new_folder}/{new_name_txt}","wb")
                            aux=f.read()
                            fp.write(aux)
                            fp.close()
                            f.close()

                    else:
                        print(f"Para el elemento : {files_names_dt[j]} no hay un txt correspondiente")
                except IndexError:
                    print("Hay más elementos dt que txt")


#Once we have all the data unzipped, we should have tons of files .dt and .txt, 
#all with the same structure --> element_tel_[1-9]_run[1-30].dt or .txt
#For energy prediction it is very important to have the .txt file, because it contains information related to 
#the simulation paramets (not only the energy but it was our case).
#To check that all the .dt have a corresponding .txt, we hace the following function:

#?? incompleto y no revisado
def dif_dt_txt(dir,faltantes=False,max_val=None,ending=(".dt",".txt")):
    #esta funcion comprueba si hay los mismo archivos para txt y dt y cuales faltan y hasta que run llegan
    #devuelve un diccionario con los telescopios y las runs de cada uno

    #si pedimos los faltantes y damos un max_val obtenemos los que no hay en cada run
    os.chdir(dir)
    file_dt=glob.glob(f"*{ending[0]}")
    file_txt=glob.glob(f"*{ending[1]}")

    #primero extraemos la informacion importante, el tel y la run
    tel_run_dt=np.array([ np.array([ re.findall("tel_([0-9]*)_",i)[0] ,re.findall("run_([0-9]*).",i)[0]],dtype="int")  for i in file_dt])
    tel_run_txt=np.array([ np.array([ re.findall("tel_([0-9]*)_",i)[0] ,re.findall("run_([0-9]*).",i)[0] ],dtype="int")  for i in file_txt])

    #una vez tenemos la info, queremos ver si son iguales
    #primero las dimensiones
    if tel_run_dt.shape[0]!=tel_run_txt.shape[0]:
        print("Error con las dimensiones, no hay los mismos")
        if tel_run_dt.shape[0] > tel_run_txt.shape[0]:
            for i in tel_run_dt:
                if not np.all(tel_run_txt==i,axis=-1).any():
                    print(f"El tel_{i[0]}_run_{i[1]}.{ending[0]} no tiene correspondiente {ending[1]}.")
        else:
            for i in tel_run_txt:
                if not np.all(tel_run_dt==i,axis=-1).any():
                    print(f"El tel_{i[0]}_run_{i[1]}.{ending[1]} no tiene correspondiente {ending[0]}.")
        return None

    #son iguales las dos listas?
    salir=False
    for i in tel_run_dt:
        if i not in tel_run_txt:
            print(f"{i} no tiene correspondiente {ending[0]}")
            salir=True
    for i in tel_run_txt:
        if i not in tel_run_dt:
            print(f"{i} no tiene correspondiente {ending[1]}")
            salir=True
    if salir:
        return None
    else:
        print(f"Para \"{os.path.basename(dir)}\" todos los {ending[1]} tienen {ending[0]} y viceversa, todo bien.")
    
    #si las dimensiones sí estan bien entonces pasamos a listar los telescopios que hay para cada elemento y las runs para cada telescopio
    telescopios=sorted(np.unique(tel_run_dt[:,0]))
    runs=[sorted(tel_run_dt[tel_run_dt[:,0]==i][:,1]) for i in telescopios]
    #por ultimo vamos a hacer un diccionario que tenga el telescopios y las runs que agrupa
    if not faltantes:
        return dict(zip(telescopios,runs))
    else:
        if max_val is None:
            print("pasa un valor maximo para las runs")
            return None
        else:
            runs_reales=np.arange(1,max_val+1)
            run_faltantes=[]
            for i in range(len(runs)):
                faltan=[]
                for j in runs_reales:
                    if j not in runs[i]:
                        faltan.append(j)
                run_faltantes.append(faltan)
            diccionario=dict(zip(telescopios,run_faltantes))
            for i in telescopios:
                if (diccionario[i]==[]):
                    diccionario.pop(i)
            return diccionario


#The process of loading data from the .dt into python is a very time consuming process, a much better solution is 
#to use .npy files with only the data we want, in the order and shape we need. 
#?? I think the storage memory could be reduced with .npz files.

#this function is extremely important because it reconstructs from the raw data in the .dt files, the images captured in the telescopes
#the raw data gives us: the event id, the telescope number, the pixels possitions, the values for the pixel activations, and an indicator
#that tells us if such pixel was active because a real signal or it was due to simulated noise (this can be done because,
#  remember, the data is simulated, the ground truth is known).

