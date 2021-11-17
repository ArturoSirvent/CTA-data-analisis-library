#Script writen by Arturo Sirvent Fresneda (2020-2021) as part of its final physics bachelor thesis in the University of Granda (UGR).   
#All the code if available free in GitHub. The use of it, is only restricted by a proper authorship acknowledgment and citation.  
#https://github.com/ArturoSirvent/CTA-data-analisis-library

#FUNCTIONS FOR CTA DATA WRANGLING ANALISIS AND MODEL CREATION/TRAINING/TESTING.

#The functions are separated into block according to its purpose.

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

#for reading the .dt etc...
import pandas as pd


#FIRST STAGE, FROM DATA FILES INTO PYTHON

#the most important/relevant thing here is the directory structure

#the data came as compressed files .tar and inside of it we had a .txt file with the details of the simulation in CORSIKA, 
#and the results of the simulation as a .dt.gz (gunzip compressed .dt file).

#the .tar compresion was for the elements: iron.tar, gamma.tar, etc.
#inside iron.tar we had: iron_tel_1_run_1.dt.gz, iron_tel_1_run_1.txt, iron_tel_1_run_2.dt.gz, iron_tel_1_run_2.txt, etc.


def extract_single_tar(dir_in,dir_out,new_folder=True):
    #this function receives de directory of a .tar file (dir_in) an unzips it into into 
    #dir_out in a folder (if final_folder=True) with the same name.
    if tarfile.is_tarfile(dir_in)==False:
        print(f"ERROR, {os.path.basename(dir_in)} IT IS NOT A TAR FILE")
        return
    else:
        with tarfile.open(dir_in) as aux_tar:
            if new_folder:
                nombre_aux=os.path.basename(dir_in).replace(".tar","")
                dir_out=f"{dir_out}/{nombre_aux}"
                if not (os.path.isdir(dir_out)):
                    os.mkdir(dir_out)
            else:
                #if not dir_out stays the same
                pass
            aux_tar.extractall(dir_out)


def extract_multiple_tar(dir_folder_with_tars,dir_final_folder):
    #this is just a function that implements extract_single_tar to a folder with all the elements compressed as .tar
    #?? it would be great to allow this function to receive a list of tars instead of a dir with them, for the case we dont
    #only have .tar files in it
    list_tar=os.listdir(dir_folder_with_tars)
    for i in list_tar:
        name_aux=f"{dir_folder_with_tars}/{i}"
        if tarfile.is_tarfile(name_aux):
            #?? it could be good to add some robust "error" management for not .tar elements
            extract_single_tar(name_aux,dir_final_folder,new_folder=True)


#Once we hace the folders with the .txt and .dt.gz , we need to unzip the .gz

#?? this function could be splited into a basic unzip function and all the loops in other one, then just 
#call it for the .txt and for the .dt separately
def unzip_gunzip(base_dir,final_dir=None,elements=None,folders=True):
    #this function attempts to unzip all the .dt.gz files, no matter the element.
    #starting from a base_dir, it will unzip all gunzip in it. Sending them to final_dir. If final_dir is not
    #specified, then all the files will be sended to extract_element folders in the same base_dir.

    #es necesaria una carpeta con carpetas para los elementos y se crearan nuevas como extract_element
    #funcion para descomprimir los elementos gunzip
    #base_dir es la carpeta padre de las otras carpetas o la contenedora de los datos talcual.
    #elementos son los [gamma,electron,silicium...]
    #folders true nos indica que hay carpetas contenedoras para cada elemento. I false, estan todos los datos
    #en el ground directory.
    #??in case of folders=False it would unzip al the elements in the same folder, but this function is not needed by now
    if folders:
        #if elements are not specified, then it will take the name of the folders as element names
        if elements is None:
            elements=[i for i in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,i)) ]
            #we dont want those we have extract_ at the beggining, incase we have them.
            elements=list(filter( re.compile("^(?!extract_).*").match,elements ))

        if final_dir is None:
            final_dir=base_dir

        for i in elements:
            new_folder=f"{final_dir}/extract_{i}"
            os.mkdir(new_folder)
            element_dir=f"{base_dir}/{i}"
            os.chdir(element_dir)
            files_names_dt=glob.glob("*.dt.gz")
            
            #?? .txt files in principle are not in zip format...
            #files_names_txt=[h.replace(".dt.gz",".txt.gz") for h in files_names_dt]
            #better like this
            files_names_txt=glob.glob("*.txt.gz")
            for j in range(len(files_names_dt)): #tqdm(range(len(files_names_dt))): ?? how to use the tqdm for progress bars if it is not in notebook?
                try:
                    if (os.path.isfile(f"{element_dir}/{files_names_dt[j]}")) and (os.path.isfile(f"{element_dir}/{files_names_txt[j]}")):
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
                        if not os.path.isfile(f"{element_dir}/{files_names_dt[j]}"):
                            print(f"File {files_names_dt[j]} not found, but {files_names_txt[j]} exists.")
                        elif not os.path.isfile(f"{element_dir}/{files_names_txt[j]}"):
                            print(f"File {files_names_txt[j]} not found, but {files_names_dt[j]} exists.")
                except IndexError:
                    print("There is more .dt elements than .txt elements, check it out.")


#Once we have all the data unzipped, we should have tons of files .dt and .txt, 
#all with the same structure --> element_tel_[1-9]_run[1-30].dt or .txt
#For energy prediction it is very important to have the .txt file, because it contains information related to 
#the simulation paramets (not only the energy but it was our case).
#To check that all the .dt have a corresponding .txt, we hace the following function:

def dif_dt_txt(dir,faltantes=False,max_val=None,ending=(".dt",".txt")):
    #this functions check wheter we have the same elements for two all the simulations
    #it returns a dictionary with the telescopes and runs in each one.

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




def lista_dt(dt_dir):
    return sorted(glob.glob(f"{dt_dir}/*.dt"))
def lista_txt(txt_dir):
    return sorted(glob.glob(f"{txt_dir}/*.txt"))


#GUARDAR LOS ARCHIVOS COMO .NPY pero sin normalizar ni nada
 
#tenemos que buscar una forma de que no guardemos archivos npy de mas de un giga (por poner un limite)
#from tqdm.notebook import tqdm
 
def multiple_dt_2_npy(lista_archivos,npy_dir,limit_size=0.35,save_events_id=False,verbose=False):
    #le pasamos una lista de directorios y los guardará descomprimidos y sin normalizar en npy_dir
    #ground_dir es el directorio base para las carpetas o para los archivos
    #npy_dir es el directorio para guardar todosl os .npy juntos, sin fantasia ni carpetas
    #folders=True es que lo .dt estan en carpetas
    #limit_size limite de peso en gigas de los .npy, por defecto esat en 350 Mb ó 0.35 Gigas

    #sin_norm in the name refers to -> without normalization

    limit_size=limit_size*1e9 # pasamos de gigas a bytes 
    npy_dir_aux=npy_dir
    num_pix_x=0
    num_pix_y=0

    #if lista_archivos is just a string and not a list, then we put it into a 1 length list
    if (type(lista_archivos)!=list) & (type(lista_archivos)==str):
        lista_archivos=[lista_archivos]

    
    #?? unnecesary change of name
    files_names=lista_archivos
    verbose_list=[]
    for j in range(len(files_names)):

        #check that all the passed files exist
        if not os.path.isfile(files_names[j]):
            print(f"The file {files_names[j]} not found")
            return
        contador_nombre=0
        dt_list=[]  
        nombre_archivo=re.findall("([a-zA-Z]*_tel_[0-9]*_run_\d\d).dt$",files_names[j])[0]
        aux_df=pd.read_csv(files_names[j],sep='  ',names=["1","2","3","4","5","6"],engine="python")
        #ahora la procesamos y la guardamos en un npy
        value_auf=aux_df[['1','3','4','5']].copy()
        del aux_df
        #tenemos que agupar los valores 
        value_auf.loc[value_auf["5"]<0,"5"]=0
        #max_aux=np.amax(value_auf["5"])
        #value_auf["5"]=value_auf["5"]/max_aux
        x_minimo=min(value_auf['3'])
        y_minimo=min(value_auf['4'])
        events=value_auf["1"].unique()
        num_pix_x_aux=value_auf["3"].unique().size
        num_pix_y_aux=value_auf["4"].unique().size
        if (num_pix_x != num_pix_x_aux) or (num_pix_y != num_pix_y_aux) : #tenemos que ser capacer de cambiar segun si observamos un telescopio u otro
            num_pix_x=num_pix_x_aux
            num_pix_y=num_pix_y_aux
            if verbose:
                #print(num_pix_x,num_pix_y)
                verbose_list.append((num_pix_x,num_pix_y))
 
            x_minimo=min(value_auf['3'])
            y_minimo=min(value_auf['4'])
            ##!!!esto puede dar problemas si resulta que para el primer evento faltan datos o algo...
            auxiliar=value_auf.loc[value_auf["1"]==events[0]][["3","4","5"]].to_numpy()
            #ahora tenemos los datos de los pixeles, podemos obtener lo que ocupa cada pixel
            size_pix_x=np.ceil((max(auxiliar[:,0])-min(auxiliar[:,0]))/(np.unique(auxiliar[:,0]).size-1))
            size_pix_y=np.ceil((max(auxiliar[:,1])-min(auxiliar[:,1]))/(np.unique(auxiliar[:,1]).size-1))
            del auxiliar
        if verbose:
            #print(nombre_archivo,end="\n")
            verbose_list.append(nombre_archivo)
 
        value_auf.loc[:,'3']=value_auf['3'].apply(lambda x: round((x-x_minimo)/size_pix_x))
        value_auf.loc[:,'4']=value_auf['4'].apply(lambda x: round((x-y_minimo)/size_pix_y))
        #event_aux=value_auf["1"].unique()
        for k in range(np.shape(events)[0]):
            #cada evento tiene que ponerse en una imagen con sus valores
            array_aux=value_auf.loc[value_auf["1"]==events[k]][["3","4","5"]]
            #lo que vamos a hacer es poner los valores en una matriz creada de antemano y guardar esa matrix
            #esos numeros vienen del maximo y el minimo valor para los pixeles, simplemente shifteamos todo
            matrix_aux=np.zeros((num_pix_x,num_pix_y)) #eran 60-5= 55 y 131-38
            matrix_aux[array_aux["3"].to_numpy(),array_aux["4"].to_numpy()]=array_aux["5"].to_numpy() 
            dt_list.append(matrix_aux)
            if limit_size!=0:
                if (np.array(dt_list).nbytes>limit_size):
                    if contador_nombre==0:
                        name_npy=f"{npy_dir_aux}/npy_sin_norm_{nombre_archivo}_{contador_nombre}.npy"
                    np.save(name_npy,np.array(dt_list))
                    del dt_list
                    dt_list=[]
                    contador_nombre+=1
        name_npy=f"{npy_dir_aux}/npy_sin_normal_{nombre_archivo}_{contador_nombre}.npy"
        np.save(name_npy,np.array(dt_list))
        del dt_list
        if save_events_id:
            name_npy_events=f"{npy_dir_aux}/id_eventos_npy_sin_normal_{nombre_archivo}.npy"
            np.save(name_npy_events,np.array(events))
    if verbose:
        return verbose_list




def dt_2_npy(base_dir,npy_base_dir=None,elements=None,save_names_list=True):
    #this function recives a directory where we have some extract_element
    #folders with the dt and txt files. Then the dt files are translated into 
    #npy files into a new folder npy_element (inside the npy_base_dir)
    #this function complets multiple_dt_2_npy with the task of giving the npy_file_names

    #the argument save_names_list=True saves the name of the dt that have been correctly preccessed,
    #because this function can take so long, and the proccess can be disturbed in half a file
    #if save_names_list=False, then it will always redo everything


    if elements is None:
        elements=[i for i in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,i)) ]
        #we do want those we have extract_ at the beggining, 
        elements=[i.replace("extract_","") for i in list(filter( re.compile("^extract_").match,elements))]

    #we go into every extract_folder and then convert each .dt into a .npy file in npy_element
    #which has been formated as required into images
    #this folder is inside a npy_data main folder (npy_base_dir)
    if npy_base_dir is None:
        npy_base_dir=f"{base_dir}/npy_data"

    if not os.path.isdir(npy_base_dir):
        os.mkdir(npy_base_dir)
    for i,j in enumerate(elements):
        #load the required names
        dt_fold_name=f"{base_dir}/extract_{j}"
        names_files=lista_dt(dt_fold_name)
        #create dest directory
        
        dest_folder_name=f"{npy_base_dir}/npy_{j}"

        if save_names_list:
            if not os.path.isdir(dest_folder_name):
                os.mkdir(dest_folder_name)
                files_done=[]
            else:
                try:
                    files_done=list(np.load(f"{dest_folder_name}/files_done_{j}.npy"))
                except:
                    files_done=[]
            #now we just give the function the list and the destination
            for k in names_files:
                #loop over the dt files for an element
                #multiple_dt_2_npy can handdle a list of multiple files, but because save_names_list=True
                #we will procced slowly one by one, passing each time a 1 length list with the file name
                if k not in files_done:
                    #if this file was not done then we proceed
                    #the k must be replaced by a 1- element list no?
                    multiple_dt_2_npy(k,dest_folder_name,save_events_id=True,verbose=False)
                    #after doing it, we save the result
                    files_done.append(k)
                    np.save(f"{dest_folder_name}/files_done_{j}.npy",files_done)

        else:
            #if we dont want to save the progress
            if not os.path.isdir(dest_folder_name):
                os.mkdir(dest_folder_name)
            #now we just give the function the list and the destination
            multiple_dt_2_npy(names_files,dest_folder_name,save_events_id=True,verbose=False)