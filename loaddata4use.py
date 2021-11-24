
#functions used after all the preprocesing into npy files etc.
#functions made for loading data into numpy objects, process it and feed it to models etc.


import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import tensorflow as tf
import re
from sklearn.utils import train_test_split
import seaborn as sns

#with the npy files we are not done. 
#Because we have many many observation, each one with and id, and each one in a diferent telescope, all them related (same event)
# and for the diferents elements we have the same thing, many events not all in all telescopes (events for diferent elements are totally independent)


def list_txts(base_txt_dir,element,run):
    #function that retrieves the txt files for an element and a run
    #in the same run we will have same events, for diferent telescopes
    #the expected directory structure is base_txt_dir-> element_folders -> txt files for all run for all telescopes
    #?? it could be good to do it more general
    regex=f"{base_txt_dir}/*/{element}_tel_*_run_{str(run).zfill(2)}.txt"
    return sorted(glob.glob(regex),key=lambda x: int(re.findall("tel_([0-9]*)_",x)[0]))
 

def extract_info_txt(txt_dir,cols=None,cols_order=True):
    #function made for info extraction from the txt files
    #it recieves a txt_dir of a txt file, and it looks for the info about the events
    #the info about the event has many columns, if cols is a list of indices, only those columns are retrieved.
    #if cols_order=True, then the column values are grouped together
    #if cols_order=False, then it is grouped be rows
    """
    cols_names=["event"
    ,"telescope"
    ,"energy"
    ,"core distance to telescope"
    ,"image size (amplitude) [p.e.]"
    ,"number of pixels in image"
    ,"width [deg.]"
    ,"length [deg.]"
    ,"distance [deg.]"
    ,"miss [deg.]"
    ,"alpha [deg.]"
    ,"orientation [deg.]"
    ,"direction [deg.]"
    ,"image c.o.g. x [deg.]"
    ,"image c.o.g. y [deg.]"
    ,"Xmax [g/cm^2]"
    ,"Hmax [m]"
    ,"Npe (true number of photo-electrons)"
    ,"(19-23): Hottest pixel amplitudes)"]
    """
    with open(txt_dir,"r") as file:
        text_aux=file.read()
        a=re.findall(r'@\+[\s\d.-]*\d',text_aux)
        a=[i.replace("@+ ","").split() for i in a]
    if cols is None:
        return np.array(a).astype("float")
    elif type(cols)==list or type(cols)==np.array:
        list_aux=[]
        if cols_order :
            
            for i in cols:
                list_aux.append(np.array([float(j[i]) for j in a]))
            return np.array(list_aux).astype("float")
        else:
            list_aux=[]
            for i in a:
                list_aux.append(np.array([i[j] for j in cols ]))
            return np.array(list_aux).astype("float")
            #??this can be done simpler using well array indexing ¿isnt? 

    else:
        #if it is only a number, then we return only that column
        return np.array([float(j[cols]) for j in a]).astype("float")


def id_events(base_txt_dir,element,run):
    #to get the id for the events we search in the txt files
    #in the npy files the images are sorted by its ids, but 
    #?? it would be great to
    #check wheter the events from both are correct one to one.
    #this function looks for all event ids and list it only once.
    #I thing the problem is solve because together with the npy data files, i save a id_npy_data_file 
    # which contains the ids for the events in the same order
    #it is possible to list only the events from a file, if we only give a string
    #as base_txt_dir
    if bool(re.search("\.txt$",base_txt_dir)):
        try:
            return np.asarray(sorted(np.unique(extract_info_txt(base_txt_dir,cols=0))))
        except:
            print("ERROR, some problem with the path to the txt file")
            return
    else: 
        events=np.array([])
        listatxt=list_txts(base_txt_dir,element,run)
        if len(listatxt)==0:
            print(f"ERROR, not found files for run {run} and element {element}.")
        for i,j in enumerate(listatxt):
            events=np.concatenate((events,extract_info_txt(j,cols=0)))
        return np.asarray(sorted(np.unique(events)))
   




def events_and_telescopes(base_txt_dir,element,run):
    #not all telescope are triggered in al events, therefore we have to list the available telescopes for each event
    #this function return a dictionary withe the event id as label and a list of telescopes that have info about it as keys.
    #the arguments are, the base txt directory, where are all the element folders with the txt files
    #with element are you interested in, and which RUN.
    event_tels={}
    listatxt=list_txts(base_txt_dir,element,run)
    if len(listatxt)==0:
        print(f"ERROR, not found files for run {run} and element {element}.")
    for i,j in enumerate(listatxt):
        tel_aux=int(re.findall("tel_([0-9]*)_",j)[0])
        events_aux=extract_info_txt(j,cols=0)
        for k in events_aux:
            if k not in event_tels.keys():
                event_tels[k]=[tel_aux]
            else:
                event_tels[k].append(tel_aux)
    return event_tels

def events_and_energy(base_txt_dir,element,run):
    #this function creates a dictionary with the event id as label and the energy as key
    #the arguments are, the base txt directory, where are all the element folders with the txt files
    #with element are you interested in, and which RUN.
    event_energy={}
    listatxt=list_txts(base_txt_dir,element,run)
    if len(listatxt)==0:
        print(f"ERROR, not found files for run {run} and element {element}.")
    #we loop though all the telescopes for a RUN and an element
    for i,j in enumerate(listatxt):
        events_aux=extract_info_txt(j,cols=[0,2])
        for p,k in enumerate(events_aux[0]):
            if k not in event_energy.keys():
                event_energy[k]=events_aux[1][p]
                
    return event_energy




#################################################################################
#################################################################################
#FUNCIONES PARA JUNTAR TODAS LAS IMÁGENES EN UNA SOLA Y HACER UN MODELO QUE CON
#UN SOLO INPUT MUY GRANDE. ESTO SE PROVÓ, PERO SE USÓ EN EL RESULTADO FINAL
#################################################################################
#################################################################################

#ahora tenemos que apañar esto a lo grande
#le decimos unos telescopios, un elemento y una run y que nos devuelva una listao array con muchas
#primero vamos a hacer que dados un id de evento y unos telescopios,
#se nos forma la imagen de input
def big_input_img(tels,element,run,base_txt_dir,base_npy_dir,return_energies=False):
    #this function receives the telescopes, the element and the run, and it returns
    # the images for which all telescopes in that element and run, had a common event 
    shape1=(55,93)
    list_tels_shape1=[4,5,6,11,58,59,60,61,85,86]

    #get id and telescopes in each one
    aux_ids=events_and_telescopes(base_txt_dir,element,run)

    if return_energies:
        energy=events_and_energy(base_txt_dir,element,run)
    set_tels=set(tels)
    aux_keys=list(aux_ids.keys())
    #for each event id, check if the telescopes we are interested in, were triggered in the event
    for i in aux_keys:
        #if they are, okey
        if set(aux_ids[i]).issuperset(set_tels):
            pass
        else:
            #if they are not, then remove such event
            if return_energies:
                del energy[i]
            del aux_ids[i]
 

    num_eventos=len(aux_ids)
    if return_energies:
        if len(energy)!= num_eventos:
            print("Algun error con la lengitud de las labels y los eventos etc.. yo que se...")
    #ahora es cuando tenemos que hacer imagen final a base de los diferentes telescopios, 
    #primero lo vamos a hacer sencillo, uno detras de otro y dejamos ceros si sobra
    #tenemos que buscar los indices par aluego buscarlo en el npy de imagenes
 
 
    #primero de todo creamos el array inicial donde los vamos a ir metiendo
    alto=55*2
    largo=93*2
 
    final_array=np.zeros((num_eventos,alto,largo))
    #vamos telescopio por telescopio
    if return_energies:
        labels_energias=[]
    posiciones=[[0,0],[55,0],[55,93],[0,93]] #rellenadas desde la esquina sup izqr counterclokwise
    for j,i in enumerate(tels):
        pos_aux=posiciones[j]
 
        #si está seguimos adelante cargando el evento
        #tenemos que cargar la lista y ver el indice para luego cargar el evento
        indices_aux=np.load(f"{base_npy_dir}/npy_{element}/id_eventos_npy_sin_normal_{element}_tel_{i}_run_{str(run).zfill(2)}.npy")
        comprobar_shape_1=indices_aux.shape[0]
        img_aux=np.load(f"{base_npy_dir}/npy_{element}/npy_sin_normal_{element}_tel_{i}_run_{str(run).zfill(2)}_0.npy")
        comprobar_shape_2=img_aux.shape[0]
        if comprobar_shape_1 != comprobar_shape_2:
            print(f"Error con las dimensiones del array de indices y el de imagenes para {element}: run {run}, tel {i}.")
        for k,id_evento in enumerate(aux_ids.keys()):
            if i in aux_ids[id_evento]:
                num_indice=np.where(indices_aux==id_evento)[0][0]
                img_aux_2=img_aux[num_indice]

                final_array[k,pos_aux[0]:(pos_aux[0]+55),pos_aux[1]:(pos_aux[1]+93)]=img_aux_2
            else:
                print("Error, se suponia que todos los eventos tenia todosl ostelescopios")
                pass
            
            if return_energies:
                if j==0:
                    try:
                        labels_energias.append(energy[id_evento])
                    except:
                        print("Algun error con las keys de los eventos y las de las energias, nu se...")
 
                #para el primer telescopio solo, no hace falta mas
 
    if return_energies:
        return final_array, labels_energias
    else:
        return final_array




#ahora hacemos para que esto lo haga para varias runs y todas juntas
def mult_runs_big_input_img(tels,element,runs,base_txt_dir,base_npy_dir,return_energies=False):
    if not return_energies:
        final=big_input_img(tels,element,runs[0],base_txt_dir,base_npy_dir)
        for i in runs[1:]:
            final=np.concatenate((final,big_input_img(tels,element,i,base_txt_dir,base_npy_dir)))
        return final
    else:
        final_array,final_labels=big_input_img(tels,element,runs[0],base_txt_dir,base_npy_dir,return_energies=True)
        for i in runs[1:]:
            aux1,aux2=big_input_img(tels,element,i,base_txt_dir,base_npy_dir,return_energies=True)
            final_array=np.concatenate((final_array,aux1))
            final_labels.extend(aux2)
        return final_array,final_labels


#una funcion para crear la lista de list runs segun nos sea necesaria para indicar las runs a tomar de cada elemento
def create_lista_list_runs(num_events,init_events=None,random_select=False,elementos=None,max_runs=None):
    #solo tenemos que indicar la cantidad de runs que queremos de cada uno de los elementos
    #tambien podemos indicar el numero maximo de runs disponibles y si los queremos aleatorios
    if type(init_events)==int:
        if init_events==0:
            init_events=1
        init_events=np.ones(len(num_events))*init_events
    if elementos is None:
        elementos=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']
    if max_runs is None:
        max_runs=[41,40,40,40,20,40,40]
    lista_master=[]
    for i,j in enumerate(num_events):
        if random_select:
            if j> max_runs[i]:
                print("No hay tantas runs.")
                list_aux=np.random.randint(low=1,high=max_runs[i]+1,size=max_runs[i])
            else:
                list_aux=np.random.randint(low=1,high=max_runs[i]+1,size=j)
        else: 
            if init_events is not None:
                if (j+init_events[i]-1)> max_runs[i]:
                    print(f"Error, para el {i} te estas pasando de run, no hay runs {np.arange(max_runs[i]+1,init_events[i]+j+1,dtype=int)}")
                    if init_events[i] > max_runs[i]:
                        list_aux=np.random.randint(low=1,high=max_runs[i]+1,size=j)
                    else:
                        list_aux=np.arange(init_events[i],1+max_runs[i],dtype=int)
                else:
                    list_aux=np.arange(init_events[i],j+init_events[i],dtype=int)
 
 
            else:
                if (j) > max_runs[i]:
                    print(f"Error, para el \"{elementos[i]}\" te estas pasando de run, no hay run {np.arange(max_runs[i]+1,j+2,dtype=int)}")
                    list_aux=np.arange(1,max_runs[i]+1,dtype=int)
                else:
                    list_aux=np.arange(1,j+1,dtype=int)
        lista_master.append(list_aux)
    return np.array(lista_master,dtype=object)



#final dataset function
#le pasamos los elementos y las runs que queremos de cada y obtenemos el dataset 
#con sus labels y todo. Las labels tambien las personalizamos
 
def data_set_longinput(tels,runs_list,base_txt_dir,base_npy_dir,labels=None,elements=None,test_size=0.2):
    if elements is None:
        elements=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']
        if labels is None:
            labels=[0,1,2,3,4,5,6]
    else:
        if labels is None:
            labels=np.arange(len(elements))
    if len(elements)!= len(labels):
        print("Error con la long de las labels y los elementos.")
        return None
    features_list=mult_runs_big_input_img(tels,elements[0],runs_list[0],base_txt_dir,base_npy_dir)
    labels_array=np.ones(features_list.shape[0])*labels[0]
    print(elements[0],features_list.shape[0],features_list.shape,labels_array.shape)
    for i,j in enumerate(elements[1:]):
        aux_element=mult_runs_big_input_img(tels,j,runs_list[i+1],base_txt_dir,base_npy_dir)
        features_list=np.concatenate((features_list,aux_element))
        labels_array=np.concatenate((labels_array,np.ones(aux_element.shape[0])*labels[i+1]))
        print(elements[i+1],aux_element.shape[0],features_list.shape,labels_array.shape)
    features_list=features_list[...,np.newaxis]
    labels_array=tf.keras.utils.to_categorical(labels_array)
    x_train,x_test,y_train,y_test=train_test_split(features_list,labels_array,random_state=42,test_size=test_size)
    return x_train,x_test,y_train,y_test

def cargar_datos(labels,tels=None):
    txt_dir="/content/drive/MyDrive/prediccion_datos_muchos_telescopios/datos_muchos_tels_seleccion_6_03_21"
    npy_dir="/content/drive/MyDrive/prediccion_datos_muchos_telescopios/datos_muchos_tels_seleccion_6_03_21/npy_data"
    if tels is None:
        tels=[4,11,5,6]
    a=create_lista_list_runs([1,7,7,8,8,8,8],init_events=np.ones(7),random_select=False)
    x_train,x_test,y_train,y_test=data_set_longinput(tels,a,txt_dir,npy_dir,labels=labels,elements=None)
    return x_train,x_test,y_train,y_test




#################################################################################
#################################################################################
#FUNCIONES PARA LA CARGA DE DATOS, LAS USADAS EN EL ARCHIVO NOTEBOOK FINAL 
#################################################################################
#################################################################################

def get_common_events(npy_element_dir,tels=None,run=None):
    #this function returns the indices in the npy files for the telescopes, element and 
    # run indicated for common events.
    # if the event 124321 is the first one common to telescopes 4,5 and 6, then the output will 
    #indicate the index of that event in its first place

    #esta funcion nos sirve para obtener los indices de eventos comunes tal que 
    #esto es necesario para que las imagenes sean del mismo evento

    #TENEMOS QUE MIRAR PARA UNA RUN, TODOS LOS TELESCOPIOS QUE NOS INTERESEN,
    #SIN IMPORTAR EL ELEMENTO, SOLO LOS TELESCOPIOS DE LA RUN Y EL EMENTO QUE TOQUE
    lista=[]
    if (type(tels)==list) or (type(tels)==np.ndarray):
        #primero miramos a ver si es una lista los telescopios
        for i in tels:
            regex=f"{npy_element_dir}/id_eventos_*_tel_{i}_run_{str(run).zfill(2)}.npy"
            aux=glob.glob(regex)
            if aux:
                lista.append(np.load(aux[0]))
            else:
                #si no lo encuentra entonces nos saldra mal la cosa
                print("ERROR")
                print(f"Para {os.path.basename(npy_element_dir)} no se encuentra el {os.path.basename(regex)}.")
                return None
    else:
        print("ERROR")
        return None
    if len(tels)==1:
        eventos_comunes=lista[0]
    else:
        sets=[set(i) for i in lista]
        #devolvemos una lista de los INDICES de los eventos que SON COMUNES y podemos coger
        eventos_comunes=sorted(list(sets[0].intersection(*sets[1:])))
        del sets
    indices_master=[]
    for i,j in enumerate(tels):
        indices=[]
        for k in eventos_comunes:
            indices.append(np.where(lista[i]==k)[0][0])
        indices_master.append(np.array(indices))
    #nos devuelve un array con tantas listas como telescopios con el indice donde estan los eventos con 
    #igual numero de evento
    #indices_master=[tel1:[eventos para la run],tel2:[eventos para la run]...]
    return np.array(indices_master)


def fill_holes(npy):
    npy_aux=npy.copy()
    if type(npy)!=np.ndarray:
        print("Error input")
        return
    #para cada elemento que sea cero lo rellenamos con la media de los vecinos
    indices=np.where(npy[1:-1,1:-1]==0)
    indices_1=indices[1]+1
    indices_0=indices[0]+1
    for i in range(indices_1.shape[0]):
        #esto se podría vectorizar
        media=(npy[indices_0[i]-1,indices_1[i]]+npy[indices_0[i],indices_1[i]-1]+npy[indices_0[i]+1,indices_1[i]]+npy[indices_0[i],indices_1[i]+1])/4
        npy_aux[indices_0[i],indices_1[i]]=media
    return npy_aux


def load_data(npy_dir_element,tels,runs,indices_events=None,only_names=False,ending=".npy"):
    #aplicamos regular expresions para extraer los documentos deseados
    #usamos glob
    #es poco optimo este uso de glob, pero es mas flexivo porque le puedo pasar los tels concretos y los runs concretos
    #si es return sin labels, nos devuelve todo, sin separar siquiera
    #el indices runs es para que solo nos devuelva los que queremos de cada archivo

    #it returns the data in the same order as the names if only_names=True
    #its a loop for runs inside a loop for tels, therefore it will be for tel 3 y 4 and runs 7 8
    #[tel3_run_7,tel3_run_8,tel4_run_7,tel4_run_8]
    lista=[]
    if ((type(tels)==list) or (type(tels)==np.ndarray)):
        #primero miramos a ver si es una lista los telescopios
        for i in tels:
            for j in runs:
                #?? it is done with glob regular expresion, and its highly dependent in the way 
                #i formated the names, it would be great to generalize
                regex=f"{npy_dir_element}/*_tel_{i}_run_{str(j).zfill(2)}_?{ending}"
                aux=glob.glob(regex)
                if aux:
                    lista.extend(aux)
                else:
                    #si no lo encuentra entonces nos saldra mal la cosa
                    print("ERROR")
                    print(f"Para {os.path.basename(npy_dir_element)} no se encuentra el {os.path.basename(regex)}.")
                    return None
    else:
        for j in runs:
            regex=f"{npy_dir_element}/*_tel_{tels}_run_{str(j).zfill(2)}_?{ending}"
            aux=glob.glob(regex)
            if aux:
                lista.extend(aux)
            else:
                #si no lo encuentra entonces nos saldra mal la cosa
                print("ERROR")
                print(f"Para {os.path.basename(npy_dir_element)} no se encuentra el {os.path.basename(regex)}.")
                return None


    if only_names:
        return lista
    else:
        if indices_events != None:
            
            if indices_events[0].size==0:
            #Esta comprobacion la hacemos porque no sabemos si no comportarten ningun evento para esa primera run
                no_salir=True
                i=1
                while no_salir:
                    if indices_events[i].size==0:
                        i+=1
                        if i==len(indices_events):
                            return None
                    else:
                        lista_npy=np.load(lista[i])[indices_events[i]]
                        no_salir=False
                        i+=1
                for m,k in enumerate(lista[i:]):
                    if indices_events[m+i].size!= 0:
                        lista_npy=np.concatenate((lista_npy,np.load(k)[indices_events[m+i]]),axis=0)
                return lista_npy
            else:
                lista_npy=np.load(lista[0])[indices_events[0]]
                for m,k in enumerate(lista[1:]):
                    if indices_events[m+1].size!= 0:
                        lista_npy=np.concatenate((lista_npy,np.load(k)[indices_events[m+1]]),axis=0)
                return lista_npy

        else:
            lista_npy=np.load(lista[0])
            for m,k in enumerate(lista[1:]):
                lista_npy=np.concatenate((lista_npy,np.load(k)),axis=0)
            return lista_npy



#??la de load_dataset_completo, es mejor no? mas completa

#esta es la funcion master que lo junta todo en uno
#aqui indicamos los elementos que queremos que estén diferenciados en la clasificacion, así
#como las runs que queremos de cada uno de ellos
def load_dataset_ensemble(base_dir,elementos_clasif,pre_name_folders="npy_",telescopios=None,lista_list_runs=None,elementos=None,test_size=0.2,same_quant=False):
    #telescopios deb ser una lista , aunque solo haya uno 1
    #esta funcion es una version de load_dataset donde puedes escoger cuantas runs de cada elemento,
    #tambien (como en la otra, puedes decidir que haya mas o menos las mismas de todas same_quant)
    #list_RUNS Ahora es una lista de listas indicando para cada telescopio los indices de las runs 
    #indicamos los elementos que queremos que sean tomados en cuenta, pero el resto se les pone una label igual

    if elementos==None:
        elementos=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']
    #para cada elemento y para cada RUN tenemos que indicar los eventos validos
    eventos_elementos=[]
    numero_eventos=[]
    for i,j in enumerate(elementos):
        dir_aux=f"{base_dir}/{pre_name_folders}{j}"
        eventos_runs=[]
        list_runs=lista_list_runs[i]
        aux_num_events=0
        for l,k in enumerate(list_runs):
            aux_events=get_common_events(dir_aux,tels=telescopios,run=k)
            print(j,k,list_runs,aux_events.shape)
            eventos_runs.append(aux_events)
            aux_num_events+=aux_events.shape[1]
        numero_eventos.append(aux_num_events)
        eventos_elementos.append(eventos_runs)
    #ahora tenemos los eventos que sí nos valen porque estan en todos los telescopios a considerar 
    x_train_list=[]
    x_test_list=[]
    print("_______________")
    last_aux_decider=0
    for l,k in enumerate(telescopios):
        for i,j in enumerate(elementos):
            dir_aux=f"{base_dir}/{pre_name_folders}{j}"
            list_runs=lista_list_runs[i]
            #TENEMOS QUE TENER UNA LISTA DE LOS INDICES QUE SÍ COMPARTEN ID, EL RESTO NO NOS INTERESA
            #lo que hacemos es pasarle una lista de los indices PARA:
            #PARA el elemento que toca y el telescopio qeu toca, para todas las runs que queremos 
            data_aux=load_data(dir_aux,tels=k,runs=list_runs,indices_events=[m[l,:] for m in eventos_elementos[i]],only_names=False)
            if data_aux != None:
                if same_quant:
                    media=int(np.median(numero_eventos))
                    data_aux=data_aux[:media]
                print(j,k,data_aux.shape)
                if l==0:
                    if j in elementos_clasif:
                        aux_decider=last_aux_decider+1
                        last_aux_decider+=1
                    else:
                        aux_decider=0
                if (i==0) :
                    if l==0:
                        label_size=data_aux.shape[0]
                        labels=np.ones(label_size)*aux_decider
                    data=data_aux
                else:
                    if l==0:
                        label_size=data_aux.shape[0]
                        labels=np.concatenate((labels,np.ones(label_size)*aux_decider),axis=0)
                    data=np.concatenate((data,data_aux),axis=0)
                    del data_aux
        #print(data.shape)
        #ahora le aumentamos el canal y las labels las volvemos categorical
        data=data[...,np.newaxis]
        if l==0:
            labels=tf.keras.utils.to_categorical(labels)
            x_train,x_test,y_train,y_test =train_test_split(data,labels,test_size=test_size,random_state=42)
            del data,labels
            y_train_list=y_train
            y_test_list=y_test
            x_train_list.append(x_train)
            x_test_list.append(x_test)
            #del y_train,y_test
        else:
            x_train,x_test =train_test_split(data,test_size=test_size,random_state=42)
            x_train_list.append(x_train)
            x_test_list.append(x_test)
            del data
        #del x_train,x_test

    if (x_train.shape[0]!=y_train.shape[0]) or (x_test.shape[0]!=y_test.shape[0]):
        print("Ha habido algún problema con las dimensiones y eso...te jodes lo siento")
        return None
    else:
        print("EXITO")
        if len(telescopios)==1:
            return x_train_list[0],x_test_list[0],y_train_list,y_test_list
        else:
            return x_train_list,x_test_list,y_train_list,y_test_list

#lo que nosotros necesitamos ahora es, poder indicar el telescopio del que queremos las runs
#los elementos que deben ir a la misma clasificación (mismo label), e indicar cuantas runs queremos de cada uno de ellos
#tambíen es importante que se pueda testar con un numero igual de features y asi tener un porcentaje de acierto real y no sesgado.

def load_dataset_completo(npy_base_dir,main_list_runs,telescopes,labels_asign=None,elements=None,pre_name_folders="npy_",test_size=0.2,same_quant="same",verbose=True,fill=False,categorical=True):
    #Main function for data loading

    #This function allows to: load common data to multiple telescopes, split the result into test and training sets, if the classes (elements) are not 
    #balanced, there is the option to stick with the minimal lenght, an aprox size, or dont care about it.
    #it also gives the option to fill the empty pixels to smooth the image, and to personalize the labels for the classification task.
    #In that sense, we can assign numbers and group classes as we want, and also make them to be categorical (like dummie variables) or just numerical.
    #main_list_runs is a list with lists, each sublist indicates the runs that are considered for each element
    #this argument is very important because with 2 or 3 runs of gamma we have a lot of data, but for Iron, we will need maybe 20 RUNS
    #and to specify correctly if we want RUNS 3,4,5 or 7,8,9 for gamma etc, this main_list_helps us.

    #telescopes must we given in a list, even if its one
    #?? this can be changed easily
    #the desired structure is a folder with element folders and a the npy files in them
    #the prename argument is the prefix to those folders, in that case we could work in a enviroment with more folders (like the ones with files dt. and .txt )etc

    #poniendo las labels nosotros podemos escoger que dos entren dentro de la misma categoria 
    #a la hora de poner las labels_asign hay que tener en cuenta que si ponemos [0,1,2,6]
    #aunque solo haya 4 clases, se pensara que hay 7 pues va de 0 hasta 6. Siempre empieza en 0 y no puede saltarse ningun numero.

    #en él se indica los elementos que vamos a usar así como las labels que van a tener 
    #por ultimo da igual qeu tomemos muchas runs de uno si ponemos same_quant=same se nos quedará la cantidad exacata de cada
    #si ponemos same_quant=aprox mas o menos igual de cada uno  (se toma la mediana de la cantidad total de datos de cada clase considerada)
    #same_quant = "element" existe porque no es lo mis que haya la misma cantida por cada elemento qeu por cada label, entonces hay que tener eso en cuenta

    if elements==None:
        elements=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']

    if labels_asign==None:
        #labels_asign=np.array([0,1,2,3,4,5,6])
        labels_asign=np.arange(0,len(elements))
    else:
        labels_asign=np.asarray(labels_asign)

    if len(labels_asign)!=len(elements):
        print("ERROR WITH THE LENGTHS OF ELEMENTS AND LABELS")
        return None

    if len(main_list_runs)!=len(elements):
        #the length of the list with the runs for each element, must be equal to the length of elements we are considering 
        print("ERROR WITH THE LENGTH OF MAIN_LIST AND THE ELEMENTS")
        return None

    if verbose:
        print("Load of names and common events")
        print("_______________")

    #para cada elemento y para cada RUN tenemos que indicar los eventos validos
    #for each element and RUN, we must list the valid/common events
    eventos_elementos=[]
    numero_eventos=[]
    for i,j in enumerate(elements):
        dir_aux=f"{npy_base_dir}/{pre_name_folders}{j}"
        eventos_runs=[]
        list_runs=main_list_runs[i]
        aux_num_events=0
        for l,k in enumerate(list_runs):
            #all this is for apply the get_common_event to all the runs we want to use
            aux_events=get_common_events(dir_aux,tels=telescopes,run=k)
            if verbose:
                print("Element: ",j,", Telescope: ",k,", Runs: ",list_runs," Shape of common events (tels,common events): ",aux_events.shape)
            eventos_runs.append(aux_events)
            aux_num_events+=aux_events.shape[1]
        numero_eventos.append(aux_num_events)
        eventos_elementos.append(eventos_runs)
    #esto es lo mas lioso
    #eventos_elementos=[gamma:[run1:[tel1:[eventos en tel1 comunes entre telescopios en run 1 para gamma],tel2:[eventos],...],
                              #run2:[tel1:[eventos],tel2:[ev...]],run3:[...]] ,electron:[run1:[...],run2:[...]],....]


    #ahora tenemos los eventos que sí nos valen porque estan en todos los telescopios a considerar 
    x_train_list=[]
    x_test_list=[]

    if verbose:
        print("_______________")
        print("Load of actual npy data:")
        print("_______________")

    
    if (same_quant=="approx") :
        cantidad=int(np.median(numero_eventos))
            
    elif (same_quant=="same") :
        cantidad=int(np.amin(numero_eventos))
    elif same_quant=="element":
        #entonces tenemos que ver cual es el numero maximo de eventos que podemos tomar para cada elemento!!
        #tal que al final haya el mismo numero para cada label
        #esto es sabe quant per label
        #debemos sumar lo que tengan misma label y ver cual es el menor de ellos
        #cual es el menor de los numeros sumados con mismo label?
        numero_eventos=np.array(numero_eventos)
        unique_class=np.unique(labels_asign)
        num_class=len(unique_class)
        suma_labels=[np.sum(numero_eventos[i==labels_asign]) for i in unique_class]
        min_val_label=np.amin(suma_labels)
        cantidad=[]
        for i,j in enumerate(labels_asign):
            #contamos cuantas veces esta la cantidad esa minima y dividimos por el
            #numero de elementos que conforman esa clase
            cantidad.append(min_val_label//np.sum([j==aux_i for aux_i in labels_asign]))

    num_eventos_labels=[]
    for l,k in enumerate(telescopes):
        for i,j in enumerate(elements):
            dir_aux=f"{npy_base_dir}/{pre_name_folders}{j}"
            list_runs=main_list_runs[i]
            #TENEMOS QUE TENER UNA LISTA DE LOS INDICES/eventos QUE SÍ COMPARTEN ID, EL RESTO NO NOS INTERESA
            #lo que hacemos es pasarle una lista de los indices PARA:
            #PARA el elemento que toca y el telescopio qeu toca, para todas las runs que queremos 
            data_aux=load_data(dir_aux,tels=k,runs=list_runs,indices_events=[m[l,:] for m in eventos_elementos[i]],only_names=False)

            #el tema del numero de eventos eta complicado porque ahora queremos el mismo de eventos para elementos con la misma label,
            #entonces por eso debemos de tener todos los de una misma label antes de reducirlo
            #pero por otro lado tambien debe haber la misma cantidad de subclases dentro de una misma label
            if same_quant=="element":
                #esto es sabe quant per label
                data_aux=data_aux[:cantidad[i]]
            elif same_quant in ["same","approx"]:
                data_aux=data_aux[:cantidad]
            elif same_quant=="all":
                pass

            if fill:
                #aplicamos la funcion que nos rellena todo
                for p in range(data_aux.shape[0]):
                    data_aux[p,:,:]=fill_holes(data_aux[p])

            if verbose:
                print("Element: ",j,", Telescope: ",k," Shape of loaded array (amount of images, size of images): ",data_aux.shape)
            if i==0 :
                if l==0:
                    label_size=data_aux.shape[0]
                    labels=np.ones(label_size)*labels_asign[i]
                data=data_aux
            else:
                if l==0:
                    label_size=data_aux.shape[0]
                    labels=np.concatenate((labels,np.ones(label_size)*labels_asign[i]),axis=0)
                data=np.concatenate((data,data_aux),axis=0)
                del data_aux
        #new we add a new channel/axis, and turn into categorical the labels if required
        data=data[...,np.newaxis]
        if l==0:
            if categorical:
                labels=tf.keras.utils.to_categorical(labels)
            x_train,x_test,y_train,y_test =train_test_split(data,labels,test_size=test_size,random_state=42)
            
            del data,labels
            y_train_list=y_train
            y_test_list=y_test
            x_train_list.append(x_train)
            x_test_list.append(x_test)
        else:
            x_train,x_test =train_test_split(data,test_size=test_size,random_state=42)
            x_train_list.append(x_train)
            x_test_list.append(x_test)
            del data

        if (x_train.shape[0]!=y_train.shape[0]) or (x_test.shape[0]!=y_test.shape[0]):
            print("Must be some problems with the dimmensions...mmmm sorry")
            return None
        del x_train,x_test

    print("SUCCESS")
    if len(telescopes)==1:
        return x_train_list[0],x_test_list[0],y_train_list,y_test_list
    else:
        return x_train_list,x_test_list,y_train_list,y_test_list





#una funcion para crear la lista de list runs segun nos sea necesaria para indicar las runs a tomar de cada elemento

def create_main_list_runs(num_events,init_events=None,random_select=False,elementos=None,max_runs=None):
    #easy way to create the main_list_runs, because it will be a list of 7 elements, with around 20 sub elements each, 
    #to create such list, we can select some options and do it automatically

    #in this functions we just need to indicate the amount of runs we want for each element,
    #also we cant enter the max number of available runs and if those runs are selected randomly 
    #function made for listing the runs we will load of each element, this is done 
    #because 3 runs of gamma have has many events as 20 for iron and so, it is unbalanced
    if elementos is None:
        elementos=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']
    if type(num_events) is int:
        num_events=np.ones(len(elementos),dtype=int)*num_events

    if type(init_events)==int:
        if init_events==0:
            init_events=1
        init_events=np.ones(len(num_events),dtype=int)*init_events


    if max_runs is None:
        max_runs=[41,40,40,40,20,40,40]
    lista_master=[]
    for i,j in enumerate(num_events):
        if random_select:
            if j> max_runs[i]:
                print("No hay tantas runs.")
                list_aux=np.random.randint(low=1,high=max_runs[i]+1,size=max_runs[i])
            else:
                list_aux=np.random.randint(low=1,high=max_runs[i]+1,size=j)
        else: 
            if init_events is not None:
                if (j+init_events[i]-1)> max_runs[i]:
                    print(f"Error, para el {i} te estas pasando de run, no hay runs {np.arange(max_runs[i]+1,init_events[i]+j+1,dtype=int)}")
                    if init_events[i] > max_runs[i]:
                        list_aux=np.random.randint(low=1,high=max_runs[i]+1,size=j)
                    else:
                        list_aux=np.arange(init_events[i],1+max_runs[i],dtype=int)
                else:
                    list_aux=np.arange(init_events[i],j+init_events[i],dtype=int)


            else:
                if (j) > max_runs[i]:
                    print(f"ERROR, with \"{elementos[i]}\" you are going over the runs, theres no run {np.arange(max_runs[i]+1,j+2,dtype=int)}")
                    list_aux=np.arange(1,max_runs[i]+1,dtype=int)
                else:
                    list_aux=np.arange(1,j+1,dtype=int)
        lista_master.append(list_aux)
    return np.array(lista_master,dtype=object)




#################################################################################
#################################################################################
#FUNCIONES PARA LA CARGA DE DATOS DE ENERGIA!!!
#################################################################################
#################################################################################


def get_common_events_energy(npy_dir_base,tels=None,run=None,array_from_txt=None,return_eventos=False):
    #esta funcion nos sirve para obtener los indices de eventos comunes tal que 
    #esto es necesario para que las imagenes sean del mismo evento

    #TENEMOS QUE MIRAR PARA UNA RUN, TODOS LOS TELESCOPIOS QUE NOS INTERESEN,
    #SIN IMPORTAR EL ELEMENTO, SOLO LOS TELESCOPIOS DE LA RUN Y EL EMENTO QUE TOQUE
    lista=[]
    if (type(tels)==list) or (type(tels)==np.ndarray):
        #primero miramos a ver si es una lista los telescopios
        for i in tels:
            regex=f"{npy_dir_base}/id_eventos_*_tel_{i}_run_{str(run).zfill(2)}.npy"
            aux=glob.glob(regex)
            if aux:
                lista.append(np.load(aux[0]))
            else:
                #si no lo encuentra entonces nos saldra mal la cosa
                print("ERROR")
                print(f"Para {os.path.basename(npy_dir_base)} no se encuentra el {os.path.basename(regex)}.")
    else:
        print("ERROR")
        return None
    if len(tels)==1:
        if array_from_txt:
            sets=[set(i) for i in array_from_txt]
            eventos_comunes=sorted(list(set(lista[0]).intersection(*sets)))
        else:
            eventos_comunes=lista[0]
    else:
        sets=[set(i) for i in lista]
        if array_from_txt:
            for i in array_from_txt:
                sets.append(set(i)) 

        #devolvemos una lista de los INDICES de los eventos que SON COMUNES y podemos coger
        eventos_comunes=sorted(list(sets[0].intersection(*sets[1:])))
        del sets
    indices_master=[]
    indices_txt_master=[]
    for i,j in enumerate(tels):
        #no solo queremos que nos devuelva los indices para buscarlo en los npy, tambien para buscarlo en los txt
        indices=[]
        indices_txt=[]
        for k in eventos_comunes:
            indices.append(np.where(lista[i]==k)[0][0])
            indices_txt.append(np.where(array_from_txt[i]==k)[0][0])
        indices_master.append(np.array(indices))
        indices_txt_master.append(np.array(indices_txt))

    if return_eventos:
        return np.array(indices_master),np.array(indices_txt_master),np.array(eventos_comunes)
    else:
        return np.array(indices_master),np.array(indices_txt_master)



#por ultimo la funcion que nos va a administrar toda la carga de datos, aqui es donde pondemos la funcion de elergir el numero de runs para cada elementos


#MODIFICACION PARA QUE HAYA MAS O MENOS LA MISMA CANTIDAD DE DATOS DE CADA UNO.
def load_dataset_energy(base_dir_npy,base_dir_txt,elementos=None,lista_list_runs=None,pre_name_folders_npy="npy_",pre_name_folders_txt="extract_",
                        telescopios=None,test_size=0.2,same_quant="same",verbose=True,fill=False):
    #LOS TELESCOPIOS EN UNA LISTA AUNQUE SEA 1
    #la estructura de datos esperada es una carpeta contenedora de las carpetas con los archivos npy
    #y prename folder es eso que va delante del nombre de la carpeta que tiene el nombre del elemento

    #los labels son las energis que se obtienen de los txt

    #por ultimo da igual qeu tomemos muchas runs de uno si ponemos same_quant=same se nos quedará la cantidad exacata de cada
    #si ponemos same_quant=aprox mas o menos igual de cada uno  (se toma la mediana de la cantidad total de datos de cada clase considerada)
    #same_quant = "element" existe porque no es lo mis que haya la misma cantida por cada elemento qeu por cada label, entonces hay que tener eso en cuenta

    if elementos==None:
        elementos=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']


    if len(lista_list_runs)!=len(elementos):
        #como lista_list_runs es una lista de las runs que vamos a tomar, pues deber haber una para cada elemento
        print("Error con la long de los elementos y las runs")
        return None


    #para cada elemento y para cada RUN tenemos que indicar los eventos validos
    energia_label=[]
    eventos_elementos=[]
    numero_eventos=[]
    for i,j in enumerate(elementos):
        dir_aux=f"{base_dir_npy}/{pre_name_folders_npy}{j}"
        eventos_runs=[]
        list_runs=lista_list_runs[i]
        aux_num_events=0
        energia_label_aux=[]
        for l,k in enumerate(list_runs):
            #todo esto es para aplicar el get_common events a todas las runs que debemos comprobar
            event_aux=get_txt_info(base_dir_txt,extension=pre_name_folders_txt,cols=0,tel=telescopios,run=k,element=j,cols_order=True)
            #ahora tambien tenemos que estar en concordancia con los eventos que se encuentran en los txt
            aux_events,aux_events_energy=get_common_events_energy(dir_aux,tels=telescopios,run=k,array_from_txt=event_aux,return_eventos=False)

            #ahora tenemos los eventos de cada txt y los que sí se van a usar
            #solo tenemos que conseguir un array con los indices
            if verbose:
                print(j,k,list_runs,aux_events.shape,aux_events_energy.shape)
            energia=get_txt_info(base_dir_txt,extension=pre_name_folders_txt,cols=2,tel=telescopios[0],run=k,element=j,cols_order=True)
            if len(aux_events_energy[0])!=0:
                energia_label_aux.extend(energia[aux_events_energy[0]])

            eventos_runs.append(aux_events)
            aux_num_events+=aux_events.shape[1]
        if len(energia_label_aux)!=aux_num_events:
            print("Error con las dimensiones que de labels y features")
            return None
        energia_label.append(np.array(energia_label_aux))
        numero_eventos.append(aux_num_events)
        eventos_elementos.append(eventos_runs)
    #esto es lo mas lioso
    #eventos_elementos=[gamma:[run1:[tel1:[eventos en tel1 comunes entre telescopios en run 1 para gamma],tel2:[eventos],...],
                              #run2:[tel1:[eventos],tel2:[ev...]],run3:[...]] ,electron:[run1:[...],run2:[...]],....]


    #ahora tenemos los eventos que sí nos valen porque estan en todos los telescopios a considerar 
    x_train_list=[]
    x_test_list=[]

    if verbose:
        print("_______________")

    
    if (same_quant=="approx") :
        cantidad=int(np.median(numero_eventos))
        print(cantidad)

            
    elif (same_quant=="same") :
        cantidad=int(np.amin(numero_eventos))
        print(cantidad)


    for l,k in enumerate(telescopios):
        for i,j in enumerate(elementos):
            dir_aux=f"{base_dir_npy}/{pre_name_folders_npy}{j}"
            list_runs=lista_list_runs[i]
            #TENEMOS QUE TENER UNA LISTA DE LOS INDICES/eventos QUE SÍ COMPARTEN ID, EL RESTO NO NOS INTERESA
            #lo que hacemos es pasarle una lista de los indices PARA:
            #PARA el elemento que toca y el telescopio qeu toca, para todas las runs que queremos 
            data_aux=load_data(dir_aux,tels=k,runs=list_runs,indices_runs=[m[l,:] for m in eventos_elementos[i]],only_names=False)

            if same_quant in ["same","approx"]:
                data_aux=data_aux[:cantidad]
                if l==0:
                    energia_label[i]=energia_label[i][:cantidad]
            elif same_quant=="all":
                pass

            if fill:
                #aplicamos la funcion que nos rellena todo
                for p in range(data_aux.shape[0]):
                    data_aux[p,:,:]=fill_holes(data_aux[p])

            if verbose:
                print(j,k,data_aux.shape)
                if l==0:
                    print(energia_label[i].shape)
            if i==0 :
                data=data_aux
            else:
                data=np.concatenate((data,data_aux),axis=0)
                del data_aux
        #ahora le aumentamos el canal y las labels las volvemos categorical
        data=data[...,np.newaxis]
        if l==0:
            energia_label=np.concatenate([h for h in energia_label])
            print(energia_label.shape,data.shape)
            x_train,x_test,y_train,y_test =train_test_split(data,energia_label,test_size=test_size,random_state=42)
            del data
            y_train_list=y_train
            y_test_list=y_test
            x_train_list.append(x_train)
            x_test_list.append(x_test)
        else:
            x_train,x_test =train_test_split(data,test_size=test_size,random_state=42)
            x_train_list.append(x_train)
            x_test_list.append(x_test)
            del data

        if (x_train.shape[0]!=y_train.shape[0]) or (x_test.shape[0]!=y_test.shape[0]):
            print("Ha habido algún problema con las dimensiones y eso...te jodes lo siento")
            return None
        del x_train,x_test

    print("EXITO")
    if len(telescopios)==1:
        return x_train_list[0],x_test_list[0],y_train_list,y_test_list
    else:
        return x_train_list,x_test_list,y_train_list,y_test_list


##############################################################################

# THIS FUNCTIONS ARE REPEATED, BUT, ARE THEY EXACTLY THE SAME??? OR DO THEY
#HAVE CHANGES BECAUSE THE DEAL WITH THE ENERGY DATA LOADING PROCCESS???

##############################################################################

#primero funcion que carga los datos al darle unos telescopios y runs 
def load_data(npy_dir,tels=None,runs=None,indices_runs=None,only_names=False,ending=".npy",test_size=0.2):
    #aplicamos regular expresions para extraer los documentos deseados
    #usamos glob
    #si no pasamos ni los tesls ni las runs, deolvemos todos los arichivos
    #es poco optimo este uso de glob, pero es mas flexivo porque le puedo pasar los tels concretos y los runs concretos
    #si es return sin labels, nos devuelve todo, sin separar siquiera

    lista=[]
    if ((type(tels)==list) or (type(tels)==np.ndarray)):
        #primero miramos a ver si es una lista los telescopios
        for i in tels:
            for j in runs:
                regex=f"{npy_dir}/*_tel_{i}_run_{str(j).zfill(2)}_?{ending}"
                aux=glob.glob(regex)
                if aux:
                    lista.extend(aux)
                else:
                    #si no lo encuentra entonces nos saldra mal la cosa
                    print("ERROR")
                    print(f"Para {os.path.basename(npy_dir)} no se encuentra el {os.path.basename(regex)}.")
    else:
        for j in runs:
            regex=f"{npy_dir}/*_tel_{tels}_run_{str(j).zfill(2)}_?{ending}"
            aux=glob.glob(regex)
            if aux:
                lista.extend(aux)
            else:
                #si no lo encuentra entonces nos saldra mal la cosa
                print("ERROR")
                print(f"Para {os.path.basename(npy_dir)} no se encuentra el {os.path.basename(regex)}.")

    if only_names:
        return lista
    else:
        if indices_runs is not None:
            if indices_runs[0].size==0:
                no_salir=True
                i=1
                while no_salir:
                    if indices_runs[i].size==0:
                        i+=1
                    else:
                        lista_npy=np.load(lista[i])[indices_runs[i]]
                        no_salir=False
                        i+=1
                for m,k in enumerate(lista[i:]):
                    if indices_runs[m+1].size!= 0:
                        lista_npy=np.concatenate((lista_npy,np.load(k)[indices_runs[m+1]]),axis=0)
                return lista_npy
            else:
                lista_npy=np.load(lista[0])[indices_runs[0]]
                for m,k in enumerate(lista[1:]):
                    if indices_runs[m+1].size!= 0:
                        lista_npy=np.concatenate((lista_npy,np.load(k)[indices_runs[m+1]]),axis=0)
                return lista_npy

        else:
            lista_npy=np.load(lista[0])
            for m,k in enumerate(lista[1:]):
                lista_npy=np.concatenate((lista_npy,np.load(k)),axis=0)
            return lista_npy


#una funcion para crear la lista de list runs 
def create_lista_list_runs(num_events,init_events=None,random_select=False,elementos=None,max_runs=None):
    #solo tenemos que indicar la cantidad de runs que queremos de cada uno de los elementos
    #tambien podemos indicar el numero maximo de runs disponibles y si los queremos aleatorios
    if type(init_events)==int:
        if init_events==0:
            init_events=1
        init_events=np.ones(len(num_events))*init_events
    if elementos is None:
        elementos=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']
    if max_runs is None:
        max_runs=[41,40,40,40,20,40,40]
    lista_master=[]
    for i,j in enumerate(num_events):
        if random_select:
            if j> max_runs[i]:
                print("No hay tantas runs.")
                list_aux=np.random.randint(low=1,high=max_runs[i]+1,size=max_runs[i])
            else:
                list_aux=np.random.randint(low=1,high=max_runs[i]+1,size=j)
        else: 
            if init_events is not None:
                if (j+init_events[i]-1)> max_runs[i]:
                    print(f"Error, para el {i} te estas pasando de run, no hay runs {np.arange(max_runs[i]+1,init_events[i]+j+1,dtype=int)}")
                    if init_events[i] > max_runs[i]:
                        list_aux=np.random.randint(low=1,high=max_runs[i]+1,size=j)
                    else:
                        list_aux=np.arange(init_events[i],1+max_runs[i],dtype=int)
                else:
                    list_aux=np.arange(init_events[i],j+init_events[i],dtype=int)


            else:
                if (j) > max_runs[i]:
                    print(f"Error, para el \"{elementos[i]}\" te estas pasando de run, no hay run {np.arange(max_runs[i]+1,j+2,dtype=int)}")
                    list_aux=np.arange(1,max_runs[i]+1,dtype=int)
                else:
                    list_aux=np.arange(1,j+1,dtype=int)
        lista_master.append(list_aux)
    return np.array(lista_master,dtype=object)

def extract_info_txt(txt_dir,cols=None,cols_order=True):
  #extraer la informacion relevante de un archivo .txt
  #si se le pasan ciertas columnas se devolvera una lista solo con esas columnas
    cols_names=["event"
    ,"telescope"
    ,"energy"
    ,"core distance to telescope"
    ,"image size (amplitude) [p.e.]"
    ,"number of pixels in image"
    ,"width [deg.]"
    ,"length [deg.]"
    ,"distance [deg.]"
    ,"miss [deg.]"
    ,"alpha [deg.]"
    ,"orientation [deg.]"
    ,"direction [deg.]"
    ,"image c.o.g. x [deg.]"
    ,"image c.o.g. y [deg.]"
    ,"Xmax [g/cm^2]"
    ,"Hmax [m]"
    ,"Npe (true number of photo-electrons)"
    ,"(19-23): Hottest pixel amplitudes)"]
    with open(txt_dir,"r") as file:
        text_aux=file.read()
        a=re.findall(r'@\+[\s\d.-]*\d',text_aux)
        a=[i.replace("@+ ","").split() for i in a]
    if cols is None:
        return np.array(a).astype("float")
    elif type(cols)==list or type(cols)==np.array:
        list_aux=[]
        if cols_order :
            for i in cols:
                list_aux.append(np.array([float(j[i]) for j in a]))
            return np.array(list_aux).astype("float")
        else:
            list_aux=[]
            for i in a:
                list_aux.append(np.array([i[j] for j in cols ]))
            return np.array(list_aux).astype("float")

    else:
        return np.array([float(j[cols]) for j in a]).astype("float")


#ahora necesitamos poder indicar el telescopio, la run y el elemento para que nos lo devulve
def get_txt_info(base_dir,extension="extract_",tel=None,run=None,element=None,cols=None,cols_order=True,ending=".txt"):
    if (type(tel)==list) or (type(tel)==np.ndarray):
        list_return=[]
        for i in tel:
            regex=f"{base_dir}/{extension}{element}/{element}_tel_{i}_run_{str(run).zfill(2)}{ending}"
            aux=glob.glob(regex)
            if aux:
                list_return_aux=extract_info_txt(aux[0],cols=cols,cols_order=cols_order)
                list_return.append(list_return_aux)
            else:
                print("Error, archivo no encontrado")
                return None
    else:
        regex=f"{base_dir}/{extension}{element}/{element}_tel_{tel}_run_{str(run).zfill(2)}{ending}"
        aux=glob.glob(regex)
        if aux:
            list_return=extract_info_txt(aux[0],cols=cols,cols_order=cols_order)
        else:
            print("Error, archivo no encontrado")
            return None

    return list_return


#How to use it?
"""
a=create_lista_list_runs(num_events=[6,39],init_events=np.ones(2),random_select=False)
npy_base="/content/drive/MyDrive/prediccion_datos_muchos_telescopios/datos_muchos_tels_seleccion_6_03_21/npy_data"
dir_base_txt="/content/drive/MyDrive/prediccion_datos_muchos_telescopios/datos_muchos_tels_seleccion_6_03_21"
x_train,x_test,y_train,y_test=load_dataset_energy(npy_base,dir_base_txt,elementos=['gamma', 'electron'],lista_list_runs=a,
                                                    telescopios=[4,5,6,11],test_size=0.2,same_quant="all",verbose=True,fill=True)

#more info on the file auxiliar_final_results_with_energy_predict.ipynb
"""