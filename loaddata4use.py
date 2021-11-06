
#functions used after all the preprocesing into npy files etc.
#functions made for loading data into numpy objects, process it and feed it to models etc.

import numpy as np
import matplotlib.pyplot as plt
import os
import glob



#with the npy files we are not done. 
#Because we have many many observation, each one with and id, and each one in a diferent telescope, all them related (same event)
# and for the diferents elements we have the same thing, many events not all in all telescopes (events for diferent elements are totally independent)


def list_txts(base_txt_dir,elemento,run):
    regex=f"{base_txt_dir}/*/{elemento}_tel_*_run_{str(run).zfill(2)}.txt"
    return sorted(glob.glob(regex),key=lambda x: int(re.findall("tel_([0-9]*)_",x)[0]))
 
def id_events(base_txt_dir,elemento,run):
    #buscamos todos los txt todos los numeros de eventos y luego los contamos solo una vez
    eventos=np.array([])
    listatxt=list_txts(base_txt_dir,elemento,run)
    if len(listatxt)==0:
        print(f"Algún error, no se han encontrado archivos de run {run} y elemento {elemento}.")
    for i,j in enumerate(listatxt):
        eventos=np.concatenate((eventos,extract_info_txt(j,cols=0)))
 
    return np.asarray(sorted(np.unique(eventos)))
 
def eventos_y_telescopios(base_txt_dir,elemento,run):
    #esta funcion nos crea un disccionario con el numero de evento y con los telescopios que forman parte
    #cuando abrimos una txt vemos que eventos hay dentro y añadimos al total los telescopios segun toque
    event_tels={}
    listatxt=list_txts(base_txt_dir,elemento,run)
    if len(listatxt)==0:
        print(f"Algún error, no se han encontrado archivos de run {run} y elemento {elemento}.")
    for i,j in enumerate(listatxt):
        tel_aux=int(re.findall("tel_([0-9]*)_",j)[0])
        events_aux=extract_info_txt(j,cols=0)
        for k in events_aux:
            if k not in event_tels.keys():
                event_tels[k]=[tel_aux]
            else:
                event_tels[k].append(tel_aux)
    return event_tels

def eventos_y_energia(base_txt_dir,elemento,run):
    #esta funcion nos crea un disccionario con el numero de evento y la nergia o algo asi que forman parte
    #cuando abrimos una txt vemos que eventos hay dentro y añadimos al total los telescopios segun toque
    event_tels={}
    #buscamos para todos los telescopios
    listatxt=list_txts(base_txt_dir,elemento,run)
    if len(listatxt)==0:
        print(f"Algún error, no se han encontrado archivos de run {run} y elemento {elemento}.")
    for i,j in enumerate(listatxt):
        events_aux=extract_info_txt(j,cols=[0,2])
        for p,k in enumerate(events_aux[0]):
            if k not in event_tels.keys():
                event_tels[k]=events_aux[1][p]
                
    return event_tels

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


#ahora tenemos que apañar esto a lo grande
#le decimos unos telescopios, un elemento y una run y que nos devuelva una listao array con muchas
#primero vamos a hacer que dados un id de evento y unos telescopios,
#se nos forma la imagen de input
def big_input_img(tels,element,run,base_txt_dir,base_npy_dir,return_energies=False):
    shape1=(55,93)
    list_tels_shape1=[4,5,6,11,58,59,60,61,85,86]

    aux_ids=eventos_y_telescopios(base_txt_dir,element,run)

    if return_energies:
        energias=eventos_y_energia(base_txt_dir,element,run)
    #solo dejamos los que tengan los 4 telescopios en la lista
    set_tels=set(tels)
    aux_keys=list(aux_ids.keys())
    for i in aux_keys:
        if set(aux_ids[i]).issuperset(set_tels):
            pass
        else:
            if return_energies:
                del energias[i]
            del aux_ids[i]
 
    num_eventos=len(aux_ids)
    if return_energies:
        if len(energias)!= num_eventos:
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
                        labels_energias.append(energias[id_evento])
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



###########################################################################3
#######################################################################3

#funciones para la carga de datos

def get_common_events(npy_dir_base,tels=None,run=None):
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
    del npy
    return npy_aux


def load_data(npy_dir,tels=None,runs=None,indices_runs=None,only_names=False,ending=".npy"):
    #aplicamos regular expresions para extraer los documentos deseados
    #usamos glob
    #si no pasamos ni los tesls ni las runs, deolvemos todos los archivos
    #es poco optimo este uso de glob, pero es mas flexivo porque le puedo pasar los tels concretos y los runs concretos
    #si es return sin labels, nos devuelve todo, sin separar siquiera
    #el indices runs es para que solo nos devuelva los que queremos de cada archivo

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
                    return None
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
                return None


    if only_names:
        return lista
    else:
        if indices_runs != None:
            
            if indices_runs[0].size==0:
            #Esta comprobacion la hacemos porque no sabemos si no comportarten ningun evento para esa primera run
                no_salir=True
                i=1
                while no_salir:
                    if indices_runs[i].size==0:
                        i+=1
                        if i==len(indices_runs):
                            return None
                    else:
                        lista_npy=np.load(lista[i])[indices_runs[i]]
                        no_salir=False
                        i+=1
                for m,k in enumerate(lista[i:]):
                    if indices_runs[m+i].size!= 0:
                        lista_npy=np.concatenate((lista_npy,np.load(k)[indices_runs[m+i]]),axis=0)
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
            data_aux=load_data(dir_aux,tels=k,runs=list_runs,indices_runs=[m[l,:] for m in eventos_elementos[i]],only_names=False)
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

def load_dataset_completo(base_dir,labels_asign=None,elementos=None,lista_list_runs=None,pre_name_folders="npy_",telescopios=None,test_size=0.2,same_quant="same",verbose=True,fill=False,categorical=True):
    #LOS TELESCOPIOS EN UNA LISTA AUNQUE SEA 1
    #la estructura de datos esperada es una carpeta contenedora de las carpetas con los archivos npy
    #y prename folder es eso que va delante del nombre de la carpeta que tiene el nombre del elemento

    #poniendo las labels nosotros podemos escoger que dos entren dentro de la misma categoria 
    #a la hora de poner las labels_asign hay que tener en cuenta que si ponemos [0,1,2,6]
    #aunque solo haya 4 clases, se pensara que hay 7 pues va de 0 hasta 6. Siempre empieza en 0 y no puede saltarse ningun numero.

    #en él se indica los elementos que vamos a usar así como las labels que van a tener 
    #por ultimo da igual qeu tomemos muchas runs de uno si ponemos same_quant=same se nos quedará la cantidad exacata de cada
    #si ponemos same_quant=aprox mas o menos igual de cada uno  (se toma la mediana de la cantidad total de datos de cada clase considerada)
    #same_quant = "element" existe porque no es lo mis que haya la misma cantida por cada elemento qeu por cada label, entonces hay que tener eso en cuenta

    if elementos==None:
        elementos=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']

    if labels_asign==None:
        labels_asign=np.array([0,1,2,3,4,5,6])
    else:
        labels_asign=np.array(labels_asign)

    if len(labels_asign)!=len(elementos):
        print("Error con la long de los elementos y los elementos")
        return None

    if len(lista_list_runs)!=len(elementos):
        #como lista_list_runs es una lista de las runs que vamos a tomar, pues deber haber una para cada elemento
        print("Error con la long de los elementos y las runs")
        return None


    #para cada elemento y para cada RUN tenemos que indicar los eventos validos
    eventos_elementos=[]
    numero_eventos=[]
    for i,j in enumerate(elementos):
        dir_aux=f"{base_dir}/{pre_name_folders}{j}"
        eventos_runs=[]
        list_runs=lista_list_runs[i]
        aux_num_events=0
        for l,k in enumerate(list_runs):
            #todo esto es para aplicar el get_common events a todas las runs que debemos comprobar
            aux_events=get_common_events(dir_aux,tels=telescopios,run=k)
            if verbose:
                print(j,k,list_runs,aux_events.shape)
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
    for l,k in enumerate(telescopios):
        for i,j in enumerate(elementos):
            dir_aux=f"{base_dir}/{pre_name_folders}{j}"
            list_runs=lista_list_runs[i]
            #TENEMOS QUE TENER UNA LISTA DE LOS INDICES/eventos QUE SÍ COMPARTEN ID, EL RESTO NO NOS INTERESA
            #lo que hacemos es pasarle una lista de los indices PARA:
            #PARA el elemento que toca y el telescopio qeu toca, para todas las runs que queremos 
            data_aux=load_data(dir_aux,tels=k,runs=list_runs,indices_runs=[m[l,:] for m in eventos_elementos[i]],only_names=False)

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
                print(j,k,data_aux.shape)

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
        #ahora le aumentamos el canal y las labels las volvemos categorical
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
            print("Ha habido algún problema con las dimensiones y eso...te jodes lo siento")
            return None
        del x_train,x_test

    print("EXITO")
    if len(telescopios)==1:
        return x_train_list[0],x_test_list[0],y_train_list,y_test_list
    else:
        return x_train_list,x_test_list,y_train_list,y_test_list





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


#funcion que nos ayuda a mostar la matrix de confusion, necesita seaborn as sns
def print_conf_matrix(matrix,elements=None,sin_diag=True,save_dir=None):
    if elements is None:
        elements=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']
    if sin_diag:
        for i in range(len(elements)):
            matrix[i,i]=0
    plt.figure(figsize=(13,9))
    sns.heatmap(matrix,annot=True,annot_kws={'size':16})

    plt.yticks(np.arange(len(elements))+0.25,elements,fontsize=14,rotation=25);
    plt.xticks(np.arange(len(elements))+0.25,elements,fontsize=14,rotation=25);
    plt.title("True label in Y-axis, predicted label in X-axis", fontsize=15)
    if save_dir != None:
        plt.tight_layout()
        plt.savefig(save_dir)


def comp_and_diplay_conf_matrix(y_test,y_predict,elements=None,sin_diag=True,norm="true",save_dir=None):
    matrix=confusion_matrix(np.argmax(y_test,axis=-1),np.argmax(y_predict,axis=-1),normalize=norm)
    print_conf_matrix(matrix,elements=elements,sin_diag=sin_diag,save_dir=save_dir)



def display_max_errores(x_test,y_test,y_predicted,true_index=None,predict_index=None,sort_max=False):
    #primero tenemos que sacar aquellos que tengan maxima discrepancia entre lo predicho y lo real
    #sort max seria para sortearlas segun los maximo errores cometidos
    indices={}
    a=0
    if (true_index is None) or (predict_index is None):
        print("Dime que elemento quieres ver sus errores")
        return None

    if sort_max:
        #solo tenemos que meter primero a los que tengan mayor certeza de prediccion y asi ya nos sacara los erroneos
        indices_sort=np.argsort(y_predicted[:,predict_index])[::-1]
        #los mayores iran delante
        y_test=y_test[indices_sort]
        y_predicted=y_predicted[indices_sort]


    for i,j in enumerate(y_test):
        true_ind=np.argmax(j)
        predict_ind=np.argmax(y_predicted[i])
        if (true_ind!=predict_ind) and ((true_ind==true_index) and (predict_ind==predict_index)):
            indices[i]=y_predicted[i][predict_index], y_predicted[i][true_index]
    return indices

def plot_errors(x_test,y_test,y_predicho,true_index,predict_index,elementos=None,sort_max=False):

    if elementos is None:
        elementos=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']
    a=display_max_errores(x_test,y_test,y_predicho,true_index=true_index,predict_index=predict_index,sort_max=sort_max)
    #vamos a ver algunos de los que se han confundido

    for i in range(0,8):
        fig=plt.figure(figsize=(10,8))
        indice=i #por orden natural
        indice_real= list(a)[indice]# el valor real en el x_test
        fig.suptitle(f"Se creyó que era {elementos[predict_index]} ({a[indice_real][0]*100:.2f}%), pero era {elementos[true_index]} ({a[indice_real][1]*100:.2f}%)",fontsize=15)
        for j in range(1,5):    
            plt.subplot(2,2,j)
            plt.imshow(x_test[j-1][indice_real][:,:,0])
            
        plt.tight_layout()