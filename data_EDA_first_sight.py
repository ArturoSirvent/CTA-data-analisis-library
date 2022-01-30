import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
#functions used for exploring the data as a first step




def pos_telescopes(txt_with_the_Data,configurations_names,plotted=True,telescope_ranges=None,plot_save_dir=None):
    #get the telescopes distributions from one of the .txt files, and according to the telescope distribution code that we specify
    #plot them if plot=True (default)
    #vamos a mirar las configuraciones según los diferentes nombres, a ver
    #we asume 3 telescope types, with numbers as follows 1-11 LSTS | 12-52 MSTS | 53-210 SST
    #this can be changed with the telescope ranges argument
    #BY DEFAUlt it is  telescope_ranges={"LST":[1,11],"MST":[12,52],"SST":[53,210]}
    if telescope_ranges is None:
        telescope_ranges={"LST":[1,11],"MST":[12,52],"SST":[53,210]}

    dict_telescope_pos={}
    for num_plot,k in enumerate(configurations_names):
        telescope_positions=[]
        telescopes_list_aux=[]
        nuevo=0
        with open(txt_with_the_Data) as txt_file:
            for line in txt_file:
                if ("# TELESCOPE" in line) and (k in line):
                    line=line.split()
                    #?? this could be done with regex in a more general way, because now we depend that the number is in possition 7.
                    #we add a column for the type of telescope big=3, medium=2, small=1
                    if line[7] not in telescopes_list_aux:
                        #we want to register each telescope only once 
                        if int(line[7]) in range(telescope_ranges["LST"][0],telescope_ranges["LST"][1]+1):
                            size=3.3
                        elif int(line[7]) in range(telescope_ranges["MST"][0],telescope_ranges["MST"][1]+1):
                            size=1.7
                        elif int(line[7]) in range(telescope_ranges["SST"][0],telescope_ranges["SST"][1]+1):
                            size=1
                        aux_list=np.array([int(line[7]),float(line[1]),float(line[2]),size])
                        telescope_positions.append(aux_list)
        dict_telescope_pos[k]=np.array(telescope_positions)

    #Now lets plot the telescope positions like a scatter plot
    #??this could be done better using axes etc...
    if plotted:
        if plot_save_dir is None:
            txt_aux="_".join(configurations_names)
            plot_save_dir=os.path.join(os.getcwd,f"pos_tels_{txt_aux}")
        plt.figure(figsize=(10*len(configurations_names),10))
        for num_plot,k in enumerate(configurations_names):
            tel_pos_aux=dict_telescope_pos[k]
            plt.subplot(1,len(configurations_names),num_plot+1)
            plt.scatter(tel_pos_aux[:,1],tel_pos_aux[:,2],s=tel_pos_aux[:,3]*300,c=tel_pos_aux[:,3],marker="H",cmap=cm.cool)
            plt.xticks([])
            plt.yticks([])
            for i,j in enumerate(zip(tel_pos_aux[:,1],tel_pos_aux[:,2])):
                text=tel_pos_aux[:,0].astype(int).astype(str)[i]
                plt.annotate(text,xy=j,ha="center")
            ax=plt.gca()
            plt.text(0.05,0.95,k,fontsize=30,transform = ax.transAxes)
            plt.tight_layout()
            plt.savefig(plot_save_dir)
    return dict_telescope_pos



#functions that gives us some plots about the incidence in the telescopes over all the runs for an elements
#we need before the npy files.

def analysis_npy_files_sep(output_dir,npy_dir=None,npy_list=None, ):
  #this function gives us a summary of the data in npy files, giving it a list, o a directory full of them.
  #output_dir must be a directory and not a name for a file
  #npy_list files must be a complete path

  if npy_dir is not None:
    #?? it would be good to do it not changing into the directory
    os.chdir(npy_dir)
    list_files=glob.glob("*.npy")
    list_files=[os.path.join(npy_dir,i) for i in list_files]
  elif npy_list is not None:
    list_files=npy_list
  else:
    print("Must provide a directory or a list of npy files")
    return 

  #now we loop thought all the files and get its content
  for i in range(len(list_files)):
    data=np.load(list_files[i])
    name_aux=list_files[i].replace(".npy","")
    name_aux_sum_img="total_sum_"+name_aux
    matrix_aux=sum(data.copy())
    plt.figure(figsize=(14,14))
    plt.subplot(3,3,2)
    plt.imshow(matrix_aux,aspect="auto")
    plt.title(name_aux_sum_img,fontsize=14)

    perc=["perc75","perc85","perc99.9"]
    vals_perc=[np.percentile(data,75),np.percentile(data,85),np.percentile(data,99.9)]
    for j in range(len(perc)):
      data_aux=data.copy()
      name_aux_sum_img="sum_"+perc[j]+"_"+name_aux
      data_aux[data<vals_perc[j]]=0
      matrix_aux=sum(data_aux)
      plt.subplot(3,3,4+j)
      plt.imshow(matrix_aux,aspect="auto")
      plt.title(name_aux_sum_img,fontsize=14)


      data_aux=data.copy()
      data_aux[data>=vals_perc[j]]=1.0
      data_aux[data<vals_perc[j]]=0
      matrix_aux=sum(data_aux)
      name_aux_sum_img="counts_"+perc[j]+"_"+name_aux
      plt.subplot(3,3,7+j)
      plt.imshow(matrix_aux,aspect="auto")
      plt.title(name_aux_sum_img,fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,name_aux)+".png")
    plt.close()



def analysis_npy_files_conjunt(output_dir,npy_dir=None,npy_list=None):
  #this function gives us a summary of the data in npy files, giving it a list, o a directory full of them.
  #this function computes the summary for all the files together, and not for each npy separately
  #output_dir can be a name and the directory will be working directory
  #npy_list files must be a complete path

  if npy_dir is not None:
    #?? it would be good to do it not changing into the directory
    os.chdir(npy_dir)
    list_files=glob.glob("*.npy")
    list_files=[os.path.join(npy_dir,i) for i in list_files]
  elif npy_list is not None:
    list_files=npy_list
  else:
    print("You must provide a directory or a list of npy files")
    return 

  #now we loop thought all the files and get its content
  for i in range(len(list_files)):
    data=np.load(list_files[i])

    if i==0:
      total_matrix_aux=np.zeros([data.shape()[1],data.shape()[2]])
      percentiles_matrix_aux=np.zeros([3,data.shape()[1],data.shape()[2]])
      counts_matrix_aux=np.zeros([3,data.shape()[1],data.shape()[2]])

    total_matrix_aux=total_matrix_aux+sum(data.copy())

    
    vals_perc=[np.percentile(data,75),np.percentile(data,85),np.percentile(data,99.9)]
    for j in range(len(vals_perc)):
      data_aux=data.copy()
      #name_aux_sum_img="sum_"+perc[j]+"_"+name_aux
      data_aux[data<vals_perc[j]]=0
      percentiles_matrix_aux[j]=percentiles_matrix_aux[j]+sum(data_aux)
      
      #lo hacemos tambien para las indicendias totales, aquellas que superan el percentil, las ponemos 
      #a valor 1 independientemente, y luego sumamos
      data_aux=data.copy()
      data_aux[data>=vals_perc[j]]=1.0
      data_aux[data<vals_perc[j]]=0
      counts_matrix_aux[j]=counts_matrix_aux[j]+sum(data_aux)

  plt.figure(figsize=(14,14))
  plt.subplot(3,3,2)
  plt.imshow(total_matrix_aux,aspect="auto")
  plt.title("total_sum",fontsize=14)

  perc=["perc75","perc85","perc99.9"]
  for j in range(len(perc)):
    plt.subplot(3,3,4+j)
    plt.imshow(percentiles_matrix_aux[j],aspect="auto")
    plt.title(f"sum_{perc[j]}",fontsize=14)
  
    name_aux_sum_img="counts_"+perc[j]+"_"+name_aux
    plt.subplot(3,3,7+j)
    plt.imshow(counts_matrix_aux,aspect="auto")
    plt.title(f"counts_{perc[j]}",fontsize=14)
  plt.tight_layout()
  plt.savefig(output_dir)
  plt.close()







#ahora voy a poner un intervalo para las energias y segun si un proceso ocurre en ese intervalo, lo incluyo en una lista
#ese intervalo de energías lo obtenemos de hacer histogram con los datos de TODAS LAS RUNS.
#VAMOS A JUNTAR RUNS PUES SUPONEMOS QUE EL ERROR NO ESTAN EN NINGUNA DE ELLAS EN CONCRETO

ground_dir="/content/drive/MyDrive/TFG arturo"
dt_dir="/content/drive/MyDrive/TFG arturo/gamma/gamma_dt"
txt_dir="/content/drive/MyDrive/TFG arturo/gamma_txt"
resultados_dir="/content/drive/MyDrive/TFG arturo/correlaciones_energia_core"
agrupacion_npy_energia="/content/drive/MyDrive/TFG arturo/correlaciones_energia_core/npy_phi"
agrupacion_graph_energia="/content/drive/MyDrive/TFG arturo/correlaciones_energia_core/graphs_phi_por_RUN"


def plot_variable_grouped(dt_dir,txt_dir,n_bins=16,variable_split_criteria=3):
  #in this function we give a directory full of files or a list of names, and it groups the observations given 
  #a variable of the simulation and a certain number of intervals for the division

  if type(dt_dir) is list:
    archivos_dt=dt_dir
  else:
    archivos_dt=os.path.join(dt_dir,os.listdir(dt_dir))

  if type(txt_dir) is list:
    archivos_txt=txt_dir
  else:
    archivos_txt=os.path.join(txt_dir,os.listdir(txt_dir))

  primero=True
  #toda la clasificacion de energias se hara segun limites en un numero determinado de intervalos
  #ahora creamos un array para ir llenandolo con los valores del evento que encaja ahí
  #seguro que con esto tendremos problemas de memoria

  #??this must be generic
  
  #valores de las posiciones de los pixeles
  y=[32, 33, 31,  0, 30, 34, 29, 35, 28, 36, 27, 37, 26, 38, 25, 39, 24,40, 23, 41,
    22, 21, 20, 42, 43, 44, 19, 18, 45, 46, 17, 16, 47, 48,15, 14, 49, 50, 13, 12,
    51, 52, 11, 10, 53, 54,  9,  8, 55, 56, 7,6, 57, 58,  5, 59]
  x=[ 84,  82,  83,  85,  86,   0,  89,  87,  88,  90,  91,  79,  77, 78,  80,  81,
    94,  92,  95,  96,  93,  74,  72,  73,  75,  76, 99,  97,  98, 100, 101,  69,
    67,  68,  70,  71, 104, 102, 103,105, 106,  64,  62,  63,  65,  66, 109, 107,
    108, 110, 111,  59, 57,  58,  60,  61, 114, 112, 113, 115, 116,  54,  52,  53,
    55, 56, 119, 117, 118, 120, 121,  49,  47,  48,  50,  51, 123, 125,126,  44,
    42,  43,  45, 122, 124, 127,  46,  40,  38,  39,  41,128, 129, 130]

  RUN=1
  clasificacion_sumatotal=[ np.zeros((len(y),len(x))) for i in range(n_of_bins)]

  for dt,txt in zip(archivos_dt,archivos_txt):
    #for each file/run, we split the values into ranges
    #vamos a extraer la informacion de cada una de las runs a la vez 
    #y vamos a unir la agrupar segun la energia los eventos
    dt_data=pd.read_csv(dt,sep='  ',names=["1","2","3","4","5","6"],engine="python")
    #procesamos los valores y despues clasificamos
    dt_data=dt_data[['1','3','4','5']].copy()
    dt_data.loc[dt_data["5"]<0]=0
    max_aux=np.amax(dt_data["5"])
    dt_data["5"]=dt_data["5"]/max_aux
    x_minimo=min(dt_data['3'])
    y_minimo=min(dt_data['4'])
    #??this must be changed to work with general telescopes, not only de big one
    dt_data['3']=dt_data['3'].apply(lambda x: round((x-x_minimo)/333))
    dt_data['4']=dt_data['4'].apply(lambda x: round((x-y_minimo)/192))

    with open(txt) as txt_file:
      txt_data=txt_file.read()

    txt_data=re.findall(r'@\+[\s\d.-]*\d',txt_data)
    txt_data=[i.replace("@+ ","").split() for i in txt_data]
    #primero el evento, segundo la energia, tercero distancia al core, cuarto el angulo (este puede estar mal)
    var_split=np.array([np.array([float(i[0]),float(i[variable_split_criteria])]) for i in txt_data])
    #esto lo vamos a hacer para el primero
    if (primero==True):
      #con esto estamos haciendo la suposicion de que mas o menos todas las energias tienen el mismo rango y con esto podemos clasificar casi todo
      #esta creencia esta fundamentada por la representacion que vemos de todas las runs, que son casi iguales las energias.
      primero=False
      ordenacion=np.histogram(var_split[:,1],bins=n_of_bins)
      #plt.hist([i[1] for i in energias],30);
      edges=ordenacion[1]
    #ahora vamos a ordenar segun los edges
    #for num_evento, val_energia in energias:
    #tenemos que ver en que intervalo está la energia de este evento
    #creo que esto con histogram se puede hacer muy facil
    for i in range(n_of_bins):
      minim=edges[i]
      maxim=edges[i+1]
      #ahora vemos cuales de los eventos estan entre estas energias
      eventos_aux=var_split[:,0][(var_split[:,1]>=minim) & (var_split[:,1]<maxim) ]
      #ahora tenemos que meter esos eventos en el array clasificación
      for event in eventos_aux:
        #ahora vamos a crear una imagen
        #error con esto seguramente falt un +1 o -1
        matrix_aux=np.zeros((len(y),len(x)))
        data_aux=dt_data[dt_data["1"]==event][["3","4","5"]]
        #we should get rid of this -5 and -39
        matrix_aux[data_aux["3"].to_numpy()-5,data_aux["4"].to_numpy()-39]=data_aux["5"].to_numpy()
        clasificacion_sumatotal[i]+=matrix_aux
    #ya tenemos clasificacion que es una lista con todos los eventos clasificados segun energias
    #SEGUIMOS EN CADA UNA DE LAS RUNS, ahora vamos a hacer varias cosas, para todas las run, para cada una ...
    #primero de nada vamos a guardar los datos
    #guardamos los limites de energia y guardamos las imagenes de eventos en dichos intervalos
    #agrupacion_npy_energia

    #la funcion devuelve los limites de agrupacion y las matrices con las agrupaciones
    #np.save(agrupacion_npy_energia+"/intervalos_de_phi_RUN_"+str(RUN)+".npy",edges)
    #np.save(agrupacion_npy_energia+"/sucesos_RUN_"+str(RUN)+"_"+str(n_of_bins)+"_bins.npy",np.array(clasificacion))
    """
    plt.figure(figsize=(18,13))
    for i in range(len(clasificacion)):
      #esto depende el numero de bins, se han tomado 16 para que sea 4 y 4
      plt.subplot(4,4,i+1)
      plt.title("Sucesos de la RUN "+str(RUN)+" para phi \n entre "+str(round(edges[i],3))+" y "+str(round(edges[i+1],3)),fontsize=14)
      sum_final=clasificacion[i]
      plt.imshow(sum_final)
    plt.tight_layout()
    dir_graph=os.path.join(agrupacion_graph_energia,"phi_RUN_"+str(RUN)+".png")
    plt.savefig(dir_graph)
    plt.close()
    RUN+=1
    """

  #np.save(agrupacion_npy_energia+"/sucesos_total_"+str(n_of_bins)+"_bins_phi.npy",np.array(clasificacion_sumatotal))
  return edges,np.array(clasificacion_sumatotal)
  """
  plt.figure(figsize=(18,13))
  for i in range(len(clasificacion_sumatotal)):
    #esto depende el numero de bins, se han tomado 9 para que sea 3 y 3
    plt.subplot(4,4,i+1)
    plt.title("Sucesos totales para core \n entre "+str(round(edges[i],3))+" y "+str(round(edges[i+1],3)),fontsize=14)
    sum_final=clasificacion_sumatotal[i]
    plt.imshow(sum_final)
  plt.tight_layout()
  dir_graph=os.path.join(agrupacion_graph_energia,"TOTAL_phi.png")
  plt.savefig(dir_graph)
  """
