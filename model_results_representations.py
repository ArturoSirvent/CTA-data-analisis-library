#once we train the model, we would like to keep track of the results, store them as json in case we try many architectures
#and access it later for visualize accuracy and other parameters. This functions aim to it.







#################################################################
#################################################################
#Funciones representacion parte clasificador notebook final
#################################################################
#################################################################

def hex_repre(matrix=None,npy_file=None,savedir=None):
  if npy_file is None and matrix is not None:
    matrix_aux=matrix
  elif npy_file is not None and matrix is None:
    datos=np.load(npy_file)
    matrix_aux=sum(datos)
  else:
    print("Error, debes pasar solo una de las dos cosas, matriz o ruta de archivo.")
    return
  plt.figure(figsize=(12,12))
  plt.scatter(*matrix_aux.nonzero(),s=280,c=matrix_aux[matrix_aux.nonzero()],marker="H",cmap="RdPu",alpha=0.75)
  plt.xticks([])
  plt.yticks([])
  plt.tight_layout()
  #plt.colorbar()
  if savedir is None:
    plt.show()
  else:
    plt.savefig(savedir)
  plt.close()



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