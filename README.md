# CTA-data-analisis-library-(incomplete)
This library gathers functions used for importation, data wrangling and model building and testing for the CTA data (specifically, data simulated by CORSIKA).


These are the functions used in the final version of the TFG. Those functions are not the final version because they are being improved for achieving higher efficiency, nonetheless, such functions are fully functional in its purpose.

The ipynb shows examples for the functions, while the .py is a simple script with these functions.

An example of the functions use can be found in this Colab Notebook (if you run it, it wont work because it needs the data which is private by now):  
https://colab.research.google.com/drive/1SP9-A2oODou3NgB8SpaVIFcGHuztXBiK?usp=sharing


### The functions workflow: 

![]("imgs/carga_inicial_datos.png")    
![]("imgs/create_model_etc.png")


### List of the functions and its arguments:

26 dec
* model_creation_functions.py
	* def create_model(input_shape=(55,93,1),filts=None,kerns=None): 
	* def funcion_actualizacion(epoch,lr): 
	* def model_1_tel(input_shape=(55,93,1),filtros=None,batch_init=True,last_layers=None,avg_pooling=False,classes=3,learning_rate=1e-5,first_model=None,first_part=False): 
	* def model_multi_tel(len_inputs=4,input_shapes=[(55,93,1)],classes=7,learning_rate=1e-5,pre_proces_model=None,filtros=None,last_dense=None,autoencoder=None,common_pre=True,batch=False): 
	* def model_multi_tel_encoder(len_inputs=4,input_shapes=[(55,93,1)],classes=3,last_dense=None,encoder=None,common_pre=True): 
	* def model_multi_tel_energy(len_inputs=4,input_shapes=[(55,93,1)],learning_rate=1e-5,pre_proces_model=None,batch_init=False,filtros=None,last_dense=None,autoencoder=None,common_pre=True): 
* loaddata4use.py
	* def list_txts(base_txt_dir,element,run): 
	* def extract_info_txt(txt_dir,cols=None,cols_order=True): 
	* def id_events(base_txt_dir,element,run): 
	* def events_and_telescopes(base_txt_dir,element,run): 
	* def events_and_energy(base_txt_dir,element,run): 
	* def big_input_img(tels,element,run,base_txt_dir,base_npy_dir,return_energies=False): 
	* def mult_runs_big_input_img(tels,element,runs,base_txt_dir,base_npy_dir,return_energies=False): 
	* def create_lista_list_runs(num_events,init_events=None,random_select=False,elementos=None,max_runs=None): 
	* def data_set_longinput(tels,runs_list,base_txt_dir,base_npy_dir,labels=None,elements=None,test_size=0.2): 
	* def cargar_datos(labels,tels=None): 
	* def get_common_events(npy_element_dir,tels=None,run=None): 
	* def fill_holes(npy): 
	* def load_data(npy_dir_element,tels,runs,indices_events=None,only_names=False,ending=".npy"): 
	* def load_dataset_ensemble(base_dir,elementos_clasif,pre_name_folders="npy_",telescopios=None,lista_list_runs=None,elementos=None,test_size=0.2,same_quant=False): 
	* def load_dataset_completo(npy_base_dir,main_list_runs,telescopes,labels_asign=None,elements=None,pre_name_folders="npy_",test_size=0.2,same_quant="same",verbose=True,fill=False,categorical=True): 
	* def create_main_list_runs(num_events,init_events=None,random_select=False,elementos=None,max_runs=None): 
	* def get_common_events_energy(npy_dir_base,tels=None,run=None,array_from_txt=None,return_eventos=False): 
	* def load_dataset_energy(base_dir_npy,base_dir_txt,elementos=None,lista_list_runs=None,pre_name_folders_npy="npy_",pre_name_folders_txt="extract_",telescopios=None,test_size=0.2,same_quant="same",verbose=True,fill=False): 
	* def load_data(npy_dir,tels=None,runs=None,indices_runs=None,only_names=False,ending=".npy",test_size=0.2): 
	* def create_lista_list_runs(num_events,init_events=None,random_select=False,elementos=None,max_runs=None): 
	* def extract_info_txt(txt_dir,cols=None,cols_order=True): 
	* def get_txt_info(base_dir,extension="extract_",tel=None,run=None,element=None,cols=None,cols_order=True,ending=".txt"): 
* data_EDA_first_sight.py
	* def pos_telescopes(txt_with_the_Data,configurations_names,plotted=True,telescope_ranges=None,plot_save_dir=None): 
	* def analysis_npy_files_sep(output_dir,npy_dir=None,npy_list=None, ): 
	* def analysis_npy_files_conjunt(output_dir,npy_dir=None,npy_list=None): 
	* def plot_variable_grouped(dt_dir,txt_dir,n_bins=16,variable_split_criteria=3): 
* model_results_representations.py
	* def hex_repre(matrix=None,npy_file=None,savedir=None): 
	* def print_conf_matrix(matrix,elements=None,sin_diag=True,save_dir=None): 
	* def comp_and_diplay_conf_matrix(y_test,y_predict,elements=None,sin_diag=True,norm="true",save_dir=None): 
	* def display_max_errores(x_test,y_test,y_predicted,true_index=None,predict_index=None,sort_max=False): 
	* def plot_errors(x_test,y_test,y_predicho,true_index,predict_index,elementos=None,sort_max=False): 
* autoencoders_functions.py
* ensemble_model_functions.py
	* def load_data(npy_dir,tels=None,runs=None,indices_runs=None,only_names=False,ending=".npy",test_size=0.2): 
	* def get_common_events(npy_dir_base,tels=None,run=None): 
	* def load_dataset_ensemble(base_dir,elementos_clasif,pre_name_folders="npy_",telescopios=None,lista_list_runs=None,elementos=None,test_size=0.2,same_quant=False):#,ponderaciones=None): 
	* def create_lista_list_runs(num_events,init_events=None,random_select=False,elementos=None,max_runs=None): 
	* def load_dataset(base_dir,pre_name_folders="npy_",telescopios=None,list_runs=None,elementos=None,test_size=0.2,normalizacion_mal=False,same_quant=True):#,ponderaciones=None): 
* unzipdata_and_first_treatments.py
	* def extract_single_tar(dir_in,dir_out,new_folder=True): 
	* def extract_multiple_tar(dir_folder_with_tars,dir_final_folder): 
	* def unzip_gunzip(base_dir,final_dir=None,elements=None,folders=True): 
	* def dif_dt_txt(dir,faltantes=False,max_val=None,ending=(".dt",".txt")): 
	* def lista_dt(dt_dir): 
	* def lista_txt(txt_dir): 
	* def multiple_dt_2_npy(lista_archivos,npy_dir,limit_size=0.35,save_events_id=False,verbose=False): 
	* def dt_2_npy(base_dir,npy_base_dir=None,elements=None,save_names_list=True): 
	* def dif_dt_txt(dirs,faltantes=False,max_val=None,ending=(".dt",".txt")): 
