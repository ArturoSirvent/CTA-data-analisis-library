# CTA-data-analisis-library-(incomplete)
This library gathers functions used for importation, data wrangling and model building and testing for the CTA data (specifically, data simulated by CORSIKA).


These are the functions used in the final version of the TFG. Those functions are not the final version because they are being improved for achieving higher efficiency, nonetheless, such functions are fully functional in its purpose.

The ipynb shows examples for the functions, while the .py is a simple script with these functions.

An example of the functions use can be found in this Colab Notebook (if you run it, wont work because it needs the data which is private by now):  
https://colab.research.google.com/drive/1SP9-A2oODou3NgB8SpaVIFcGHuztXBiK?usp=sharing


17 nov

* model_creation_functions.py
	* funcion_actualizacion(epoch,lr) 
* loaddata4use.py
	* list_txts(base_txt_dir,element,run) 
	* extract_info_txt(txt_dir,cols=None,cols_order=True) 
	* id_events(base_txt_dir,element,run) 
	* events_and_telescopes(base_txt_dir,element,run) 
	* events_and_energy(base_txt_dir,element,run) 
	* big_input_img(tels,element,run,base_txt_dir,base_npy_dir,return_energies=False) 
	* mult_runs_big_input_img(tels,element,runs,base_txt_dir,base_npy_dir,return_energies=False) 
	* create_lista_list_runs(num_events,init_events=None,random_select=False,elementos=None,max_runs=None) 
	* data_set_longinput(tels,runs_list,base_txt_dir,base_npy_dir,labels=None,elements=None,test_size=0.2) 
	* cargar_datos(labels,tels=None) 
	* get_common_events(npy_element_dir,tels=None,run=None) 
	* fill_holes(npy) 
	* load_data(npy_dir_element,tels,runs,indices_events=None,only_names=False,ending=".npy") 
	* load_dataset_ensemble(base_dir,elementos_clasif,pre_name_folders="npy_",telescopios=None,lista_list_runs=None,elementos=None,test_size=0.2,same_quant=False) 
	* load_dataset_completo(npy_base_dir,main_list_runs,telescopes,labels_asign=None,elements=None,pre_name_folders="npy_",test_size=0.2,same_quant="same",verbose=True,fill=False,categorical=True) 
	* create_main_list_runs(num_events,init_events=None,random_select=False,elementos=None,max_runs=None) 
* data_EDA_first_sight.py
	* pos_telescopes(txt_with_the_Data,configurations_names,plotted=True,telescope_ranges=None,plot_save_dir=None) 
	* analysis_npy_files_sep(output_dir,npy_dir=None,npy_list=None, ) 
	* analysis_npy_files_conjunt(output_dir,npy_dir=None,npy_list=None) 
	* plot_variable_grouped(dt_dir,txt_dir,n_bins=16,variable_split_criteria=3) 
* list_funciones_in_files.py
	* find_files(base_dir) 
	* get_func_names_and_args(list_paths,get_args=False) 
	* master_get_funcs(base_dir,get_args=True) 
* model_results_representations.py
	* hex_repre(matrix=None,npy_file=None,savedir=None) 
	* print_conf_matrix(matrix,elements=None,sin_diag=True,save_dir=None) 
	* comp_and_diplay_conf_matrix(y_test,y_predict,elements=None,sin_diag=True,norm="true",save_dir=None) 
	* display_max_errores(x_test,y_test,y_predicted,true_index=None,predict_index=None,sort_max=False) 
	* plot_errors(x_test,y_test,y_predicho,true_index,predict_index,elementos=None,sort_max=False) 
* autoencoders_functions.py
* ensemble_model_functions.py
	* load_data(npy_dir,tels=None,runs=None,indices_runs=None,only_names=False,ending=".npy",test_size=0.2) 
	* get_common_events(npy_dir_base,tels=None,run=None) 
	* load_dataset_ensemble(base_dir,elementos_clasif,pre_name_folders="npy_",telescopios=None,lista_list_runs=None,elementos=None,test_size=0.2,same_quant=False) 
	* create_lista_list_runs(num_events,init_events=None,random_select=False,elementos=None,max_runs=None) 
	* load_dataset(base_dir,pre_name_folders="npy_",telescopios=None,list_runs=None,elementos=None,test_size=0.2,normalizacion_mal=False,same_quant=True) 
* unzipdata_and_first_treatments.py
	* extract_single_tar(dir_in,dir_out,new_folder=True) 
	* extract_multiple_tar(dir_folder_with_tars,dir_final_folder) 
	* unzip_gunzip(base_dir,final_dir=None,elements=None,folders=True) 
	* lista_dt(dt_dir) 
	* lista_txt(txt_dir) 
	* multiple_dt_2_npy(lista_archivos,npy_dir,limit_size=0.35,save_events_id=False,verbose=False) 
	* dt_2_npy(base_dir,npy_base_dir=None,elements=None,save_names_list=True) 
