	f = fopen("Output_maxime/Data/DARTS/Nb_conflit_donnee.csv", "w");
	fprintf(f, "N,Nb conflits,Nb conflits critiques\n");
	fclose(f);
	
	f = fopen("Output_maxime/Data/DARTS/Choice_during_scheduling.csv", "w");
	fprintf(f, "N,Return NULL, Return task, Return NULL because main task list empty,Nb of random selection,nb_1_from_free_task_not_found\n");
	fclose(f);
	
	f = fopen("Output_maxime/Data/DARTS/Choice_victim_selector.csv", "w");
	fprintf(f, "N,victim_selector_refused_not_on_node,victim_selector_refused_cant_evict,victim_selector_return_refused,victim_selector_return_unvalid,victim_selector_return_data_not_in_planned_and_pulled,victim_evicted_compteur,victim_selector_compteur,victim_selector_return_no_victim,victim_selector_belady\n");
	fclose(f);
	
	f = fopen("Output_maxime/Data/DARTS/Misc.csv", "w");
	fprintf(f, "N,Nb refused tasks,Nb new task initialized\n");
	fclose(f);
	
	f = fopen("Output_maxime/Data/DARTS/DARTS_time.csv", "w");
	fprintf(f, "N,time_total_selector,time_total_evicted,time_total_belady,time_total_schedule,time_total_choose_best_data,time_total_fill_planned_task_list,time_total_initialisation,time_total_randomize, time_total_pick_random_task,time_total_least_used_data_planned_task,time_total_createtolasttaskfinished\n");
	fclose(f);
