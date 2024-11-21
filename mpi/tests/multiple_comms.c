#include <starpu_mpi.h>
#include "helper.h"

int main(int argc, char ** argv){
	//we will only use three processes for now
	int my_rank, size, rsize;
	int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	MPI_Comm new_world;
	MPI_Comm_dup(MPI_COMM_WORLD, &new_world);
	MPI_Comm_set_name(new_world, "application duplicated comm");
	starpu_mpi_comm_register(new_world);
	if(size < 2){
		fprintf(stderr, "We need at least 2 processes.\n");
    starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}
	//we want an even number of processes
	rsize = size%2 != 0 ? size-1 : size;
	if(my_rank < rsize ){
		starpu_data_handle_t send_handles[rsize];
		starpu_data_handle_t recv_handles[rsize];
		
		int * stabs[rsize];
		int * rtabs[rsize];
		for(int i = 0; i < rsize; i++){
			stabs[i] = malloc(sizeof(int)*100);
			rtabs[i] = malloc(sizeof(int)*100);
		}
		for(int i = 0; i < 100; i++)
			stabs[my_rank][i] = my_rank * 100 + i;
		for(int i = 0; i < rsize; i++){
			starpu_variable_data_register( send_handles + i, STARPU_MAIN_RAM, (uintptr_t)stabs[i], 100* sizeof(int));
			starpu_variable_data_register( recv_handles + i, STARPU_MAIN_RAM, (uintptr_t)rtabs[i], 100 *sizeof(int));
		}
		starpu_mpi_req reqs[4];

		if(my_rank%2 == 0){
			starpu_mpi_isend(send_handles[my_rank], &reqs[0], (my_rank+1)%rsize, 12, MPI_COMM_WORLD);
			starpu_mpi_isend(send_handles[my_rank], &reqs[1], my_rank==0?rsize-1:my_rank-1, 12, new_world);

			starpu_mpi_irecv(recv_handles[(my_rank+1)%rsize], &reqs[2], (my_rank+1)%rsize, 13,MPI_COMM_WORLD);
			starpu_mpi_irecv(recv_handles[my_rank==0?rsize-1:my_rank-1], &reqs[3], my_rank==0?rsize-1:my_rank-1, 13,new_world);
		}
		else{
			starpu_mpi_irecv(recv_handles[my_rank==0?rsize-1:my_rank-1], &reqs[0], my_rank==0?rsize-1:my_rank-1, 12, MPI_COMM_WORLD);
			starpu_mpi_irecv(recv_handles[(my_rank+1)%rsize], &reqs[1], (my_rank+1)%rsize, 12, new_world);

			starpu_mpi_isend(send_handles[my_rank], &reqs[2], my_rank==0?rsize-1:my_rank-1, 13, MPI_COMM_WORLD);
			starpu_mpi_isend(send_handles[my_rank], &reqs[3], (my_rank+1)%rsize, 13, new_world);
		}
		int nb_req=4;
		while (nb_req)
		{
			int r=0;
			for(r=0 ; r<4 ; r++)
			{
				if (reqs[r])
				{
					int finished = 0;
					MPI_Status status;
					ret = starpu_mpi_test(&reqs[r], &finished, &status);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_test");
					STARPU_ASSERT(finished != -1);
					if (finished)
					{
						reqs[r] = NULL;
						nb_req--;
					}
				}
			}
		}
		for(int i = 0; i < rsize; i++){
			starpu_data_unregister(send_handles[i]);
			starpu_data_unregister(recv_handles[i]);
			free(stabs[i]);
			free(rtabs[i]);
		}
	}
	starpu_mpi_barrier(MPI_COMM_WORLD);
	starpu_mpi_shutdown();
	return 0;
}
