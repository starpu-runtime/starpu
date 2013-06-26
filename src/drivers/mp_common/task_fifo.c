#include "task_fifo.h"

void task_fifo_init(struct mp_task_fifo* fifo){
  fifo->first = fifo->last = NULL;
  pthread_mutex_init(&(fifo->mutex), NULL);
}

int task_fifo_is_empty(struct mp_task_fifo* fifo){
  return fifo->first == NULL;
}

void task_fifo_append(struct mp_task_fifo* fifo, struct mp_task * task){
  pthread_mutex_lock(&(fifo->mutex));
  if(task_fifo_is_empty(fifo)){
    fifo->first = fifo->last = task;
  }
  else{
    fifo->last->next = task;
    fifo->last = task;
  }
  task->next = NULL;
  pthread_mutex_unlock(&(fifo->mutex));
}

void task_fifo_pop(struct mp_task_fifo* fifo){
  pthread_mutex_lock(&(fifo->mutex));
  if(!task_fifo_is_empty(fifo)){
    if(fifo->first == fifo->last)
      fifo->first = fifo->last = NULL;
    else
      fifo->first = fifo->first->next; 
  }
  pthread_mutex_unlock(&(fifo->mutex));
}
