#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

int main(){
  if(fork()==0){
    printf("poop 1\n");
  }else{
    printf("poop 2\n");
  }
  
}
