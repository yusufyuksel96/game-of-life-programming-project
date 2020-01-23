#!/usr/bin/env python3

#name : Yusuf Yüksel
#compilation status : Compiling
#working status : Working
#usage : mpirun -np [M] --oversubscribe python3 game.py input.txt output.txt [T]
#periodic and checkered version

from mpi4py import MPI
import numpy as np
import sys
import math

#update of cell state according to game of life rules
def cell_update(temp_cell,cell,cell_row):

    for i in range(1,cell_row+1):
                
        for j in range(1,cell_row+1):

            #calculating sum of neighbors
            sum_creature = 0 
            sum_creature += temp_cell[i][j-1] #right neighbor
            sum_creature += temp_cell[i][j+1] #left neighbor
            sum_creature += temp_cell[i-1][j] #up  neighbor
            sum_creature += temp_cell[i+1][j] #dowm neighbor
            sum_creature += temp_cell[i-1][j-1] #left up neighbor
            sum_creature += temp_cell[i-1][j+1] #right up neighbor
            sum_creature += temp_cell[i+1][j+1] #right down neighbor
            sum_creature += temp_cell[i+1][j-1] #left down neighbor

            #game of life conditions
            if(sum_creature<2 and temp_cell[i][j] == 1):
                cell[i-1][j-1] = 0
            if(sum_creature > 3 and temp_cell[i][j] == 1):
                cell[i-1][j-1] = 0
            if(sum_creature == 3 and temp_cell[i][j] == 0):
                cell[i-1][j-1] = 1

#get row of rank
def row_r(c_size,rank):
    row = int((rank-1)/c_size)
    return row

#get rank of right cell
def right(rank,row,c_size):
    right = (rank%c_size) + (row*c_size) +1
    return right

#get rank of left cell
def left(rank,row,c_size):
    left= ((rank-2) %c_size) +(row*c_size)+1
    return left

#get rank of up cell
def up(rank,c_size):
    up= (rank-c_size)% (c_size*c_size)
    if up == 0:
        up = c_size * c_size
    return up

#get rank of down cell
def down(rank,c_size):
    down= (rank+c_size)%(c_size*c_size)
    if down == 0:
        down = c_size * c_size
    return down

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#In order to find number of worker processes
size = comm.Get_size()-1 

input_file = sys.argv[1]
output_file = sys.argv[2]
time_steps = int(sys.argv[3])

table = np.loadtxt(input_file, dtype = int) 

#get number of rows of table which equal number of columns
row_table = np.size(table,0)

#c_size equals square root of number of worker processes
c_size = int(math.sqrt(size))

#calculate checkered version of rows and colums of cell
cell_row = int(row_table/c_size)

rank_counter = 1

if rank == 0:

    for i in range(0,c_size):
        for j in range(0,c_size): 
            #sending cells to worker process in terms of checkered version
            comm.send(table[i*cell_row : (i+1) * cell_row , j*cell_row : (j+1) * cell_row], dest = rank_counter )
            rank_counter+=1

    rank_counter= 1

    for i in range(0,c_size):
        for j in range(0,c_size):
            #receiving last version of cells from worker process 
            table[i*cell_row : (i+1) * cell_row , j*cell_row : (j+1) * cell_row] = comm.recv(source = rank_counter)
            rank_counter+=1

    #printing results to output file
    np.savetxt(output_file, table,fmt ='%1d')

else :

    cell = comm.recv(source=0)
    row = row_r(c_size,rank)
    
    if((rank%2) == 1) :

        for step in range(0,time_steps):

            temp_cell = np.zeros((cell_row + 2, cell_row + 2),dtype=int) # in order to calculate results of cell state for each iteration
            temp_cell[1:cell_row+1,1:cell_row+1] = cell 
        
        #right
            comm.send(cell[:,cell_row-1] , dest = right(rank,row,c_size) ,tag =1)   
            temp_cell[1:(cell_row+1) ,-1] =comm.recv(source=right(rank,row,c_size),tag =2)
       
        
        #left
            comm.send(cell[:,0],dest=left(rank,row,c_size),tag=3)
            temp_cell[1:(cell_row+1),0]=comm.recv(source=left(rank,row,c_size),tag=4)
        
        #left up
            temp_rank = up(rank,c_size)
            temp_row = row_r(c_size,temp_rank)
            comm.send(cell[0][0],dest=left(temp_rank,temp_row,c_size),tag=5)
            temp_cell[0][0]= comm.recv(source=left(temp_rank,temp_row,c_size),tag=6)
        
        #right up 
            temp_rank = up(rank,c_size)
            temp_row = row_r(c_size,temp_rank)
            comm.send(cell[0][cell_row-1],dest=right(temp_rank,temp_row,c_size),tag=7)
            temp_cell[0][cell_row+1] = comm.recv(source=right(temp_rank,temp_row,c_size),tag=8)
              
        #left down
            temp_rank = down(rank,c_size)
            temp_row = row_r(c_size,temp_rank)
            comm.send(cell[cell_row-1][0],dest=left(temp_rank,temp_row,c_size),tag=9)
            temp_cell[cell_row+1][0]=comm.recv(source=left(temp_rank,temp_row,c_size),tag=10)

        #right down
            temp_rank = down(rank,c_size)
            temp_row = row_r(c_size,temp_rank)
            comm.send(cell[cell_row-1][cell_row-1], dest=right(temp_rank,temp_row,c_size),tag=11)
            temp_cell[cell_row+1][cell_row+1]=comm.recv(source=right(temp_rank,temp_row,c_size),tag=12)

        #down
            comm.send(cell[cell_row-1] , dest= down(rank,c_size),tag=13)
            temp_cell[0:1,1:cell_row+1] = comm.recv(source =up(rank,c_size),tag=13 )
            
        #up
            comm.send(cell[0] , dest= up(rank,c_size),tag=14)
            temp_cell[cell_row+1 : cell_row+2 , 1:cell_row+1] = comm.recv(source = down(rank,c_size),tag=14)

            cell_update(temp_cell,cell,cell_row) #updating a cell in terms of neighbors
            

        comm.send(cell,dest = 0)

    else:
        
        for step in range(0,time_steps):

            temp_cell = np.zeros((cell_row + 2, cell_row + 2),dtype=int) # in order to calculate results of cell state for each iteration
            temp_cell[1:cell_row+1,1:cell_row+1] = cell 

            #left
            temp_cell[1:(cell_row+1),0]=comm.recv(source=left(rank,row,c_size),tag=1)
            comm.send(cell[:,0],dest=left(rank,row,c_size),tag=2)
            
            #right
            temp_cell[1:cell_row+1 , -1] =comm.recv(source=right(rank,row,c_size),tag=3)
            comm.send(cell[:,(cell_row-1)],dest=right(rank,row,c_size),tag=4)
            
            #right down
            temp_rank = down(rank,c_size)
            temp_row = row_r(c_size,temp_rank)
            temp_cell[cell_row+1][cell_row+1]=comm.recv(source=right(temp_rank,temp_row,c_size),tag=5)
            comm.send(cell[cell_row-1][cell_row-1],dest=right(temp_rank,temp_row,c_size),tag=6)
            
            #left down
            temp_rank = down(rank,c_size)
            temp_row = row_r(c_size,temp_rank)
            temp_cell[cell_row+1][0]=comm.recv(source=left(temp_rank,temp_row,c_size),tag=7)
            comm.send(cell[cell_row-1][0] , dest=left(temp_rank,temp_row,c_size),tag=8)

            #right up
            temp_rank = up(rank,c_size)
            temp_row = row_r(c_size,temp_rank)
            temp_cell[0][cell_row+1] = comm.recv(source=right(temp_rank,temp_row,c_size),tag=9)
            comm.send(cell[0][cell_row-1],dest=right(temp_rank,temp_row,c_size),tag=10)
  
            #left up
            temp_rank = up(rank,c_size)
            temp_row = row_r(c_size,temp_rank)
            temp_cell[0][0]= comm.recv(source=left(temp_rank,temp_row,c_size),tag=11)
            comm.send(cell[0][0],dest=left(temp_rank,temp_row,c_size),tag=12)
            
            #down
            comm.send(cell[cell_row-1] , dest= down(rank,c_size),tag=15)
            temp_cell[0:1,1:cell_row+1] = comm.recv(source =up(rank,c_size),tag=15 )
            
            #up
            comm.send(cell[0] , dest= up(rank,c_size),tag=16)
            temp_cell[cell_row+1 : cell_row+2 , 1:cell_row+1] = comm.recv(source = down(rank,c_size),tag=16)

            cell_update(temp_cell,cell,cell_row) #updating a cell in terms of neighbors

        comm.send(cell,dest = 0)




        
            
        










        

