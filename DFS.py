class DFS():
    ## Constructor de la clase DFS
    def __init__(self, arr):
        self.arr = arr
        pass
    
    ## Extrae la posicion de un espacio vacio
    def find_empty_location(self,arr,l):
        for row in range(9):
            for col in range(9):
                if(arr[row,col]==0):
                    l[0]=row
                    l[1]=col
                    return True
        return False

    ## Devuelve verdadero si se encuentra el numero en la fila
    def used_in_row(self,arr,row,num):
        for i in range(9):   
            if(arr[row,i] == num):  
                return True
        return False

    ## Devuelve verdadero si se encuentra el numero en la columna
    def used_in_col(self,arr,col,num):
        for i in range(9):  
            if(arr[i,col] == num):  
                return True
        return False

    ## Devuelve verdadero si se encuentra en el cajon de 3x3
    def used_in_box(self,arr,row,col,num):
        for i in range(3):
            for j in range(3):
                if(arr[i+row,j+col] == num):     
                    return True 
        return False

    ## Evalua que todas las condiciones del sudoku se cumplan
    def check_location_is_safe(self,arr,row,col,num):
        return (not self.used_in_row(arr,row,num) 
                and not self.used_in_col(arr,col,num) 
                and not self.used_in_box(arr,row-row%3,col-col%3,num))
        
    ## Resuelve el sudoku
    def solve_sudoku(self,arr):
        l=[0,0]     
        if(not self.find_empty_location(arr,l)):
            return True     
        row=l[0]
        col=l[1]     
        for num in range(1,10): 
            if(self.check_location_is_safe(arr,row,col,num)): 
                arr[row,col] = num
                if(self.solve_sudoku(arr)): 
                    return True
                arr[row,col] = 0 
        return False
