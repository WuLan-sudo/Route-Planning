import math
import numpy as np
import matplotlib.pyplot as plt
def init(row,col,x_interval,y_interval,obstacle_list):
    graph_matrix = []
    for i in range(row*col):
        element = []
        if i == 0:
            if 1 not in obstacle_list:
                element.append(1)
            if col not in obstacle_list:
                element.append(col)
        elif i == col-1:
            if col-2 not in obstacle_list:
                element.append(col-2)
            if 2*col-1 not in obstacle_list:
                element.append(2*col-1)
        elif i == col*(row-1):
            if col*(row-2) not in obstacle_list:
                element.append(col*(row-2))
            if col*(row-1)+1 not in obstacle_list:
                element.append(col*(row-1)+1)
        elif i == col*row-1:
            if col*(row-1)-1 not in obstacle_list:
                element.append(col*(row-1)-1)
            if col*row-2 not in obstacle_list:
                element.append(col*row-2)
        elif i%col == 0:
            if i-col not in obstacle_list:
                element.append(i-col)
            if i+1 not in obstacle_list:
                element.append(i+1)
            if i+col not in obstacle_list:
                element.append(i+col)
        elif i%col == col-1:
            if i-col not in obstacle_list:
                element.append(i-col)
            if i-1 not in obstacle_list:
                element.append(i-1)
            if i+col not in obstacle_list:
                element.append(i+col)
        elif i<col:
            if i-1 not in obstacle_list:
                element.append(i-1)
            if i+1 not in obstacle_list:
                element.append(i+1)
            if i+col not in obstacle_list:
                element.append(i+col)
        elif i>(row-1)*col:
            if i-col not in obstacle_list:
                element.append(i-col)
            if i-1 not in obstacle_list:
                element.append(i-1)
            if i+1 not in obstacle_list:
                element.append(i+1)
        else:
            if i-col not in obstacle_list:
                element.append(i-col)
            if i-1 not in obstacle_list:
                element.append(i-1)
            if i+1 not in obstacle_list:
                element.append(i+1)
            if i+col not in obstacle_list:
                element.append(i+col)
        graph_matrix.append(element)
    # print(graph_matrix)
    plot_graph(row,col,x_interval,y_interval,obstacle_list)
    return graph_matrix

def obstacle_init(row,col,x_interval,y_interval,ob_x,ob_y,ob_r):
    obstacle_list = []
    for i in range(row*col):
        tem = 0
        contour = contour_point(i,row,col,x_interval,y_interval)
        for point in contour:
            for ele in range(len(ob_x)):
                if (point[0]-ob_x[ele])**2 + (point[1]-ob_y[ele])**2 < ob_r[ele]**2:
                    tem = 1
                    # print('obstacl',i,'inner_circle_contour_point',point)
                    break
            if tem == 1:
                break
        if tem == 1:
            obstacle_list.append(i)
    return obstacle_list

def center_point(n,row,col,x_interval,y_interval):
    x = np.linspace(x_interval[0],x_interval[1],col+1)
    y = np.linspace(y_interval[1],y_interval[0],row+1)
    return (0.5*(x[n%col]+x[n%col+1]), 0.5*(y[n/col]+y[n/col+1]))

def contour_point(n,row,col,x_interval,y_interval):
    x = np.linspace(x_interval[0],x_interval[1],col+1)
    y = np.linspace(y_interval[1],y_interval[0],row+1)
    contour_list = []
    contour_list.append([float(x[n%col]),float(y[n/col])])
    contour_list.append([float(x[n%col+1]),float(y[n/col])])
    contour_list.append([float(x[n%col]),float(y[n/col+1])])
    contour_list.append([float(x[n%col+1]),float(y[n/col+1])])
    for i in [0.2,0.4,0.6,0.8]:
        contour_list.append([i*x[n%col]+(1-i)*x[n%col+1],y[n/col]])
        contour_list.append([i*x[n%col]+(1-i)*x[n%col+1],y[n/col+1]])
        contour_list.append([x[n%col],i*y[n/col]+(1-i)*y[n/col+1]])
        contour_list.append([x[n%col+1],i*y[n/col]+(1-i)*y[n/col+1]])
    contour_list.append(center_point(n,row,col,x_interval,y_interval))
    return contour_list

def plot_graph(row,col,x_interval,y_interval,obstacle_list,route1 = [],route2 = []):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(x_interval[0],x_interval[1],col+1)
    y = np.linspace(y_interval[1],y_interval[0],row+1)
    for i in range(len(x)):
        ax.plot((x[i],x[i]),(y[0],y[row]),'k')
    for i in range(len(y)):
        ax.plot((x[0],x[col]),(y[i],y[i]),'k')
    for i in obstacle_list:
        x_1 = np.linspace(x[i%col],x[i%col+1],50)
        y_1 = y[i/col]
        y_2 = y[i/col + 1]
        ax.fill_between(x_1, y_1, y_2, where=y_1 > y_2,facecolor='green',alpha=0.3)
    a = contour_point(0,row,col,x_interval,y_interval)
    listx = [i[0] for i in a]
    listy = [i[1] for i in a]
    if route1 != []:
        route_x = []
        route_y = []
        for i in route1:
            pt = center_point(i,row,col,x_interval,y_interval)
            # ax.scatter(pt[0],pt[1],'r')
            route_x.append(pt[0])
            route_y.append(pt[1])
        ax.plot(route_x,route_y,'b',linestyle='--',marker='o')
    if route2 != []:
        route_x = []
        route_y = []
        for i in route2:
            pt = center_point(i,row,col,x_interval,y_interval)
            # ax.scatter(pt[0],pt[1],'r')
            route_x.append(pt[0])
            route_y.append(pt[1])
        ax.plot(route_x,route_y,'r',linestyle='--',marker='>')
            
    # ax.scatter(listx,listy,'b')
    plt.show()


def Dijikstra(row,col,start,end,graph_matrix,obstacle_list):
    min_route_len = []
    route_list = []
    new_added_list = []
    new_added_list.append(start)
    for i in range(row*col):
        min_route_len.append(-1)
        route_list.append([])
    for j in obstacle_list:
        min_route_len[j] = 0
    min_route_len[start] = 0
    route_list[start].append(start)
    i = 0
    while -1 in min_route_len:
        new_added_list2 = []
        i += 1
        for ele in new_added_list:
            # print(i,"new added",ele)
            for neigh in graph_matrix[ele]:
                if neigh not in new_added_list2:
                    new_added_list2.append(neigh)
                if min_route_len[neigh] == -1:
                    min_route_len[neigh] = min_route_len[ele] + 1
                    for temp in route_list[ele]:
                        route_list[neigh].append(temp)
                    route_list[neigh].append(neigh)
                    # print('neighbour',neigh,'min_route_len',min_route_len[neigh],'route',route_list[neigh])
        new_added_list = new_added_list2
    # print(route_list)
    return route_list[end]

def Dijikstra_2UGV_version(row,col,x_interval,y_interval,ob_x,ob_y,ob_r):
    obstacle_list = obstacle_init(row,col,x_interval,y_interval,ob_x,ob_y,ob_r)
    graph_matrix = init(row,col,x_interval,y_interval,obstacle_list)
    start1 = input("Input the starting block for vehicle 1:")
    end1 = input("Input the ending block for vehicle 1:")
    start2 = input("Input the starting block for vehicle 2:")
    end2 = input("Input the ending block for vehicle 2:")
    route_list1 = Dijikstra(row,col,start1,end1,graph_matrix,obstacle_list)
    route_list2 = Dijikstra(row,col,start2,end2,graph_matrix,obstacle_list)
    plot_graph(row,col,x_interval,y_interval,obstacle_list,route_list1,route_list2)

# a = obstacle_init(3,3,[0,3],[0,3],[1.5],[1.5],[0.5])
# G = init(3,3,[0,3],[0,3],a)
# Dijikstra(3,3,0,8,G,a)
Dijikstra_2UGV_version(20,20,[0,20],[0,20],[3,14,19],[3,14,19],[2,3,1])
#four input: 381  53  0  138
