import matplotlib.pyplot as plt
import csv

def read_csv(path: str):
    outliers_x: list[int] = []
    outliers_y: list[int] = []

    x: list[int] = []
    y: list[int] = []


    with open(path) as csvfile:
        reader = csv.reader(csvfile)        
        for row in reader:
            r, g, _, _, c = row

            if c == "-1":
                outliers_x.append(int(r))
                outliers_y.append(int(g))
                continue

            x.append(int(r))
            y.append(int(g))


    return (x, y, outliers_x, outliers_y)



if __name__ == "__main__": 
    x, y, ox, oy = read_csv("output/dbscan.csv")
    colors = ["#21f573", "#333333"]

    fig, ax = plt.subplots()

    ax.text(0, 200, r'$\epsilon=25,\ minP=700$')

    ax.set_title("DBSCAN")
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")

    ax.scatter(x, y, s=7, linewidths=0, c=colors[0], label="points")
    ax.scatter(ox, oy, s=7, linewidths=0, c=colors[1], label="noise")
    ax.legend()
    plt.show()