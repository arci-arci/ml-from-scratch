import matplotlib.pyplot as plt
import csv
from itertools import groupby

COLORS = [
    '#1f77b4', 
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    "#5f2013",
    "#611249",
    "#3a3939",
    "#7a7a16",
    '#17becf',
    '#eaf619',
    "#0e7179",
    "#0c6834",
    "#a01a8e",
    "#751236",
    "#466e0e"
]

type CsvData = list[tuple[str, list[list[str]]]]

def read_csv(path: str) -> CsvData:
    data: list[list[str]] = []

    with open(path) as csvfile:
        reader = csv.reader(csvfile)        
        for row in reader:
            rc, gc, bc, ac, r, g, _, _ = row
            center = f"{rc}#{gc}#{bc}#{ac}"
            item = f"{r}#{g}"

            data.append([center, item])

    data.sort(key=lambda x: x[0])
    grouped_data = groupby(data, lambda x: x[0])

    return [(key, list(data)) for (key, data) in grouped_data]

def convert(data: CsvData):
    cx: list[int] = []
    cy: list[int] = []

    x: list[list[int]] = []
    y: list[list[int]] = []


    for key, items in data:
        rc, gc, _, _ = key.split("#")
        cx.append(int(rc))
        cy.append(int(gc))

        local_x: list[int] = []
        local_y: list[int] = []


        for item in items: 
            r, g = item[1].split("#")
            local_x.append(int(r))
            local_y.append(int(g))

        x.append(local_x)
        y.append(local_y)
    
    return cx, cy, x, y


if __name__ == "__main__":
    data = read_csv("output/cluster.csv")
    cx, cy, x, y = convert(data)
    
    fig, ax = plt.subplots()
    ax.set_title("K-Means Segmentation")
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")

    ax.text(0, 200, r'$k=16$')

    for k, color in list(zip(range(len(cx)), COLORS)):
        ax.scatter(x[k], y[k], s=1.5, linewidths=0, c=color)

    for a, b in list(zip(cx, cy)):
        ax.scatter(a, b, s=65.0, c="#0c0217", linewidths=1.5, edgecolors="#e4def1")

    
    plt.show()