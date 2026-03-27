package example

import (
	"fmt"
	"ml/clustering/kmeans"
	"ml/common/imgreader"
)

func RunKMeans() {
	path := "images/image.png"
	pixel, err := imgreader.GetPixelFromImage(path)

	if err != nil {
		panic(err)
	}

	options := kmeans.KMeansOptions{
		Dataset:   pixel,
		K:         16,
		Threshold: 0.02,
	}

	clusters, iteration := kmeans.KMeans(options)
	fmt.Println("Iterations: ", iteration)
	kmeans.Save(clusters, "chart/output/cluster.csv")
}
