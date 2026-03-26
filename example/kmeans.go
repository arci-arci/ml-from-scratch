package example

import (
	"fmt"
	"ml/clustering"
	"ml/common/imgreader"
)

func RunKMeans() {
	path := "images/image.png"
	pixel, err := imgreader.GetPixelFromImage(path)

	if err != nil {
		panic(err)
	}

	options := clustering.KMeansOptions{
		Dataset:   pixel,
		K:         16,
		Threshold: 0.02,
	}

	clusters, iteration := clustering.KMeans(options)
	fmt.Println("Iterations: ", iteration)
	clustering.Save(clusters, "chart/output/cluster.csv")
}
