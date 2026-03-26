package clustering

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand/v2"
	"ml/common/imgreader"
	"os"
	"slices"
	"strconv"
)

type Cluster struct {
	Centroide     imgreader.Pixel
	ClusterNumber int
	Items         []imgreader.Pixel
}

type KMeansOptions struct {
	Dataset   []imgreader.Pixel
	K         int
	Threshold float64
}

func KMeans(options KMeansOptions) ([]Cluster, int) {
	if options.K > len(options.Dataset) {
		panic("k must be less or equal than the number of observations")
	}

	means := initialize(options.Dataset, options.K)
	clusters := createCluster(means)
	stop := false
	iteration := 0

	for !stop {
		for _, pixel := range options.Dataset {
			minDistance := math.Inf(0)
			clusterRef := -1

			for cIdx, cluster := range clusters {
				distance := math.Pow(cluster.Centroide.Distance(pixel), 2)

				if distance < minDistance {
					minDistance = distance
					clusterRef = cIdx
				}
			}

			clusters[clusterRef].Items = append(clusters[clusterRef].Items, pixel)
		}

		newClusters := nextClusters(clusters)
		distances := diffClusters(clusters, newClusters)
		stop = canStop(distances, options.Threshold)
		clusters = newClusters
		iteration += 1
	}

	return clusters, iteration
}

func PrintClusters(clusters []Cluster, limit int) {
	for _, c := range clusters {
		fmt.Printf("Cluster %v\n", c.ClusterNumber)
		fmt.Printf("  Centroid %v\n", c.Centroide)

		for i := range len(c.Items[:limit]) {
			fmt.Printf("  %v => %v\n", i, c.Items[i])
		}

		fmt.Printf("-----------------------------\n")
	}
}

func Save(clusters []Cluster, path string) {
	file, err := os.Create(path)

	if err != nil {
		panic(err)
	}

	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for _, c := range clusters {
		clusterData := []string{
			strconv.FormatInt(int64(c.Centroide.R), 10),
			strconv.FormatInt(int64(c.Centroide.G), 10),
			strconv.FormatInt(int64(c.Centroide.B), 10),
			strconv.FormatInt(int64(c.Centroide.A), 10),
		}

		for _, item := range c.Items {
			itemData := []string{
				strconv.FormatInt(int64(item.R), 10),
				strconv.FormatInt(int64(item.G), 10),
				strconv.FormatInt(int64(item.B), 10),
				strconv.FormatInt(int64(item.A), 10),
			}

			record := append(clusterData, itemData...)
			err := writer.Write(record)
			if err != nil {
				panic(err)
			}
		}

	}

}

// cr, cg, cb, ca, ir, ig, ib, ia

func nextClusters(clusters []Cluster) []Cluster {
	newClusters := make([]Cluster, 0, len(clusters))

	for _, c := range clusters {
		sum := imgreader.Pixel{}

		for _, item := range c.Items {
			sum = sum.Sum(item)
		}

		newMean := sum.Div(len(c.Items))
		newCluster := Cluster{
			Centroide:     newMean,
			Items:         c.Items,
			ClusterNumber: c.ClusterNumber,
		}

		newClusters = append(newClusters, newCluster)
	}

	return newClusters
}

func createCluster(means []imgreader.Pixel) []Cluster {
	clusters := make([]Cluster, 0, len(means))

	for idx, mean := range means {
		cluster := Cluster{
			Centroide:     mean,
			ClusterNumber: idx,
		}

		clusters = append(clusters, cluster)
	}

	return clusters
}

func initialize(dataset []imgreader.Pixel, k int) (means []imgreader.Pixel) {
	// The Initialization step
	// implement the Forgy method which
	// randomly chooses k observations from the
	// dataset and uses these as the initial means

	acc := make([]int, 0, k)
	generated := 0
	means = make([]imgreader.Pixel, 0, k)

	for generated < k {
		idx := rand.IntN(len(dataset))

		if slices.Contains(acc, idx) {
			continue
		}

		acc = append(acc, idx)
		means = append(means, dataset[idx])
		generated += 1
	}

	return means
}

func diffClusters(oldClusters []Cluster, newClusters []Cluster) []float64 {
	distances := make([]float64, 0, len(oldClusters))

	for i := range len(oldClusters) {
		oldC := oldClusters[i]
		newC := newClusters[i]
		d := oldC.Centroide.Distance(newC.Centroide)

		distances = append(distances, d)
	}

	return distances
}

func canStop(distances []float64, threshold float64) bool {
	for _, d := range distances {
		if d > threshold {
			return false
		}
	}

	return true
}
