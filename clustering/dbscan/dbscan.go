package dbscan

import (
	"encoding/csv"
	"ml/common/imgreader"
	"os"
	"slices"
	"strconv"
)

type DBSCANOptions struct {
	Dataset   []imgreader.Pixel
	Epsilon   float64
	MinPoints int
}

type DBSCANOutput struct {
	Labels         []int
	ClustersAmount int
}

func DBScan(options DBSCANOptions) DBSCANOutput {
	clusterCounter := 0
	size := len(options.Dataset)

	// if labels[idx] > 0 than the current point
	// is part of a dense cluster. Otherwise, if
	// labels[idx] == -1, the current point is "noise"
	// If labels[idx] == 0 than the current node is not
	// visited

	labels := make([]int, size)

	for idx, p := range options.Dataset {
		if labels[idx] > 0 {
			continue
		}

		neighboors := findNeighboors(p, options)
		if len(neighboors) < options.MinPoints {
			// p it's an outlier
			labels[idx] = -1
			continue
		}

		clusterCounter += 1
		labels[idx] = clusterCounter

		// Expand cluster
		for iterIdx, nIdx := range neighboors {
			if nIdx == idx || labels[iterIdx] > 0 {
				continue
			}

			if labels[iterIdx] == -1 || labels[iterIdx] == 0 {
				// A noise point, or a directly rechable point
				// is now part of a cluster
				labels[iterIdx] = clusterCounter
			}

			// Find new core points and add them inside the "queue"
			expandNeighboors := findNeighboors(options.Dataset[nIdx], options)
			if len(expandNeighboors) >= options.MinPoints {
				neighboors = addNeighboors(neighboors, expandNeighboors)
			}
		}
	}

	return DBSCANOutput{
		Labels:         labels,
		ClustersAmount: clusterCounter,
	}
}

func Save(output DBSCANOutput, dataset []imgreader.Pixel, path string) {
	file, err := os.Create(path)

	if err != nil {
		panic(err)
	}

	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for idx := range output.Labels {
		label := output.Labels[idx]
		pixel := dataset[idx]

		data := []string{
			strconv.FormatInt(int64(pixel.R), 10),
			strconv.FormatInt(int64(pixel.G), 10),
			strconv.FormatInt(int64(pixel.B), 10),
			strconv.FormatInt(int64(pixel.A), 10),
			strconv.FormatInt(int64(label), 10),
		}

		err := writer.Write(data)
		if err != nil {
			panic(err)
		}
	}

}

func findNeighboors(point imgreader.Pixel, options DBSCANOptions) []int {
	neighboors := make([]int, 0, options.MinPoints)

	for idx, q := range options.Dataset {
		if point.Distance(q) <= options.Epsilon {
			neighboors = append(neighboors, idx)
		}
	}

	return neighboors
}

func addNeighboors(where []int, src []int) []int {
	for _, v := range src {
		if !slices.Contains(where, v) {
			where = append(where, v)
		}
	}

	return where
}
