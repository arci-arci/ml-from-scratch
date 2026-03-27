package example

import (
	"ml/clustering/dbscan"
	"ml/common/imgreader"
)

func RunDBScan() {
	path := "images/scaled_image.png"
	pixel, err := imgreader.GetPixelFromImage(path)

	if err != nil {
		panic(err)
	}

	options := dbscan.DBSCANOptions{
		Dataset:   pixel,
		Epsilon:   150,
		MinPoints: 35,
	}

	output := dbscan.DBScan(options)
	dbscan.Save(output, pixel, "chart/output/dbscan.csv")
}
