package example

import (
	"fmt"
	"ml/common/imgreader"
)

func RunKMeans() {
	path := "images/image.png"
	pixel, err := imgreader.GetPixelFromImage(path)

	if err != nil {
		panic(err)
	}

	fmt.Println(pixel)
}
