package imgreader

import (
	"image"
	"image/png"
	"io"
	"os"
)

type Pixel struct {
	R int
	G int
	B int
}

func GetPixelFromImage(path string) ([][]Pixel, error) {
	image.RegisterFormat("png", "png", png.Decode, png.DecodeConfig)
	file, err := os.Open(path)

	if err != nil {
		return nil, err
	}

	defer file.Close()
	pixels, err := getPixels(file)

	if err != nil {
		return nil, err
	}

	return pixels, nil
}

func getPixels(file io.Reader) ([][]Pixel, error) {
	img, _, err := image.Decode(file)

	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	var pixels [][]Pixel

	for y := range height {
		var row []Pixel

		for x := range width {
			r, g, b, _ := img.At(x, y).RGBA()
			row = append(row, rgbaToPixel(r, g, b))
		}

		pixels = append(pixels, row)
	}

	return pixels, nil
}

func rgbaToPixel(r uint32, g uint32, b uint32) Pixel {
	return Pixel{
		R: int(r / 257),
		G: int(g / 257),
		B: int(b / 257),
	}
}
